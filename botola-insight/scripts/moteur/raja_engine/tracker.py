"""Suivi multi-joueurs à l'intérieur d'un plan (type ByteTrack simplifié).

- Filtre de Kalman à vitesse constante sur (cx, cy, w, h).
- Compensation du mouvement de caméra (homographie image→image) avant association.
- Association en deux temps : détections fortes puis faibles (IoU + distance des pieds).
- AUCUNE ré-identification automatique entre deux plans : chaque changement de plan termine
  toutes les pistes. Relier deux pistes de plans différents exige une preuve (réponse humaine,
  numéro lu) gérée par identity.py. C'est volontairement prudent.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)


class KalmanBox:
    def __init__(self, box):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0], np.float64)
        self.P = np.diag([10, 10, 10, 10, 100, 100]).astype(np.float64)
        self.F = np.eye(6)
        self.F[0, 4] = self.F[1, 5] = 1.0
        self.Q = np.diag([1, 1, 1, 1, 4, 4]).astype(np.float64)
        self.Hm = np.zeros((4, 6))
        self.Hm[:4, :4] = np.eye(4)
        self.R = np.diag([4, 4, 9, 9]).astype(np.float64)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 2)
        self.x[3] = max(self.x[3], 4)

    def warp(self, H):
        """Applique le mouvement de caméra au centre et à la vitesse estimés."""
        if H is None:
            return
        p = np.array([self.x[0], self.x[1], 1.0])
        q = H @ p
        q = q[:2] / q[2]
        p2 = np.array([self.x[0] + self.x[4], self.x[1] + self.x[5], 1.0])
        q2 = H @ p2
        q2 = q2[:2] / q2[2]
        s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        self.x[0], self.x[1] = q
        self.x[4], self.x[5] = q2 - q
        self.x[2] *= s
        self.x[3] *= s

    def update(self, box):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        y = z - self.Hm @ self.x
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.Hm) @ self.P

    def box(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Track:
    _next = 1

    def __init__(self, box, score, t, frame_idx, shot_id):
        self.id = Track._next
        Track._next += 1
        self.kf = KalmanBox(box)
        self.shot_id = shot_id
        self.obs = []  # (t, frame_idx, box, score, matched)
        self.add(box, score, t, frame_idx)
        self.missed = 0
        self.active = True

    def add(self, box, score, t, frame_idx):
        self.obs.append((float(t), int(frame_idx), [float(v) for v in box], float(score)))

    @property
    def last_t(self):
        return self.obs[-1][0]


class ShotTracker:
    def __init__(self, high=0.45, low=0.15, iou_thr=0.2, max_missed_s=1.0, min_hits=3):
        self.high, self.low, self.iou_thr = high, low, iou_thr
        self.max_missed_s = max_missed_s
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self.shot_id = 0

    def new_shot(self, shot_id):
        self.finished.extend(self.tracks)
        self.tracks = []
        self.shot_id = shot_id

    def _match(self, tracks, boxes, thr):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        tb = np.array([tr.kf.box() for tr in tracks])
        iou = iou_matrix(tb, boxes)
        # distance des pieds normalisée par la hauteur (utile quand l'IoU tombe à 5 i/s)
        tf = np.stack([(tb[:, 0] + tb[:, 2]) / 2, tb[:, 3]], 1)
        bf = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
        hh = np.maximum((tb[:, 3] - tb[:, 1])[:, None], 8)
        dist = np.linalg.norm(tf[:, None] - bf[None], axis=2) / hh
        cost = 1.0 - iou + 0.5 * np.clip(dist, 0, 2)
        valid = (iou >= thr) | (dist <= 0.6)
        cost[~valid] = 1e3
        r, c = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(r, c) if cost[i, j] < 1e3]
        mt = {i for i, _ in matches}
        md = {j for _, j in matches}
        return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]

    def step(self, t, frame_idx, boxes: np.ndarray, scores: np.ndarray, cam_H=None) -> list[tuple[int, int]]:
        """Renvoie la liste (indice détection, id piste) des associations de cette image."""
        for tr in self.tracks:
            tr.kf.warp(cam_H)
            tr.kf.predict()
        hi = np.where(scores >= self.high)[0]
        lo = np.where((scores >= self.low) & (scores < self.high))[0]
        out = []
        m1, ut, ud = self._match(self.tracks, boxes[hi] if len(hi) else np.zeros((0, 4)), self.iou_thr)
        for i, j in m1:
            tr = self.tracks[i]
            d = hi[j]
            tr.kf.update(boxes[d])
            tr.add(boxes[d], scores[d], t, frame_idx)
            tr.missed = 0
            out.append((int(d), tr.id))
        rem = [self.tracks[i] for i in ut]
        m2, ut2, _ = self._match(rem, boxes[lo] if len(lo) else np.zeros((0, 4)), 0.35)
        for i, j in m2:
            tr = rem[i]
            d = lo[j]
            tr.kf.update(boxes[d])
            tr.add(boxes[d], scores[d], t, frame_idx)
            tr.missed = 0
            out.append((int(d), tr.id))
        matched_ids = {tid for _, tid in out}
        for tr in self.tracks:
            if tr.id not in matched_ids:
                tr.missed += 1
        for j in ud:
            d = hi[j]
            tr = Track(boxes[d], scores[d], t, frame_idx, self.shot_id)
            self.tracks.append(tr)
            out.append((int(d), tr.id))
        alive = []
        for tr in self.tracks:
            if t - tr.last_t > self.max_missed_s:
                self.finished.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        return out

    def close(self):
        self.finished.extend(self.tracks)
        self.tracks = []
        return [tr for tr in self.finished if len(tr.obs) >= self.min_hits]
