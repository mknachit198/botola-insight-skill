"""Détection de personnes et du ballon avec YOLOX (ONNX, licence Apache-2.0).

Exécuté par onnxruntime sur CPU : aucune dépendance Ultralytics, aucun appel réseau.
Les poids COCO génériques détectent « person » (classe 0) et « sports ball » (classe 32).
Ils ne sont pas spécialisés football : le rappel sur les petits joueurs et surtout sur
le ballon (4–8 px en 854×480) est limité ; il est mesuré par `evaluate.py`, pas supposé.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

PERSON = 0
BALL = 32


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int  # 0 personne, 32 ballon

    @property
    def box(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def foot(self):
        return ((self.x1 + self.x2) / 2.0, self.y2)


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    idx = cv2.dnn.NMSBoxes(
        [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
        [float(s) for s in scores], 0.0, thr)
    return [int(i) for i in np.array(idx).reshape(-1)]


class YoloxDetector:
    def __init__(self, model_path: str, threads: int | None = None):
        import onnxruntime as ort  # importé ici pour que le reste du moteur fonctionne sans
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2))
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = (int(inp.shape[2]), int(inp.shape[3]))  # h, w
        self._grids = self._make_grids()
        self.model_path = model_path

    def _make_grids(self):
        h, w = self.size
        grids, strides = [], []
        for s in (8, 16, 32):
            hs, ws = h // s, w // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hs * ws, 1), s))
        return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)

    def _infer(self, img: np.ndarray):
        h, w = self.size
        r = min(h / img.shape[0], w / img.shape[1])
        resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
        pad = np.full((h, w, 3), 114, dtype=np.uint8)
        pad[: resized.shape[0], : resized.shape[1]] = resized
        blob = pad.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.input_name: blob})[0][0]
        grids, strides = self._grids
        out[:, :2] = (out[:, :2] + grids) * strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * strides
        return out, r

    def detect(self, frame: np.ndarray, person_thr=0.25, ball_thr=0.10, tiles: int = 1,
               roi: tuple[int, int, int, int] | None = None) -> list[Detection]:
        """tiles=1 : image entière ; tiles=2 : deux tuiles carrées chevauchantes (meilleur rappel
        des petits joueurs, coût x2). roi : (x1,y1,x2,y2) facultatif en pixels."""
        H, W = frame.shape[:2]
        crops = []
        if tiles <= 1:
            crops.append((0, 0, frame))
        else:
            side = H
            xs = np.linspace(0, max(0, W - side), tiles).astype(int)
            for x in xs:
                crops.append((int(x), 0, frame[:, x:x + side]))
        boxes, scores, classes = [], [], []
        for ox, oy, img in crops:
            out, r = self._infer(img)
            obj = out[:, 4:5] * out[:, 5:]
            for cls, thr in ((PERSON, person_thr), (BALL, ball_thr)):
                sc = obj[:, cls]
                keep = sc >= thr
                if not keep.any():
                    continue
                o = out[keep]
                b = np.empty((len(o), 4), np.float32)
                b[:, 0] = (o[:, 0] - o[:, 2] / 2) / r + ox
                b[:, 1] = (o[:, 1] - o[:, 3] / 2) / r + oy
                b[:, 2] = (o[:, 0] + o[:, 2] / 2) / r + ox
                b[:, 3] = (o[:, 1] + o[:, 3] / 2) / r + oy
                boxes.append(b); scores.append(sc[keep]); classes.append(np.full(len(o), cls))
        if not boxes:
            return []
        boxes = np.concatenate(boxes); scores = np.concatenate(scores); classes = np.concatenate(classes)
        dets: list[Detection] = []
        for cls in (PERSON, BALL):
            m = classes == cls
            if not m.any():
                continue
            bb, ss = boxes[m], scores[m]
            for i in _nms(bb, ss, 0.45):
                x1, y1, x2, y2 = np.clip(bb[i], [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(ss[i]), int(cls)))
        # doublons partiels (cadre du seul buste à l'intérieur d'un cadre complet) : suppression
        persons = sorted([d for d in dets if d.cls == PERSON], key=lambda d: -d.score)
        kept = []
        for d in persons:
            dup = False
            for k in kept:
                ix = max(0.0, min(d.x2, k.x2) - max(d.x1, k.x1))
                iy = max(0.0, min(d.y2, k.y2) - max(d.y1, k.y1))
                area = (d.x2 - d.x1) * (d.y2 - d.y1)
                if area > 0 and ix * iy / area > 0.8 and (d.y2 - d.y1) < 0.75 * (k.y2 - k.y1):
                    dup = True
                    break
            if not dup:
                kept.append(d)
        dets = kept + [d for d in dets if d.cls != PERSON]
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            dets = [d for d in dets if rx1 <= (d.x1 + d.x2) / 2 <= rx2 and ry1 <= (d.y1 + d.y2) / 2 <= ry2]
        return dets
