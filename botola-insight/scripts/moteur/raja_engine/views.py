"""Terrain, type de plan et changements de plan.

Ces mesures servent à décider où le suivi est exploitable : un plan large du terrain
permet le suivi et (après calibration) la mesure en mètres ; un gros plan aide à identifier
un joueur mais ne fournit ni position ni présence sur le terrain.
"""
from __future__ import annotations

import cv2
import numpy as np

GREEN_LO = np.array([28, 50, 40], np.uint8)
GREEN_HI = np.array([80, 255, 255], np.uint8)


def green_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, GREEN_LO, GREEN_HI)


def pitch_region(frame: np.ndarray, scale: float = 0.25) -> tuple[np.ndarray, float]:
    """Masque (taille d'origine) de la zone de jeu : enveloppe convexe de la plus grande
    composante verte. Renvoie (masque uint8, fraction verte de l'image)."""
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = green_mask(small)
    frac = float((g > 0).mean())
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g)
    mask = np.zeros_like(g)
    if n > 1:
        i = 1 + int(np.argmax(st[1:, 4]))
        if st[i, 4] > 0.03 * g.size:
            cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(np.vstack(cnts))
            cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask, frac


def hist_signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 1.0, 0, cv2.NORM_L1)
    return h


def is_cut(prev_sig, sig, thr: float = 0.45) -> tuple[bool, float]:
    if prev_sig is None:
        return True, 1.0
    d = float(cv2.compareHist(prev_sig, sig, cv2.HISTCMP_BHATTACHARYYA))
    return d > thr, d


def classify_view(green_frac: float, person_heights_rel: list[float]) -> str:
    """wide | medium | close | other — règles simples, documentées, évaluées dans evaluate.py."""
    med = float(np.median(person_heights_rel)) if person_heights_rel else None
    big = max(person_heights_rel) if person_heights_rel else 0.0
    if green_frac < 0.12 and big < 0.3:
        return "other"
    if big >= 0.45 or (med is not None and med >= 0.30):
        return "close"
    if green_frac >= 0.35 and (med is None or med < 0.17):
        return "wide"
    if green_frac >= 0.20:
        return "medium"
    return "close" if big >= 0.25 else "other"


class CameraMotion:
    """Mouvement de caméra entre deux images (homographie sur des points du terrain,
    joueurs exclus). Sert à compenser le panoramique pour le suivi et à propager une
    calibration du terrain à l'intérieur d'un plan."""

    def __init__(self, max_corners: int = 400):
        self.max_corners = max_corners
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def step(self, frame: np.ndarray, pitch_mask: np.ndarray | None, person_boxes: list) -> tuple[np.ndarray | None, int]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = None
        inliers = 0
        if self.prev_gray is not None:
            mask = np.full(gray.shape, 255, np.uint8) if pitch_mask is None else pitch_mask.copy()
            for b in person_boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(mask, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
            mask[: int(0.22 * mask.shape[0]), : int(0.32 * mask.shape[1])] = 0  # tableau incrusté
            pts = cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, 0.01, 7, mask=self._prev_mask)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
                ok = st.reshape(-1) == 1
                if ok.sum() >= 12:
                    H, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
                    inliers = int(inl.sum()) if inl is not None else 0
                    if H is not None and inliers < 10:
                        H = None
            self._prev_mask = mask
        else:
            m = np.full(gray.shape, 255, np.uint8) if pitch_mask is None else pitch_mask.copy()
            for b in person_boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(m, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
            m[: int(0.22 * m.shape[0]), : int(0.32 * m.shape[1])] = 0
            self._prev_mask = m
        self.prev_gray = gray
        return H, inliers


class MotionComposer:
    """Compose les mouvements successifs dans l'ordre image précédente → courante.

    Une seule estimation manquante invalide l'intervalle entier. Une coupure
    remet à zéro l'accumulation et la prochaine image détectée est une origine.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = np.eye(3)
        self.valid = True
        self.has_origin = False
        self.inliers = None

    def add(self, H, inliers=0):
        if not self.has_origin:
            return
        if H is None:
            self.valid = False
            return
        H = np.asarray(H, float).reshape(3, 3)
        if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
            self.valid = False
            return
        self.acc = H @ self.acc
        self.inliers = inliers if self.inliers is None else min(self.inliers, inliers)

    def take(self):
        H = self.acc.copy() if self.has_origin and self.valid else None
        n = self.inliers or 0
        self.acc = np.eye(3)
        self.valid = True
        self.has_origin = True
        self.inliers = None
        return H, n
