"""Extraction des recadrages dans la vidéo originale, en un seul passage séquentiel.
La vidéo n'est ni modifiée ni réencodée : seules de petites images JPEG sont écrites."""
from __future__ import annotations

import cv2
import numpy as np


def padded(box, W, H, pad_x=0.30, pad_top=0.12, pad_bot=0.06):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (int(max(0, x1 - pad_x * w)), int(max(0, y1 - pad_top * h)),
            int(min(W, x2 + pad_x * w)), int(min(H, y2 + pad_bot * h)))


def extract(video_path, requests, context=False):
    """requests : liste de dicts {key, fi, box}. Renvoie {key: crop BGR} (+ image entière
    réduite sous key+'#ctx' si context=True). Décodage séquentiel ; grab() saute le reste."""
    wanted = {}
    for r in requests:
        wanted.setdefault(int(r["fi"]), []).append(r)
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Vidéo illisible : {video_path}")
    out, idx, last = {}, 0, max(wanted)
    try:
        while idx <= last:
            if idx in wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                H, W = frame.shape[:2]
                for r in wanted[idx]:
                    x1, y1, x2, y2 = padded(r["box"], W, H)
                    out[r["key"]] = frame[y1:y2, x1:x2].copy()
                    if context:
                        out[r["key"] + "#ctx"] = context_view(frame, r["box"])
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()
    return out


def context_view(frame, box, width=480):
    """Image entière réduite avec la boîte entourée (repère de position dans le plan)."""
    H, W = frame.shape[:2]
    s = width / W
    small = cv2.resize(frame, (width, int(round(H * s))), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = [int(round(v * s)) for v in box]
    cv2.rectangle(small, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 2)
    return small


def upscale(crop, target_h=360):
    if crop is None or crop.size == 0:
        return np.zeros((target_h, target_h // 3, 3), np.uint8)
    s = target_h / crop.shape[0]
    return cv2.resize(crop, (max(1, int(round(crop.shape[1] * s))), target_h), interpolation=cv2.INTER_CUBIC)
