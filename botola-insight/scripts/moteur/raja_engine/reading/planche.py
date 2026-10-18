"""Planches de lecture : grille de recadrages agrandis, numérotés c1..cN, avec l'instant."""
from __future__ import annotations

import cv2
import numpy as np

from .crops import upscale


def compose(crops, labels, title, cols=6, cell_w=200, cell_h=380, header=34):
    n = len(crops)
    rows = max(1, (n + cols - 1) // cols)
    img = np.full((header + rows * (cell_h + 22), cols * cell_w, 3), 235, np.uint8)
    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    for k, (c, lab) in enumerate(zip(crops, labels)):
        r, q = divmod(k, cols)
        up = upscale(c, cell_h)
        if up.shape[1] > cell_w:
            s = cell_w / up.shape[1]
            up = cv2.resize(up, (cell_w, max(1, int(up.shape[0] * s))), interpolation=cv2.INTER_AREA)
        y0 = header + r * (cell_h + 22)
        x0 = q * cell_w + (cell_w - up.shape[1]) // 2
        img[y0:y0 + up.shape[0], x0:x0 + up.shape[1]] = up
        cv2.putText(img, lab, (q * cell_w + 4, y0 + cell_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 160), 1,
                    cv2.LINE_AA)
    return img
