"""Lecture opportuniste des numéros de maillot (plans moyens et rapprochés seulement).

En 854×480, un numéro n'est lisible que sur des joueurs assez grands à l'écran. Une
lecture isolée n'identifie jamais un joueur : il faut au moins deux lectures concordantes
sur la même piste, un numéro présent dans la composition et une couleur d'équipe cohérente.
Les lectures sont conservées comme indices (« auto_number ») et non comme vérité.
"""
from __future__ import annotations

import cv2
import numpy as np

from .scoreboard import HAS_TESSERACT, pytesseract

MIN_BOX_H_REL = 0.28


def read_number(frame: np.ndarray, box) -> tuple[int | None, float, str]:
    if not HAS_TESSERACT:
        return None, 0.0, ""
    H = frame.shape[0]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < MIN_BOX_H_REL * H or w < 12:
        return None, 0.0, ""
    crop = frame[max(0, y1 + int(0.15 * h)):max(0, y1 + int(0.55 * h)), max(0, x1 + int(0.12 * w)):max(0, x2 - int(0.12 * w))]
    if crop.size == 0:
        return None, 0.0, ""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, 120.0 / max(1, g.shape[0]))
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    best = (None, 0.0, "")
    for inv in (False, True):
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if inv:
            b = 255 - b
        b = cv2.copyMakeBorder(b, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        try:
            d = pytesseract.image_to_data(b, config="--psm 7 -c tessedit_char_whitelist=0123456789",
                                          output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        for txt, conf in zip(d.get("text", []), d.get("conf", [])):
            txt = (txt or "").strip()
            try:
                c = float(conf)
            except Exception:
                c = -1
            if txt.isdigit() and 1 <= len(txt) <= 2 and c > best[1]:
                best = (int(txt), c / 100.0, txt)
    return best
