"""Calibration du terrain et mesures physiques sur segments visibles.

Méthode :
1. Ancrage humain : sur une image d'un plan large, l'utilisateur clique au moins 4 repères
   du terrain (coins de surface, points de penalty, rond central…). Homographie image→terrain.
2. Propagation dans le même plan par les homographies image→image du mouvement de caméra
   (points du gazon, joueurs exclus) ; arrêt dès que l'alignement des lignes blanches se
   dégrade, qu'une homographie manque ou que le plan change.
3. Distances/vitesses calculées seulement sur les images calibrées d'un tronçon identifié.
   Les mètres « observés » ne sont jamais extrapolés à 90 minutes ; une sensibilité (bruit de
   ±2 px sur les clics) est publiée, ce n'est pas un intervalle de confiance.
Une seule homographie du but (dossier V2) ne suffit pas et n'est pas réutilisée ici.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def landmarks(L: float = 105.0, W: float = 68.0) -> dict[str, tuple[float, float]]:
    """Repères en mètres : x le long du terrain (gauche négatif à l'écran de la caméra
    principale), y en travers (tribune opposée négative)."""
    h, w = L / 2, W / 2
    lm = {"centre": (0.0, 0.0), "milieu_haut": (0.0, -w), "milieu_bas": (0.0, w),
          "rond_central_haut": (0.0, -9.15), "rond_central_bas": (0.0, 9.15),
          "rond_central_gauche": (-9.15, 0.0), "rond_central_droite": (9.15, 0.0)}
    # points du rond central à 45° : repères de calcul (non proposés au clic, sans intersection de lignes),
    # pour transcrire un ajustement sur le rond central quand seule la ligne médiane est par ailleurs visible
    d = 9.15 / math.sqrt(2)
    lm.update({"rond_central_45_gauche_haut": (-d, -d), "rond_central_45_droite_haut": (d, -d),
               "rond_central_45_gauche_bas": (-d, d), "rond_central_45_droite_bas": (d, d)})
    for side, s in (("gauche", -1), ("droite", 1)):
        gx = s * h
        lm.update({
            f"coin_{side}_haut": (gx, -w), f"coin_{side}_bas": (gx, w),
            f"surface_{side}_ligne_but_haut": (gx, -20.16), f"surface_{side}_ligne_but_bas": (gx, 20.16),
            f"surface_{side}_angle_haut": (gx - s * 16.5, -20.16), f"surface_{side}_angle_bas": (gx - s * 16.5, 20.16),
            f"six_metres_{side}_ligne_but_haut": (gx, -9.16), f"six_metres_{side}_ligne_but_bas": (gx, 9.16),
            f"six_metres_{side}_angle_haut": (gx - s * 5.5, -9.16), f"six_metres_{side}_angle_bas": (gx - s * 5.5, 9.16),
            f"point_penalty_{side}": (gx - s * 11.0, 0.0),
            f"arc_{side}_haut": (gx - s * 16.5, -7.31), f"arc_{side}_bas": (gx - s * 16.5, 7.31),
            f"poteau_{side}_haut": (gx, -3.66), f"poteau_{side}_bas": (gx, 3.66),
        })
    return lm


LANDMARK_LABELS = {
    "centre": "Point central", "milieu_haut": "Ligne médiane × touche (loin)", "milieu_bas": "Ligne médiane × touche (près)",
    "rond_central_haut": "Rond central × ligne médiane (loin)", "rond_central_bas": "Rond central × ligne médiane (près)",
    "rond_central_gauche": "Rond central : point le plus à gauche (axe du terrain)",
    "rond_central_droite": "Rond central : point le plus à droite (axe du terrain)",
}
for _side in ("gauche", "droite"):
    for _k, _lab in [("coin_{s}_haut", "Coin {S} (loin)"), ("coin_{s}_bas", "Coin {S} (près)"),
                     ("surface_{s}_ligne_but_haut", "Surface {S} : ligne de but (loin)"),
                     ("surface_{s}_ligne_but_bas", "Surface {S} : ligne de but (près)"),
                     ("surface_{s}_angle_haut", "Surface {S} : angle avant (loin)"),
                     ("surface_{s}_angle_bas", "Surface {S} : angle avant (près)"),
                     ("six_metres_{s}_ligne_but_haut", "6 m {S} : ligne de but (loin)"),
                     ("six_metres_{s}_ligne_but_bas", "6 m {S} : ligne de but (près)"),
                     ("six_metres_{s}_angle_haut", "6 m {S} : angle avant (loin)"),
                     ("six_metres_{s}_angle_bas", "6 m {S} : angle avant (près)"),
                     ("point_penalty_{s}", "Point de penalty {S}"),
                     ("arc_{s}_haut", "Arc {S} × surface (loin)"), ("arc_{s}_bas", "Arc {S} × surface (près)"),
                     ("poteau_{s}_haut", "Poteau {S} (loin) au sol"), ("poteau_{s}_bas", "Poteau {S} (près) au sol")]:
        LANDMARK_LABELS[_k.format(s=_side)] = _lab.format(S=_side)


def model_lines(L: float = 105.0, W: float = 68.0) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    h, w = L / 2, W / 2
    lines = [((-h, -w), (h, -w)), ((-h, w), (h, w)), ((-h, -w), (-h, w)), ((h, -w), (h, w)), ((0, -w), (0, w))]
    for s in (-1, 1):
        gx = s * h
        lines += [((gx, -20.16), (gx - s * 16.5, -20.16)), ((gx, 20.16), (gx - s * 16.5, 20.16)),
                  ((gx - s * 16.5, -20.16), (gx - s * 16.5, 20.16)),
                  ((gx, -9.16), (gx - s * 5.5, -9.16)), ((gx, 9.16), (gx - s * 5.5, 9.16)),
                  ((gx - s * 5.5, -9.16), (gx - s * 5.5, 9.16))]
    return lines


MAX_FIT_RMSE_M = 1.0  # au-delà, les clics se contredisent (repère mal nommé ou mal placé)


def fit_homography(points: list[dict], W_img: int, H_img: int, L=105.0, Wp=68.0) -> dict:
    lm = landmarks(L, Wp)
    src, dst, names = [], [], []
    for p in points:
        if p["landmark"] in lm:
            src.append([p["x"] * W_img, p["y"] * H_img])
            dst.append(lm[p["landmark"]])
            names.append(p["landmark"])
    if len(src) < 4:
        return {"ok": False, "reason": "au moins 4 repères nécessaires"}
    src, dst = np.array(src, np.float64), np.array(dst, np.float64)
    # au moins 4 repères dont aucun trio n'est aligné sur le terrain (sinon l'homographie est indéterminée)
    from itertools import combinations
    best = []
    for idx in combinations(range(len(dst)), 4):
        q = dst[list(idx)]
        def area(a, b, c):
            u, v = q[b] - q[a], q[c] - q[a]
            return abs(u[0] * v[1] - u[1] * v[0]) / 2
        if all(area(a, b, c) > 1.0 for a, b, c in combinations(range(4), 3)):
            best = idx
            break
    if not best:
        return {"ok": False, "reason": "repères dégénérés : il faut 4 repères dont 3 ne sont jamais alignés sur le terrain"}
    Hm, _ = cv2.findHomography(src, dst, 0)
    if Hm is None:
        return {"ok": False, "reason": "repères dégénérés (alignés ?)"}
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    res = np.linalg.norm(proj - dst, axis=1)
    rmse = float(np.sqrt(np.mean(res ** 2)))
    if rmse > MAX_FIT_RMSE_M:
        return {"ok": False, "reason": f"repères incohérents entre eux (écart moyen {rmse:.1f} m > {MAX_FIT_RMSE_M} m)",
                "fit_rmse_m": rmse, "n_points": len(src)}
    # validation croisée « en laisser un » quand ≥ 5 points : erreur sur un repère non utilisé
    loo = []
    if len(src) >= 5:
        for i in range(len(src)):
            m = np.ones(len(src), bool)
            m[i] = False
            Hi, _ = cv2.findHomography(src[m], dst[m], 0)
            if Hi is not None:
                q = cv2.perspectiveTransform(src[i:i + 1].reshape(-1, 1, 2), Hi).reshape(2)
                loo.append(float(np.linalg.norm(q - dst[i])))
    return {"ok": True, "H": Hm.tolist(), "n_points": len(src), "landmarks": names,
            "fit_rmse_m": float(np.sqrt(np.mean(res ** 2))),
            "leave_one_out_error_m": None if not loo else {"median": float(np.median(loo)), "max": float(np.max(loo))},
            "checked": len(loo) > 0}


def line_alignment(frame: np.ndarray, H_img2pitch: np.ndarray, L=105.0, Wp=68.0, samples_per_m: float = 1.0) -> float | None:
    """Part des points des lignes du modèle (projetées dans l'image) tombant sur un pixel clair.
    Sert de contrôle de dérive de la propagation, relativement à l'image d'ancrage."""
    try:
        Hinv = np.linalg.inv(np.array(H_img2pitch, np.float64))
    except np.linalg.LinAlgError:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8)
    white = cv2.dilate(white, np.ones((5, 5), np.uint8))
    hits = tot = 0
    Hh, Ww = white.shape
    for (x1, y1), (x2, y2) in model_lines(L, Wp):
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) * samples_per_m))
        pts = np.stack([np.linspace(x1, x2, n), np.linspace(y1, y2, n)], 1).reshape(-1, 1, 2)
        img = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
        for u, v in img:
            if 0 <= u < Ww and int(0.22 * Hh) <= v < Hh:  # hors tableau incrusté
                tot += 1
                hits += int(white[int(v), int(u)] > 0)
    return None if tot < 20 else hits / tot


def propagate(frames: list[dict], anchor_t: float, H_anchor: np.ndarray, get_frame=None, check_every_s: float = 1.0,
              min_rel_alignment: float = 0.5, L=105.0, Wp=68.0) -> dict:
    """frames : enregistrements frames.jsonl du MÊME plan, triés. Renvoie {t: H_img2pitch} et un journal."""
    frs = sorted(frames, key=lambda f: f["t"])
    idx = min(range(len(frs)), key=lambda i: abs(frs[i]["t"] - anchor_t))
    Hs = {frs[idx]["t"]: np.array(H_anchor, np.float64)}
    base_align = None
    if get_frame is not None:
        f0 = get_frame(frs[idx]["t"])
        base_align = None if f0 is None else line_alignment(f0, H_anchor, L, Wp)
    log = {"anchor_t": frs[idx]["t"], "anchor_alignment": base_align, "stops": []}

    def ok_align(t, Hc, last_check):
        if get_frame is None or base_align is None or t - last_check < check_every_s:
            return True, last_check
        f = get_frame(t)
        a = None if f is None else line_alignment(f, Hc, L, Wp)
        if a is None or a < min_rel_alignment * base_align:
            log["stops"].append({"t": t, "alignment": a, "reason": "dérive (lignes mal alignées)"})
            return False, t
        return True, t

    cur = Hs[frs[idx]["t"]]
    last = frs[idx]["t"]
    for i in range(idx + 1, len(frs)):
        Hc = frs[i].get("H")
        if Hc is None or frs[i]["view"] not in ("wide", "medium"):
            log["stops"].append({"t": frs[i]["t"], "reason": "mouvement de caméra non estimé"})
            break
        cur = cur @ np.linalg.inv(np.array(Hc, np.float64).reshape(3, 3))
        good, last = ok_align(frs[i]["t"], cur, last)
        if not good:
            break
        Hs[frs[i]["t"]] = cur
    cur = Hs[frs[idx]["t"]]
    last = frs[idx]["t"]
    for i in range(idx, 0, -1):
        Hc = frs[i].get("H")
        if Hc is None or frs[i - 1]["view"] not in ("wide", "medium"):
            log["stops"].append({"t": frs[i - 1]["t"], "reason": "mouvement de caméra non estimé (arrière)"})
            break
        cur = cur @ np.array(Hc, np.float64).reshape(3, 3)
        good, last = ok_align(frs[i - 1]["t"], cur, last)
        if not good:
            break
        Hs[frs[i - 1]["t"]] = cur
    log["calibrated_frames"] = len(Hs)
    log["calibrated_span_s"] = [min(Hs), max(Hs)]
    return {"H": Hs, "log": log}


def to_pitch(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), np.array(H, np.float64)).reshape(2)
    return float(p[0]), float(p[1])


def kinematics(samples: list[tuple[float, float, float]], max_dt: float = 0.45, smooth_s: float = 0.8,
               sprint_kmh: float = 25.2, sprint_min_s: float = 1.0) -> dict:
    """samples : [(t, x_m, y_m)] d'un tronçon identifié et calibré. Distance observée,
    vitesse max lissée, sprints. Les trous > max_dt coupent la mesure (pas d'interpolation)."""
    if len(samples) < 3:
        return {"distance_m": 0.0, "covered_s": 0.0, "vmax_kmh": None, "sprints": 0, "n": len(samples)}
    s = np.array(sorted(samples), np.float64)
    t, x, y = s[:, 0], s[:, 1], s[:, 2]
    # lissage : moyenne glissante centrée et SYMÉTRIQUE à l'intérieur de chaque tronçon
    # continu (fenêtre réduite aux bords) : pas de biais sur un mouvement rectiligne
    xs, ys = x.copy(), y.copy()
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.2
    hw = max(1, int(round(smooth_s / 2 / max(step, 1e-3))))
    runs, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] - t[i - 1] > max_dt:
            runs.append((start, i))
            start = i
    for a, b in runs:
        for i in range(a, b):
            k = min(hw, i - a, b - 1 - i)
            xs[i], ys[i] = x[i - k:i + k + 1].mean(), y[i - k:i + k + 1].mean()
    dist = covered = 0.0
    speeds = []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > max_dt:
            continue
        d = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        v = d / dt * 3.6
        if v > 40:  # physiquement implausible pour un footballeur : rejet (erreur de pied/détection)
            continue
        dist += d
        covered += dt
        speeds.append((t[i], v))
    vmax = None
    sprints = 0
    if speeds and covered >= 2.0:
        st = np.array(speeds)
        # vitesse maximale SOUTENUE : moyenne sur des fenêtres d'au moins 0,6 s dans 1 s
        sustained = []
        for i in range(len(st)):
            m = (st[:, 0] >= st[i, 0] - 1.0) & (st[:, 0] <= st[i, 0])
            if np.ptp(st[m, 0]) >= 0.6:
                sustained.append(float(st[m, 1].mean()))
        if sustained:
            vmax = float(np.max(sustained))
        # sprint : vitesse ≥ seuil sans interruption pendant au moins sprint_min_s
        run_start = None
        counted = False
        prev_t = None
        for tt, v in speeds:
            if v >= sprint_kmh and (prev_t is None or tt - prev_t <= max_dt):
                if run_start is None:
                    run_start, counted = tt, False
                if not counted and tt - run_start >= sprint_min_s:
                    sprints += 1
                    counted = True
            else:
                run_start, counted = None, False
            prev_t = tt
    return {"distance_m": round(float(dist), 1), "covered_s": round(float(covered), 2), "vmax_kmh": None if vmax is None else round(vmax, 1),
            "sprints": sprints, "n": len(samples)}
