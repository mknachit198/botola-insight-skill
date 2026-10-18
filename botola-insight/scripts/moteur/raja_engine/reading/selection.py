"""Choix des instants à recadrer pour lire un numéro, sans estimation de pose.

Indices purement géométriques (aucune apparence de la personne) :
- dos probable : en plan large, des pieds qui MONTENT dans l'image (y qui diminue, mouvement
  de caméra compensé) indiquent un joueur qui s'éloigne de la caméra et montre son dos ;
- taille de la boîte (plus grande = chiffres plus lisibles) ;
- isolement : torse non recouvert par une autre boîte, aucun croisement à cet instant ;
- boîte entière dans l'image.
Les instants sont répartis dans la chaîne (une case de temps par recadrage).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

W_BACK, W_SIZE, W_ISO, W_BORDER = 0.35, 0.30, 0.25, 0.10


def frame_index(frames):
    return {(f["shot"], f["t"]): f for f in frames}


def _foot(box):
    return np.array([(box[0] + box[2]) / 2.0, box[3], 1.0])


def vertical_motion(track_obs, fidx, shot):
    """Vitesse verticale des pieds en hauteurs de corps par seconde, caméra compensée.
    Négative = le joueur monte dans l'image (s'éloigne). None si non mesurable."""
    v = [None] * len(track_obs)
    for i in range(1, len(track_obs)):
        a, b = track_obs[i - 1], track_obs[i]
        dt = b["t"] - a["t"]
        if dt <= 0 or dt > 0.45:
            continue
        f = fidx.get((shot, b["t"]))
        if f is None or f.get("H") is None:
            continue
        H = np.array(f["H"], np.float64).reshape(3, 3)
        p = H @ _foot(a["box"])
        p = p / p[2]
        h = max(8.0, b["box"][3] - b["box"][1])
        v[i] = float((b["box"][3] - p[1]) / h / dt)
    # lissage sur ±2 observations
    out = []
    for i in range(len(v)):
        w = [x for x in v[max(0, i - 2): i + 3] if x is not None]
        out.append(float(np.mean(w)) if w else None)
    return out


def score_observations(tr, fidx, W=1920, H=1080):
    """Score de lisibilité attendue pour chaque observation d'une piste (0..1)."""
    obs = tr["obs"]
    vy = vertical_motion(obs, fidx, tr["shot"])
    cross_t = {c["t"] for c in tr.get("crossings", []) if c["iou"] > 0.02}
    rows = []
    for i, o in enumerate(obs):
        x1, y1, x2, y2 = o["box"]
        h = y2 - y1
        size = float(np.clip((h - 60) / 80, 0, 1))
        f = fidx.get((tr["shot"], o["t"]), {})
        # recouvrement du torse (15–55 % de la hauteur) par une autre boîte
        tx1, tx2 = x1 + 0.15 * (x2 - x1), x2 - 0.15 * (x2 - x1)
        ty1, ty2 = y1 + 0.15 * h, y1 + 0.55 * h
        tarea = max(1.0, (tx2 - tx1) * (ty2 - ty1))
        occ = 0.0
        for row in f.get("p", []):
            if row[5] == tr["id"]:
                continue
            ix = max(0.0, min(tx2, row[2]) - max(tx1, row[0]))
            iy = max(0.0, min(ty2, row[3]) - max(ty1, row[1]))
            occ = max(occ, ix * iy / tarea)
        iso = float(np.clip(1 - 2 * occ, 0, 1)) * (0.4 if o["t"] in cross_t else 1.0)
        border_ok = 0.0 if (x1 < 3 or x2 > W - 3 or y1 < 3 or y2 > H - 3) else 1.0
        back = 0.3 if vy[i] is None else float(np.clip(-vy[i] / 0.5, 0, 1))
        s = W_BACK * back + W_SIZE * size + W_ISO * iso + W_BORDER * border_ok
        rows.append({"i": i, "t": o["t"], "score": round(s, 4), "back": round(back, 3), "size": round(size, 3),
                     "iso": round(iso, 3), "border_ok": border_ok, "vy": None if vy[i] is None else round(vy[i], 3)})
    return rows


def pick(tr, fidx, n=12, min_iso=0.2, exclude=(), exclude_tol=0.3, use_back=True):
    """n instants répartis : la chaîne est découpée en n cases de temps égales ; dans chaque
    case, l'observation de meilleur score (torse assez dégagé). Cases vides ignorées.
    exclude : instants déjà recadrés (seconde passe) ; use_back=False ignore l'indice de dos."""
    rows = score_observations(tr, fidx)
    if not use_back:
        for r in rows:
            r["score"] = round(r["score"] - W_BACK * r["back"], 4)
    if exclude:
        rows = [r for r in rows if all(abs(r["t"] - x) > exclude_tol for x in exclude)] or rows
    if not rows:
        return []
    t0, t1 = rows[0]["t"], rows[-1]["t"]
    span = max(t1 - t0, 1e-6)
    bins = defaultdict(list)
    for r in rows:
        k = min(n - 1, int((r["t"] - t0) / span * n))
        bins[k].append(r)
    chosen = []
    for k in sorted(bins):
        cands = [r for r in bins[k] if r["iso"] >= min_iso] or bins[k]
        chosen.append(max(cands, key=lambda r: r["score"]))
    return chosen


def evenly(tr, n):
    """n observations équiréparties dans la chaîne (mode annotation : pas de sélection)."""
    obs = tr["obs"]
    if len(obs) <= n:
        return list(range(len(obs)))
    return sorted({int(round(x)) for x in np.linspace(0, len(obs) - 1, n)})
