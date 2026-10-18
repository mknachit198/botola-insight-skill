"""Recalage temporel entre deux enregistrements du même match (ex. 854×480 et 1920×1080).

Méthode : à plusieurs instants répartis sur la vidéo source, on lit quelques secondes d'images
dans les deux fichiers, réduites en vignettes grises 96×54, et on cherche le décalage (pas d'une
image à 30 i/s) qui minimise l'écart moyen entre vignettes. Le décalage n'est retenu que s'il est
net (écart nettement plus faible qu'aux décalages voisins) ; les points incohérents sont signalés.
Résultat : une table de points (t_source → t_cible) et une interpolation par morceaux."""
from __future__ import annotations

import cv2
import numpy as np


def _thumbs(path: str, t0: float, dur: float, step: float | None = None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t0) * 1000)
    out = []
    n = int(dur * fps)
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        g = cv2.cvtColor(cv2.resize(f, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append((t, g))
    cap.release()
    return out


def offset_at(src: str, dst: str, t: float, guess: float = 0.0, search: float = 3.0, win: float = 4.0) -> dict:
    """Décalage d tel que dst(t + d) ≈ src(t), cherché dans [guess − search, guess + search]."""
    a = _thumbs(src, t, win)
    b = _thumbs(dst, t + guess - search, win + 2 * search)
    if len(a) < 10 or len(b) < 10:
        return {"t": t, "ok": False, "reason": "lecture impossible"}
    a_t = np.array([x[0] for x in a]); a_g = np.stack([x[1] for x in a])
    b_t = np.array([x[0] for x in b]); b_g = np.stack([x[1] for x in b])
    # contraste minimal : un plan fixe sans mouvement ne permet pas de recaler finement
    motion = float(np.mean(np.abs(np.diff(a_g, axis=0)))) if len(a_g) > 1 else 0.0
    cands = np.arange(guess - search, guess + search + 1e-9, 1 / 30.0)
    errs = []
    for d in cands:
        idx = np.clip(np.searchsorted(b_t, a_t + d), 0, len(b_t) - 1)
        ok = np.abs(b_t[idx] - (a_t + d)) < 0.04
        errs.append(np.mean(np.abs(a_g[ok] - b_g[idx[ok]])) if ok.sum() > 0.6 * len(a_t) else np.inf)
    errs = np.array(errs)
    k = int(np.argmin(errs))
    best = float(errs[k])
    others = errs[np.abs(cands - cands[k]) > 0.5]
    second = float(np.min(others)) if len(others) and np.isfinite(others).any() else np.inf
    sharp = second / max(best, 1e-6)
    return {"t": round(t, 2), "offset": round(float(cands[k]), 3), "err": round(best, 2), "next_err": round(second, 2),
            "sharpness": round(sharp, 2), "motion": round(motion, 2), "ok": bool(np.isfinite(best) and sharp > 1.5 and motion > 1.0)}


def align(src: str, dst: str, times: list[float], guess: float = 0.0, search: float = 3.0, tol: float = 0.1) -> dict:
    """Décalage par morceaux : un enregistrement peut sauter (image figée, coupure). Les points nets
    sont regroupés en morceaux de décalage constant (± tol) ; chaque saut entre deux morceaux est
    localisé par dichotomie (précision ~5 s)."""
    pts = [offset_at(src, dst, t, guess, search) for t in times]
    good = [p for p in pts if p["ok"]]
    # morceaux successifs
    pieces = []
    for p in good:
        if pieces and abs(p["offset"] - pieces[-1]["offset"]) <= tol:
            pieces[-1]["points"].append(p)
        else:
            pieces.append({"offset": p["offset"], "points": [p]})
    for pc in pieces:
        pc["offset"] = float(np.median([p["offset"] for p in pc["points"]]))
    jumps = []
    for a, b in zip(pieces, pieces[1:]):
        lo, hi = a["points"][-1]["t"], b["points"][0]["t"]
        wide = max(search, abs(b["offset"] - a["offset"]) + 1.0)
        mid_guess = (a["offset"] + b["offset"]) / 2
        while hi - lo > 5.0:
            m = (lo + hi) / 2
            r = offset_at(src, dst, m, mid_guess, wide)
            if not r["ok"]:  # plan fixe : on essaie un peu plus loin
                r = offset_at(src, dst, m + 2.5, mid_guess, wide)
            if r["ok"] and abs(r["offset"] - a["offset"]) <= tol:
                lo = m
            elif r["ok"] and abs(r["offset"] - b["offset"]) <= tol:
                hi = m
            else:
                break
        jumps.append({"from_t": round(lo, 2), "to_t": round(hi, 2), "before": a["offset"], "after": b["offset"]})
    for p in pts:
        p["consistent"] = bool(p["ok"] and any(p in pc["points"] for pc in pieces))
    offs = np.array([p["offset"] for p in good]) if good else np.array([])
    return {"points": pts, "pieces": [{"offset": pc["offset"], "from_t": pc["points"][0]["t"], "to_t": pc["points"][-1]["t"],
                                        "n": len(pc["points"])} for pc in pieces],
            "jumps": jumps, "median_offset": float(np.median(offs)) if len(offs) else None,
            "n_ok": len(good), "n_consistent": sum(p["consistent"] for p in pts)}


def mapper(points: list[dict], jumps: list[dict] | None = None):
    """Fonction t_source → t_cible. Décalage constant par morceau ; aux sauts, on bascule au milieu
    de l'intervalle de localisation. Sans saut connu, interpolation entre points cohérents."""
    good = sorted((p["t"], p["offset"]) for p in points if p.get("consistent"))
    if not good:
        return None
    if jumps:
        cuts = sorted(((j["from_t"] + j["to_t"]) / 2, j["after"]) for j in jumps)
        first = good[0][1]

        def f(t):
            d = first
            for c, after in cuts:
                if t >= c:
                    d = after
            return float(t + d)
        return f
    ts = np.array([g[0] for g in good]); ds = np.array([g[1] for g in good])
    return lambda t: float(t + np.interp(t, ts, ds))
