"""Ballon : filtrage temporel des détections, proximité joueur–ballon, candidats de possession.

Le ballon mesure 4 à 8 px en 854×480 : le détecteur générique le manque souvent et confond
parfois des taches blanches. On ne garde que des positions cohérentes dans le temps
(chaîne d'au moins 3 détections compatibles) et on ne remplit pas les trous de plus de 0,6 s.
La couverture (part du direct où le ballon est localisé) est toujours publiée avec les mesures.

Proximité ≠ possession : la proximité est une distance (en hauteurs de joueur tant que le
terrain n'est pas calibré) ; une « possession candidate » exige un ballon au pied du joueur
sur au moins deux images consécutives et reste un candidat non validé.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def ball_tracks(frames: list[dict], max_speed_px_s: float = 900.0, min_len: int = 3, max_gap_s: float = 0.6) -> dict:
    """Renvoie {t: (x, y, score)} pour les images où le ballon est retenu."""
    by_shot = defaultdict(list)
    for fr in frames:
        if fr["view"] not in ("wide", "medium"):
            continue
        by_shot[fr["shot"]].append(fr)
    kept = {}
    for shot, frs in by_shot.items():
        chains = []  # chaque chaîne : liste (t, x, y, s)
        for fr in sorted(frs, key=lambda f: f["t"]):
            cands = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[4]) for b in fr["b"]]
            used = set()
            for ch in sorted(chains, key=lambda c: -len(c)):
                lt, lx, ly, _ = ch[-1]
                dt = fr["t"] - lt
                if dt > max_gap_s or dt <= 0:
                    continue
                best, bd = None, None
                for i, (x, y, s) in enumerate(cands):
                    if i in used:
                        continue
                    d = np.hypot(x - lx, y - ly)
                    if d <= max_speed_px_s * dt + 15 and (bd is None or d < bd):
                        best, bd = i, d
                if best is not None:
                    used.add(best)
                    x, y, s = cands[best]
                    ch.append((fr["t"], x, y, s))
            for i, (x, y, s) in enumerate(cands):
                if i not in used:
                    chains.append([(fr["t"], x, y, s)])
        for ch in chains:
            if len(ch) >= min_len:
                for t, x, y, s in ch:
                    if t not in kept or kept[t][2] < s:
                        kept[t] = (x, y, s)
    return kept


def proximity(frames: list[dict], balls: dict, near_h: float = 1.0, touch_h: float = 0.45) -> dict:
    """Pour chaque piste : secondes avec ballon proche (≤ near_h hauteurs du joueur entre pieds
    et ballon) et images « au pied » (≤ touch_h). Renvoie aussi la couverture."""
    per_track = defaultdict(lambda: {"near_s": 0.0, "touch_frames": 0, "touch_times": [], "near_times": []})
    nearest = {}
    live_wide = 0
    located = 0
    step = None
    ts = sorted(f["t"] for f in frames)
    if len(ts) > 2:
        step = float(np.median(np.diff(ts)))
    step = step or 0.2
    for fr in frames:
        if fr["view"] not in ("wide", "medium"):
            continue
        live_wide += 1
        b = balls.get(fr["t"])
        if b is None:
            continue
        located += 1
        bx, by = b[0], b[1]
        best = None
        for x1, y1, x2, y2, sc, tid, feat in fr["p"]:
            if tid is None:
                continue
            h = max(y2 - y1, 8)
            fx, fy = (x1 + x2) / 2, y2
            d = np.hypot(bx - fx, by - fy) / h
            if best is None or d < best[0]:
                best = (d, tid)
            if d <= near_h:
                per_track[tid]["near_s"] += step
                per_track[tid]["near_times"].append(fr["t"])
            if d <= touch_h:
                per_track[tid]["touch_frames"] += 1
                per_track[tid]["touch_times"].append(fr["t"])
        if best is not None:
            nearest[fr["t"]] = {"track_id": best[1], "dist_h": round(float(best[0]), 3)}
    return {"per_track": {k: dict(v) for k, v in per_track.items()}, "nearest": nearest,
            "coverage": {"wide_medium_frames": live_wide, "ball_located_frames": located,
                         "ball_located_fraction": round(located / max(1, live_wide), 3), "step_s": step}}


def possession_chain(prox: dict, tracks: dict, touch_h: float = 0.45, min_frames: int = 2, max_link_s: float = 4.0) -> list[dict]:
    """Candidats de passes : contrôle par A puis contrôle par B (A ≠ B) dans un délai court.
    Résultat brut, non validé, utilisé seulement comme candidat (portée engine_candidates)."""
    seq = []
    for t in sorted(prox["nearest"]):
        n = prox["nearest"][t]
        if n["dist_h"] <= touch_h:
            seq.append((t, n["track_id"]))
    # regrouper en contrôles (même piste sur images consécutives)
    controls = []
    for t, tid in seq:
        if controls and controls[-1]["track_id"] == tid and t - controls[-1]["t_end"] <= 0.5:
            controls[-1]["t_end"] = t
            controls[-1]["frames"] += 1
        else:
            controls.append({"track_id": tid, "t_start": t, "t_end": t, "frames": 1})
    controls = [c for c in controls if c["frames"] >= min_frames]
    out = []
    for a, b in zip(controls, controls[1:]):
        if a["track_id"] == b["track_id"] or b["t_start"] - a["t_end"] > max_link_s:
            continue
        ta, tb = tracks.get(a["track_id"], {}), tracks.get(b["track_id"], {})
        if ta.get("shot") != tb.get("shot"):
            continue
        same = ta.get("team") is not None and ta.get("team") == tb.get("team")
        out.append({"from_track": a["track_id"], "to_track": b["track_id"], "t_release": a["t_end"],
                    "t_receive": b["t_start"], "from_team": ta.get("team"), "to_team": tb.get("team"),
                    "outcome_candidate": "completee" if same else "interceptee_ou_perdue",
                    "status": "candidate_not_validated"})
    return out
