"""Étape 1 : filtrage, séparation des tenues et raccords dans un même plan.

Les numéros ne servent jamais à choisir un raccord. Les observations brutes restent
disponibles ; les identités continuent de respecter les croisements et les trous.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracks import assign_teams, build_tracks, load_frames
from .util import read_json, write_json

VERSION = "s1-1"
REFINEMENT_VERSION = "s1-2"


def remove_short_tracks(frames, min_hits=3):
    counts = Counter(p[5] for f in frames for p in f["p"] if p[5] is not None)
    keep = {tid for tid, n in counts.items() if n >= min_hits}
    for f in frames:
        for p in f["p"]:
            if p[5] not in keep:
                p[5] = None
        f["num"] = [r for r in f.get("num", []) if r[0] in keep]
    return sum(n < min_hits for n in counts.values())


def split_by_team(frames, model, min_run=2):
    """Une autre tenue persistante ouvre une nouvelle piste, lectures OCR comprises."""
    seq = defaultdict(list)
    for fi, f in enumerate(frames):
        for pi, p in enumerate(f["p"]):
            if p[5] is None:
                continue
            lab = None
            if p[6] is not None:
                c, _ = model.predict(np.asarray(p[6], np.float32))
                lab = {"raja": "R", "raja_gk": "R", "opponent": "O", "opponent_gk": "O"}.get(model.label(c))
            seq[p[5]].append((fi, pi, lab))
    next_id = max(seq, default=0) + 1
    changes = {}
    splits = 0
    for tid, observations in seq.items():
        runs, leading = [], []
        for fi, pi, lab in observations:
            if lab is None:
                (runs[-1][2] if runs else leading).append((fi, pi))
            elif runs and runs[-1][0] == lab:
                runs[-1][1] += 1
                runs[-1][2].append((fi, pi))
            else:
                runs.append([lab, 1, leading + [(fi, pi)]])
                leading = []
        merged = []
        for run in runs:
            if merged and (run[1] < min_run or merged[-1][0] == run[0]):
                merged[-1][2].extend(run[2])
                if merged[-1][0] == run[0]:
                    merged[-1][1] += run[1]
            else:
                merged.append(run)
        for run in merged[1:]:
            for fi, pi in run[2]:
                frames[fi]["p"][pi][5] = next_id
                changes[(fi, tid)] = next_id
            next_id += 1
            splits += 1
    for fi, f in enumerate(frames):
        f["num"] = [[changes.get((fi, r[0]), r[0]), *r[1:]] for r in f.get("num", [])]
    return splits


def short_links(frames, model, max_gap=1.0, margin=0.35):
    """Affectation globale, avec abstention et sans chevauchement temporel.

    Aucun raccord entre deux plans, à travers une homographie manquante, ou entre
    deux pistes visibles simultanément. Les coûts suivent l'essai panolink.
    """
    tracks = build_tracks(frames)
    assign_teams(tracks, model)
    transforms = {}
    C, sequence, last_shot, was_wide = np.eye(3), 0, None, False
    for f in frames:
        wide = f["view"] in ("wide", "medium")
        if not wide or not was_wide or f["shot"] != last_shot or f.get("H") is None:
            sequence += 1
            C = np.eye(3)
        else:
            H = np.asarray(f["H"], float).reshape(3, 3)
            try:
                candidate = C @ np.linalg.inv(H)
                if not np.isfinite(candidate).all():
                    raise ValueError("homographie non finie")
                C = candidate
            except (np.linalg.LinAlgError, ValueError):
                sequence += 1
                C = np.eye(3)
        if wide:
            transforms[(f["shot"], f["t"])] = (sequence, C.copy())
        last_shot, was_wide = f["shot"], wide
    trajectories = {}
    for tid, tr in tracks.items():
        pts = []
        for o in tr["obs"]:
            item = transforms.get((tr["shot"], o["t"]))
            if item is None or o["view"] not in ("wide", "medium"):
                continue
            sid, C = item
            x1, y1, x2, y2 = o["box"]
            p = C @ [(x1 + x2) / 2, y2, 1.0]
            if abs(p[2]) < 1e-8:
                continue
            pos = p[:2] / p[2]
            h = (y2 - y1) * np.sqrt(abs(np.linalg.det(C[:2, :2])))
            if np.isfinite(pos).all() and np.isfinite(h) and h > 0:
                pts.append((o["t"], sid, pos, h))
        if len(pts) >= 3 and tr.get("team") in ("raja", "opponent"):
            trajectories[tid] = pts
    # Résolution indépendante par équipe et repère : pas de matrice quadratique du match entier.
    ends, starts = defaultdict(list), defaultdict(list)
    for tid, pts in trajectories.items():
        tr = tracks[tid]
        # Ne pas raccorder un gros plan à un plan de terrain.
        if pts[0][0] != tr["t_start"] or pts[-1][0] != tr["t_end"]:
            continue
        for side, target in (("start", starts), ("end", ends)):
            endpoint = pts[0] if side == "start" else pts[-1]
            sel = [p for p in pts if p[1] == endpoint[1]]
            sel = sel[:3] if side == "start" else sel[-3:]
            P = np.median([p[2] for p in sel], axis=0)
            h = float(np.median([p[3] for p in sel]))
            tail = [p for p in pts if p[1] == endpoint[1] and p[0] >= endpoint[0] - 1]
            v = (tail[-1][2] - tail[0][2]) / max(1e-3, tail[-1][0] - tail[0][0]) if side == "end" and len(tail) >= 2 else np.zeros(2)
            target[(tr["shot"], endpoint[1], tr["team"])].append((tid, endpoint[0], P, h, v))
    links = {}
    big, no_link = 1e6, 2.5
    for key, a_list in ends.items():
        b_list = starts.get(key, [])
        if not b_list:
            continue
        cost = np.full((len(a_list), len(b_list)), big)
        for i, (a, ta, pa, ha, va) in enumerate(a_list):
            for j, (b, tb, pb, hb, _) in enumerate(b_list):
                gap = tb - ta
                if a == b or not (0 < gap <= max_gap + 1e-6) or not (0.6 <= hb / ha <= 1.6):
                    continue
                h = (ha + hb) / 2
                pred = pa + va * min(gap, 1.0) if np.linalg.norm(va) / h < 4.5 else pa
                d = float(np.linalg.norm(pb - pred)) / h
                if d <= 0.6 + 1.5 * gap:
                    cost[i, j] = d + 0.1 * gap
        full = np.full((len(a_list), len(b_list) + len(a_list)), big)
        full[:, :len(b_list)] = cost
        for i in range(len(a_list)):
            full[i, len(b_list) + i] = no_link
        rr, cc = linear_sum_assignment(full)
        for i, j in zip(rr, cc):
            if j >= len(b_list) or cost[i, j] >= no_link:
                continue
            row = np.delete(cost[i], j)
            col = np.delete(cost[:, j], i)
            if (len(row) and row.min() - cost[i, j] < margin) or (len(col) and col.min() - cost[i, j] < margin):
                continue
            links[a_list[i][0]] = b_list[j][0]
    return links


def reject_opposite_kit_observations(frames, model):
    """Écarte une tenue adverse isolée sans casser toute la continuité physique.

    La coupe à deux images tolère une erreur de couleur isolée ; cette tolérance
    ne doit pas attribuer l'image d'un adversaire aux statistiques du Raja.
    """
    tracks = build_tracks(frames)
    assign_teams(tracks, model)
    family = {"raja": "R", "raja_gk": "R", "opponent": "O", "opponent_gk": "O"}
    removed = 0
    for f in frames:
        if f["view"] not in ("wide", "medium"):
            continue
        rejected = set()
        for p in f["p"]:
            tid = p[5]
            if tid is None or p[6] is None:
                continue
            expected = family.get(tracks[tid].get("team"))
            cluster, _ = model.predict(np.asarray(p[6], np.float32))
            observed = family.get(model.label(cluster))
            if expected and observed and observed != expected:
                p[5] = None
                rejected.add(tid)
                removed += 1
        f["num"] = [r for r in f.get("num", []) if r[0] not in rejected]
    return removed


def refine_frames(frames, model):
    frames = copy.deepcopy(frames)
    removed_before = remove_short_tracks(frames)
    splits = split_by_team(frames, model)
    removed_after = remove_short_tracks(frames)
    opposite_kit_removed = reject_opposite_kit_observations(frames, model)
    removed_after_outliers = remove_short_tracks(frames)
    links = short_links(frames, model)
    previous = {b: a for a, b in links.items()}
    canonical = {}
    for f in frames:
        origins = {}
        for p in f["p"]:
            tid = p[5]
            if tid is None:
                continue
            if tid not in canonical:
                head = tid
                while head in previous:
                    head = previous[head]
                canonical[tid] = head
            p[5] = canonical[tid]
            origins[str(p[5])] = tid
        f["track_origins"] = origins
        f["num"] = [[canonical.get(r[0], r[0]), *r[1:]] for r in f.get("num", [])]
    return frames, {"version": REFINEMENT_VERSION, "splits": splits, "short_tracks_removed_before": removed_before,
                    "short_tracks_removed_after": removed_after, "links": {str(a): b for a, b in links.items()},
                    "opposite_kit_observations_removed": opposite_kit_removed,
                    "short_tracks_removed_after_outliers": removed_after_outliers,
                    "max_gap_s": 1.0, "min_hits": 3, "team_min_run": 2,
                    "cross_shot_links": False, "identity_gap_limit_s": 0.6}


def write_frames(path, frames):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in frames:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def refine_file(work, model):
    work = Path(work)
    raw = work / "frames.jsonl"
    target = work / "frames_tracked.jsonl"
    st = raw.stat()
    fingerprint = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                   "teams": hashlib.sha256(json.dumps(model.to_json(), sort_keys=True).encode()).hexdigest(),
                   "version": REFINEMENT_VERSION}
    report = read_json(work / "tracking_refine.json", {})
    if target.exists() and report.get("input") == fingerprint:
        return target, report
    begin = time.perf_counter()
    frames, report = refine_frames(load_frames(raw), model)
    write_frames(target, frames)
    report.update(input=fingerprint, compute_s=round(time.perf_counter() - begin, 3))
    write_json(work / "tracking_refine.json", report)
    return target, report
