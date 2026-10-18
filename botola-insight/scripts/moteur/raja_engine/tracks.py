"""Construction des pistes à partir de frames.jsonl, équipes et zones de risque d'échange.

Une « zone de risque » est un instant où deux pistes de la même équipe se croisent ou se
recouvrent : au-delà de ce point, l'identité d'une piste n'est plus garantie. Une réponse
humaine donnée sur une image ne se propage donc que jusqu'aux zones de risque les plus
proches (avant et après), jamais sur toute la piste par principe.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .teams import TeamModel
from .tracker import iou_matrix
from .util import read_jsonl


def load_frames(path):
    return read_jsonl(path)


def build_tracks(frames: list[dict]) -> dict[int, dict]:
    tracks: dict[int, dict] = {}
    for fr in frames:
        for row in fr["p"]:
            x1, y1, x2, y2, sc, tid, feat = row
            if tid is None:
                continue
            tr = tracks.setdefault(tid, {"id": tid, "shot": fr["shot"], "obs": [], "views": Counter(),
                                         "numbers": [], "seg_kind": fr.get("seg_kind")})
            tr["obs"].append({"t": fr["t"], "box": [x1, y1, x2, y2], "score": sc, "view": fr["view"],
                              "feat": feat, "W": fr.get("W"), "Himg": fr.get("Himg")})
            tr["views"][fr["view"]] += 1
        for tid, n, conf in fr.get("num", []):
            if tid in tracks:
                tracks[tid]["numbers"].append({"t": fr["t"], "number": n, "conf": conf})
    for tr in tracks.values():
        tr["obs"].sort(key=lambda o: o["t"])
        tr["t_start"] = tr["obs"][0]["t"]
        tr["t_end"] = tr["obs"][-1]["t"]
        tr["n_obs"] = len(tr["obs"])
        hs = [o["box"][3] - o["box"][1] for o in tr["obs"]]
        tr["median_h"] = float(np.median(hs))
        gaps = np.diff([o["t"] for o in tr["obs"]]) if len(tr["obs"]) > 1 else np.array([0.0])
        tr["max_gap_s"] = float(gaps.max()) if len(gaps) else 0.0
        tr["views"] = dict(tr["views"])
    return tracks


def fit_teams(frames: list[dict], k: int = 7, max_samples: int = 8000, seed: int = 0) -> tuple[TeamModel, dict]:
    feats = []
    for fr in frames:
        if fr["view"] not in ("wide", "medium"):
            continue
        for row in fr["p"]:
            if row[6] is not None:
                feats.append(row[6])
    feats = np.array(feats, np.float32)
    model = TeamModel(k)
    info = {"samples": int(len(feats))}
    if len(feats) < 40:
        info["status"] = "insufficient"
        return model, info
    rng = np.random.default_rng(seed)
    if len(feats) > max_samples:
        feats = feats[rng.choice(len(feats), max_samples, replace=False)]
    labels, counts = model.fit(feats, seed)
    info.update(status="ok", cluster_sizes=counts.tolist())
    return model, info


def assign_teams(tracks: dict, model: TeamModel) -> None:
    for tr in tracks.values():
        votes = Counter()
        clusters = Counter()
        for o in tr["obs"]:
            if o["feat"] is None or model.centers is None:
                continue
            c, margin = model.predict(np.array(o["feat"], np.float32))
            clusters[c] += 1
            votes[model.label(c)] += 1
        n = sum(votes.values())
        if n == 0:
            tr["team"], tr["team_conf"], tr["team_votes"] = None, 0.0, {}
            continue
        lab, cnt = votes.most_common(1)[0]
        tr["team"] = lab
        tr["team_conf"] = round(cnt / n, 3)
        tr["team_votes"] = dict(votes)
        tr["clusters"] = {int(k): v for k, v in clusters.items()}


KNOWN = {"raja", "raja_gk", "opponent", "opponent_gk", "referee"}


def risk_points(tracks: dict, iou_thr: float = 0.15, foot_thr: float = 0.6) -> None:
    """Ajoute tr['crossings'] (description complète, sans coupure, cf. perception.py) et
    tr['risk_times'] : instants où la règle (iou_thr, foot_thr) couperait.

    v1 : la décision de couper appartient à la couche 2 (identity_resolver). risk_times
    n'est conservé que pour le chemin historique (identity.py, solveur global-0.2) et pour
    mesurer le point zéro. Valeurs par défaut = règle historique stricte du moteur."""
    from .perception import crossing_cuts, describe_crossings
    describe_crossings(tracks)
    for tr in tracks.values():
        tr["risk_times"] = crossing_cuts(tr, iou_thr, foot_thr)


def safe_window(tr: dict, t: float) -> tuple[float, float]:
    """Fenêtre [a, b] autour de t sans croisement de même équipe ni trou > 0,6 s.
    Chemin historique ; en v1 la propagation est décidée par identity_resolver."""
    obs_t = [o["t"] for o in tr["obs"]]
    a, b = tr["t_start"], tr["t_end"]
    for rt in tr.get("risk_times", []):
        if rt < t - 1e-6:
            a = max(a, rt)
        elif rt > t + 1e-6:
            b = min(b, rt)
            break
        else:  # l'instant même de l'ancrage est un croisement : fenêtre minimale
            return t, t
    # trous de suivi
    for t1, t2 in zip(obs_t, obs_t[1:]):
        if t2 - t1 > 0.6:
            if t2 <= t:
                a = max(a, t2)
            elif t1 >= t:
                b = min(b, t1)
                break
    return a, b


def box_at(tr: dict, t: float, tol: float = 0.3):
    best = None
    for o in tr["obs"]:
        d = abs(o["t"] - t)
        if d <= tol and (best is None or d < best[0]):
            best = (d, o)
    return None if best is None else best[1]


def stitch(tracks: dict, frames: list[dict], max_gap_s: float = 1.2) -> dict:
    """Raccorde, dans un même plan, une piste qui s'arrête à une piste qui commence juste après,
    quand c'est la SEULE continuation plausible (même équipe, position compensée du mouvement de
    caméra, taille cohérente) et réciproquement. Renvoie {id_piste: id_suivante}.
    Les raccords servent seulement à prolonger une identité donnée par l'utilisateur ; ils sont
    marqués comme tels et jamais utilisés pour créer une identité."""
    import numpy as np
    by_shot = defaultdict(list)
    for fr in frames:
        by_shot[fr["shot"]].append(fr)
    for v in by_shot.values():
        v.sort(key=lambda f: f["t"])

    def warp(p, shot, t0, t1):
        x = np.array([p[0], p[1], 1.0])
        for fr in by_shot[shot]:
            if t0 < fr["t"] <= t1 + 1e-6:
                if fr.get("H") is None:
                    return None
                x = np.array(fr["H"], np.float64).reshape(3, 3) @ x
                x = x / x[2]
        return x[:2]

    def team(tr):
        return tr.get("team_override") or tr.get("team")

    ends = defaultdict(list)
    starts = defaultdict(list)
    for tr in tracks.values():
        if team(tr) not in KNOWN or tr.get("team_conf", 0) < 0.8:
            continue
        ends[tr["shot"]].append(tr)
        starts[tr["shot"]].append(tr)
    cand_next = defaultdict(list)
    cand_prev = defaultdict(list)
    for shot, trs in ends.items():
        for a in trs:
            oa = a["obs"][-1]
            if oa["view"] not in ("wide", "medium"):
                continue
            ha = oa["box"][3] - oa["box"][1]
            fa = ((oa["box"][0] + oa["box"][2]) / 2, oa["box"][3])
            for b in starts[shot]:
                if b["id"] == a["id"] or team(b) != team(a):
                    continue
                ob = b["obs"][0]
                gap = ob["t"] - oa["t"]
                if not (0 < gap <= max_gap_s) or ob["view"] not in ("wide", "medium"):
                    continue
                hb = ob["box"][3] - ob["box"][1]
                if not (0.7 <= hb / max(ha, 1e-6) <= 1.4):
                    continue
                pred = warp(fa, shot, oa["t"], ob["t"])
                if pred is None:
                    continue
                fb = np.array([(ob["box"][0] + ob["box"][2]) / 2, ob["box"][3]])
                d = float(np.linalg.norm(fb - pred)) / max(ha, hb)
                if d <= 0.5 + 2.5 * gap:
                    cand_next[a["id"]].append((d, b["id"]))
                    cand_prev[b["id"]].append((d, a["id"]))
    links = {}
    for a, cs in cand_next.items():
        if len(cs) == 1:
            b = cs[0][1]
            if len(cand_prev[b]) == 1:
                links[a] = b
    return links
