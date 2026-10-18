"""Couche 1 — perception : pistes (tracklets) et description des croisements.

Aucune décision d'identité ici. Un croisement est DÉCRIT (instant, piste croisée,
recouvrement IoU, distance des pieds rapportée à la hauteur, même équipe ou non) ;
c'est la couche 2 (identity_resolver) qui décide s'il coupe ou s'il pèse.

Format d'une ligne de tracklets.jsonl :
{"id": 207, "shot": 14, "team": "raja", "team_conf": 0.97, "t_start": 101.2, "t_end": 148.4,
 "obs": [{"t": 101.2, "fi": 3036, "box": [x1, y1, x2, y2], "view": "wide", "H_ok": true}],
 "crossings": [{"t": 112.4, "with": 231, "with_team": "raja", "same_team": true,
                "iou": 0.21, "foot_dist_h": 0.4, "view": "wide"}],
 "gaps": [{"from": 120.0, "to": 121.4, "s": 1.4}]}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tracker import iou_matrix

VERSION = "perception-1.0"
KNOWN = {"raja", "raja_gk", "opponent", "opponent_gk", "referee"}
# Seuils purement DESCRIPTIFS : on consigne tout contact plausible, la règle vient après.
DESCRIBE_FOOT_H = 1.0
DET_STEP_S = 0.2


def _team(tr):
    return tr.get("team_override") or tr.get("team")


def describe_crossings(tracks: dict, foot_desc: float = DESCRIBE_FOOT_H) -> None:
    """Ajoute tr['crossings'] : chaque contact (IoU > 0 ou pieds < foot_desc × hauteur en plan
    large) avec une autre piste du même plan, sans rien couper. Valeurs non arrondies."""
    by_t = defaultdict(list)
    views = {}
    for tr in tracks.values():
        tr["crossings"] = []
        for o in tr["obs"]:
            by_t[(tr["shot"], o["t"])].append((tr["id"], o["box"]))
            views[(tr["shot"], o["t"])] = o["view"]
    for (shot, t), items in by_t.items():
        if len(items) < 2:
            continue
        ids = [i for i, _ in items]
        boxes = np.array([b for _, b in items], np.float32)
        iou = iou_matrix(boxes, boxes)
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
        hh = np.maximum(boxes[:, 3] - boxes[:, 1], 8)
        view = views.get((shot, t))
        wide = view in ("wide", "medium")
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                d = float(np.linalg.norm(feet[a] - feet[b]) / max(hh[a], hh[b]))
                io = float(iou[a, b])
                if not (io > 0 or (wide and d < foot_desc)):
                    continue
                ta, tb = tracks[ids[a]], tracks[ids[b]]
                la, lb = _team(ta), _team(tb)
                # équipes différentes et toutes deux connues : pas de confusion d'identité
                same = not (la in KNOWN and lb in KNOWN and la != lb)
                for me, other, lo in ((ta, tb, lb), (tb, ta, la)):
                    me["crossings"].append({"t": t, "with": other["id"], "with_team": lo, "same_team": same,
                                            "iou": io, "foot_dist_h": d if wide else None, "view": view})
    for tr in tracks.values():
        tr["crossings"].sort(key=lambda c: (c["t"], c["with"]))


def crossing_cuts(tr: dict, iou_thr: float, foot_thr: float = 0.0, close_iou_thr: float = 0.45,
                  any_team: bool = False) -> list[float]:
    """Instants où une RÈGLE donnée couperait la piste (même équipe seulement, sauf any_team).
    Règle historique du moteur : iou_thr=0,15, foot_thr=0,6. Règle « fort recouvrement » :
    iou_thr=0,4, foot_thr=0. Aucune coupure : iou_thr > 1. any_team=True : un recouvrement avec
    un adversaire ou un arbitre compte aussi (le suivi peut sauter d'un Raja à un autre Raja
    pendant qu'un adversaire les masque : chaînes 157 et 174 de la partie 7)."""
    cuts = set()
    for c in tr.get("crossings", []):
        if not c["same_team"] and not any_team:
            continue
        wide = c["view"] in ("wide", "medium")
        near = (c["foot_dist_h"] is not None and c["foot_dist_h"] < foot_thr) if wide else False
        if c["iou"] > (iou_thr if wide else close_iou_thr) or near:
            cuts.add(c["t"])
    return sorted(cuts)


def gaps(tr: dict, step: float = DET_STEP_S) -> list[dict]:
    ts = [o["t"] for o in tr["obs"]]
    return [{"from": a, "to": b, "s": round(b - a, 3)} for a, b in zip(ts, ts[1:]) if b - a > 1.5 * step + 1e-6]


def tracklets(tracks: dict, frames: list[dict], teams: set | None = None) -> list[dict]:
    """Sérialise les pistes au format de la couche 1 (croisements décrits, trous listés)."""
    if any("crossings" not in tr for tr in tracks.values()):
        describe_crossings(tracks)
    meta = {(f["shot"], f["t"]): f for f in frames}
    out = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        if teams is not None and _team(tr) not in teams:
            continue
        obs = []
        for o in tr["obs"]:
            f = meta.get((tr["shot"], o["t"]), {})
            obs.append({"t": o["t"], "fi": f.get("fi"), "box": [round(float(v), 1) for v in o["box"]],
                        "view": o["view"], "H_ok": f.get("H") is not None, "kit": o.get("kit")})
        out.append({"id": tid, "shot": tr["shot"], "team": _team(tr), "team_conf": tr.get("team_conf"),
                    "t_start": tr["t_start"], "t_end": tr["t_end"], "n_obs": len(obs), "obs": obs,
                    "crossings": [{**c, "iou": round(c["iou"], 3),
                                   "foot_dist_h": None if c["foot_dist_h"] is None else round(c["foot_dist_h"], 3)}
                                  for c in tr["crossings"]],
                    "gaps": gaps(tr), "perception_version": VERSION})
    return out


def write_tracklets(path, items) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_tracklets(path) -> dict[int, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                it = json.loads(line)
                out[it["id"]] = it
    return out


def load_part(frames_path, teams_json, views=("wide", "medium")):
    """Charge un frames_tracked.jsonl, attribue les équipes et décrit les croisements.
    Les observations hors plan large/moyen sont écartées (comme dans les mesures v7)."""
    from .teams import TeamModel
    from .tracks import assign_teams, build_tracks
    from .util import read_json, read_jsonl
    frames = read_jsonl(frames_path)
    frames = [{**f, "p": f["p"] if f["view"] in views else []} for f in frames]
    model = TeamModel.from_json(read_json(teams_json))
    tracks = build_tracks(frames)
    assign_teams(tracks, model)
    observation_kits(tracks, model)
    describe_crossings(tracks)
    return frames, model, tracks


def observation_kits(tracks: dict, model) -> None:
    """Étiquette de TENUE par observation (couleur du maillot, comme pour l'équipe de la piste).
    Sert à repérer, dans une piste Raja, les images où la boîte a glissé sur un arbitre, un
    adversaire ou le gardien. Équipement uniquement, aucune caractéristique physique."""
    for tr in tracks.values():
        for o in tr["obs"]:
            if o.get("feat") is None or model.centers is None:
                o["kit"] = None
                continue
            c, _ = model.predict(np.array(o["feat"], np.float32))
            o["kit"] = model.label(c)
