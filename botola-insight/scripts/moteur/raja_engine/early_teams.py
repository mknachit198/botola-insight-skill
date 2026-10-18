"""Apprentissage anticipé des couleurs d'équipe pendant que la détection tourne encore.

Ajuste le modèle de couleurs sur les images déjà analysées d'une session, produit les planches
d'exemples et crée les questions « couleurs » (mêmes identifiants que l'étape teams), afin que
l'utilisateur puisse répondre sans attendre la fin du calcul. L'étape teams réutilisera ensuite
ce modèle (teams.json) et les réponses.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .pipeline import MatchWorkspace
from .teams import LABEL_FR, LABELS
from .tracks import fit_teams, load_frames
from .util import write_json
from .videoio import VideoSource


def run(work: Path, match_cfg: Path, period: int = 1, min_frames: int = 600) -> dict:
    ws = MatchWorkspace(work, match_cfg)
    s = next(x for x in ws.sessions() if x.meta.get("period") == period)
    frames = load_frames(s.dir / "frames.jsonl")
    if len(frames) < min_frames:
        return {"status": "trop tôt", "frames": len(frames)}
    model, info = fit_teams(frames)
    if info["status"] != "ok":
        return {"status": "insuffisant", **info}
    video = VideoSource(s.meta["video_path"])
    s._team_mosaics(video, frames, model)
    video.close()
    write_json(ws.root / "teams.json", {**model.to_json(), "fit_info": {**info, "frames_used": len(frames),
                                                                        "note": "ajusté pendant la détection (anticipé)"},
                                        "fitted_on_session": s.dir.name})
    ids = []
    for c in range(len(model.centers)):
        tid = f"team_colors-{ws.cfg['match_key']}-cluster{c}"
        ws.store.add({"id": tid, "kind": "team_colors", "video": None, "period": None, "time": 0.0, "bbox": None,
                      "match_clock": None, "clock_status": None, "blocking": True, "priority": 90,
                      "question": "À qui correspondent les maillots de ce groupe ?",
                      "reason": f"Proposition automatique : {LABEL_FR[model.label(c)]}.",
                      "choices": [{"value": k, "label": LABEL_FR[k]} for k in LABELS],
                      "evidence": {"image": f"evidence/team_cluster_{c}.jpg"}, "origin": "engine", "cluster": c,
                      "proposed": LABEL_FR[model.label(c)]})
        ids.append(tid)
    ws.store.save()
    return {"status": "ok", "frames": len(frames), "clusters": len(ids), "sizes": info.get("cluster_sizes"),
            "proposals": {c: model.label(c) for c in range(len(model.centers))}}


if __name__ == "__main__":
    import json
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1), ensure_ascii=False))
