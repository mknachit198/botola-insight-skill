"""Import du dossier V2 (événements relus, demandes en attente, xG du but) dans le dossier du match.

Rien n'est recompté ni requalifié automatiquement : les événements gardent leur source
« v2_assistant_review », leur fiabilité V2 et leur statut compté / en attente.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .events import EVENT_CHOICES, import_v2
from .util import read_json, write_json


def run(v2_dir: Path, match_dir: Path, store=None) -> dict:
    v2_dir, match_dir = Path(v2_dir), Path(match_dir)
    res = read_json(v2_dir / "resultats.json")
    journal = read_json(v2_dir / "journal_relecture.json")
    cfg = read_json(match_dir / "match.json")
    events = import_v2(res, cfg["match_key"])
    write_json(match_dir / "v2_events.json", events)
    geo = res.get("shot_geometry", {})
    xg_items = []
    if geo.get("xg_point") is not None:
        lo, hi = geo.get("xg_location_sensitivity_5_95", [None, None])
        common = {"evidence": [{"clip": "05_But_26m28_26m38.mp4", "clock": "26:33"}], "shots_located": 1}
        xg_items.append({"jersey_number": 19, "key": "xg", "value": geo["xg_point"],
                         "method": ("Régression logistique (distance, angle, tête, coup franc) entraînée sur StatsBomb Open Data "
                                    "(Mondial 2018 + Euro 2020), test Mondial 2022 ; position du tir par calibration ponctuelle "
                                    "d'une image du ralenti (dossier V2)."),
                         "reason": (f"Seul tir localisé (but de 26:33) ; sensibilité spatiale {lo:.2f}–{hi:.2f}, "
                                    "pas un intervalle de confiance ; modèle non validé sur la Botola."), **common})
        xg_items.append({"jersey_number": 14, "key": "xa", "value": geo["xg_point"],
                         "method": "xA = xG du tir directement assisté (passe de Bakiz sur le but de 26:33).",
                         "reason": "Une seule passe décisive localisée ; ne pas additionner xG et xA.", **common})
    write_json(match_dir / "xg_items.json", xg_items)
    # extraits des demandes en attente, pour pouvoir y répondre sans les vidéos complètes
    clips_dir = match_dir / "v2_clips"
    clips_dir.mkdir(exist_ok=True)
    copied = []
    for r in res.get("requests", []):
        src = v2_dir / "clips" / r["clip"]
        if src.exists():
            shutil.copy(src, clips_dir / r["clip"])
            copied.append(r["clip"])
    n_tasks = 0
    if store is not None:
        jobs = {j["name"]: j for j in journal.get("jobs", [])}
        fps = {v["period"]: v["fingerprint"] for v in cfg.get("videos", [])}
        v2map = {"chraibi09": "chraibi_volley", "bettaieb18": "bettaieb_acrobatie", "keeper84": "keeper84_pending",
                 "keeper91": "keeper91_pending", "centre21": "centre21", "barkok88": "barkok_foul_won"}
        for r in res.get("requests", []):
            job = jobs.get(r["clip"])
            if not job:
                continue
            ev = next((e for e in events if e["id"] == v2map.get(r["id"])), None)
            task = {"id": f"v2req-{r['id']}", "kind": "event_review", "video": fps.get(job["half"]), "period": job["half"],
                    "time": float(job["internal_start"]), "end": float(job["internal_start"] + job["duration"]), "bbox": None,
                    "match_clock": r["clock"], "clock_status": "observed_manual_v2", "blocking": False, "priority": 50,
                    "question": r["question"],
                    "reason": ("Demande V2 en attente." + (f" Lecture V2 : {ev['type']} / {ev['outcome']} ({ev['reliability']})." if ev else "")),
                    "choices": EVENT_CHOICES, "player_choice": True,
                    "evidence": {"v2_clip": f"v2_clips/{r['clip']}"}, "origin": "imported_v2_requests"}
            _, st = store.add(task)
            n_tasks += st == "created"
        store.save()
    return {"events": len(events), "counted": sum(e["counted_in_metrics"] for e in events), "xg_items": len(xg_items),
            "clips_copied": copied, "tasks_created": n_tasks}
