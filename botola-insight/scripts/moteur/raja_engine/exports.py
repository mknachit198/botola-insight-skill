"""Exports : JSON complet, CSV lisibles, et charges utiles conformes à l'API WordPress V1.

Le portail (API_MOTEUR.md) n'accepte que des tâches `identity` dans /video-session et les
portées visible_segments / reviewed_events / model_estimate dans /analysis. Les autres types
de questions et les candidats automatiques restent dans l'interface locale du moteur.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .util import write_json

PORTAL_SCOPES = {"visible_segments", "reviewed_events", "model_estimate"}


def write_player_csv(stats: dict, path: Path):
    keys = []
    for p in stats["players"]:
        for k in p["metrics"]:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        head = ["numero", "joueur", "statut_composition"]
        for k in keys:
            head += [k, f"{k}__fiabilite", f"{k}__portee"]
        w.writerow(head)
        for p in stats["players"]:
            row = [p["jersey_number"], p["player_name"], p.get("participation_status")]
            for k in keys:
                m = p["metrics"].get(k)
                if m is None:
                    row += ["", "", ""]  # vide = inconnu, jamais zéro
                else:
                    row += [m["value"], m["reliability"], m["scope"]]
            w.writerow(row)


def write_events_csv(events: list[dict], path: Path):
    cols = ["source", "type", "periode", "chrono", "statut_chrono", "numero", "joueur", "issue", "fiabilite",
            "preuve", "note"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(cols)
        for e in events:
            w.writerow([e.get(c, "") if e.get(c) is not None else "" for c in
                        ["source", "type", "period", "clock", "clock_status", "jersey_number", "player_name", "outcome",
                         "reliability", "evidence", "note"]])


def flatten_events(stats: dict, v2_events, human_events, banner_events, goal_events) -> list[dict]:
    out = []
    for e in v2_events:
        out.append({"source": "v2_assistant_review", "type": e["type"], "clock": e["match_clock"], "clock_status": "observed_manual_v2",
                    "jersey_number": e.get("jersey_number"), "player_name": e.get("player_name"), "outcome": e.get("outcome"),
                    "reliability": e.get("reliability"), "evidence": e.get("clip"),
                    "note": ("compté" if e.get("counted_in_metrics") else "en attente, non compté")})
    for e in human_events:
        claude = e.get("answer_origin") == "claude"
        out.append({"source": "claude_review" if claude else "human_review", "type": e["type"], "period": e.get("period"),
                    "clock": e.get("match_clock"),
                    "clock_status": e.get("clock_status"), "jersey_number": e.get("jersey_number"),
                    "player_name": e.get("player_name"), "outcome": e.get("outcome"), "reliability": "moyenne" if claude else "élevée",
                    "evidence": e.get("task_id"), "note": e.get("note")})
    for b in banner_events:
        for p in b.get("players", []):
            out.append({"source": "auto_banner", "type": b["type"], "period": b.get("period"), "clock": b.get("match_clock_banner_start"),
                        "clock_status": b.get("clock_status"), "jersey_number": p.get("jersey_number"),
                        "player_name": p.get("player_name") or p.get("name_read"), "outcome": None,
                        "reliability": "élevée" if not b.get("needs_review") else "à vérifier",
                        "evidence": f"fichier {b['file_start']:.1f}–{b['file_end']:.1f} s", "note": b.get("note")})
    for g in goal_events:
        out.append({"source": "scoreboard", "type": "but_equipe", "period": g.get("period"), "clock": g.get("match_clock_score_update"),
                    "clock_status": g.get("clock_status"), "jersey_number": None, "player_name": g.get("team"),
                    "outcome": "-".join(map(str, g.get("score_after", []))), "reliability": "élevée" if not g.get("needs_review") else "à vérifier",
                    "evidence": f"fichier {g['file_t_score_update']:.1f} s", "note": g.get("note")})
    return out


def portal_analysis_payload(stats: dict) -> dict:
    metrics = []
    for p in stats["players"]:
        for k, m in p["metrics"].items():
            if m["scope"] not in PORTAL_SCOPES or m["value"] is None:
                continue
            v = float(m["value"])
            if v < 0:
                continue
            cov = m.get("coverage") or {}
            method = m["method"]
            if cov:
                method += " Couverture : " + ", ".join(f"{a}={b}" for a, b in cov.items() if not isinstance(b, (dict, list)))
            item = {"jersey_number": p["jersey_number"], "key": k, "label": m["label"], "value": v, "unit": m["unit"],
                    "scope": m["scope"], "method": method[:900],
                    "reliability_reason": f"{m['reliability']} — {m['reliability_reason']}"[:900]}
            if m["unit"] in ("m", "km", "km/h"):
                item["calibrated_pitch"] = bool(m.get("calibrated_pitch"))
                if not item["calibrated_pitch"]:
                    continue
            metrics.append(item)
    return {"version": 1, "match_key": stats["match_key"], "metrics": metrics}


def portal_video_session_payload(match_key: str, period: int, video_identity: dict, duration: float, store) -> dict:
    tasks = []
    for t in sorted(store.tasks.values(), key=lambda x: x.get("time") or 0):
        if t["kind"] != "identity" or t.get("video") != video_identity["fingerprint"] or t.get("origin") == "imported_corrections":
            continue
        tasks.append({"id": t["id"], "kind": "identity", "time": float(t["time"]), "bbox": t.get("bbox"),
                      "match_clock": t.get("match_clock") if t.get("clock_status") == "observed_engine" else None,
                      "reason": t.get("reason", "Identité à confirmer")})
    return {"version": 1, "match_key": match_key, "period": period,
            "video": {"name": video_identity["name"], "size": video_identity["size"],
                      "fingerprint": video_identity["fingerprint"], "fingerprint_method": video_identity["fingerprint_method"],
                      "duration_seconds": float(duration)},
            "tasks": tasks}


def portal_corrections_payload(video_identity: dict, store) -> dict:
    tasks = []
    for t in store.tasks.values():
        if t["kind"] != "identity" or t.get("video") != video_identity["fingerprint"] or t.get("origin") == "imported_corrections":
            continue
        tasks.append({"id": t["id"], "kind": "identity", "time": float(t["time"]), "bbox": t.get("bbox"),
                      "status": t["status"] if t["status"] in ("pending", "resolved", "unknown") else "pending",
                      "response": t.get("response")})
    return {"version": 1, "video": {k: video_identity[k] for k in ("name", "size", "fingerprint", "fingerprint_method")},
            "tasks": tasks}


def write_all(out_dir: Path, stats: dict, events: list[dict]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "stats_joueurs.json", stats)
    write_player_csv(stats, out_dir / "stats_par_joueur.csv")
    write_events_csv(events, out_dir / "evenements.csv")
    write_json(out_dir / "import_portail_analysis.json", portal_analysis_payload(stats))
