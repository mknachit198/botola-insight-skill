"""Report des réponses d'une session vidéo vers une autre session du même match
(ex. enregistrement 854×480 → même diffusion en 1920×1080).

- Les instants sont convertis par le recalage (align.py) ; les cadres et les repères de
  calibration sont normalisés (0–1), donc valables d'une résolution à l'autre si le cadrage est
  identique (même diffusion : vérifié par l'écart des vignettes lors du recalage).
- Seules les réponses liées à une vidéo sont reportées : identités, calibrations, composition.
  Les actions (ralentis) ne le sont pas : elles sont comptées par chrono de match, une seule fois,
  quelle que soit la vidéo. Les conflits ne le sont pas : ils dépendent des pistes de suivi et
  seront recalculés.
- Chaque réponse garde son origine (humaine, Claude…) et sa fiabilité ; la tâche reportée porte
  « transferred_from » et une note.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .questions import task_id

KINDS = ("identity", "calibration", "roster")
STD = {"value", "label", "note", "reviewedAt", "scope", "origin"}


def _shot_at(frames_path: Path):
    ts, shots = [], []
    with open(frames_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ts.append(r["t"]); shots.append(r["shot"])
    ts = np.array(ts)

    def f(t):
        i = int(np.argmin(np.abs(ts - t)))
        return shots[i] if abs(ts[i] - t) < 0.5 else None
    return f


def transfer(ws, src_session, dst_session, tmap, kinds=KINDS) -> dict:
    store = ws.store
    src_fp, dst_fp = src_session.fp, dst_session.fp
    shot_at = _shot_at(dst_session.dir / "frames.jsonl") if (dst_session.dir / "frames.jsonl").exists() else (lambda t: None)
    stats = {"transferred": 0, "already": 0, "skipped_pending": 0, "by_kind": {}}
    for t in list(store.tasks.values()):
        if t.get("video") != src_fp or t["kind"] not in kinds:
            continue
        if t["status"] == "pending" or t.get("response") is None:
            stats["skipped_pending"] += 1
            continue
        tn = round(tmap(float(t["time"])), 3)
        new = {k: v for k, v in t.items() if k not in ("id", "status", "response", "history", "video", "time", "end",
                                                        "track_id", "shot", "transferred_to")}
        new.update(video=dst_fp, time=tn, track_id=None, origin="transfert", transferred_from=t["id"])
        if t.get("end") is not None:
            new["end"] = round(tmap(float(t["end"])), 3)
        if t["kind"] == "calibration":
            new["shot"] = shot_at(tn)
        new["id"] = task_id(dst_fp, t["kind"], tn, t.get("bbox"))
        ex, status = store.add(new)
        if ex["status"] != "pending" and (ex.get("response") or {}).get("transferred_from") == t["id"]:
            stats["already"] += 1
            continue
        r = t["response"]
        extra = {k: v for k, v in r.items() if k not in STD}
        extra["transferred_from"] = t["id"]
        note = (r.get("note") or "")
        note = (note + " " if note else "") + "[réponse reportée depuis l'enregistrement 854×480, même diffusion]"
        store.answer(ex["id"], r.get("value"), label=r.get("label") or "", note=note, status=t["status"],
                     extra=extra, origin=r.get("origin") or "human")
        t["transferred_to"] = ex["id"]
        stats["transferred"] += 1
        stats["by_kind"][t["kind"]] = stats["by_kind"].get(t["kind"], 0) + 1
    store.save()
    return stats
