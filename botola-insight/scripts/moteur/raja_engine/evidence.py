"""Couche 1 — preuves d'identité (evidence.jsonl), une ligne par preuve, jamais fusionnées.

{"id": "vlm_plank:207:131.8:back", "type": "number_read", "track": 207, "t": 131.8,
 "player": 30, "confidence": 0.9, "source": "vlm_plank", "provenance": "planches/p207.jpg#c3",
 "holdout": false}

`player` est le numéro de maillot déclaré dans la composition du match (jamais un nom déduit
d'une apparence). Une preuve vaut pour UNE observation (piste, instant) ; sa propagation
est décidée par la couche 2. Les producteurs sont indépendants et remplaçables.
"""
from __future__ import annotations

import json
from pathlib import Path

TYPES = {"number_read", "banner_card", "banner_sub", "goalkeeper_kit", "captain_armband",
         "commentary_mention", "official_event", "user_answer", "replay_read"}
# Types qui désignent un candidat pondéré et non une étiquette (jamais publiés seuls).
SOFT_TYPES = {"commentary_mention", "official_event"}
REQUIRED = ("id", "type", "track", "t", "player", "confidence", "source", "provenance", "holdout")


class EvidenceError(ValueError):
    pass


def make(type: str, track: int, t: float, player, confidence: float, source: str, provenance: str,
         holdout: bool = False, part: str | None = None, **extra) -> dict:
    key = f"{source}:{track}:{t:.2f}" + (f":{part}" if part else "")
    ev = {"id": key, "type": type, "track": int(track), "t": round(float(t), 3), "player": player,
          "confidence": round(float(confidence), 3), "source": source, "provenance": provenance,
          "holdout": bool(holdout)}
    if part:
        ev["part"] = part
    ev.update(extra)
    validate(ev)
    return ev


def validate(ev: dict) -> None:
    for k in REQUIRED:
        if k not in ev:
            raise EvidenceError(f"champ manquant {k!r} dans {ev.get('id')}")
    if ev["type"] not in TYPES:
        raise EvidenceError(f"type inconnu {ev['type']!r}")
    if not 0.0 <= float(ev["confidence"]) <= 1.0:
        raise EvidenceError("confidence hors [0, 1]")
    if ev["type"] == "commentary_mention":
        # Un nom prononcé donne des candidats pondérés, jamais une étiquette.
        if not isinstance(ev["player"], (list, dict)):
            raise EvidenceError("commentary_mention : player doit être une liste/dict de candidats")
    elif ev["player"] is not None and not isinstance(ev["player"], int):
        raise EvidenceError("player doit être un numéro entier ou null")
    if ev.get("physical_traits_used"):
        raise EvidenceError("identification par caractéristiques physiques interdite")


def write(path, items) -> None:
    seen = set()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for ev in items:
            validate(ev)
            if ev["id"] in seen:
                raise EvidenceError(f"identifiant dupliqué {ev['id']}")
            seen.add(ev["id"])
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")


def read(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ev = json.loads(line)
                validate(ev)
                out.append(ev)
    return out


def from_legacy_dataset(dataset: dict, source: str = "legacy_v7_annotation") -> list[dict]:
    """Les 175 lectures du relais v7 (dataset.json du solveur), rattachées aux pistes réelles.
    Provenance : groupe d'origine et instant global ; confiance historique non calibrée (0,9)."""
    by_id = {u["id"]: u for u in dataset["nodes"]}
    out, seen = [], set()
    for a in dataset["anchors"]:
        track = by_id[a["node"]]["parent_track"]
        key = (track, round(a["local_t"], 2))
        if key in seen:  # deux recadrages historiques peuvent tomber sur la même observation
            continue
        seen.add(key)
        out.append(make("number_read", track, a["local_t"], int(a["number"]), 0.9, source,
                        f"experiences_partie7/reading/readings.json#groupe{a['group']}@{a['global_t']}",
                        part="legacy", legacy_iou=round(a["iou"], 3)))
    return out
