"""Infrastructure des commandes ; aucun algorithme du moteur n'est modifié."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Avant tout import ONNX : empêche aussi l'événement d'initialisation 1DS.
# https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
ENGINE = HERE / "moteur"
sys.path.insert(0, str(ENGINE))
VERSION = "1.0.0"
LIMIT = "Critère de l'étape 1 non atteint ; scores d'identité non calibrés."


def read(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"Clé JSON dupliquée : {k}")
            out[k] = v
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def slug(value):
    value = str(value)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
        raise ValueError("Identifiant attendu : lettres, chiffres, tirets, 1 à 80 caractères.")
    return value


def match(path):
    cfg = read(path)
    missing = [k for k in ("match_key", "opponent", "roster") if not cfg.get(k)]
    if missing:
        raise ValueError("Demander à l'utilisateur : " + ", ".join(missing))
    slug(cfg["match_key"])
    if not cfg.get("team_name") and not cfg["roster"].get("team"):
        raise ValueError("Nom de l'équipe suivie manquant (team_name / roster.team).")
    for key in ("roster", "opponent_roster"):
        if key not in cfg:
            continue
        players = cfg[key].get("players", [])
        if not players:
            raise ValueError(f"Demander la composition et les numéros : {key}.players est vide.")
        nums = set()
        for p in players:
            n = p.get("jersey_number")
            if type(n) is not int or not 0 <= n <= 99 or n in nums:
                raise ValueError(f"Numéro invalide ou dupliqué dans {key} : {n}")
            nums.add(n)
            if not p.get("player_name") or p.get("role") not in ("G", "D", "M", "A"):
                raise ValueError(f"Nom/poste manquant ou invalide pour le n°{n} ({key}).")
            if type(p.get("starter")) is not bool:
                raise ValueError(f"Titularisation à préciser pour le n°{n} ({key}).")
    return cfg


def team_name(cfg, label):
    focus = cfg.get("team_name") or cfg["roster"]["team"]
    return focus if label in ("raja", "raja_gk") else cfg["opponent"]


def load_teams(path):
    from raja_engine.teams import LABELS, TeamModel
    d = read(path)
    if not d.get("centers"):
        raise ValueError("Modèle de couleurs sans centres.")
    labels = d.get("cluster_labels", {})
    if set(labels) != {str(i) for i in range(len(d["centers"]))}:
        raise ValueError("Chaque groupe de couleurs doit avoir une étiquette.")
    if not set(labels.values()) <= set(LABELS):
        raise ValueError("Étiquette de couleur inconnue.")
    return TeamModel.from_json(d)


def record(out, command, inputs, parameters, outputs):
    data = {"command": command, "skill_version": VERSION,
            "inputs": {str(Path(p).resolve()): sha(p) for p in inputs if p is not None},
            "parameters": parameters,
            "outputs": {str(Path(p).resolve()): sha(p) for p in outputs}, "limitations": [LIMIT]}
    write(Path(out) / (command + ".manifest.json"), data)
    return data


def validate_readings(meta_path, readings_path):
    meta, data = read(meta_path), read(readings_path)
    if set(data) - {"reader", "protocol", "readings", "notes"}:
        raise ValueError("Champs de lectures inconnus ; respecter lectures.schema.json.")
    if not isinstance(data.get("reader"), str) or not data["reader"].strip():
        raise ValueError("reader doit identifier la session/le lecteur.")
    readings = data.get("readings")
    if not isinstance(readings, dict):
        raise ValueError("readings doit être un objet JSON.")
    index = {str(m["track"]): {str(c["c"]) for c in m["crops"]} for m in meta["planches"]}
    for tid, crops in readings.items():
        if not re.fullmatch(r"[0-9]+", tid) or tid not in index or not isinstance(crops, dict):
            raise ValueError(f"Piste inconnue ou lectures invalides : {tid}")
        for k, r in crops.items():
            if not re.fullmatch(r"[1-9][0-9]*", k) or k not in index[tid]:
                raise ValueError(f"Case absente : piste {tid}, c{k}")
            if r is None:
                continue
            if not isinstance(r, dict) or set(r) != {"number", "part", "conf"}:
                raise ValueError(f"Lecture {tid}/c{k} : seules number, part et conf sont admises.")
            if type(r["number"]) is not int or not 0 <= r["number"] <= 99:
                raise ValueError(f"Numéro invalide : {tid}/c{k}")
            if r["part"] not in ("back", "front", "short") or r["conf"] not in ("high", "medium", "low"):
                raise ValueError(f"Partie/confidence invalide : {tid}/c{k}")
    return meta, data


def rule_e():
    return read(SKILL / "assets/configs_v8.json")["configs"]["E_union"]["rule"]


def engine_integrity():
    expected = read(SKILL / "assets/engine_sha256.json")
    bad = [p for p, h in expected.items() if not (ENGINE / p).is_file() or sha(ENGINE / p) != h]
    if bad:
        raise ValueError("Moteur de référence modifié : " + ", ".join(bad))
    return len(expected)
