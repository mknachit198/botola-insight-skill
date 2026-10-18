"""File de questions humaines persistante.

- questions.json : état courant des demandes (une entrée par identifiant stable).
- answers.jsonl  : journal en ajout seul de TOUTES les réponses données (jamais réécrit).
  Une nouvelle réponse à la même demande s'ajoute ; l'historique reste consultable.
- Les identifiants sont déterministes (fichier + type + temps + cadre) : relancer le moteur
  recrée les mêmes demandes et retrouve les mêmes réponses.
- Une demande existante dont le temps ou le cadre changerait est refusée (comme le portail).
- Les réponses importées (corrections V1) sont recopiées telles quelles, avec leur origine.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from .util import append_jsonl, now_iso, read_json, read_jsonl, write_json

KINDS = {
    "identity": "Identité d'un joueur dans un cadre",
    "roster": "Correction de composition",
    "team_colors": "Couleurs des équipes",
    "clock_read": "Lecture manuelle du chrono",
    "event_review": "Nature d'une action / d'un ralenti",
    "banner_check": "Bandeau (carton, remplacement) mal lu",
    "calibration": "Points du terrain pour la calibration",
    "conflict": "Conflit entre deux réponses",
}


def task_id(video_fp: str, kind: str, t: float, bbox=None, extra: str = "") -> str:
    key = json.dumps([video_fp, kind, round(float(t), 2), None if bbox is None else [round(float(v), 4) for v in bbox], extra])
    return f"{kind}-{video_fp[:8]}-{int(round(t * 10)):06d}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


class QuestionStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.path = self.folder / "questions.json"
        self.log = self.folder / "answers.jsonl"
        self._lock = threading.RLock()
        self.tasks: dict[str, dict] = {t["id"]: t for t in read_json(self.path, {"tasks": []})["tasks"]}
        self._replay_log()

    # ------------------------------------------------------------------ persistance
    def _replay_log(self):
        """Réapplique le journal : si questions.json a été régénéré, aucune réponse n'est perdue."""
        for rec in read_jsonl(self.log):
            t = self.tasks.get(rec["task_id"])
            if t is None:
                continue
            if t.get("response") != rec["response"] or t.get("status") != rec["status"]:
                if t.get("response") is not None and t["response"] != rec["response"]:
                    t.setdefault("history", []).append(t["response"])
                t["response"] = rec["response"]
                t["status"] = rec["status"]

    def save(self):
        """Écrit l'état en FUSIONNANT avec le fichier sur disque : une demande ajoutée par un autre
        processus (interface, autre partie, outil d'import) n'est jamais effacée."""
        with self._lock:
            disk = {t["id"]: t for t in read_json(self.path, {"tasks": []})["tasks"]}
            for tid, t in disk.items():
                mine = self.tasks.get(tid)
                if mine is None:
                    self.tasks[tid] = t
                elif t.get("response") and (not mine.get("response") or
                                            (t["response"].get("reviewedAt") or "") > (mine["response"].get("reviewedAt") or "")):
                    mine["response"], mine["status"] = t["response"], t["status"]
            tasks = sorted(self.tasks.values(), key=lambda x: (x.get("period") or 0, x.get("time") or 0, x["id"]))
            write_json(self.path, {"version": 1, "updated_at": now_iso(), "tasks": tasks})

    def _latest_logged(self, tid: str):
        last = None
        for rec in read_jsonl(self.log):
            if rec.get("task_id") == tid:
                last = rec
        return last

    # ------------------------------------------------------------------ création
    def add(self, task: dict) -> tuple[dict, str]:
        """Ajoute une demande. Renvoie (demande, 'created'|'existing'|'conflict')."""
        with self._lock:
            ex = self.tasks.get(task["id"])
            if ex is not None:
                if ex.get("time") != task.get("time") or ex.get("bbox") != task.get("bbox") or ex.get("kind") != task.get("kind"):
                    return ex, "conflict"
                # mise à jour des champs descriptifs seulement (preuves régénérées, priorité)
                for k in ("evidence", "priority", "reason", "question", "choices", "match_clock", "clock_status", "proposed", "end"):
                    if k in task:
                        ex[k] = task[k]
                return ex, "existing"
            task = dict(task)
            task.setdefault("status", "pending")
            task.setdefault("response", None)
            # une réponse déjà journalisée pour cet identifiant est réappliquée (jamais perdue)
            if task["status"] == "pending":
                rec = self._latest_logged(task["id"])
                if rec is not None:
                    task["status"], task["response"] = rec["status"], rec["response"]
            task.setdefault("blocking", False)
            task.setdefault("created_at", now_iso())
            self.tasks[task["id"]] = task
            return task, "created"

    def import_corrections(self, corrections: dict, roster_lookup, period: int, source_name: str) -> dict:
        """Importe un export de corrections V1 (raja-corrections_*.json) sans altérer les réponses."""
        n_new = n_kept = 0
        for t in corrections.get("tasks", []):
            task = {
                "id": t["id"], "kind": t["kind"], "time": t["time"], "end": t.get("end"), "bbox": t.get("bbox"),
                "match_clock": t.get("match_clock"), "clock_status": t.get("clock_status"),
                "period": t.get("period", period), "reason": t.get("reason"), "track_id": None,
                "video": corrections.get("video", {}).get("fingerprint"),
                "origin": "imported_corrections", "imported_from": source_name, "blocking": False,
                "question": t.get("reason"),
            }
            ex, status = self.add(task)
            if status == "created":
                ex["status"] = t.get("status", "pending")
                ex["response"] = t.get("response")
                n_new += 1
                if ex["response"] is not None:
                    append_jsonl(self.log, {"task_id": ex["id"], "status": ex["status"], "response": ex["response"],
                                            "at": now_iso(), "origin": "import:" + source_name})
            else:
                n_kept += 1
        self.save()
        return {"imported": n_new, "already_present": n_kept}

    # ------------------------------------------------------------------ réponses
    def answer(self, tid: str, value, label: str = "", note: str = "", status: str = "resolved",
               extra: dict | None = None, origin: str = "human") -> dict:
        if status not in ("resolved", "unknown", "skipped"):
            raise ValueError("statut invalide")
        with self._lock:
            t = self.tasks[tid]
            resp = {"value": value, "label": label, "note": note[:2000], "reviewedAt": now_iso(),
                    "scope": "observation_only" if t["kind"] in ("identity", "roster") else "task_only",
                    "origin": origin}
            if extra:
                resp.update(extra)
            if t.get("response") is not None:
                t.setdefault("history", []).append(t["response"])
            t["response"] = resp
            t["status"] = status
            append_jsonl(self.log, {"task_id": tid, "status": status, "response": resp, "at": resp["reviewedAt"]})
            self.save()
            return t

    # ------------------------------------------------------------------ consultation
    def pending(self, blocking_only=False, kinds=None):
        out = [t for t in self.tasks.values() if t["status"] == "pending"
               and (not blocking_only or t.get("blocking")) and (kinds is None or t["kind"] in kinds)]
        return sorted(out, key=lambda x: (not x.get("blocking"), -(x.get("priority") or 0), x.get("time") or 0))

    def resolved(self, kind=None):
        return [t for t in self.tasks.values() if t["status"] == "resolved" and (kind is None or t["kind"] == kind)]

    def counts(self):
        c = {}
        for t in self.tasks.values():
            c.setdefault(t["kind"], {}).setdefault(t["status"], 0)
            c[t["kind"]][t["status"]] += 1
        return c
