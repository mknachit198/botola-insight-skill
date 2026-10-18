"""Composition du match et règles de correspondance numéro ↔ joueur."""
from __future__ import annotations

import difflib
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    return "".join(ch for ch in s if ch.isalnum() or ch == " ").strip()


class Roster:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.players = cfg["roster"]["players"]
        self.by_number = {p["jersey_number"]: p for p in self.players}
        self.by_source_number = {p.get("source_capture_jersey_number", p["jersey_number"]): p for p in self.players}

    def from_answer_value(self, value) -> dict | None:
        """Les réponses V1 utilisent le numéro de la capture (ex. 20 pour Dahmani, qui porte 11)."""
        try:
            n = int(value)
        except (TypeError, ValueError):
            return None
        return self.by_source_number.get(n) or self.by_number.get(n)

    def match_banner_line(self, number: int | None, name: str | None) -> tuple[dict | None, float, str]:
        """Rapproche une ligne de bandeau lue (bruitée) d'un joueur de la composition."""
        cand = []
        nn = _norm(name or "")
        for p in self.players:
            last = _norm(p["player_name"]).split(" ")[-1]
            full = _norm(p["player_name"])
            sim = max(difflib.SequenceMatcher(None, nn, last).ratio(), difflib.SequenceMatcher(None, nn, full).ratio()) if nn else 0.0
            num_ok = number is not None and number == p["jersey_number"]
            score = sim + (0.6 if num_ok else 0.0)
            cand.append((score, sim, num_ok, p))
        cand.sort(key=lambda x: -x[0])
        best = cand[0]
        second = cand[1][0] if len(cand) > 1 else 0
        if best[2] and best[1] >= 0.5:
            return best[3], min(1.0, best[0] / 1.6), "numero_et_nom"
        if best[1] >= 0.75 and best[0] - second > 0.2:
            return best[3], best[1] * 0.8, "nom_seul"
        if best[2] and best[1] < 0.5:
            return best[3], 0.4, "numero_seul_nom_divergent"
        return None, 0.0, "non_rapproche"

    def choices(self, clock_s: float | None = None):
        """Choix de réponse. Si le chrono est connu, les joueurs déclarés sur le terrain à cet
        instant passent en tête ; les autres restent proposés (la composition peut être fausse)
        avec une mention."""
        def eligible(p):
            if clock_s is None:
                return True
            a, b = self.declared_window(p)
            return a is not None and a - 120 <= clock_s <= b + 120
        out = []
        for p in sorted(self.players, key=lambda p: (not eligible(p), p["jersey_number"])):
            lab = f"{p['jersey_number']} · {p['player_name']}"
            if not eligible(p):
                lab += " (hors période déclarée)"
            out.append({"value": str(p.get("source_capture_jersey_number", p["jersey_number"])), "label": lab,
                        "jersey_number": p["jersey_number"]})
        out += [{"value": "opponent", "label": "Joueur adverse"}, {"value": "referee", "label": "Arbitre"},
                {"value": "staff", "label": "Autre (staff, ramasseur…)"}]
        return out

    def declared_window(self, p: dict) -> tuple[float | None, float | None]:
        """Fenêtre de présence DÉCLARÉE (minutes de la composition, ±1 min de tolérance).
        Sert seulement à signaler des incohérences, jamais à prouver une présence."""
        a = 0.0 if p.get("starter") else (p.get("entered_minute_declared") or None)
        b = p.get("left_minute_declared")
        a_s = None if a is None else max(0.0, (a - 1) * 60.0)
        b_s = None if b is None else b * 60.0 + 60.0
        if not p.get("starter") and a is None:
            return None, None
        return a_s, (b_s if b_s is not None else 10 ** 6)
