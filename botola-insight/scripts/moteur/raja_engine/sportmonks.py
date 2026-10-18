"""Données Sportmonks (API Football v3) : lecture, normalisation et confrontation avec la vidéo.

Deux façons d'obtenir le match, sans jamais écrire la clé sur disque :
1. depuis le portail WordPress (recommandé) : la clé reste côté serveur ; l'extension a déjà
   synchronisé le match (`/matches/{id}/sync`) et `/dashboard` renvoie `api_data` ;
2. directement, avec la variable d'environnement SPORTMONKS_TOKEN.

Règles : la provenance « sportmonks » est conservée à part ; une divergence avec la vidéo est
signalée, jamais arbitrée silencieusement ; une donnée absente de l'abonnement reste inconnue.
Sources de structure : docs.sportmonks.com (v3 — events, lineups.details, statistics).
"""
from __future__ import annotations

import json
import os
import urllib.request

EVENT_TYPES = {14: "GOAL", 15: "OWNGOAL", 16: "PENALTY", 17: "MISSED_PENALTY", 18: "SUBSTITUTION", 19: "YELLOWCARD",
               20: "REDCARD", 21: "YELLOWREDCARD", 10: "VAR"}
INCLUDE = "participants;scores;statistics.type;lineups.details.type;events.type"


def fetch_direct(fixture_id: int, timeout: int = 20) -> dict:
    token = os.environ.get("SPORTMONKS_TOKEN", "")
    if not token:
        raise RuntimeError("SPORTMONKS_TOKEN non défini.")
    url = f"https://api.sportmonks.com/v3/football/fixtures/{int(fixture_id)}?include={INCLUDE}"
    req = urllib.request.Request(url, headers={"Authorization": token, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["data"]


def fetch_from_portal(wp_match_id: int, timeout: int = 30) -> dict | None:
    import base64
    base = os.environ.get("RI_WP_BASE_URL", "").rstrip("/")
    user, pw = os.environ.get("RI_WP_USER", ""), os.environ.get("RI_WP_APP_PASSWORD", "")
    if not base.startswith("https://") or not user or not pw:
        raise RuntimeError("Portail non configuré (RI_WP_BASE_URL, RI_WP_USER, RI_WP_APP_PASSWORD).")
    req = urllib.request.Request(f"{base}/wp-json/raja-insight/v1/dashboard",
                                 headers={"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    for m in d.get("matches", []):
        if int(m.get("id", 0)) == int(wp_match_id):
            return m.get("api_data")
    return None


def _dev(obj) -> str | None:
    t = obj.get("type") or {}
    return t.get("developer_name") or EVENT_TYPES.get(obj.get("type_id"))


def normalize(fixture: dict, raja_team_id: int) -> dict:
    lineup = []
    for l in fixture.get("lineups") or []:
        details = {}
        for d in l.get("details") or []:
            key = _dev(d) or f"type_{d.get('type_id')}"
            val = (d.get("data") or {}).get("value")
            details[key] = val
        lineup.append({"player_id": l.get("player_id"), "player_name": l.get("player_name"), "jersey_number": l.get("jersey_number"),
                       "team_id": l.get("team_id"), "starter": l.get("type_id") == 11, "details": details})
    events = []
    for e in fixture.get("events") or []:
        kind = _dev(e)
        ev = {"type": kind, "minute": e.get("minute"), "extra_minute": e.get("extra_minute"), "team_id": e.get("participant_id"),
              "player_id": e.get("player_id"), "player_name": e.get("player_name"),
              "related_player_id": e.get("related_player_id"), "related_player_name": e.get("related_player_name"),
              "result": e.get("result"), "addition": e.get("addition")}
        if kind == "SUBSTITUTION":  # Sportmonks : player = entrant, related_player = sortant
            ev["player_in"], ev["player_out"] = e.get("player_name"), e.get("related_player_name")
        events.append(ev)
    return {"fixture_id": fixture.get("id"), "starting_at": fixture.get("starting_at"), "raja_team_id": raja_team_id,
            "raja_lineup": [p for p in lineup if p["team_id"] == raja_team_id], "events": events,
            "raja_events": [e for e in events if e["team_id"] == raja_team_id]}


def _name_key(s: str | None) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return s.split()[-1] if s.split() else ""


def cross_check(norm: dict, roster, banner_events: list, goal_events: list, v2_events: list) -> list[dict]:
    """Confronte composition, buts, cartons et remplacements. Renvoie une liste d'écarts/accords."""
    out = []
    api_by_num = {p["jersey_number"]: p for p in norm["raja_lineup"] if p.get("jersey_number") is not None}
    for p in roster.players:
        a = api_by_num.get(p["jersey_number"])
        if a is None:
            out.append({"check": "composition", "status": "absent_api", "player": p["player_name"], "jersey_number": p["jersey_number"]})
        elif _name_key(a["player_name"]) != _name_key(p["player_name"]):
            out.append({"check": "composition", "status": "divergence", "jersey_number": p["jersey_number"],
                        "video_roster": p["player_name"], "sportmonks": a["player_name"]})
        else:
            out.append({"check": "composition", "status": "accord", "jersey_number": p["jersey_number"], "player": p["player_name"]})
    api_goals = [e for e in norm["raja_events"] if e["type"] in ("GOAL", "PENALTY")]
    vid_goals = [g for g in goal_events if g.get("team") == "raja"]
    out.append({"check": "buts_raja", "status": "accord" if len(api_goals) == len(vid_goals) else "divergence",
                "sportmonks": [(e["minute"], e["player_name"]) for e in api_goals],
                "video_scoreboard": [g.get("match_clock_score_update") for g in vid_goals],
                "note": "le tableau vidéo est mis à jour après le but : minutes à rapprocher, pas à égaler"})
    api_y = sorted(_name_key(e["player_name"]) for e in norm["raja_events"] if e["type"] == "YELLOWCARD")
    vid_y = sorted(_name_key(p.get("player_name")) for b in banner_events if b["type"] == "yellow_card"
                   for p in b["players"] if p.get("team") == "raja" and p.get("player_name"))
    vid_y += sorted(_name_key(e["player_name"]) for e in v2_events if e["type"] == "carton_jaune")
    out.append({"check": "cartons_jaunes_raja", "status": "accord" if sorted(set(api_y)) == sorted(set(vid_y)) else "divergence",
                "sportmonks": api_y, "video": sorted(set(vid_y))})
    api_subs = [(e["minute"], _name_key(e.get("player_out")), _name_key(e.get("player_in"))) for e in norm["raja_events"] if e["type"] == "SUBSTITUTION"]
    out.append({"check": "remplacements_raja", "status": "info", "sportmonks": api_subs,
                "declared": roster.cfg["roster"].get("substitution_groups_declared")})
    return out
