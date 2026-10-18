"""Fichier unique du match pour le site public (extension WordPress Raja Insight 2).

Construit à partir des mêmes données que le rapport (report.build(ws, None)) : rien n'est
recalculé ici, rien n'est inventé. Une valeur inconnue reste absente (affichée « — »)."""
from __future__ import annotations

import re

from .publish import site_bundle_experimental
from .report import COLS, SRC_FR, build as report_data
from .util import parse_clock, read_json

FORMAT = "raja-insight-site"
VERSION = 1

# métriques publiées : celles du rapport (les candidats automatiques du moteur restent internes)
PUBLIC_KEYS = [k for _, cols in COLS for k, _ in cols]
SHORT = {k: l for _, cols in COLS for k, l in cols}
GROUP = {k: g for g, cols in COLS for k, _ in cols}
REL_ORDER = {"élevée": 0, "moyenne": 1, "faible": 2}
# libellés de source pour le public (le rapport interne s'adresse à l'analyste : « votre réponse »)
PUBLIC_SRC = {"votre réponse": "relecture humaine", "votre réponse (joueur V2)": "relecture humaine",
              "relecture V2": "relecture vidéo", "revue visuelle de Claude": "relecture assistée par IA",
              "bandeau de diffusion": "bandeau de la diffusion", "tableau de score": "tableau de score de la diffusion",
              "modèle": "modèle statistique"}
# entrées/sorties : déjà dans le statut du joueur
HIDDEN_KEYS = {"sub_in_banner", "sub_out_banner"}


def _pub_src(txt):
    parts = [x.strip() for x in (txt or "").split(",") if x.strip()]
    out = []
    for x in parts:
        y = PUBLIC_SRC.get(x, x)
        if y not in out:
            out.append(y)
    return ", ".join(out)


def _clock_seconds(c):
    if not c:
        return None
    v, a = parse_clock(c.split("–")[0].split("(")[0].strip())
    return None if v is None else v + (a or 0)


def _status(p):
    dm = p.get("declared_minutes") or {}
    st = p.get("participation_status")
    if st == "starter":
        s = "Titulaire"
    elif st == "entered":
        s = f"Entré à la {dm.get('entered')}′" if dm.get("entered") else "Entré en jeu"
    elif st in ("no_entry", "no_entry_reported"):
        s = "Non entré"
    else:
        s = "—"
    if dm.get("left"):
        s += f", sorti à la {dm['left']}′"
    return s


def build_bundle(ws, date: str | None = None, date_source: str | None = None) -> dict:
    d = report_data(ws, None)
    cfg = ws.cfg
    opp = cfg.get("opponent") or "Adversaire"

    # actions par joueur (une ligne par action)
    actions = {}
    for clock, num, name, labs, src in d["events_rows"]:
        actions.setdefault(num, []).append({"clock": clock, "label": labs, "source": _pub_src(src)})

    players = []
    for p in d["players"]:
        mets = []
        for k in PUBLIC_KEYS:
            if k in HIDDEN_KEYS:
                continue
            m = p["metrics"].get(k)
            if not m or m.get("value") is None:
                continue
            v = m["value"]
            if isinstance(v, float):
                v = round(v, 2) if k in ("xg", "xa") else round(v, 1)
            mets.append({"key": k, "label": SHORT[k], "group": GROUP[k], "value": v, "unit": m.get("unit") or "",
                         "reliability": m.get("reliability") or "faible", "method": m.get("method") or "",
                         "reason": m.get("reliability_reason") or "",
                         "sources": [_pub_src(SRC_FR.get(s, s)) for s in m.get("sources", [])]})
        players.append({"number": p["jersey_number"], "name": p["player_name"], "role": p.get("role") or "",
                        "status": _status(p), "played": p.get("participation_status") in ("starter", "entered"),
                        "metrics": mets, "actions": actions.get(p["jersey_number"], [])})

    # buteur : action « Buts » du Raja la plus proche du but lu sur le tableau (± 3 min)
    goal_actions = [(a, pl) for pl in players for a in pl["actions"] if "Buts" in a["label"]]
    facts = []
    for clock, typ, detail, source, period in d["facts"]:
        team = "raja" if detail.startswith("Raja") else "opponent"
        who = re.sub(r"^(Raja|" + re.escape(opp) + r") — ", "", detail).replace(" (non compté pour le Raja)", "")
        if team == "opponent":
            nums = re.findall(r"(?:^|, )(\d+) ", who + " ")
            who = ("n° " + ", ".join(nums)) if nums else ""
        if typ == "But":
            who = ""
        f = {"clock": clock, "period": period, "type": typ, "team": team, "detail": who,
             "source": "tableau de score de la diffusion" if typ == "But" else "bandeau de la diffusion"}
        if typ == "But":
            t = _clock_seconds(clock)
            near = [(abs((_clock_seconds(a["clock"]) or 10 ** 6) - t), a, pl) for a, pl in goal_actions] if t is not None else []
            near = [x for x in near if x[0] <= 180]
            if near:
                _, a, pl = min(near, key=lambda x: x[0])
                f["scorer"] = {"number": pl["number"], "name": pl["name"], "source": a["source"]}
        f["needs_review"] = "à vérifier" in who
        facts.append(f)

    pos = read_json(ws.out / "possession_fenetres.json", {}) or {}
    windows = []
    for w in pos.get("fenetres", []):
        known = w.get("secondes_connues") or 0
        lab = w.get("fenetre") or "?"
        lab = f"{lab.split('-')[0]}′–{lab.split('-')[1]}′" if "-" in lab else lab.replace("+", "′+")
        windows.append({"period": w.get("periode"), "label": lab, "raja": w.get("part_raja") if known >= 30 else None,
                        "known_s": round(known, 1)})
    res = pos.get("resume") or {}

    coverage = []
    for pr in d["periods"]:
        gaps = [f"{g['clock_before']} → {g['clock_after']} : {'image figée' if g['kind'] == 'frozen' else 'coupure'}"
                for g in pr["gaps"]]
        coverage.append({"period": pr["period"], "live_min": round(pr["live_s"] / 60, 1), "gaps": gaps,
                         "attributed_frames": pr["attributed"], "raja_frames": pr["raja_frames"],
                         "ball_located": pr["ball"]})

    score = d.get("score_raw")
    bundle = {
        "format": FORMAT, "version": VERSION, "match_key": cfg["match_key"],
        "title": f"Raja — {opp}", "home": "Raja Club Athletic", "away": opp,
        "competition": cfg.get("competition") or "", "season": cfg.get("season") or "",
        "round": cfg.get("round") or "", "date": date or cfg.get("date"),
        "date_source": date_source or ("composition du match" if cfg.get("date") else None),
        "score": None if not score else {"raja": score[0], "opponent": score[1], "source": "tableau de score de la vidéo"},
        "facts": facts, "players": players,
        "possession": {"windows": windows, "raja": (res.get("match") or {}).get("part_raja"),
                       "coverage": (res.get("match") or {}).get("couverture"),
                       "by_period": {k: v.get("part_raja") for k, v in (res.get("par_periode") or {}).items()}},
        "coverage": coverage, "divergences": d["divergences"],
        "generated_at": d["stats"].get("generated_at"), "engine_version": d["stats"].get("engine_version"),
    }
    # Décision de l'utilisateur du 18/10/2026 (écart au cahier v1 §1, site seulement) : possession
    # estimée, passes, distances, vitesses, xG/xA affichées comme estimations expérimentales, en
    # fiabilité faible. Pour revenir au cahier strict : site_bundle_v1(bundle).
    return site_bundle_experimental(bundle)


def validate(b: dict) -> list[str]:
    """Contrôles minimaux, les mêmes que ceux de l'extension WordPress."""
    err = []
    if b.get("format") != FORMAT or b.get("version") != VERSION:
        err.append("format/version")
    if not b.get("match_key") or not b.get("title"):
        err.append("match_key/title")
    for p in b.get("players", []):
        if not isinstance(p.get("number"), int) or not p.get("name"):
            err.append(f"joueur invalide {p.get('number')}")
        for m in p.get("metrics", []):
            if not isinstance(m.get("value"), (int, float)) or m["value"] < 0:
                err.append(f"valeur invalide {p.get('number')} {m.get('key')}")
            if m.get("reliability") not in REL_ORDER:
                err.append(f"fiabilité invalide {p.get('number')} {m.get('key')}")
    return err
