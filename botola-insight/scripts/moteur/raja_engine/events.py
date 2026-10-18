"""Événements détectés automatiquement ou soumis à revue humaine.

Sources, jamais mélangées sans étiquette :
- scoreboard  : changement de score lu sur le tableau (but d'une équipe, pas de buteur) ;
- auto_banner : bandeaux de diffusion (carton, remplacement) lus par OCR et rapprochés de la
                composition (numéro + nom) ;
- human_review: réponses aux questions « que montre ce ralenti ? » ;
- v2_assistant_review : événements du dossier V2, relus par l'assistant, non confirmés par
                l'utilisateur ; importés avec cette mention ;
- engine_candidate : candidats issus du ballon et du suivi (passes), non validés.
"""
from __future__ import annotations

import re
import difflib
from collections import Counter
from pathlib import Path

from .questions import task_id
from .util import fmt_clock, parse_clock

EVENT_CHOICES = [
    {"value": "but", "label": "But"},
    {"value": "tir_cadre", "label": "Tir cadré (arrêté / contré sur la ligne)"},
    {"value": "tir_non_cadre", "label": "Tir non cadré"},
    {"value": "tir_contre", "label": "Tir contré par un joueur"},
    {"value": "arret_gardien", "label": "Arrêt du gardien"},
    {"value": "intervention_gardien", "label": "Sortie / intervention du gardien (poing, prise aérienne)"},
    {"value": "faute_commise", "label": "Faute COMMISE par le joueur du Raja choisi"},
    {"value": "faute_subie", "label": "Faute SUBIE par le joueur du Raja choisi (au profit du Raja)"},
    {"value": "carton_jaune", "label": "Carton jaune"},
    {"value": "carton_rouge", "label": "Carton rouge"},
    {"value": "hors_jeu", "label": "Hors-jeu"},
    {"value": "occasion_centre", "label": "Centre / occasion sans tir"},
    {"value": "duel", "label": "Duel / tacle"},
    {"value": "rediffusion", "label": "Rediffusion d'une action déjà répondue (ne pas recompter)"},
    {"value": "autre", "label": "Autre"},
    {"value": "rien", "label": "Rien à compter (ambiance, public, réaction)"},
]


def refresh_event_choices(store) -> int:
    """Met les questions d'action déjà créées à jour avec la liste de choix courante.

    Les questions créées avant la séparation « faute commise / faute subie » (et « carton » →
    jaune/rouge) gardaient l'ancienne liste en mémoire : la page proposait encore « Faute » seul,
    y compris pour corriger une réponse. Les réponses déjà données ne sont pas touchées ; les
    questions de précision (champ `refines`) gardent leurs propres choix."""
    changed = 0
    for t in store.tasks.values():
        if t.get("kind") != "event_review" or t.get("refines"):
            continue
        if t.get("choices") != EVENT_CHOICES:
            t["choices"] = EVENT_CHOICES
            changed += 1
    if changed:
        store.save()
    return changed


def _banner_clock_s(b):
    c, _ = parse_clock(b.get("match_clock_banner_start"))
    return c


def check_raja_consistency(banners: list[dict], roster) -> list[dict]:
    """Contrôle chaque bandeau de remplacement « Raja » contre l'état du terrain.

    Le bandeau affiche un numéro et un nom bruité ; le rapprochement par numéro seul attribue
    au Raja des bandeaux de l'adversaire (même numéro, autre joueur). Règles du football :
    pas de retour d'un joueur remplacé, on n'entre pas si l'on est déjà sur le terrain, on ne
    sort pas si l'on n'y est pas. Un bandeau qui viole ces règles n'est PAS compté pour le Raja ;
    il est marqué « adversaire probable » et soumis à confirmation. Un bandeau sortie et son
    bandeau entrée apparié (moins de 30 s) appartiennent à la même équipe.

    État du terrain : titulaires, entrées/sorties déclarées (±3 min), puis bandeaux sûrs, dans
    l'ordre du chrono. Idempotent : `needs_review_reading` garde le verdict de la seule lecture."""
    order = sorted([b for b in banners if b.get("duplicate_of") is None],
                   key=lambda b: (b.get("period") or 0, _banner_clock_s(b) if _banner_clock_s(b) is not None else 1e9))
    for b in banners:
        b.setdefault("needs_review_reading", bool(b.get("needs_review")))
        b["needs_review"] = False if b.get("human_override") else b["needs_review_reading"]
        b.pop("raja_check", None)
        if b.get("team_probable") == "opponent" and not b.get("human_override"):
            b.pop("team_probable")
    entered_at, left_at = {}, {}
    for p in roster.players:
        n = p["jersey_number"]
        entered_at[n] = 0.0 if p.get("starter") else (p["entered_minute_declared"] * 60.0 if p.get("entered_minute_declared") else None)
        left_at[n] = p["left_minute_declared"] * 60.0 if p.get("left_minute_declared") else None
    banner_in, banner_out = {}, {}
    tol = 180.0

    def on_pitch(n, c):
        a = banner_in.get(n, entered_at.get(n))
        z = banner_out.get(n, left_at.get(n))
        return a is not None and a - tol <= c and (z is None or c <= z + tol)

    for b in order:
        if b["type"] not in ("substitution_out", "substitution_in"):
            continue
        c = _banner_clock_s(b)
        if c is None:
            continue
        reasons = []
        if b.get("human_override"):
            for pl in b.get("players", []):
                if pl.get("team") == "raja" and pl.get("jersey_number") is not None:
                    (banner_in if b["type"] == "substitution_in" else banner_out).setdefault(pl["jersey_number"], c)
            continue
        for pl in b.get("players", []):
            n = pl.get("jersey_number")
            if pl.get("team") == "opponent" or n is None:
                continue
            name = pl.get("player_name")
            if b["type"] == "substitution_in":
                if n in banner_out and banner_out[n] < c - 30:
                    reasons.append(f"{name} déjà remplacé à {fmt_clock(banner_out[n])} : pas de retour possible")
                elif n not in banner_out and left_at.get(n) is not None and left_at[n] < c - tol:
                    reasons.append(f"{name} remplacé (sortie déclarée {int(left_at[n] // 60)}') : pas de retour possible")
                elif n in banner_in and banner_in[n] < c - 30:
                    reasons.append(f"{name} déjà entré à {fmt_clock(banner_in[n])}")
                elif entered_at.get(n) is not None and n not in banner_in and entered_at[n] < c - tol:
                    reasons.append(f"{name} déjà sur le terrain (" + ("titulaire" if entered_at[n] == 0 else
                                   f"entrée déclarée {int(entered_at[n] // 60)}'") + ")")
            else:
                if n in banner_out and banner_out[n] < c - 30:
                    reasons.append(f"{name} déjà sorti à {fmt_clock(banner_out[n])}")
                elif not on_pitch(n, c):
                    reasons.append(f"{name} n'est pas sur le terrain à ce moment (composition déclarée)")
        if reasons:
            b["raja_check"] = {"status": "incoherent", "reasons": reasons}
            continue
        if b.get("needs_review"):
            continue
        for pl in b.get("players", []):
            n = pl.get("jersey_number")
            if n is None or pl.get("team") == "opponent":
                continue
            (banner_in if b["type"] == "substitution_in" else banner_out).setdefault(n, c)
    # propagation à la paire sortie/entrée, puis verdict
    for b in banners:
        if b.get("raja_check", {}).get("status") == "incoherent":
            for o in banners:
                if o is b or o.get("duplicate_of") is not None or o.get("human_override"):
                    continue
                pair = {o["type"], b["type"]} == {"substitution_out", "substitution_in"} and \
                    abs(o["file_start"] - b["file_start"]) <= 30 + max(o["file_end"] - o["file_start"], b["file_end"] - b["file_start"])
                if pair and "raja_check" not in o:
                    o["raja_check"] = {"status": "incoherent", "reasons": ["bandeau apparié (même remplacement) incohérent pour le Raja"]}
    # lignes qui ne correspondent à aucun joueur du Raja (numéro absent de l'effectif, nom inconnu),
    # ou dont le nom lu figure déjà sur un bandeau jugé adverse
    known = {p["jersey_number"] for p in roster.players}
    adverse_names = {_norm_name(p.get("name_read")) for b in banners if b.get("raja_check", {}).get("status") == "incoherent"
                     for p in b.get("players", []) if p.get("name_read")}
    adverse_names.discard("")
    for b in banners:
        if "raja_check" in b or b.get("human_override") or b.get("duplicate_of") is not None or not b.get("needs_review"):
            continue
        lines = [p for p in b.get("players", []) if p.get("team") != "opponent"]
        if not lines:
            continue
        r = []
        if all(p.get("match_method") == "non_rapproche" and p.get("jersey_number_read") not in known for p in lines):
            r.append("aucun joueur du Raja ne porte " + ", ".join(f"le n°{p.get('jersey_number_read')}" for p in lines) +
                     " et le nom lu ne correspond à personne")
        else:
            same = [p.get("name_read") for p in lines if p.get("match_method") in ("numero_seul_nom_divergent", "non_rapproche")
                    and any(difflib.SequenceMatcher(None, _norm_name(p.get("name_read")), a).ratio() >= 0.85 for a in adverse_names)]
            if same and len(same) == len(lines):
                r.append("même nom lu (" + ", ".join(same) + ") que sur un bandeau jugé adverse")
        if r:
            b["raja_check"] = {"status": "incoherent", "reasons": r}
    for b in banners:
        if b.get("raja_check", {}).get("status") == "incoherent":
            b["needs_review"] = True
            b["team_probable"] = "opponent"
    return banners


def _norm_name(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z]", "", s)


BANNER_CHECK_CHOICES = [
    {"value": "opponent", "label": "Bandeau de l'ADVERSAIRE (Zemamra) : rien à compter pour le Raja"},
    {"value": "yellow_card", "label": "Carton jaune (Raja)"}, {"value": "red_card", "label": "Carton rouge (Raja)"},
    {"value": "substitution_out", "label": "Remplacement Raja : sortie"},
    {"value": "substitution_in", "label": "Remplacement Raja : entrée"},
    {"value": "other", "label": "Autre bandeau (information, entraîneur…)"},
]


def banner_check_task(fp, period, b) -> dict:
    lines = [(p.get("jersey_number_read"), p.get("name_read")) for p in b["players"]]
    rc = b.get("raja_check") or {}
    q = "Ce bandeau a été mal lu. Que montre-t-il ?"
    reason = f"Lecture : {b['type']}, lignes {lines}"
    proposed = None
    if rc.get("status") == "incoherent":
        q = "Bandeau probablement de l'adversaire : le confirmez-vous ?"
        reason = "Lu « " + " / ".join(f"{n}. {nm}" for n, nm in lines) + " ». Pour le Raja ce serait impossible : " + \
                 " ; ".join(rc["reasons"]) + "."
        proposed = "Bandeau de l'adversaire"
    return {"id": task_id(fp, "banner_check", b["file_start"]), "kind": "banner_check", "video": fp, "period": period,
            "time": b["file_start"], "end": b["file_end"], "bbox": None, "match_clock": b.get("match_clock_banner_start"),
            "clock_status": b.get("clock_status"), "blocking": False, "priority": 60,
            "question": q, "reason": reason, "proposed": proposed, "choices": BANNER_CHECK_CHOICES, "player_choice": True,
            "evidence": {"clip_window": [max(0, b["file_start"] - 1), b["file_end"] - b["file_start"] + 2]}, "origin": "engine"}


def apply_banner_answers(banners: list[dict], store, fp, roster) -> int:
    """Réponses humaines aux questions « bandeau » : elles remplacent la lecture automatique."""
    n = 0
    for b in banners:
        t = store.tasks.get(task_id(fp, "banner_check", b["file_start"]))
        if not t or t.get("status") != "resolved" or not t.get("response"):
            continue
        b.setdefault("needs_review_reading", bool(b.get("needs_review")))
        r = t["response"]
        v = r.get("value")
        b["human_override"] = {"value": v, "player": r.get("player"), "task_id": t["id"]}
        b["needs_review"] = False
        if v in ("opponent", "other"):
            b["team_probable"] = "opponent" if v == "opponent" else None
            b["type"] = b["type"] if v == "opponent" else "banner_info"
            for p in b["players"]:
                p["team"] = "opponent" if v == "opponent" else None
        elif v in ("yellow_card", "red_card", "substitution_out", "substitution_in"):
            b["type"] = v
            pl = roster.from_answer_value(str(r["player"])) if r.get("player") not in (None, "") else None
            if pl:
                b["players"] = [{"team": "raja", "jersey_number": pl["jersey_number"], "player_name": pl["player_name"],
                                 "match_method": "humain", "match_confidence": 1.0}]
        n += 1
    return n


FOUL_PRECISION_CHOICES =[c for c in EVENT_CHOICES if c["value"] in ("faute_commise", "faute_subie", "rien")]


def foul_followups(store, roster) -> list[str]:
    """Une réponse « Faute » (ancien choix, sans sens) ne peut pas être comptée comme commise ou subie :
    une question de précision est créée (une seule fois par action). Tant qu'elle n'a pas de réponse,
    l'action reste comptée « faute impliquée », jamais commise ni subie."""
    created = []
    for t in list(store.tasks.values()):
        if t.get("kind") != "event_review" or t.get("refines") or t.get("status") != "resolved":
            continue
        r = t.get("response") or {}
        if r.get("value") != "faute":
            continue
        pid = "precision-" + t["id"]
        if pid in store.tasks:
            continue
        p = roster.from_answer_value(str(r["player"])) if r.get("player") not in (None, "") else None
        who = p["player_name"] if p else "un joueur (non précisé)"
        task = {k: t.get(k) for k in ("video", "period", "time", "end", "match_clock", "clock_status", "evidence")}
        task.update(id=pid, kind="event_review", bbox=None, refines=t["id"], blocking=False, priority=95,
                    question=f"Vous avez répondu « Faute » ({who}). Faute commise par le Raja ou subie par le Raja ?",
                    reason="Précision demandée : l'ancien choix « Faute » ne disait pas dans quel sens.",
                    choices=FOUL_PRECISION_CHOICES, player_choice=True, origin="engine")
        store.add(task)
        created.append(pid)
    if created:
        store.save()
    return created


def _clock_for(timeline, t: float, lookback_s: float = 40.0) -> tuple[str | None, str]:
    st = timeline.at(t)
    if st.get("live") and st.get("clock"):
        return st["clock"], ("interpolated" if st.get("interpolated") else "observed_engine")
    prev = [s for s in timeline.segments if s["kind"] == "live" and s["file_end"] <= t and t - s["file_end"] <= lookback_s]
    if prev:
        s = prev[-1]
        return fmt_clock(int(t + s["offset"])), "extrapolated_from_previous_live_segment"
    return None, "unknown"


def _phases(times, icons, min_len_s):
    """Découpe un bandeau en phases d'icône stable (ex. sortants en rouge puis entrants en vert)."""
    phases = []
    for t, ic in zip(times, icons):
        if phases and (ic is None or ic == phases[-1]["icon"] or phases[-1]["icon"] is None):
            phases[-1]["times"].append(t)
            if phases[-1]["icon"] is None:
                phases[-1]["icon"] = ic
        else:
            phases.append({"icon": ic, "times": [t]})
    return [p for p in phases if p["times"][-1] - p["times"][0] >= min_len_s]


def banner_events(samples: list[dict], video, reader, roster, timeline, min_len_s: float = 1.0,
                  home_side: str = "raja") -> list[dict]:
    """Regroupe les images à bandeau, sépare les phases d'icône, lit 5 images par phase et vote."""
    times = [s["t"] for s in samples if s.get("banner")]
    groups = []
    for t in times:
        if groups and t - groups[-1][-1] <= 1.0:
            groups[-1].append(t)
        else:
            groups.append([t])
    out = []
    for g in groups:
        if g[-1] - g[0] < min_len_s:
            continue
        icons = []
        for t in g:
            f = video.frame_at(t)
            icons.append(None if f is None else reader.banner_icon(f)["icon"])
        for ph in _phases(g, icons, min_len_s):
            pt = ph["times"]
            a, b = pt[0] + 0.2 * (pt[-1] - pt[0]), pt[-1] - 0.2 * (pt[-1] - pt[0])
            picks = [a + (b - a) * i / 4 for i in range(5)]
            lines_votes: dict[int, Counter] = {}
            names: dict[int, list] = {}
            teams, raws, sides = Counter(), [], Counter()
            for t in picks:
                f = video.frame_at(t)
                if f is None:
                    continue
                r = reader.read_banner(f)
                raws.append(r.get("raw", ""))
                if r.get("team"):
                    teams[r["team"]] += 1
                if r.get("side"):
                    sides[r["side"]] += 1
                good = [ln for ln in r["lines"] if ln.get("jersey_number") is not None
                        and sum(ch.isalpha() for ch in (ln.get("name") or "")) >= 3]
                for i, ln in enumerate(good[:3]):
                    lines_votes.setdefault(i, Counter())[ln.get("jersey_number")] += 1
                    names.setdefault(i, []).append(ln.get("name") or "")
            team = teams.most_common(1)[0][0] if teams else None
            # position de l'étiquette (gauche = domicile, droite = extérieur) : vote sur les 5 images
            side = None
            if sides and sides.most_common(1)[0][1] >= 3 and len(sides) == 1:
                side = sides.most_common(1)[0][0]
            home_is_raja = (home_side or "raja") == "raja"
            side_team = None if side is None else ("RCA" if (side == "home") == home_is_raja else "RCAZ")
            if side_team and team and side_team != team:
                team = None  # désaccord OCR / position : aucune équipe retenue, la cohérence et l'humain tranchent
            elif side_team:
                team = side_team
            lines = []
            for i in sorted(lines_votes):
                num, k = lines_votes[i].most_common(1)[0]
                if k < 2:  # une ligne lue sur une seule image n'est pas retenue
                    continue
                nm = Counter(names[i]).most_common(1)[0][0]
                lines.append({"jersey_number_read": num, "votes": k, "name_read": nm})
            icon = ph["icon"]
            n_lines = len(lines)
            if n_lines == 0:
                # bandeau sans ligne « N. NOM » : entraîneur, animation de but, information
                kind = "banner_info"
            elif icon == "red" and n_lines >= 1:
                kind = "substitution_out"
            elif icon == "green" and n_lines >= 1:
                kind = "substitution_in"
            elif icon == "yellow" and n_lines == 1:
                kind = "yellow_card"
            elif icon == "red_small" and n_lines == 1:
                kind = "red_card_possible"
            else:
                kind = "banner_unclassified"
            players = []
            for ln in lines:
                if team == "RCAZ":
                    players.append({"team": "opponent", "jersey_number": ln["jersey_number_read"], "name_read": ln["name_read"]})
                    continue
                p, conf, how = roster.match_banner_line(ln["jersey_number_read"], ln["name_read"])
                players.append({"team": "raja" if team == "RCA" or p is not None else None,
                                "jersey_number": None if p is None else p["jersey_number"],
                                "player_name": None if p is None else p["player_name"], "match_confidence": round(conf, 2),
                                "match_method": how, **ln})
            clock, cstat = _clock_for(timeline, pt[0])
            ok = kind == "banner_info" or (kind not in ("banner_unclassified", "red_card_possible") and bool(players) and
                                            all(p.get("team") == "opponent" or p.get("match_method") == "numero_et_nom" for p in players))
            out.append({"source": "auto_banner", "type": kind, "file_start": pt[0], "file_end": pt[-1], "team_tag": team,
                        "team_side_votes": dict(sides), "icon": icon, "players": players, "match_clock_banner_start": clock, "clock_status": cstat,
                        "needs_review": not ok, "ocr_raw": raws,
                        "note": "heure d'affichage du bandeau, postérieure au fait de jeu"})
    # doublons (même bandeau rediffusé, extraits qui se recouvrent) : même type et mêmes joueurs à < 90 s
    for i, ev in enumerate(out):
        for prev in out[:i]:
            if prev.get("duplicate_of") is None and prev["type"] == ev["type"] and \
                    sorted(p.get("jersey_number") or -1 for p in prev["players"]) == sorted(p.get("jersey_number") or -1 for p in ev["players"]):
                from .util import parse_clock
                c1, _ = parse_clock(prev.get("match_clock_banner_start"))
                c2, _ = parse_clock(ev.get("match_clock_banner_start"))
                if c1 is not None and c2 is not None and abs(c1 - c2) <= 90:
                    ev["duplicate_of"] = prev["file_start"]
                    ev["needs_review"] = False
                    break
    out = [e for e in out if e.get("duplicate_of") is None] + [e for e in out if e.get("duplicate_of") is not None]
    # appariement des remplacements : un bandeau « sortie » suivi d'un bandeau « entrée »
    for i, ev in enumerate(out):
        if ev["type"] == "substitution_out" and ev.get("duplicate_of") is None:
            for nxt in out[i + 1:i + 3]:
                if nxt["type"] == "substitution_in" and nxt["file_start"] - ev["file_end"] <= 30:
                    ev["paired_in"] = nxt["players"]
                    nxt["paired_out"] = ev["players"]
                    ev["pairing_note"] = "groupe sortie/entrée lu sur bandeaux ; correspondance ligne à ligne non certifiée"
                    break
    return out


def clean_score_events(score_events: list[dict], revert_s: float = 180.0) -> tuple[list[dict], list[dict]]:
    """Écarte les lectures de score aberrantes : valeur > 9, saut impossible, ou changement
    annulé par la lecture suivante en moins de `revert_s` (ex. « 10-0 » lu pour « 0-0 »)."""
    reads = [e for e in score_events if e.get("score") and max(e["score"]) <= 9]
    rejected = [e for e in score_events if e.get("score") and max(e["score"]) > 9]
    seq = []
    for e in reads:
        if seq and tuple(e["score"]) == tuple(seq[-1]["score"]):
            continue
        if len(seq) >= 2 and tuple(e["score"]) == tuple(seq[-2]["score"]) and e["t"] - seq[-1]["t"] <= revert_s:
            rejected.append(seq.pop())
            continue
        seq.append(e)
    out = []
    for i, e in enumerate(seq):
        prev = out[-1]["score"] if out else None
        if prev is not None:
            dh, da = e["score"][0] - prev[0], e["score"][1] - prev[1]
            if sorted((dh, da)) != [0, 1]:
                rejected.append(e)
                continue
        out.append({**e, "previous": prev, "kind": "change" if prev is not None else "initial"})
    return out, rejected


def goal_events(score_events: list[dict], timeline, home_side: str = "raja") -> list[dict]:
    out = []
    score_events, _rej = clean_score_events(score_events)
    for e in score_events:
        if e.get("kind") != "change" or not e.get("previous") or not e.get("score"):
            continue
        (h0, a0), (h1, a1) = e["previous"], e["score"]
        if (h1, a1) == (h0, a0):
            continue
        scorer_side = "home" if h1 > h0 else "away"
        team = home_side if scorer_side == "home" else ("opponent" if home_side == "raja" else "raja")
        clock, cstat = _clock_for(timeline, e["t"])
        plausible = (h1 - h0 in (0, 1)) and (a1 - a0 in (0, 1)) and (h1 >= h0 and a1 >= a0)
        out.append({"source": "scoreboard", "type": "goal_team", "team": team, "score_after": [h1, a1],
                    "file_t_score_update": e["t"], "match_clock_score_update": clock, "clock_status": cstat,
                    "plausible_increment": plausible, "needs_review": not plausible,
                    "note": "le tableau est mis à jour après le but ; le buteur n'est pas lu sur le tableau"})
    return out


def replay_review_tasks(timeline, video_fp: str, period: int, store, budget: int = 60, min_s: float = 3.0,
                        max_s: float = 90.0) -> list[dict]:
    created = []
    cands = []
    for s in timeline.segments:
        replay_like = s["kind"] == "no_scoreboard" or (s["kind"] == "discontinuity" and s.get("scoreboard_absent_fraction", 0) >= 0.6)
        if not replay_like or not (min_s <= s["file_end"] - s["file_start"] <= max_s):
            continue
        prev_clock = s.get("clock_before")
        cands.append((s, prev_clock))
    for s, prev_clock in cands[:budget]:
        t0 = s["file_start"]
        tid = task_id(video_fp, "event_review", t0)
        task = {"id": tid, "kind": "event_review", "video": video_fp, "period": period, "time": round(t0, 3),
                "end": round(s["file_end"], 3), "bbox": None, "match_clock": prev_clock,
                "clock_status": "last_live_clock_before_replay" if prev_clock else "unknown",
                "reason": (f"Passage sans tableau de {s['file_end'] - t0:.0f} s après le direct"
                           + (f" ({prev_clock})" if prev_clock else "") + " : probablement un ralenti."),
                "question": "Quelle action ce passage montre-t-il, et quel joueur du Raja est concerné ?",
                "choices": EVENT_CHOICES, "player_choice": True, "blocking": False,
                "priority": round(s["file_end"] - t0, 1),
                "evidence": {"clip_window": [max(0.0, t0 - 8.0), min(20.0, s["file_end"] - t0 + 8.0)]},
                "origin": "engine"}
        t, st = store.add(task)
        if st == "created":
            created.append(t)
    store.save()
    return created


def import_v2(results: dict, match_key: str) -> list[dict]:
    """Événements du dossier V2 : conservés avec leur provenance et leur périmètre."""
    out = []
    for e in results.get("events", []):
        clock = e.get("clock") or ""
        m = re.match(r"^\s*(\d{1,2}):(\d{2})(?:\s*\+\s*(\d{1,2}):(\d{2}))?", clock.replace("90:00 + ", "90:00 +"))
        start = None
        added = None
        if clock.startswith("90:00 +"):
            mm = re.match(r"90:00\s*\+\s*(\d+):(\d+)", clock)
            start, added = 5400, int(mm.group(1)) * 60 + int(mm.group(2)) if mm else None
        elif m:
            start = int(m.group(1)) * 60 + int(m.group(2))
        out.append({"source": "v2_assistant_review", "id": e["id"], "type": e["type"], "outcome": e.get("outcome"),
                    "jersey_number": e.get("jersey"), "player_name": e.get("player"), "match_clock": clock,
                    "clock_s": start, "added_s": added, "reliability": e.get("reliability"),
                    "counted_in_metrics": bool(e.get("counted_in_metrics")), "clip": e.get("clip"),
                    "reason": e.get("reason"), "human_event_reviewed": bool(e.get("human_event_reviewed")),
                    "scope": "selected_sequences_only", "full_match_count": False})
    return out


def v2_request_tasks(results: dict, journal: dict, offsets: dict, video_fps: dict, store, clips_dir: str | None = None) -> list[dict]:
    """Transforme les 6 demandes V2 en demandes « event_review » rattachées au bon fichier.
    offsets : {période: décalage chrono−fichier} issus de la timeline (pas supposés)."""
    jobs = {j["name"]: j for j in journal.get("jobs", [])}
    created = []
    for r in results.get("requests", []):
        job = jobs.get(r.get("clip"))
        if not job:
            continue
        period = job["half"]
        fp = video_fps.get(period)
        base, add = parse_clock(r["clock"].replace("90:00 + ", "90:00 +"))
        t = None
        if base is not None and period in offsets and add is None:
            t = base - offsets[period]
        if t is None:
            t = float(job["internal_start"]) + 4.0
        tid = f"v2req-{r['id']}"
        task = {"id": tid, "kind": "event_review", "video": fp, "period": period, "time": round(t, 2),
                "end": round(float(job["internal_start"]) + job["duration"], 2), "bbox": None,
                "match_clock": r["clock"], "clock_status": "observed_manual_v2", "reason": "Demande V2 non résolue.",
                "question": r["question"], "choices": EVENT_CHOICES, "player_choice": True, "blocking": False,
                "priority": 50.0, "evidence": {"v2_clip": r.get("clip")}, "origin": "imported_v2_requests"}
        tt, st = store.add(task)
        if st == "created":
            created.append(tt)
    store.save()
    return created
