"""Agrégation par joueur : valeur, unité, portée, méthode, fiabilité, couverture, preuves.

Conventions (cahier des charges) :
- une statistique non mesurée vaut None (« inconnue »), jamais 0 ;
- une somme d'événements repérés n'est pas un total du match : la portée l'indique ;
- les sources ne sont pas fusionnées sans étiquette (humain, bandeau, tableau, V2, candidat) ;
- une estimation (xG, extrapolation) n'est jamais présentée comme une observation.

Fiabilité (règles, pas de pourcentage inventé) :
- élevée  : identité confirmée par l'utilisateur à l'instant mesuré, ou bandeau lu avec numéro
            ET nom concordants, ou changement de score lu sur le tableau ;
- moyenne : identité propagée dans un tronçon de suivi sans croisement, numéro lu deux fois,
            gardien déduit du maillot, événement V2 relu par l'assistant ;
- faible  : candidat automatique (ballon/possession), modèle (xG), couverture très partielle.
"""
from __future__ import annotations

from collections import defaultdict

from .util import fmt_clock

UNIFY = {"yellow_cards_video": "yellow_cards", "yellow_cards_banner": "yellow_cards", "cards_reviewed": "cards_reviewed"}


def _clock_s(text):
    """« 26:33–26:34 » → 1593 ; « 90:00 + 1:15 » → 5475 ; sinon None."""
    import re
    if not text:
        return None
    m = re.match(r"\s*(\d{1,3}):(\d{2})(?:\s*\+\s*(\d{1,2}):(\d{2}))?", str(text))
    if not m:
        return None
    v = int(m.group(1)) * 60 + int(m.group(2))
    if m.group(3):
        v += int(m.group(3)) * 60 + int(m.group(4))
    return v


SCOPE_VISIBLE = "visible_segments"
SCOPE_EVENTS = "reviewed_events"
SCOPE_MODEL = "model_estimate"
SCOPE_CANDIDATES = "engine_candidates"  # local : non exporté vers le portail (contrat V1)

EVENT_MAP_HUMAN = {
    "but": ["goals", "shots", "shots_on_target"],
    "tir_cadre": ["shots", "shots_on_target"],
    "tir_non_cadre": ["shots"],
    "tir_contre": ["shots", "shots_blocked"],
    "arret_gardien": ["saves"],
    "intervention_gardien": ["keeper_interventions"],
    "faute": ["fouls_involved"],            # ancien choix, sens non précisé
    "faute_commise": ["fouls_committed"],
    "faute_subie": ["fouls_won"],
    "carton": ["cards_reviewed"],           # ancien choix, couleur non précisée
    "carton_jaune": ["yellow_cards"],
    "carton_rouge": ["red_cards"],
    "duel": ["duels"],
    "occasion_centre": ["chance_or_cross"],
    "hors_jeu": ["offsides"],
}
FAMILY_HUMAN = {"but": "shot", "tir_cadre": "shot", "tir_non_cadre": "shot", "tir_contre": "shot", "arret_gardien": "keeper",
                "intervention_gardien": "keeper", "faute": "foul", "faute_commise": "foul", "faute_subie": "foul", "carton": "card",
                "carton_jaune": "card", "carton_rouge": "card", "duel": "duel", "occasion_centre": "cross", "hors_jeu": "offside"}
FAMILY_V2 = {"tir": "shot", "passe": "pass", "passe_avant_tir": "pass_shot", "arret_sur_tir": "keeper", "prise_de_balle": "keeper",
             "intervention_aerienne_au_poing": "keeper", "intervention_aerienne": "keeper", "plongeon_sur_tir": "keeper",
             "tacle": "duel", "duel_au_sol_participe": "duel", "duel_offensif": "duel", "centre_contre": "cross",
             "faute_obtenue": "foul", "carton_jaune": "card"}
EVENT_MAP_V2 = {
    ("tir", "but"): ["goals", "shots", "shots_on_target"],
    ("tir", "cadre_arrete"): ["shots", "shots_on_target"],
    ("tir", "non_cadre"): ["shots"],
    ("passe", "completee"): ["passes_completed"],
    ("passe_avant_tir", "completee"): ["passes_completed", "shot_assists"],
    ("arret_sur_tir", "capte"): ["saves"],
    ("prise_de_balle", "ballon_securise"): ["keeper_claims"],
    ("intervention_aerienne_au_poing", "ballon_repousse"): ["keeper_punches"],
    ("intervention_aerienne", "ballon_non_securise"): ["keeper_unsecured"],
    ("tacle", "ballon_sorti"): ["tackles"],
    ("duel_au_sol_participe", None): ["duels"],
    ("duel_offensif", "stoppe_par_tacle"): ["duels"],
    ("centre_contre", "ballon_sorti"): ["crosses_blocked"],
    ("centre_contre", "corner"): ["crosses_blocked"],
    ("faute_obtenue", "coup_franc"): ["fouls_won"],
    ("carton_jaune", "averti"): ["yellow_cards_video"],
}
LABELS = {
    "visible_wide_s": "Temps à l'écran en plan large/moyen (identité attribuée)",
    "visible_close_s": "Temps à l'écran en gros plan (identité attribuée)",
    "tracked_segments": "Tronçons de suivi attribués",
    "ball_near_s": "Temps proche du ballon (ballon localisé)",
    "touch_candidates": "Contacts candidats (ballon au pied)",
    "pass_candidates": "Passes candidates (suivi + ballon)",
    "distance_observed_m": "Distance observée (segments calibrés)",
    "vmax_kmh": "Vitesse max soutenue (segments calibrés)",
    "sprints_observed": "Sprints observés (≥ 25,2 km/h, ≥ 1 s)",
    "goals": "Buts", "shots": "Tirs", "shots_on_target": "Tirs cadrés", "passes_completed": "Passes réussies",
    "shot_assists": "Dernières passes avant tir", "saves": "Arrêts", "keeper_claims": "Prises de balle (gardien)",
    "keeper_punches": "Interventions au poing", "keeper_unsecured": "Sorties non sécurisées",
    "keeper_interventions": "Interventions du gardien", "tackles": "Tacles", "duels": "Duels",
    "crosses_blocked": "Centres contrés", "fouls_won": "Fautes obtenues", "fouls_involved": "Fautes (impliqué)",
    "yellow_cards": "Cartons jaunes", "yellow_cards_video": "Cartons jaunes (vidéo)", "yellow_cards_banner": "Cartons jaunes (bandeau diffusion)",
    "cards_reviewed": "Cartons (couleur non précisée)", "red_cards": "Cartons rouges", "fouls_committed": "Fautes commises",
    "offsides": "Hors-jeu", "chance_or_cross": "Centres / occasions sans tir", "shots_blocked": "Tirs contrés",
    "sub_out_banner": "Sortie (bandeau)", "sub_in_banner": "Entrée (bandeau)", "xg": "xG (modèle pilote)",
    "xa": "xA (xG du tir directement assisté)",
}
UNITS = {"visible_wide_s": "s", "visible_close_s": "s", "ball_near_s": "s", "distance_observed_m": "m", "vmax_kmh": "km/h"}


def _metric(key, value, scope, method, reliability, reason, coverage=None, sources=None, evidence=None, **extra):
    return {"key": key, "label": LABELS.get(key, key), "value": value, "unit": UNITS.get(key, "nb"),
            "scope": scope, "method": method, "reliability": reliability, "reliability_reason": reason,
            "coverage": coverage, "sources": sources or [], "evidence": evidence or [], **extra}


def session_player_data(period: int, timeline, frames: list[dict], tracks: dict, identity: dict, prox: dict | None,
                        calib: dict | None, pass_cands: list | None, step: float) -> dict:
    """Quantités brutes par joueur pour UN fichier (une période)."""
    out = defaultdict(lambda: defaultdict(float))
    ev = defaultdict(list)
    seg_by_track = defaultdict(list)
    for s in identity["segments"]:
        seg_by_track[s["track_id"]].append(s)

    def player_at(tid, t):
        for s in seg_by_track.get(tid, []):
            if s["t_from"] - 1e-6 <= t <= s["t_to"] + 1e-6:
                return s
        return None

    for s in identity["segments"]:
        p = s["player"]
        tr = tracks[s["track_id"]]
        obs = [o for o in tr["obs"] if s["t_from"] - 1e-6 <= o["t"] <= s["t_to"] + 1e-6]
        wide = sum(1 for o in obs if o["view"] in ("wide", "medium"))
        close = sum(1 for o in obs if o["view"] == "close")
        out[p]["visible_wide_s"] += wide * step
        out[p]["visible_close_s"] += close * step
        out[p]["tracked_segments"] += 1
        out[p][f"rel_{s['reliability']}_s"] += (wide + close) * step
        out[p][f"src_{s['source']}"] += 1
        st = timeline.at(obs[0]["t"]) if obs else {}
        en = timeline.at(obs[-1]["t"]) if obs else {}
        ev[p].append({"kind": "tracked_segment", "period": period, "file_from": s["t_from"], "file_to": s["t_to"],
                      "clock_from": st.get("clock"), "clock_to": en.get("clock"), "source": s["source"],
                      "reliability": s["reliability"], "task_id": s.get("task_id"), "track_id": s["track_id"]})
        # proximité ballon dans ce tronçon
        if prox:
            pt = prox["per_track"].get(s["track_id"]) or prox["per_track"].get(str(s["track_id"]))
            if pt:
                n_near = sum(1 for t in pt["near_times"] if s["t_from"] - 1e-6 <= t <= s["t_to"] + 1e-6)
                n_touch = sum(1 for t in pt["touch_times"] if s["t_from"] - 1e-6 <= t <= s["t_to"] + 1e-6)
                out[p]["ball_near_s"] += n_near * step
                out[p]["touch_frames"] += n_touch
                out[p]["prox_eval_s"] += wide * step
        # physique sur images calibrées
        if calib and calib.get("H"):
            from .pitch import kinematics, to_pitch
            samples = []
            for o in obs:
                Hm = calib["H"].get(o["t"])
                if Hm is None or o["view"] not in ("wide", "medium"):
                    continue
                x1, y1, x2, y2 = o["box"]
                samples.append((o["t"],) + to_pitch(Hm, (x1 + x2) / 2, y2))
            if len(samples) >= 3:
                k = kinematics(samples)
                out[p]["distance_observed_m"] += k["distance_m"]
                out[p]["calibrated_s"] += k["covered_s"]
                out[p]["sprints_observed"] += k["sprints"]
                if k["vmax_kmh"] is not None:
                    out[p]["vmax_kmh"] = max(out[p].get("vmax_kmh", 0.0), k["vmax_kmh"])
    # passes candidates nommées (les deux pistes identifiées)
    for pc in pass_cands or []:
        a = player_at(pc["from_track"], pc["t_release"])
        b = player_at(pc["to_track"], pc["t_receive"])
        if a is None:
            continue
        p = a["player"]
        out[p]["pass_candidates"] += 1
        if pc["outcome_candidate"] == "completee":
            out[p]["pass_candidates_completed"] += 1
        ev[p].append({"kind": "pass_candidate", "period": period, "file_t": pc["t_release"],
                      "clock": timeline.at(pc["t_release"]).get("clock"), "to_player": None if b is None else b["player"],
                      "outcome_candidate": pc["outcome_candidate"], "status": "candidate_not_validated"})
    return {"players": {k: dict(v) for k, v in out.items()}, "evidence": {k: v for k, v in ev.items()}}


def build_actions(v2_events, human_events, banner_events):
    """Regroupe les observations en ACTIONS uniques.
    - ralenti qualifié par vous : l'action a eu lieu avant le dernier chrono de direct affiché,
      fenêtre [chrono − 60 s, chrono + 5 s] ;
    - événement V2 ou demande V2 : chrono précis (± 5 s) ; bandeau de carton : affiché après le
      fait, fenêtre [chrono − 120 s, chrono] ;
    - même famille d'action + fenêtres qui se recouvrent + même joueur → même action ;
    - même famille, joueurs différents : si l'une des sources est la relecture V2, votre réponse
      l'emporte et l'écart est consigné ; deux de vos réponses sur deux joueurs différents
      restent deux actions ;
    - deux ralentis qualifiés par vous, même joueur, même famille, chronos à moins de 20 s :
      même action rediffusée ; « Rediffusion » n'est jamais comptée."""
    acts = []
    human_v2 = {h.get("dedup_v2") for h in human_events if h.get("dedup_v2")}
    for e in v2_events:
        if not e.get("counted_in_metrics") or e["id"] in human_v2 or e.get("jersey_number") is None:
            continue
        c = _clock_s(e.get("match_clock"))
        keys = EVENT_MAP_V2.get((e["type"], e.get("outcome"))) or EVENT_MAP_V2.get((e["type"], None)) or []
        acts.append({"family": FAMILY_V2.get(e["type"]), "player": e["jersey_number"], "keys": [UNIFY.get(k, k) for k in keys],
                     "win": None if c is None else (c - 5, c + 5), "c": c, "src": "v2_assistant_review", "prio": 1,
                     "replay": False, "label": e.get("match_clock"), "detail": f"V2 : {e['type']} / {e.get('outcome')}"})
    for h in human_events:
        if h["type"] in ("rien", "autre", "rediffusion") or h["type"] not in EVENT_MAP_HUMAN and h["type"] not in FAMILY_HUMAN:
            continue
        c = _clock_s(h.get("match_clock"))
        from_v2 = str(h.get("task_id", "")).startswith("v2req-")
        if h.get("answer_origin") == "claude":
            # réponse de Claude (revue visuelle) : jamais présentée comme la vôtre, priorité sous vos réponses
            src, prio, who = "claude_review", 2, "revue visuelle de Claude"
        else:
            src = "human_review" if h.get("player_source", "human") == "human" else "human_review_player_v2"
            prio, who = (3 if src == "human_review" else 2), "votre réponse"
        acts.append({"family": FAMILY_HUMAN.get(h["type"]), "player": h.get("jersey_number"),
                     "keys": EVENT_MAP_HUMAN.get(h["type"], []),
                     "win": None if c is None else ((c - 5, c + 5) if from_v2 else (c - 60, c + 5)), "c": c,
                     "src": src, "prio": prio, "replay": not from_v2,
                     "label": (h.get("match_clock") or "?") + ("" if from_v2 else " (ralenti, action juste avant)"),
                     "detail": f"{who} : {h['type']}" + (f" — {h['outcome']}" if h.get("outcome") else "")})
    for b in banner_events:
        if b.get("needs_review") or b.get("duplicate_of") is not None:
            continue
        c = _clock_s(b.get("match_clock_banner_start"))
        for pl in b.get("players", []):
            if pl.get("team") != "raja" or pl.get("jersey_number") is None:
                continue
            fam, keys, win = {"yellow_card": ("card", ["yellow_cards"], None), "substitution_out": ("sub_out", ["sub_out_banner"], None),
                              "substitution_in": ("sub_in", ["sub_in_banner"], None)}.get(b["type"], (None, [], None))
            if not keys:
                continue
            acts.append({"family": fam, "player": pl["jersey_number"], "keys": keys,
                         "win": None if c is None else ((c - 120, c) if fam == "card" else (c - 5, c + 5)), "c": c,
                         "src": "auto_banner", "prio": 3, "replay": False,
                         "label": (b.get("match_clock_banner_start") or "?") + " (bandeau)", "detail": f"bandeau : {b['type']}"})
    acts.sort(key=lambda a: (a["win"][0] if a["win"] else 1e9))
    clusters, conflicts = [], []

    def overlap(w1, w2):
        return w1 and w2 and w1[0] <= w2[1] and w2[0] <= w1[1]

    for a in acts:
        target = None
        if a["family"] and a["win"]:
            for cl in clusters:
                if cl["family"] != a["family"] or not overlap(cl["win"], a["win"]):
                    continue
                both_replays = a["replay"] and cl["replay_only"]
                if both_replays:
                    if cl["player"] == a["player"] and a["c"] is not None and abs(a["c"] - cl["c"]) <= 20:
                        target = cl
                        break
                    continue
                if cl["player"] == a["player"] or a["player"] is None or cl["player"] is None or min(cl["prio"], a["prio"]) == 1:
                    target = cl
                    break
        if target is None:
            clusters.append({"family": a["family"], "player": a["player"], "keys": list(a["keys"]), "win": a["win"], "c": a["c"],
                             "prio": a["prio"], "sources": {a["src"]}, "replay_only": a["replay"], "clock_label": a["label"],
                             "detail": [a["detail"]]})
            continue
        target["sources"].add(a["src"])
        target["detail"].append(a["detail"])
        target["replay_only"] = target["replay_only"] and a["replay"]
        if a["player"] is not None and target["player"] is not None and a["player"] != target["player"]:
            conflicts.append({"type": "attribution", "family": a["family"], "clock": target["clock_label"],
                              "kept": a["player"] if a["prio"] > target["prio"] else target["player"],
                              "dropped": target["player"] if a["prio"] > target["prio"] else a["player"],
                              "note": "votre réponse l'emporte sur la relecture V2"})
        if a["prio"] > target["prio"] or (target["player"] is None and a["player"] is not None):
            target["player"] = a["player"] if a["player"] is not None else target["player"]
            target["keys"] = list(a["keys"])
            target["prio"] = a["prio"]
            target["clock_label"] = a["label"] if not a["replay"] else target["clock_label"]
        if target["family"] == "card" and any("yellow_cards" in x for x in [a["keys"], target["keys"]]) and "red_cards" not in target["keys"]:
            target["keys"] = ["yellow_cards"]
    for cl in clusters:
        cl["detail"] = " ; ".join(cl["detail"])
    return clusters, conflicts


def aggregate(match_cfg: dict, roster, sessions: list[dict], v2_events: list[dict], human_events: list[dict],
              banner_events: list[dict], goal_events: list[dict], xg_items: list[dict]) -> dict:
    """sessions : [{period, data: session_player_data(...), coverage: {...}}]"""
    players = []
    totals = defaultdict(lambda: defaultdict(float))
    evidence = defaultdict(list)
    for s in sessions:
        for p, d in s["data"]["players"].items():
            p = int(p)  # clés JSON → numéro de maillot
            for k, v in d.items():
                if k == "vmax_kmh":
                    totals[p][k] = max(totals[p].get(k, 0.0), v)
                else:
                    totals[p][k] += v
        for p, e in s["data"]["evidence"].items():
            evidence[int(p)].extend(e)
    actions, action_conflicts = build_actions(v2_events, human_events, banner_events)
    cov_ball = [s["coverage"].get("ball_located_fraction") for s in sessions if s["coverage"].get("ball_located_fraction") is not None]
    ball_frac = min(cov_ball) if cov_ball else None
    analysed_live = sum(s["coverage"].get("live_seconds", 0.0) for s in sessions)
    calibrated_any = any(s["coverage"].get("calibrated_frames", 0) > 0 for s in sessions)

    for pl in sorted(roster.players, key=lambda x: x["jersey_number"]):
        n = pl["jersey_number"]
        t = totals.get(n, {})
        m = {}
        tracked = t.get("tracked_segments", 0)
        rel_hi = t.get("rel_élevée_s", 0.0)
        rel_mid = t.get("rel_moyenne_s", 0.0)
        rel = "élevée" if tracked and rel_hi >= rel_mid else ("moyenne" if tracked else None)
        cov = {"live_seconds_analysed_match": round(analysed_live, 1),
               "note": "temps attribué à ce joueur ; le reste du temps de jeu n'est pas observé ou pas attribué"}
        if tracked:
            # 0 s attribué ≠ 0 s visible : une valeur nulle est publiée comme inconnue
            m["visible_wide_s"] = _metric("visible_wide_s", round(t["visible_wide_s"], 1) if t.get("visible_wide_s") else None, SCOPE_VISIBLE,
                                          "Somme des images de plans larges/moyens où une piste attribuée à ce joueur est visible × pas d'échantillonnage.",
                                          rel, "Attribution par réponses humaines ou numéro/maillot, propagée sans croisement.", cov)
            m["visible_close_s"] = _metric("visible_close_s", round(t["visible_close_s"], 1) if t.get("visible_close_s") else None, SCOPE_VISIBLE,
                                           "Images en gros plan attribuées ; ne prouve pas une présence sur le terrain.", rel,
                                           "Un gros plan peut précéder une entrée en jeu.", cov)
            m["tracked_segments"] = _metric("tracked_segments", int(tracked), SCOPE_VISIBLE, "Tronçons de suivi attribués.",
                                            rel, "Voir sources d'identité.", cov,
                                            sources=[k[4:] for k in t if k.startswith("src_")])
            if t.get("prox_eval_s", 0) > 0 and ball_frac is not None:
                rb = "faible" if ball_frac < 0.5 else "moyenne"
                m["ball_near_s"] = _metric("ball_near_s", round(t.get("ball_near_s", 0.0), 1), SCOPE_VISIBLE,
                                           "Images où le ballon est localisé à moins d'une hauteur de joueur des pieds (distance image, terrain non calibré).",
                                           rb, f"Ballon localisé sur {ball_frac:.0%} des images larges/moyennes analysées ; proximité ≠ possession.",
                                           {**cov, "ball_located_fraction": ball_frac, "evaluated_s": round(t.get("prox_eval_s", 0.0), 1)})
                m["touch_candidates"] = _metric("touch_candidates", int(t.get("touch_frames", 0)), SCOPE_CANDIDATES,
                                                "Images où le ballon est au pied (≤ 0,45 hauteur de joueur).", "faible",
                                                "Candidats non validés ; dépend fortement de la détection du ballon.", cov)
            if t.get("pass_candidates", 0) > 0:
                m["pass_candidates"] = _metric("pass_candidates", int(t["pass_candidates"]), SCOPE_CANDIDATES,
                                               "Changement de joueur au ballon (deux contrôles successifs) dans le même plan.",
                                               "faible", "Candidats non validés ; ne pas lire comme un total de passes.", cov,
                                               completed_candidates=int(t.get("pass_candidates_completed", 0)))
            if t.get("calibrated_s", 0) > 0:
                m["distance_observed_m"] = _metric("distance_observed_m", round(t["distance_observed_m"], 1), SCOPE_VISIBLE,
                                                   "Somme des déplacements lissés sur images calibrées (homographie ancrée par l'utilisateur, propagée dans le plan).",
                                                   "moyenne" if rel == "élevée" else "faible",
                                                   f"Mesuré sur {t['calibrated_s']:.1f} s seulement ; jamais extrapolé à 90 min.",
                                                   {**cov, "calibrated_seconds": round(t["calibrated_s"], 1)}, calibrated_pitch=True)
                if t.get("vmax_kmh"):
                    m["vmax_kmh"] = _metric("vmax_kmh", round(t["vmax_kmh"], 1), SCOPE_VISIBLE,
                                            "Vitesse moyenne maximale sur 1 s, positions lissées.", "faible",
                                            "Sensible à l'erreur de calibration et au point de contact au sol.",
                                            {**cov, "calibrated_seconds": round(t["calibrated_s"], 1)}, calibrated_pitch=True)
                m["sprints_observed"] = _metric("sprints_observed", int(t.get("sprints_observed", 0)), SCOPE_VISIBLE,
                                                "Vitesse ≥ 25,2 km/h pendant ≥ 1 s sur images calibrées.", "faible",
                                                "Mesure partielle (segments calibrés).", {**cov, "calibrated_seconds": round(t["calibrated_s"], 1)})
        # événements : actions fusionnées au niveau du match (build_actions), comptées pour ce joueur
        by_key = defaultdict(list)
        for cl in actions:
            if cl["player"] != n:
                continue
            for k in cl["keys"]:
                by_key[k].append(cl)
        for k, cl in by_key.items():
            src = sorted(set().union(*[c["sources"] for c in cl]))
            if all(c["sources"] & {"auto_banner", "human_review"} for c in cl):
                rel_e, why = "élevée", "Chaque occurrence est confirmée par un bandeau de diffusion (numéro + nom) ou par votre réponse."
            elif all(c["sources"] & {"auto_banner", "human_review", "human_review_player_v2"} for c in cl):
                rel_e, why = "moyenne", "Action qualifiée par vous ; l'attribution au joueur vient de la relecture V2."
            elif any("claude_review" in c["sources"] and not (c["sources"] & {"auto_banner", "human_review"}) for c in cl):
                rel_e, why = "moyenne", ("Au moins une occurrence vient de la revue visuelle des ralentis par Claude "
                                         "(non confirmée par vous).")
            else:
                rel_e, why = "moyenne", "Au moins une occurrence vient seulement de la relecture V2 par l'assistant (non confirmée par vous)."
            m[k] = _metric(k, len(cl), SCOPE_EVENTS,
                           "Actions repérées et qualifiées ; une même action vue par plusieurs sources (V2, ralenti, bandeau) ou "
                           "rediffusée est comptée une fois. Minimum observé, pas un total certifié.",
                           rel_e, why, {"full_match_count": False}, sources=src,
                           evidence=[{"clock": c["clock_label"], "sources": sorted(c["sources"]), "detail": c["detail"]} for c in cl])
        for x in xg_items:
            if x.get("jersey_number") == n:
                m[x["key"]] = _metric(x["key"], round(x["value"], 3), SCOPE_MODEL, x["method"], "faible", x["reason"],
                                      {"shots_located": x.get("shots_located", 1), "full_match_total": False},
                                      sources=["model_estimate"], evidence=x.get("evidence", []))
        players.append({"jersey_number": n, "player_name": pl["player_name"], "role": pl.get("role"),
                        "starter": pl.get("starter"), "participation_status": pl.get("participation_status"),
                        "declared_minutes": {"entered": pl.get("entered_minute_declared"), "left": pl.get("left_minute_declared"),
                                             "source": "composition déclarée (captures utilisateur)", "verified": False},
                        "metrics": m, "evidence": sorted(evidence.get(n, []), key=lambda e: (e.get("period") or 0, e.get("file_from") or e.get("file_t") or 0))})
    unknown = ["passes tentées/réussies (totaux du match)", "duels totaux et issues", "distance totale 90 min",
               "xG/xA totaux du match", "temps de jeu mesuré (entrée/sortie certifiées)"]
    return {"match_key": match_cfg["match_key"], "players": players, "team_events": goal_events,
            "actions": [{k: (sorted(v) if isinstance(v, set) else v) for k, v in a.items() if k != "members"} for a in actions],
            "action_conflicts": action_conflicts,
            "coverage": {"sessions": [s["coverage"] for s in sessions], "ball_located_fraction_min": ball_frac,
                         "calibration_available": calibrated_any},
            "not_measured": unknown,
            "conventions": {"unknown_is_null": True, "events_are_minima": True,
                            "reliability_levels": ["élevée", "moyenne", "faible"],
                            "no_invented_percentages": True}}
