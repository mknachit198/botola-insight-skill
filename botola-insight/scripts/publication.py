"""Couche de sortie : seuils du moteur, JSON WordPress 2.2 et paquet lisible."""
from __future__ import annotations

import copy
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone

from common import LIMIT, VERSION, match, read, record, slug, team_name, write
from raja_engine.publish import FROZEN_SITE_METRICS, THRESHOLDS, publish_rows, site_bundle_experimental
from raja_engine.timeline import Timeline


def number(v, label, maximum=1e7):
    if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= maximum:
        raise ValueError(f"Valeur numérique invalide : {label}")
    return v


def write_csv(path, header, rows):
    # Les sorties peuvent contenir du texte utilisateur : neutraliser les formules de tableur.
    def cell(v):
        return "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(header)
        w.writerows([[cell(v) for v in row] for row in rows])


def packet(data, out):
    d = site_bundle_experimental(data)
    d.setdefault("possession", {"windows": [], "raja": None, "coverage": None})
    d["possession"].update(experimental=True, reliability="faible")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "01_match.json", d)
    focus = d.get("focus_team", d["home"])
    opponent = d.get("opponent_team", d["away"])
    write_csv(out / "02_faits_du_match.csv",
              ["periode", "chrono", "type", "equipe", "detail", "buteur_numero", "buteur_nom", "source", "a_verifier"],
              [[f["period"], f["clock"], f["type"], focus if f["team"] == "raja" else opponent, f["detail"],
                (f.get("scorer") or {}).get("number", ""), (f.get("scorer") or {}).get("name", ""), f["source"],
                "oui" if f.get("needs_review") else "non"] for f in d["facts"]])
    write_csv(out / "03_joueurs_valeurs.csv",
              ["numero", "nom", "poste", "statut", "a_joue", "groupe", "statistique", "cle", "valeur", "unite",
               "fiabilite", "experimental", "sources", "methode", "raison_de_la_fiabilite"],
              [[p["number"], p["name"], p["role"], p["status"], "oui" if p["played"] else "non", m.get("group", ""),
                m.get("label", m["key"]), m["key"], m["value"], m.get("unit", ""), m["reliability"],
                "oui" if m.get("experimental") else "non", ", ".join(m.get("sources", [])), m.get("method", ""), m.get("reason", "")]
               for p in d["players"] for m in p["metrics"]])
    write_csv(out / "04_joueurs_actions.csv", ["numero", "nom", "chrono", "action", "source"],
              [[p["number"], p["name"], a["clock"], a["label"], a.get("source", "")] for p in d["players"] for a in p["actions"]])
    write_csv(out / "05_couverture_video.csv",
              ["periode", "minutes_de_direct_analysees", "observations_equipe_identifiees_pct", "images_ballon_repere_pct", "interruptions"],
              [[c["period"], c["live_min"], round(100 * c["attributed_frames"] / c["raja_frames"], 1) if c.get("raja_frames") else "",
                "" if c.get("ball_located") is None else round(100 * c["ball_located"], 1), " ; ".join(c["gaps"])] for c in d["coverage"]])
    (out / "06_points_en_suspens.txt").write_text("\n".join("- " + x for x in d["divergences"]) + "\n", encoding="utf-8")
    pos = d["possession"]
    write_csv(out / "07_possession_estimee_experimentale.csv",
              ["periode", "tranche", "part_equipe_suivie_estimee", "secondes_ballon_repere", "fiabilite", "experimental"],
              [[w["period"], w["label"], "" if w.get("raja") is None else w["raja"], w.get("known_s", ""), "faible", "oui"] for w in pos["windows"]]
              + [["partie", "ensemble", "" if pos.get("raja") is None else pos["raja"], "", "faible", "oui"]])
    (out / "00_CONSIGNES_A_COLLER.md").write_text(
        "# Consignes d'analyse\n\n" + LIMIT + "\n\n"
        "Décrire uniquement cette partie et les données présentes. L'inconnu reste inconnu. "
        "La couverture mesure les observations publiées, pas leur exactitude. Les scores ne sont pas calibrés. "
        "Un temps visible n'est ni un temps de jeu ni une couverture de toutes les apparitions. "
        "Les estimations de possession, passes, distances, vitesses, xG et xA sont toujours expérimentales, "
        "de fiabilité faible. Une valeur absente n'est pas zéro. Ne pas déduire d'identité du physique ou de l'équipement. "
        "Les résultats de référence portent sur le Raja ; les autres clubs de Botola ne sont pas validés.\n",
        encoding="utf-8")


def packet_command(args):
    d = read(args.site)
    if d.get("format") != "raja-insight-site" or d.get("version") != 1:
        raise ValueError("Format site_match.json version 1 attendu.")
    packet(d, args.out)
    print(str(args.out))


def publish(args):
    cfg, res = match(args.match), read(args.resolution)
    slug(args.part)
    if not re.fullmatch(r"[a-z0-9-]{3,80}", cfg["match_key"]):
        raise ValueError("WordPress 2.2 exige match_key en minuscules/chiffres/tirets, 3 à 80 caractères.")
    if res.get("match_key") != cfg["match_key"]:
        raise ValueError("Résolution issue d'un autre match.")
    label = res.get("team_label", "raja")
    if label not in ("raja", "raja_gk"):
        raise ValueError("L'export WordPress comporte une seule équipe de joueurs. Pour publier l'adversaire, le configurer comme équipe suivie dans un match séparé.")
    roster = {p["jersey_number"]: p for p in cfg["roster"]["players"]}
    timeline = Timeline(read(args.timeline))
    rows = res["observations"]
    seen = set()
    for r in rows:
        number(r["p"], "probabilité", 1)
        number(r["t"], "temps fichier")
        key = (r["track"], round(r["t"], 3))
        if key in seen:
            raise ValueError("Observation dupliquée dans la résolution.")
        seen.add(key)
        if r["player"] is not None and r["player"] not in roster:
            raise ValueError("Identité hors composition dans la résolution.")
    annotated = publish_rows(rows, "visible_time")
    live = [r for r in annotated if timeline.at(r["t"]).get("live")]
    by_number = Counter(r["published"] for r in live if r["published"] is not None)
    unknown = len(live) - sum(by_number.values())
    public_rows = [{"track": r["track"], "t": r["t"], "player": r["published"],
                    "p": r["p"], "threshold": .90, "unit": r["unit"]} for r in live]
    players = []
    for n, p in roster.items():
        count = by_number[n]
        declared_played = bool(p["starter"] or p.get("entered_minute_declared") is not None)
        # WP 2.2 traduit played=false par « sur le banc, sans entrer ». Ne jamais
        # convertir une absence d'identification en cette affirmation.
        if not count and not declared_played:
            continue
        status = "Titulaire déclaré" if p["starter"] else "Entrée déclarée" if declared_played else "Identifié"
        status += " ; identifié sur cette partie" if count else " ; non identifié sur cette partie"
        players.append({"number": n, "name": p["player_name"], "role": p["role"],
                        "status": status, "played": True, "metrics": ([{
                            "key": "visible_s", "label": "Temps visible identifié", "group": "Observations de cette partie",
                            "value": round(count / 5.0, 2), "unit": "s", "reliability": "faible",
                            "sources": ["Suivi à 5 i/s et lectures des numéros"],
                            "method": "Une observation publiée = 0,2 s ; seuil p ≥ 0,90 ; direct seulement.",
                            "reason": "Scores non calibrés ; suivi incomplet ; ne mesure pas le temps de jeu."}] if count else []),
                        "actions": []})
    pmap = {p["number"]: p for p in players}
    if not players:
        raise ValueError("Aucune participation déclarée ni identité publiée : demander la composition avant l'export WordPress.")
    event_rows = publish_rows(rows, "events")
    emap = {(r["track"], round(r["t"], 3)): r["published"] for r in event_rows}
    facts, suppressed = [], 0
    if args.events:
        for e in read(args.events)["events"]:
            if e.get("reviewed") is not True or not e.get("source"):
                raise ValueError("Chaque événement exige une relecture explicite et une source.")
            if "player" in e or "scorer" in e or "number" in e:
                raise ValueError("L'identité d'un événement vient de la couche 2 à p ≥ 0,95, pas d'un champ libre.")
            clock = timeline.at(number(e["t"], "temps événement"))
            if not clock.get("live"):
                suppressed += 1
                continue
            n = emap.get((e["track"], round(e["t"], 3)))
            kind = {"goal": "But"}.get(e["type"], e["type"])
            fact = {"period": clock["period"], "clock": clock["clock"], "type": kind,
                    "team": "raja", "detail": e.get("detail", ""), "source": e["source"],
                    "needs_review": n is None, "scorer": None}
            if n is not None:
                if kind == "But":
                    fact["scorer"] = {"number": n, "name": roster[n]["player_name"], "source": e["source"]}
                pmap[n]["actions"].append({"clock": clock["clock"], "label": e.get("label", e["type"]), "source": e["source"]})
            facts.append(fact)
    possession = {"windows": [], "raja": None, "coverage": None, "experimental": True, "reliability": "faible"}
    if args.experimental:
        estimates = read(args.experimental)
        for p in estimates.get("players", []):
            n = p["number"]
            if n not in roster:
                raise ValueError("Estimation pour un numéro hors composition.")
            if n not in pmap:
                raise ValueError("Estimation d'un joueur sans participation déclarée ni identité publiée : demander la source de participation.")
            for raw in p["metrics"]:
                m = copy.deepcopy(raw)
                if m["key"] not in FROZEN_SITE_METRICS or not m.get("sources") or not m.get("method"):
                    raise ValueError("Estimation : clé expérimentale connue, sources et méthode requises.")
                number(m["value"], m["key"])
                pmap[n]["metrics"].append(m)
        if estimates.get("possession"):
            p = estimates["possession"]
            if not p.get("source") or not p.get("method"):
                raise ValueError("Possession : source et méthode requises.")
            for k in ("raja", "coverage"):
                if p.get(k) is not None: number(p[k], k, 1)
            for w in p.get("windows", []):
                if w.get("raja") is not None: number(w["raja"], "part possession", 1)
                if w.get("known_s") is not None: number(w["known_s"], "secondes ballon", 4000)
            possession.update(p)
    periods = sorted({s.get("period") for s in timeline.segments if s.get("period") is not None})
    coverage = []
    for period in periods:
        segments = [s for s in timeline.segments if s.get("period") == period]
        seconds = sum(s["file_end"] - s["file_start"] for s in segments if s["kind"] in ("live", "live_banner", "live_interpolated", "stoppage"))
        rr = [r for r in live if timeline.at(r["t"])["period"] == period]
        coverage.append({"period": period, "live_min": round(seconds / 60, 3),
                         "raja_frames": len(rr), "attributed_frames": sum(r["published"] is not None for r in rr),
                         "ball_located": None, "gaps": [f"{s['kind']} : {s['file_start']:.1f}–{s['file_end']:.1f} s fichier"
                                                        for s in segments if s["kind"] not in ("live", "live_banner", "live_interpolated", "stoppage")]})
    focus, opponent = team_name(cfg, label), cfg["opponent"]
    venue = cfg.get("focus_venue")
    actual_home, actual_away = (opponent, focus) if venue == "away" else (focus, opponent)
    limitations = [LIMIT, "Analyse limitée à la partie " + args.part + ".",
                  "Les tests de référence portent sur le Raja ; les autres clubs de Botola restent à valider.",
                  "Couverture des observations détectées, pas mesure d'exactitude ni temps de jeu.",
                  "Possession, passes, distances, vitesses, xG et xA : expérimental, fiabilité faible ; absents sans source."]
    if venue not in ("home", "away", "neutral"):
        limitations.append("Ordre d'affichage : équipe suivie puis adversaire ; domicile/extérieur non renseigné.")
    limitations.append("Fiches WordPress : participation déclarée ou identité publiée ; les autres joueurs ne sont pas déclarés absents et restent dans le bilan des non-identifiés.")
    if suppressed:
        limitations.append(f"{suppressed} événements hors direct exclus.")
    score = copy.deepcopy(cfg.get("score_declared"))
    if score is not None:
        for k in ("raja", "opponent"): number(score[k], "score " + k, 40)
        if not score.get("source"):
            raise ValueError("Source du score déclaré manquante.")
    bundle = {"format": "raja-insight-site", "version": 1, "match_key": cfg["match_key"],
              "title": cfg.get("title") or focus + " – " + opponent, "home": focus, "away": opponent,
              "actual_home": actual_home if venue in ("home", "away") else None,
              "actual_away": actual_away if venue in ("home", "away") else None,
              "focus_team": focus, "opponent_team": opponent, "focus_venue": venue,
              "competition": cfg.get("competition", "Botola"), "season": cfg.get("season", ""), "round": cfg.get("round", ""),
              "date": cfg.get("date") if cfg.get("date_verified") else None,
              "date_source": cfg.get("date_source", ""), "score": score,
              "players": players, "facts": facts, "coverage": coverage, "possession": possession,
              "divergences": limitations, "generated_at": datetime.now(timezone.utc).isoformat(),
              "engine_version": "botola-skill-" + VERSION, "part": args.part}
    bundle = site_bundle_experimental(bundle)
    bundle["possession"].update(experimental=True, reliability="faible")
    bundle["publication"].update(thresholds=THRESHOLDS, calibrated=False, step1_passed=False)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "site_match.json", bundle)
    write(args.out / "identities_published.json", {"observations": public_rows, "threshold": .90})
    summary = {"part": args.part, "team": focus, "observations": len(live), "published": sum(by_number.values()),
               "unknown": unknown, "coverage_pct": round(100 * sum(by_number.values()) / len(live), 2) if live else None,
               "players_identified": len(by_number), "roster_players_not_identified": [n for n in roster if n not in by_number],
               "events": len(facts), "events_outside_live_excluded": suppressed, "limitations": limitations,
               "step1_passed": False, "calibrated": False}
    write(args.out / "summary.json", summary)
    cov = "non mesurable" if summary["coverage_pct"] is None else str(summary["coverage_pct"]) + " %"
    (args.out / "COMPTE_RENDU.md").write_text(
        f"# {bundle['title']} — partie {args.part}\n\n{LIMIT}\n\n"
        f"Couverture publiée : {cov} des {len(live)} observations détectées de {focus}. "
        f"{unknown} observations restent inconnues. {len(by_number)} joueurs sont identifiés au moins une fois.\n\n"
        "La liste des numéros non identifiés figure dans summary.json ; cela ne prouve pas leur absence du terrain.\n\n"
        + "\n".join("- " + x for x in limitations) + "\n", encoding="utf-8")
    packet(bundle, args.out / "paquet_analyse")
    record(args.out, "publish", [args.match, args.timeline, args.resolution, args.events, args.experimental],
           {"thresholds": THRESHOLDS, "part": args.part}, [args.out / "site_match.json", args.out / "summary.json"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def report(args):
    print((args.publication / "COMPTE_RENDU.md").read_text(encoding="utf-8"))
