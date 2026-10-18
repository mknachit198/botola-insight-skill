"""Rapport HTML autonome du match (publiable), construit uniquement à partir des sorties du moteur."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .timeline import Timeline
from .util import fmt_clock, read_json

COLS = [
    ("Présence filmée", [("visible_wide_s", "Plan large (s)"), ("visible_close_s", "Gros plan (s)")]),
    ("Actions", [("goals", "Buts"), ("shots", "Tirs"), ("shots_on_target", "Cadrés"), ("shots_blocked", "Contrés"),
                 ("shot_assists", "Passes avant tir"), ("passes_completed", "Passes réussies"), ("chance_or_cross", "Centres / occasions"),
                 ("crosses_blocked", "Centres contrés"), ("duels", "Duels"), ("tackles", "Tacles"), ("fouls_committed", "Fautes commises"),
                 ("fouls_won", "Fautes obtenues"), ("fouls_involved", "Fautes (sens inconnu)")]),
    ("Gardien", [("saves", "Arrêts"), ("keeper_claims", "Prises"), ("keeper_punches", "Poings"), ("keeper_unsecured", "Sorties non sécurisées"),
                 ("keeper_interventions", "Interventions")]),
    ("Discipline et changements", [("yellow_cards", "Jaunes"), ("sub_out_banner", "Sortie"), ("sub_in_banner", "Entrée")]),
    ("Modèle", [("xg", "xG"), ("xa", "xA")]),
    ("Physique (segments calibrés)", [("distance_observed_m", "Distance (m)"), ("vmax_kmh", "Vmax (km/h)")]),
]
SRC_FR = {"auto_banner": "bandeau de diffusion", "human_review": "votre réponse", "claude_review": "revue visuelle de Claude", "human_review_player_v2": "votre réponse (joueur V2)",
          "v2_assistant_review": "relecture V2", "scoreboard": "tableau de score", "model_estimate": "modèle"}
EV_FR = {"yellow_card": "Carton jaune", "substitution_out": "Sortie", "substitution_in": "Entrée", "red_card_possible": "Carton rouge ?"}


def _e(x):
    return html.escape(str(x)) if x is not None else ""


def build(ws, out_path: Path | None) -> dict:
    stats = read_json(ws.out / "stats_joueurs.json")
    sessions = sorted(ws.sessions(), key=lambda s: s.meta["period"])
    periods = []
    facts = []
    for s in sessions:
        tl = read_json(s.dir / "timeline.json")
        ev = read_json(s.dir / "events.json", {"banners": [], "goals": []})
        st = s.meta.get("stages", {})
        live = sum(x["file_end"] - x["file_start"] for x in tl["segments"] if x["kind"] in ("live", "live_banner", "live_interpolated"))
        stop = [x for x in tl["segments"] if x["kind"] == "stoppage"]
        gaps = []
        for x in tl["segments"]:
            if x["kind"] in ("discontinuity", "frozen") and x.get("clock_before") and x.get("clock_after") and x["clock_before"] != x["clock_after"]:
                gaps.append(x)
        gaps += [{"clock_before": c["clock_before"], "clock_after": c["clock_after"], "kind": "cut"} for c in tl.get("hard_cuts", [])]
        repl = sum(1 for x in tl["segments"] if x["kind"] == "no_scoreboard" and 3 <= x["file_end"] - x["file_start"] <= 90)
        idq = st.get("identity", {})
        periods.append({"period": s.meta["period"], "live_s": live,
                        "stoppage": [(fmt_clock(x["base"]), x.get("added_start"), x.get("added_end")) for x in stop],
                        "gaps": gaps, "replays": repl, "qa": idq.get("tracking_qa", {}),
                        "ball": idq.get("ball_located_fraction"), "attributed": idq.get("attributed_frames"),
                        "raja_frames": idq.get("raja_track_frames"), "frames": st.get("analyze", {}).get("frames"),
                        "speed": st.get("analyze", {}).get("speed_x_vs_realtime")})
        for g in ev.get("goals", []):
            facts.append((g.get("match_clock_score_update"), "But", f"{'Raja' if g['team'] == 'raja' else 'Adversaire'} — score {g['score_after'][0]}-{g['score_after'][1]}",
                          "tableau de score (mise à jour après le but)", s.meta["period"]))
        for b in ev.get("banners", []):
            if b.get("duplicate_of") is not None or b["type"] == "banner_info":
                continue
            opp = b.get("team_tag") == "RCAZ" or (b["players"] and all(p.get("team") == "opponent" for p in b["players"]))
            if opp:
                names = ", ".join(f"{p.get('jersey_number') or p.get('jersey_number_read') or '?'} {(p.get('name_read') or '').title()}" for p in b["players"])
                who = (ws.cfg.get("opponent") or "Adversaire") + " — " + names + " (non compté pour le Raja)"
            else:
                names = ", ".join(f"{p.get('jersey_number') or p.get('jersey_number_read') or '?'} {p.get('player_name') or p.get('name_read') or ''}" for p in b["players"])
                who = "Raja — " + names
            facts.append((b.get("match_clock_banner_start"), EV_FR.get(b["type"], "Bandeau"), who + (" — à vérifier" if b.get("needs_review") else ""),
                          "bandeau de diffusion (heure d'affichage ; équipe lue par la position de l'étiquette)", s.meta["period"]))
    facts.sort(key=lambda f: (f[4], f[0] or ""))
    players = stats["players"]
    used = [(g, [(k, l) for k, l in cols if any(p["metrics"].get(k) and p["metrics"][k]["value"] is not None for p in players)]) for g, cols in COLS]
    used = [(g, cols) for g, cols in used if cols]
    q = ws.store.counts()
    answered = sum(v.get("resolved", 0) + v.get("unknown", 0) for v in q.values())
    pending = sum(v.get("pending", 0) for v in q.values())
    events_rows = []
    for p in players:
        for k, m in p["metrics"].items():
            if m["scope"] != "reviewed_events":
                continue
            for e in m.get("evidence", []):
                events_rows.append((e.get("clock") or "", p["jersey_number"], p["player_name"], m["label"], ", ".join(SRC_FR.get(s, s) for s in e.get("sources", [e.get("source")]))))
    # une ligne par action (même chrono, même joueur) : « Buts, Tirs, Tirs cadrés » ensemble
    grouped = {}
    for c, n, name, lab, src in events_rows:
        g = grouped.setdefault((c, n), [c, n, name, [], set()])
        if lab not in g[3]:
            g[3].append(lab)
        g[4].update(x.strip() for x in src.split(","))
    events_rows = [(c, n, name, ", ".join(labs), ", ".join(sorted(srcs))) for c, n, name, labs, srcs in grouped.values()]

    def _key(r):
        from .util import parse_clock
        v, a = parse_clock(r[0].split("–")[0].strip()) if r[0] else (None, None)
        return (v if v is not None else 10 ** 6) + (a or 0)
    events_rows.sort(key=_key)

    def cell(m):
        if not m or m["value"] is None:
            return '<td class="n unk">—</td>'
        v = m["value"]
        v = f"{v:.2f}" if isinstance(v, float) and m["key"] in ("xg", "xa") else (f"{v:g}" if isinstance(v, (int, float)) else v)
        tip = f"{m['method']} — {m['reliability']} : {m['reliability_reason']} Sources : {', '.join(SRC_FR.get(s, s) for s in m.get('sources', []))}"
        return f'<td class="n" title="{_e(tip)}">{_e(v)}<i class="r {_e(m["reliability"])}"></i></td>'

    head1 = "".join(f'<th colspan="{len(c)}" class="grp">{_e(g)}</th>' for g, c in used)
    head2 = "".join(f'<th class="n">{_e(l)}</th>' for g, c in used for k, l in c)
    rows = []
    for p in players:
        part = {"starter": "titulaire", "entered": f"entré {p['declared_minutes']['entered']}′", "no_entry_reported": "non entré (déclaré)"}.get(p["participation_status"], p["participation_status"] or "")
        if p["declared_minutes"].get("left"):
            part += f", sorti {p['declared_minutes']['left']}′"
        rows.append(f"<tr><td class=\"num\">{p['jersey_number']}</td><td class=\"pl\">{_e(p['player_name'])}<small>{_e(part)}</small></td>"
                    + "".join(cell(p["metrics"].get(k)) for g, c in used for k, l in c) + "</tr>")
    per_html = []
    for pr in periods:
        def _dur(g):
            from .util import parse_clock
            a, _ = parse_clock(g["clock_before"])
            b, _ = parse_clock(g["clock_after"])
            return "" if a is None or b is None else f" ({(b - a) // 60} min {(b - a) % 60:02d} s)"
        gaps = "".join(f"<li>{_e(g['clock_before'])} → {_e(g['clock_after'])}{_dur(g)} : {'image figée' if g['kind'] == 'frozen' else 'coupure'}, jeu non observable</li>" for g in pr["gaps"]) or "<li>aucune</li>"
        stop = ", ".join(f"{b} +{a // 60}:{a % 60:02d} → +{z // 60}:{z % 60:02d}" for b, a, z in pr["stoppage"] if a is not None and z is not None) or "—"
        qa = pr["qa"]
        per_html.append(f"""<div class="per"><h3>Partie {pr['period']}</h3><dl>
<dt>Direct analysé</dt><dd>{pr['live_s'] / 60:.1f} min ({pr['frames'] or '—'} images analysées)</dd>
<dt>Temps additionnel lu</dt><dd>{_e(stop)}</dd>
<dt>Ruptures de diffusion</dt><dd><ul>{gaps}</ul></dd>
<dt>Ralentis repérés</dt><dd>{pr['replays']}</dd>
<dt>Ballon localisé</dt><dd>{'—' if pr['ball'] is None else f"{pr['ball']:.0%} des images larges"}</dd>
<dt>Pistes de joueurs</dt><dd>durée médiane {qa.get('median_track_duration_s', '—')} s ; {qa.get('impure_fraction', 0):.0%} à couleur instable</dd>
<dt>Images Raja attribuées</dt><dd>{pr['attributed'] or 0} sur {pr['raja_frames'] or '—'}</dd></dl></div>""")
    facts_html = "".join(f"<tr><td class=\"mono\">{_e(c or '?')}</td><td>{_e(t)}</td><td>{_e(d)}</td><td class=\"src\">{_e(s)}</td></tr>" for c, t, d, s, _ in facts)
    ev_html = "".join(f"<tr><td class=\"mono\">{_e(c)}</td><td>{n} {_e(name)}</td><td>{_e(lab)}</td><td class=\"src\">{_e(src)}</td></tr>" for c, n, name, lab, src in events_rows)
    contra = stats.get("contradictions") or []
    # score : dernier score lu sur le tableau après un but ; sinon inconnu (jamais 0-0 par défaut)
    final_score = "—"
    last = None
    for s_ in sessions:
        for g in read_json(s_.dir / "events.json", {"goals": []}).get("goals", []):
            if g.get("score_after") and (last is None or (s_.meta["period"], g.get("file_t_score_update") or 0) > last[0]):
                last = ((s_.meta["period"], g.get("file_t_score_update") or 0), g["score_after"])
    if last:
        final_score = f"{last[1][0]}-{last[1][1]}"
    names = {p["jersey_number"]: p["player_name"] for p in ws.roster.players} if hasattr(ws.roster, "players") else {}
    FAM = {"shot": "tir", "cross": "centre / occasion", "foul": "faute", "pass": "passe", "duel": "duel"}

    def contra_txt(c):
        if c.get("type") == "attribution":
            k, d = c.get("kept"), c.get("dropped")
            return (f"{c.get('clock')} — {FAM.get(c.get('family'), c.get('family'))} : {c.get('note')} "
                    f"(retenu : {k} {names.get(k, '')} ; écarté : {d} {names.get(d, '')}).")
        return c.get("note") or c.get("detail") or json.dumps(c, ensure_ascii=False)
    contra_items = [contra_txt(c) for c in contra]
    # conflits d'identité (deux pistes, même joueur, même moment) et décision appliquée
    for s_ in sessions:
        idn = read_json(s_.dir / "identity.json", {}) or {}
        tl_ = Timeline(read_json(s_.dir / "timeline.json"))
        for c in idn.get("conflicts", []):
            a, b, dec = c.get("a", {}), c.get("b", {}), c.get("decision") or {}
            clock = tl_.at(a.get("anchor_t", 0)).get("clock") or f"fichier {a.get('anchor_t', 0):.0f} s"
            contra_items.append(f"{clock} (période {s_.meta['period']}) — deux joueurs nommés {a.get('player_name')} au même moment : "
                                f"{dec.get('why', 'à trancher')}.")
    sm = stats.get("sportmonks") or {}
    if out_path is None:  # données seules (export du site)
        return {"periods": periods, "facts": facts, "players": players, "events_rows": events_rows,
                "divergences": contra_items, "score": final_score, "score_raw": last[1] if last else None,
                "answered": answered, "pending": pending, "stats": stats}
    doc = TEMPLATE.format(
        score=final_score,
        periods="".join(per_html), facts=facts_html or '<tr><td colspan="4">—</td></tr>',
        head1=head1, head2=head2, rows="".join(rows), events=ev_html or '<tr><td colspan="4">—</td></tr>',
        answered=answered, pending=pending, generated=_e(stats.get("generated_at")), version=_e(stats.get("engine_version")),
        contra="".join(f"<li>{_e(c)}</li>" for c in contra_items) or "<li>aucune détectée automatiquement</li>",
        sportmonks=_e(sm.get("status") or ("connecté : " + str(sm.get("fixture_id")))))
    Path(out_path).write_text(doc, encoding="utf-8")
    return {"players": len(players), "facts": len(facts), "events": len(events_rows)}


TEMPLATE = """<title>Rapport Raja–Zemamra</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono&display=swap">
<style>
/* Rapport d'analyste : bandeau de résultat, couverture par partie, faits lus, tableau par joueur, chronologie. */
:root{{--bg:#f4f7f4;--panel:#fff;--ink:#132119;--muted:#5b6c62;--line:#d5dfd8;--accent:#0a6b39;--hi:#0a6b39;--mid:#9a7a00;--lo:#b2432f;
--display:"Barlow Condensed","Arial Narrow",sans-serif;--body:"Source Sans 3","Segoe UI",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0e1612;--panel:#151f19;--ink:#e5eee7;--muted:#9aafa2;--line:#283630;--accent:#3cc47a;--hi:#5fd795;--mid:#e3c45a;--lo:#ef8a76;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0e1612;--panel:#151f19;--ink:#e5eee7;--muted:#9aafa2;--line:#283630;--accent:#3cc47a;--hi:#5fd795;--mid:#e3c45a;--lo:#ef8a76;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}}
.wrap{{max-width:1240px;margin:0 auto;display:flex;flex-direction:column;gap:22px}}
h1{{font:700 40px/1 var(--display);margin:0;text-wrap:balance}}h2{{font:700 26px/1.1 var(--display);margin:0 0 10px}}h3{{font:700 20px/1.1 var(--display);margin:0 0 8px}}
.hero{{display:flex;flex-wrap:wrap;gap:12px 28px;align-items:flex-end;border-bottom:2px solid var(--ink);padding-bottom:14px}}
.score{{font:700 56px/1 var(--display);letter-spacing:.02em}}.score small{{display:block;font:600 13px var(--body);color:var(--muted);letter-spacing:0}}
.lede{{max-width:70ch;color:var(--muted);margin:0}}
.pers{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:14px}}
.per{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}}
dl{{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0;font-size:14px}}dt{{color:var(--muted)}}dd{{margin:0;min-width:0}}dd ul{{margin:0;padding-left:18px}}
.tw{{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-weight:600;color:var(--muted);white-space:nowrap;background:var(--panel)}}th.grp{{text-align:center;border-left:1px solid var(--line);color:var(--ink);font:700 14px var(--display);letter-spacing:.04em;text-transform:uppercase}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}td.unk{{color:var(--line)}}td.num{{font:13px var(--mono);color:var(--muted)}}
td.pl{{white-space:nowrap;font-weight:600}}td.pl small{{display:block;font-weight:400;color:var(--muted);font-size:12px}}
td.mono{{font:13px var(--mono);white-space:nowrap}}td.src{{color:var(--muted);font-size:12.5px}}
.r{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-left:5px;vertical-align:middle;background:var(--line)}}.r.élevée{{background:var(--hi)}}.r.moyenne{{background:var(--mid)}}.r.faible{{background:var(--lo)}}
.legend{{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 16px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:18px}}
.note{{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:6px;padding:12px 14px;font-size:14px;max-width:none}}
.note ul{{margin:6px 0 0;padding-left:18px}}
footer{{color:var(--muted);font-size:12.5px}}
</style>
<div class="wrap">
<header class="hero"><div><h1>Raja–Zemamra</h1><p class="lede">Analyse vidéo du match complet (deux enregistrements Arryadia, 854×480). Chrono de référence : tableau incrusté en haut à gauche.</p></div>
<div class="score">{score}<small>score lu sur le tableau</small></div></header>

<section><h2>Couverture</h2><div class="pers">{periods}</div></section>

<section class="grid2">
<div><h2>Faits lus automatiquement</h2><div class="tw"><table><thead><tr><th>Chrono</th><th>Fait</th><th>Détail</th><th>Source</th></tr></thead><tbody>{facts}</tbody></table></div></div>
<div class="note"><b>Comment lire les chiffres</b><ul>
<li>« — » signifie inconnu ou non observé, jamais zéro.</li>
<li>Les actions sont des <b>minima repérés</b> (ralentis qualifiés, relecture V2, bandeaux), pas des totaux certifiés du match.</li>
<li>Pastille : <i class="r élevée"></i> élevée (bandeau lu ou votre réponse), <i class="r moyenne"></i> moyenne (relecture V2, identité propagée), <i class="r faible"></i> faible (modèle, candidat automatique). Survolez une valeur pour sa méthode.</li>
<li>Les minutes de jeu sont celles de la composition déclarée ; les bandeaux de remplacement donnent l'heure d'affichage.</li>
<li>Aucune distance ni vitesse sans plan calibré ; aucune extrapolation à 90 minutes.</li></ul></div>
</section>

<section><h2>Statistiques par joueur</h2>
<div class="tw"><table><thead><tr><th></th><th></th>{head1}</tr><tr><th>N°</th><th>Joueur</th>{head2}</tr></thead><tbody>{rows}</tbody></table></div>
<p class="legend"><span>{answered} questions répondues, {pending} en attente : chaque réponse supplémentaire complète ce tableau.</span><span>Sportmonks : {sportmonks}.</span></p></section>

<section class="grid2">
<div><h2>Actions attribuées</h2><div class="tw"><table><thead><tr><th>Chrono</th><th>Joueur</th><th>Action</th><th>Source</th></tr></thead><tbody>{events}</tbody></table></div></div>
<div><h2>Divergences signalées</h2><div class="note"><ul>{contra}</ul></div></div>
</section>
<footer>Moteur Raja Insight {version} · généré le {generated} · méthodes et limites : docs/METHODE.md et docs/EVALUATION.md du moteur.</footer>
</div>
"""
