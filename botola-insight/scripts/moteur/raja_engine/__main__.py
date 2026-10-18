"""Ligne de commande du moteur Raja Insight.

  python -m raja_engine serve   --match matches/raja-zemamra/match.json --work work
  python -m raja_engine run     --match ... --work ... --video FICHIER.mp4 [--period 1] [--batch]
  python -m raja_engine import-v2 --match ... --work ... --v2 CHEMIN/Raja_Zemamra_Mesures
  python -m raja_engine answer  --match ... --work ... --task ID --value VALEUR [--note ...]
  python -m raja_engine status  --match ... --work ...
  python -m raja_engine aggregate --match ... --work ...
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="raja_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--match", required=True, type=Path)
        p.add_argument("--work", required=True, type=Path)

    p = sub.add_parser("run"); common(p)
    p.add_argument("--video", required=True)
    p.add_argument("--period", type=int)
    p.add_argument("--batch", action="store_true", help="ne pas s'arrêter pour les identités (questions mises en file)")
    p.add_argument("--auto-accept-teams", action="store_true", help="accepter la proposition de couleurs sans confirmation (moins fiable)")
    p.add_argument("--sans-questions", action="store_true",
                   help="aucune question d'identité pour l'utilisateur : les identités viennent des numéros lus ; "
                        "les autres questions sont relues par Claude puis fermées avec close-pending")
    p.add_argument("--auto-calibration-experimental", action="store_true",
                   help="calibration automatique des plans larges (expérimentale : ~1 sur 5 fausse, voir docs/EVALUATION.md § 13)")
    p.add_argument("--det-fps", type=float)
    p.add_argument("--camera-fps", type=float, help="cadence du mouvement de caméra (15 par défaut)")
    p.add_argument("--restart-analysis", action="store_true", help="archiver les résultats vidéo de cette session puis recalculer le suivi ; conserve les réponses")
    p.add_argument("--tiles", type=int)
    p.add_argument("--model")
    p.add_argument("--templates")
    p.add_argument("--until")
    p.add_argument("--redo-from", help="recalculer à partir de cette étape (l'analyse image par image est conservée)")
    p = sub.add_parser("serve"); common(p)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p = sub.add_parser("import-v2"); common(p)
    p.add_argument("--v2", required=True, type=Path)
    p = sub.add_parser("answer"); common(p)
    p.add_argument("--task", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--player")
    p.add_argument("--note", default="")
    p.add_argument("--status", default="resolved")
    p = sub.add_parser("status"); common(p)
    p = sub.add_parser("aggregate"); common(p)
    p = sub.add_parser("review-page"); common(p)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-mb", type=float, default=12.0)
    p.add_argument("--kinds", help="types séparés par des virgules (ex. team_colors,identity)")
    p = sub.add_parser("possession"); common(p)
    p.add_argument("--out", default=None, help="CSV minute par minute (défaut : output/possession_par_minute.csv)")
    p = sub.add_parser("apply-answers"); common(p)
    p.add_argument("--file", required=True, type=Path, help="JSON : liste de réponses de la page (ou {documents:[...]})")
    p = sub.add_parser("align"); common(p)
    p.add_argument("--src", required=True, help="vidéo de référence (ex. 854×480)")
    p.add_argument("--dst", required=True, help="autre enregistrement de la même diffusion (ex. 1080p)")
    p.add_argument("--guess", type=float, default=0.0, help="décalage approximatif dst − src (s)")
    p.add_argument("--search", type=float, default=3.0)
    p.add_argument("--every", type=float, default=150.0, help="un point de contrôle toutes les N secondes")
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("transfer-answers"); common(p)
    p.add_argument("--from-session", required=True)
    p.add_argument("--to-session", required=True)
    p.add_argument("--align", required=True, type=Path, help="fichier produit par la commande align")
    p = sub.add_parser("close-pending"); common(p)
    p.add_argument("--keep", default="", help="identifiants de questions à laisser ouvertes (séparés par des virgules)")
    p.add_argument("--note", default="mode sans questions : laissée sans réponse, valeur inconnue")
    p = sub.add_parser("supersede"); common(p)
    p.add_argument("--session", required=True, help="session remplacée (n'est plus comptée)")
    p.add_argument("--by", required=True, help="session qui la remplace")
    p = sub.add_parser("site-bundle"); common(p)
    p.add_argument("--out", required=True, type=Path, help="fichier unique du match pour le site (extension Raja Insight 2)")
    p.add_argument("--date", help="date du match AAAA-MM-JJ si elle n'est pas dans la configuration")
    p.add_argument("--date-source", help="source de la date (ex. « FotMob et YSScores »)")
    p = sub.add_parser("report"); common(p)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("sportmonks"); common(p)
    p.add_argument("--fixture-id", type=int, help="appel direct (variable SPORTMONKS_TOKEN)")
    p.add_argument("--from-portal", type=int, help="ID WordPress du match déjà synchronisé par le portail")
    p.add_argument("--file", type=Path, help="fichier JSON de match Sportmonks déjà téléchargé")
    a = ap.parse_args(argv)

    from .pipeline import MatchWorkspace
    ws = MatchWorkspace(a.work, a.match)
    if a.cmd == "run":
        opts = {}
        if a.batch:
            opts["interactive_pause"] = False
        if a.auto_accept_teams:
            opts["auto_accept_team_proposals"] = True
        if a.auto_calibration_experimental:
            opts["auto_calibration"] = True
        if a.sans_questions:
            opts["identity_budget"] = 0
            opts["interactive_pause"] = False
        for k, v in (("det_fps", a.det_fps), ("camera_fps", a.camera_fps), ("tiles", a.tiles), ("model", a.model), ("templates_file", a.templates)):
            if v is not None:
                opts[k] = v
        s = ws.session_for(a.video, a.period, opts)
        if a.restart_analysis:
            s.restart_analysis()
        if a.redo_from:
            s.invalidate_from(a.redo_from)

        def prog(frac, info):
            sys.stderr.write(f"\r{info.get('stage')}: {frac:6.1%}  vitesse ×{info.get('speed_x', 0):.2f}   ")
            sys.stderr.flush()
        meta = s.run(prog, until=a.until)
        sys.stderr.write("\n")
        print(json.dumps({"status": meta["status"], "waiting": meta.get("waiting"),
                          "stages": {k: v.get("status") for k, v in meta.get("stages", {}).items()}}, ensure_ascii=False, indent=1))
        if meta["status"] == "done":
            st = ws.aggregate()
            print("Statistiques écrites dans", ws.out)
    elif a.cmd == "serve":
        from .server import create_app
        app = create_app(ws)
        print(f"Interface : http://{a.host}:{a.port}/  (Ctrl+C pour arrêter)")
        app.run(host=a.host, port=a.port, threaded=True, debug=False)
    elif a.cmd == "import-v2":
        from .v2import import run as v2run
        print(json.dumps(v2run(a.v2, ws.root, ws.store), ensure_ascii=False, indent=1))
    elif a.cmd == "answer":
        extra = {"player": a.player} if a.player else None
        t = ws.store.answer(a.task, a.value, label=a.value, note=a.note, status=a.status, extra=extra)
        print(json.dumps({"id": t["id"], "status": t["status"]}, ensure_ascii=False))
    elif a.cmd == "status":
        out = {"questions": ws.store.counts(), "sessions": []}
        for s in ws.sessions():
            out["sessions"].append({"dir": s.dir.name, "status": s.meta.get("status"), "waiting": s.meta.get("waiting"),
                                    "stages": {k: v.get("status") for k, v in s.meta.get("stages", {}).items()}})
        print(json.dumps(out, ensure_ascii=False, indent=1))
    elif a.cmd == "review-page":
        from .review_page import collect, render_html
        qs = collect(ws, max_bytes=int(a.max_mb * 1e6), kinds=set(a.kinds.split(",")) if a.kinds else None)
        a.out.write_text(render_html(ws, qs, ws.roster.choices(), "Questions Raja–Zemamra"), encoding="utf-8")
        from collections import Counter
        print(json.dumps({"questions": len(qs), "by_kind": Counter(q["kind"] for q in qs),
                          "size_mb": round(a.out.stat().st_size / 1e6, 2)}, ensure_ascii=False))
    elif a.cmd == "possession":
        import csv
        from .momentum import match_tables
        from .util import write_json
        res = match_tables(ws)
        out = a.out or str(ws.out / "possession_par_minute.csv")
        ws.out.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res["minutes"][0].keys()))
            w.writeheader()
            w.writerows(res["minutes"])
        write_json(ws.out / "possession_fenetres.json", {"fenetres": res["fenetres"], "resume": res["resume"]})
        print(json.dumps({"csv": out, "resume": {k: v for k, v in res["resume"].items() if k != "methode"}}, ensure_ascii=False))
    elif a.cmd == "apply-answers":
        from .review_page import apply_answers
        d = json.load(open(a.file, encoding="utf-8"))
        rows = d if isinstance(d, list) else [x.get("data", x) for x in d.get("documents", d.get("docs", []))]
        res = apply_answers(ws, rows)
        from .events import foul_followups
        res["foul_precision_questions_created"] = foul_followups(ws.store, ws.roster)
        print(json.dumps(res, ensure_ascii=False))
    elif a.cmd == "align":
        from .align import align
        from .videoio import VideoSource
        dur = VideoSource(a.src).duration
        times = [float(x) for x in np.arange(30.0, max(31.0, dur - 30.0), a.every)]
        res = align(a.src, a.dst, times, a.guess, a.search)
        res.update(src=a.src, dst=a.dst)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "points"}, ensure_ascii=False))
    elif a.cmd == "transfer-answers":
        from .align import mapper
        from .transfer import transfer
        al = json.load(open(a.align, encoding="utf-8"))
        tmap = mapper(al["points"], al.get("jumps"))
        if tmap is None:
            raise SystemExit("recalage inutilisable : aucun point cohérent")
        src = ws.session_by_name(a.from_session); dst = ws.session_by_name(a.to_session)
        print(json.dumps(transfer(ws, src, dst, tmap), ensure_ascii=False))
    elif a.cmd == "close-pending":
        # ferme les questions encore en attente (statut « passée ») : rien n'attend plus l'utilisateur.
        # Réversible : une réponse donnée plus tard sur la page est appliquée normalement.
        keep = {x for x in a.keep.split(",") if x}
        # les questions bloquantes (couleurs des équipes, chrono) ne sont jamais fermées ainsi : Claude y répond
        blocking = [t["id"] for t in ws.store.pending() if t.get("blocking")]
        closed = [t["id"] for t in ws.store.pending() if t["id"] not in keep and not t.get("blocking")]
        for tid in closed:
            ws.store.answer(tid, None, note=a.note, status="skipped", origin="mode_sans_questions")
        print(json.dumps({"closed": len(closed), "kept": sorted(keep), "blocking_left": blocking}, ensure_ascii=False))
    elif a.cmd == "supersede":
        old = ws.session_by_name(a.session); new = ws.session_by_name(a.by)
        old.meta["superseded_by"] = new.dir.name
        old.save()
        print(json.dumps({"superseded": old.dir.name, "by": new.dir.name}, ensure_ascii=False))
    elif a.cmd == "site-bundle":
        from .site_export import build_bundle, validate
        ws.aggregate()
        b = build_bundle(ws, a.date, a.date_source)
        err = validate(b)
        if err:
            raise SystemExit("fichier du site invalide : " + "; ".join(err))
        a.out.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"out": str(a.out), "players": len(b["players"]), "facts": len(b["facts"]),
                          "metrics": sum(len(p["metrics"]) for p in b["players"])}, ensure_ascii=False))
    elif a.cmd == "report":
        from .report import build
        ws.aggregate()
        print(json.dumps(build(ws, a.out), ensure_ascii=False))
    elif a.cmd == "sportmonks":
        from . import sportmonks as sm
        from .util import now_iso, write_json
        if a.file:
            d = json.load(open(a.file, encoding="utf-8"))
            fx, src = d.get("data", d), {"mode": "fichier", "path": str(a.file)}
        elif a.from_portal:
            fx, src = sm.fetch_from_portal(a.from_portal), {"mode": "portail", "wp_match_id": a.from_portal}
        elif a.fixture_id:
            fx, src = sm.fetch_direct(a.fixture_id), {"mode": "direct", "fixture_id": a.fixture_id}
        else:
            raise SystemExit("Indiquer --fixture-id, --from-portal ou --file.")
        if not fx:
            raise SystemExit("Aucune donnée Sportmonks pour ce match (synchroniser d'abord dans le portail).")
        write_json(ws.root / "sportmonks_fixture.json", fx)
        write_json(ws.root / "sportmonks_meta.json", {**src, "fetched_at": now_iso()})
        ws.aggregate()
        print("Données Sportmonks enregistrées et confrontées ; voir output/stats_joueurs.json (clé sportmonks).")
    elif a.cmd == "aggregate":
        ws.aggregate()
        print("Statistiques écrites dans", ws.out)


if __name__ == "__main__":
    main()
