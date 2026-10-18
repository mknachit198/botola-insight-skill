"""Interface web locale du moteur : chargement de la vidéo, suivi du traitement, questions
humaines (parcours avec pause/reprise), statistiques par joueur, chronologie et exports.

Lancement : python -m raja_engine serve --match matches/raja-zemamra/match.json --work work
Par défaut l'interface n'écoute que sur 127.0.0.1 (machine locale).
"""
from __future__ import annotations

import subprocess
import threading
import traceback
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from . import portal
from .pipeline import STAGES, MatchWorkspace
from .util import read_json, write_json

HERE = Path(__file__).resolve().parent


def create_app(ws: MatchWorkspace) -> Flask:
    app = Flask(__name__, static_folder=str(HERE / "web"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3
    runner = {"thread": None, "session": None, "error": None, "log": []}
    lock = threading.Lock()

    def sessions():
        return {s.dir.name: s for s in ws.sessions()}

    def start(sess, before=None):
        with lock:
            if runner["thread"] is not None and runner["thread"].is_alive():
                return False
            runner["session"] = sess
            runner["error"] = None

            def job():
                try:
                    if before:
                        before()
                    meta = sess.run()
                    if meta.get("status") == "done":
                        ws.aggregate()
                except Exception as e:  # erreur visible dans l'interface
                    runner["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
                    sess.meta["status"] = "error"
                    sess.meta["error"] = str(e)
                    sess.save()

            runner["thread"] = threading.Thread(target=job, daemon=True)
            runner["thread"].start()
            return True

    @app.get("/")
    def index():
        return send_from_directory(HERE / "web", "index.html")

    @app.get("/api/state")
    def state():
        out = []
        for name, s in sessions().items():
            live = runner["session"] is not None and runner["session"].dir == s.dir and runner["thread"] is not None and runner["thread"].is_alive()
            prog = runner["session"].progress if live else None
            out.append({"id": name, "period": s.meta.get("period"), "video": s.meta.get("identity", {}).get("name"),
                        "known_video": bool(s.meta.get("known_video")), "status": s.meta.get("status"),
                        "running": live, "progress": prog, "waiting": s.meta.get("waiting"), "error": s.meta.get("error"),
                        "stages": [{"name": st, **s.meta.get("stages", {}).get(st, {"status": "pending"})} for st in STAGES],
                        "options": s.meta.get("options")})
        pend = ws.store.pending()
        return jsonify({"match": {"key": ws.cfg["match_key"], "title": ws.cfg.get("title")}, "sessions": out,
                        "questions": ws.store.counts(), "pending": len(pend),
                        "pending_blocking": len([t for t in pend if t.get("blocking")]),
                        "runner_error": runner["error"], "portal_configured": portal.configured(),
                        "has_stats": (ws.out / "stats_joueurs.json").exists()})

    @app.post("/api/sessions")
    def create_session():
        period = request.form.get("period") or (request.json or {}).get("period") if request.is_json else request.form.get("period")
        period = int(period) if period else None
        if "file" in request.files:
            f = request.files["file"]
            up = ws.root / "uploads"
            up.mkdir(exist_ok=True)
            dest = up / Path(f.filename).name
            f.save(dest)
            path = str(dest)
        else:
            path = (request.json or {}).get("video_path") if request.is_json else request.form.get("video_path")
        if not path or not Path(path).exists():
            return jsonify({"error": "Fichier vidéo introuvable."}), 400
        opts = {}
        body = request.json if request.is_json else request.form
        for k in ("det_fps", "tiles"):
            if body.get(k):
                opts[k] = float(body[k]) if k == "det_fps" else int(body[k])
        if body.get("model"):
            opts["model"] = body["model"]
        s = ws.session_for(path, period, opts)
        started = start(s)
        return jsonify({"id": s.dir.name, "started": started})

    @app.post("/api/sessions/<sid>/<action>")
    def session_action(sid, action):
        s = sessions().get(sid)
        if s is None:
            abort(404)
        if action == "run":
            return jsonify({"started": start(s)})
        if action == "stop":
            if runner["session"] is not None and runner["session"].dir == s.dir:
                runner["session"].stop()
            return jsonify({"ok": True})
        if action == "continue":  # reprendre sans répondre aux identités
            s.meta["skip_identity_pause"] = True
            s.save()
            return jsonify({"started": start(s)})
        if action == "recompute":  # nouvelles réponses : recalcul identités → statistiques
            def before():
                s.meta["skip_identity_pause"] = True
                s.invalidate_from("identity")
            return jsonify({"started": start(s, before)})
        if action == "recalibrate":
            def before():
                s.invalidate_from("calibration")
            return jsonify({"started": start(s, before)})
        abort(400)

    @app.get("/api/questions")
    def questions():
        status = request.args.get("status", "pending")
        kind = request.args.get("kind")
        items = [t for t in ws.store.tasks.values() if (status == "all" or t["status"] == status) and (not kind or t["kind"] == kind)]
        items.sort(key=lambda x: (not x.get("blocking"), x.get("period") or 0, x.get("time") or 0))
        return jsonify({"items": items, "roster": ws.roster.choices()})

    @app.post("/api/questions/<tid>/answer")
    def answer(tid):
        if tid not in ws.store.tasks:
            abort(404)
        b = request.json or {}
        extra = {k: b[k] for k in ("player", "outcome", "points") if b.get(k) not in (None, "", [])}
        t = ws.store.answer(tid, b.get("value"), label=b.get("label", ""), note=b.get("note", ""),
                            status=b.get("status", "resolved"), extra=extra)
        # une réponse bloquante débloque la session en attente
        for s in sessions().values():
            w = s.meta.get("waiting") or {}
            if tid in (w.get("task_ids") or []) and w.get("stage") in ("teams", "clock_templates"):
                if not [x for x in w["task_ids"] if ws.store.tasks.get(x, {}).get("status") == "pending"]:
                    start(s)
        return jsonify(t)

    def _video_for(task):
        for s in sessions().values():
            if s.meta.get("identity", {}).get("fingerprint") == task.get("video"):
                return s
        return None

    @app.get("/clip/<tid>")
    def clip(tid):
        t = ws.store.tasks.get(tid)
        if t is None:
            abort(404)
        ev = t.get("evidence") or {}
        if ev.get("v2_clip"):
            p = (ws.root / ev["v2_clip"]).resolve()
            if p.exists() and ws.root.resolve() in p.parents:
                return send_file(p, mimetype="video/mp4")
        s = _video_for(t)
        if s is None:
            return jsonify({"error": "Vidéo source non chargée dans cette interface."}), 404
        start_t, dur = ev.get("clip_window") or [max(0.0, (t.get("time") or 0) - 2.0), 5.0]
        out = ws.root / "evidence" / "clips" / f"{tid}.mp4"
        if not out.exists():
            from .videoio import extract_clip
            extract_clip(s.meta["video_path"], start_t, dur, str(out))
        return send_file(out, mimetype="video/mp4")

    @app.get("/media/<path:rel>")
    def media(rel):
        p = (ws.root / rel).resolve()
        if ws.root.resolve() not in p.parents or not p.exists():
            abort(404)
        return send_file(p)

    @app.get("/api/stats")
    def stats():
        if request.args.get("refresh") or not (ws.out / "stats_joueurs.json").exists():
            if any(s.done("session_stats") for s in ws.sessions()):
                ws.aggregate()
        return jsonify(read_json(ws.out / "stats_joueurs.json", {"players": []}))

    @app.get("/api/timeline/<sid>")
    def timeline(sid):
        s = sessions().get(sid)
        if s is None:
            abort(404)
        return jsonify({"timeline": read_json(s.dir / "timeline.json"), "events": read_json(s.dir / "events.json"),
                        "duration": read_json(s.dir / "video_info.json", {}).get("duration_seconds")})

    @app.get("/download/<name>")
    def download(name):
        p = (ws.out / name).resolve()
        if ws.out.resolve() not in p.parents or not p.exists():
            abort(404)
        return send_file(p, as_attachment=True)

    @app.get("/api/exports")
    def exports():
        files = sorted(p.name for p in ws.out.glob("*") if p.is_file()) if ws.out.exists() else []
        return jsonify({"files": files, "portal_configured": portal.configured()})

    @app.post("/api/portal/<kind>")
    def push(kind):
        b = request.json or {}
        name = {"analysis": "import_portail_analysis.json"}.get(kind) or b.get("file")
        if not name:
            abort(400)
        payload = read_json(ws.out / name)
        try:
            return jsonify(portal.post(int(b["match_id"]), kind, payload))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app
