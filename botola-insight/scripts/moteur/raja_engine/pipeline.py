"""Orchestration : une session par fichier vidéo (une période), un dossier de travail par match.

Chaque étape écrit son résultat et son statut dans session.json ; une relance reprend à la
première étape non terminée (l'analyse image par image reprend à la dernière image écrite).
Une étape qui a besoin d'une réponse humaine bloquante met la session en attente
(`waiting_human`) ; la réponse est enregistrée dans la file de questions du match et la
session repart de cette étape.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import analyze as analyze_mod
from . import ball as ball_mod
from . import events as events_mod
from . import identity as identity_mod
from . import pitch as pitch_mod
from . import stats as stats_mod
from . import tracks as tracks_mod
from .clockscan import bootstrap_templates, scan
from .exports import flatten_events, portal_corrections_payload, portal_video_session_payload, write_all
from .fingerprint import sampled_fingerprint
from .questions import QuestionStore, task_id
from .roster import Roster
from .scoreboard import ScoreboardReader
from .teams import LABEL_FR, LABELS, TeamModel
from .timeline import Timeline
from .util import fmt_clock, now_iso, read_json, write_json
from .videoio import VideoSource, save_frame

ENGINE_VERSION = "0.9.8"
ACTION_FAMILY = {"but": "tir", "tir_cadre": "tir", "tir_non_cadre": "tir", "tir_contre": "tir",
                 "arret_gardien": "gardien", "intervention_gardien": "gardien", "occasion_centre": "centre",
                 "duel": "duel", "faute": "faute", "faute_commise": "faute", "faute_subie": "faute"}
V2_FAMILY = {"tir": "tir", "arret_sur_tir": "gardien", "prise_de_balle": "gardien", "intervention_aerienne": "gardien",
             "intervention_aerienne_au_poing": "gardien", "plongeon_sur_tir": "gardien", "centre_contre": "centre",
             "duel_au_sol_participe": "duel", "duel_offensif": "duel", "tacle": "duel"}
STAGES = ["identify", "import_prior", "clock_templates", "scan", "timeline", "analyze", "teams", "identity",
          "events", "calibration", "session_stats"]
DEFAULT_OPTIONS = {
    "model": "yolox_tiny.onnx", "det_fps": 5.0, "camera_fps": 15.0, "tiles": 2, "scan_step": 0.5, "identity_budget": 40,
    "replay_review_budget": 60, "calibration_requests": 4, "auto_accept_team_proposals": False,
    "interactive_pause": True, "templates_file": None,
    # calibration automatique : EXPÉRIMENTALE, désactivée par défaut (docs/EVALUATION.md § 13 :
    # environ 1 calibration acceptée sur 5 est fausse, et le critère d'acceptation ne la détecte pas)
    "auto_calibration": False,
}


class NeedHuman(Exception):
    def __init__(self, stage, task_ids, message):
        super().__init__(message)
        self.stage, self.task_ids, self.message = stage, task_ids, message


class MatchWorkspace:
    def __init__(self, root: Path, match_cfg_path: Path):
        self.cfg = read_json(match_cfg_path)
        self.cfg_dir = Path(match_cfg_path).parent
        self.root = Path(root) / self.cfg["match_key"]
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / "match.json").exists():
            shutil.copy(match_cfg_path, self.root / "match.json")
        self.store = QuestionStore(self.root / "questions")
        from .events import refresh_event_choices
        refresh_event_choices(self.store)
        self.roster = Roster(self.cfg)
        self.out = self.root / "output"

    def sessions(self, include_superseded: bool = False) -> list["Session"]:
        """Sessions du match. Une session remplacée (ex. 854×480 remplacée par la même diffusion en
        1080p, réponses reportées) n'est plus comptée, sauf demande explicite."""
        out = []
        for d in sorted((self.root / "sessions").glob("*")):
            if (d / "session.json").exists():
                s = Session(self, d)
                if s.meta.get("superseded_by") and not include_superseded:
                    continue
                out.append(s)
        return out

    def session_by_name(self, name: str) -> "Session":
        d = self.root / "sessions" / name
        if not (d / "session.json").exists():
            raise SystemExit(f"session inconnue : {name}")
        return Session(self, d)

    def session_for(self, video_path: str, period: int | None = None, options: dict | None = None) -> "Session":
        ident = sampled_fingerprint(video_path)
        known = next((v for v in self.cfg.get("videos", []) if v["fingerprint"] == ident["fingerprint"] and v["size"] == ident["size"]), None)
        if period is None:
            period = known["period"] if known else 1
        d = self.root / "sessions" / f"p{period}_{ident['fingerprint'][:10]}"
        d.mkdir(parents=True, exist_ok=True)
        s = Session(self, d)
        if not s.meta.get("video_path"):
            s.meta.update(video_path=str(Path(video_path).resolve()), period=period, identity=ident,
                          known_video=known, options={**DEFAULT_OPTIONS, **(options or {})}, created_at=now_iso(),
                          stages={}, status="created", engine_version=ENGINE_VERSION)
            s.save()
        else:
            s.meta["video_path"] = str(Path(video_path).resolve())
            if options:
                s.meta["options"].update(options)
            s.save()
        return s

    # ------------------------------------------------------------------ niveau match
    def aggregate(self) -> dict:
        sessions = [s for s in self.sessions() if s.meta.get("stages", {}).get("session_stats", {}).get("status") == "done"]
        v2 = read_json(self.root / "v2_events.json", [])
        human_events = self.human_events()
        banners, goals, sdata, xg_items = [], [], [], []
        for s in sessions:
            ev = read_json(s.dir / "events.json", {})
            sb = ev.get("banners", [])
            for b in sb:
                b["period"] = s.meta["period"]
            events_mod.apply_banner_answers(sb, self.store, s.fp, self.roster)
            banners.extend(sb)
            for g in ev.get("goals", []):
                g["period"] = s.meta["period"]
                goals.append(g)
            sdata.append(read_json(s.dir / "session_stats.json"))
        events_mod.check_raja_consistency(banners, self.roster)
        xg_items = read_json(self.root / "xg_items.json", [])
        stats = stats_mod.aggregate(self.cfg, self.roster, sdata, v2, human_events, banners, goals, xg_items)
        stats["engine_version"] = ENGINE_VERSION
        stats["generated_at"] = now_iso()
        stats["questions"] = self.store.counts()
        stats["contradictions"] = self.contradictions(banners, goals, v2) + stats.get("action_conflicts", [])
        fx = read_json(self.root / "sportmonks_fixture.json")
        team_id = (self.cfg.get("sportmonks") or {}).get("raja_team_id")
        if fx and team_id:
            from . import sportmonks as sm
            norm = sm.normalize(fx, int(team_id))
            stats["sportmonks"] = {"fixture_id": norm["fixture_id"], "starting_at": norm["starting_at"],
                                   "cross_check": sm.cross_check(norm, self.roster, banners, goals, v2),
                                   "source": read_json(self.root / "sportmonks_meta.json", {})}
            by_num = {p["jersey_number"]: p for p in norm["raja_lineup"]}
            for pl in stats["players"]:
                a = by_num.get(pl["jersey_number"])
                pl["sportmonks"] = None if a is None else {"player_id": a["player_id"], "player_name": a["player_name"],
                                                           "starter": a["starter"], "details": a["details"],
                                                           "provenance": "sportmonks"}
        else:
            stats["sportmonks"] = {"status": "non connecté", "note": "aucune donnée API importée : champs API inconnus"}
        events = flatten_events(stats, v2, human_events, banners, goals)
        write_all(self.out, stats, events)
        for s in sessions:
            ident = s.meta["identity"]
            dur = read_json(s.dir / "video_info.json", {}).get("duration_seconds") or 0
            write_json(self.out / f"import_portail_video_session_p{s.meta['period']}.json",
                       portal_video_session_payload(self.cfg["match_key"], s.meta["period"], ident, dur, self.store))
            write_json(self.out / f"import_portail_corrections_p{s.meta['period']}.json",
                       portal_corrections_payload(ident, self.store))
        return stats

    def human_events(self) -> list[dict]:
        v2map = {"v2req-chraibi09": "chraibi_volley", "v2req-bettaieb18": "bettaieb_acrobatie",
                 "v2req-keeper84": "keeper84_pending", "v2req-keeper91": "keeper91_pending",
                 "v2req-centre21": "centre21", "v2req-barkok88": "barkok_foul_won"}
        out = []
        # précisions demandées après coup (ex. « faute » → commise ou subie) : elles remplacent
        # le type de la réponse d'origine, sans effacer celle-ci du journal
        precis = {t["refines"]: t["response"] for t in self.store.resolved("event_review") if t.get("refines")}
        for t in self.store.resolved("event_review"):
            if t.get("refines"):
                continue
            r = dict(t["response"] or {})
            if t["id"] in precis:
                pr = precis[t["id"]]
                r["value"] = pr.get("value")
                if pr.get("player"):
                    r["player"] = pr["player"]
                r["outcome"] = (r.get("outcome") or "") + " [précisé]"
            if r.get("value") in (None, "rien"):
                if t["id"] in v2map:
                    out.append({"type": "rien", "dedup_v2": v2map[t["id"]], "task_id": t["id"]})
                continue
            p = self.roster.from_answer_value(r.get("player"))
            player_source = "human"
            if p is None and t["id"] in v2map:
                # l'utilisateur a qualifié l'action sans choisir de joueur : on garde le joueur V2
                # seulement si c'est bien l'auteur de ce type d'action (tireur pour un tir…)
                v2e = next((e for e in read_json(self.root / "v2_events.json", []) if e["id"] == v2map[t["id"]]), None)
                fam_h = ACTION_FAMILY.get(r["value"])
                if v2e and v2e.get("jersey_number") and fam_h and fam_h == V2_FAMILY.get(v2e["type"]):
                    p = self.roster.by_number.get(v2e["jersey_number"])
                    player_source = "v2_assistant_review"
            out.append({"type": r["value"], "jersey_number": None if p is None else p["jersey_number"],
                        "player_name": None if p is None else p["player_name"], "match_clock": t.get("match_clock"),
                        "clock_status": t.get("clock_status"), "period": t.get("period"), "task_id": t["id"],
                        "outcome": r.get("outcome"), "note": r.get("note"), "dedup_v2": v2map.get(t["id"]),
                        "player_source": player_source,
                        "answer_origin": "claude" if str((t["response"] or {}).get("origin", "")).startswith("claude") else "human"})
        return out

    def contradictions(self, banners, goals, v2) -> list[dict]:
        out = []
        decl = self.cfg.get("score_declared") or {}
        if goals:
            last = max(goals, key=lambda g: (g.get("period") or 0, g["file_t_score_update"]))
            sa = last["score_after"]
            if decl and [decl.get("raja"), decl.get("opponent")] != sa:
                out.append({"type": "score", "declared": decl, "scoreboard": sa})
        for b in banners:
            if b["type"] in ("substitution_out", "substitution_in") and not b.get("needs_review"):
                for p in b["players"]:
                    if p.get("team") != "raja":
                        continue
                    pl = self.roster.by_number.get(p.get("jersey_number"))
                    if not pl:
                        continue
                    declared = pl.get("left_minute_declared") if b["type"] == "substitution_out" else pl.get("entered_minute_declared")
                    clock = b.get("match_clock_banner_start")
                    if declared and clock:
                        mm = int(clock.split(":")[0])
                        if abs(mm + 1 - declared) > 2:
                            out.append({"type": "substitution_minute", "player": pl["player_name"], "declared_minute": declared,
                                        "banner_clock": clock, "note": "bandeau affiché après le changement ; écart > 2 min"})
        return out


class Session:
    def __init__(self, ws: MatchWorkspace, folder: Path):
        self.ws = ws
        self.dir = Path(folder)
        self.meta = read_json(self.dir / "session.json", {})
        self._stop = threading.Event()
        self.progress = {"stage": None, "fraction": 0.0}

    def save(self):
        write_json(self.dir / "session.json", self.meta)

    @property
    def fp(self):
        return self.meta["identity"]["fingerprint"]

    def stop(self):
        self._stop.set()

    def _mark(self, stage, status, **info):
        self.meta.setdefault("stages", {})[stage] = {"status": status, "at": now_iso(), **info}
        self.save()

    def done(self, stage):
        return self.meta.get("stages", {}).get(stage, {}).get("status") == "done"

    def invalidate_from(self, stage):
        """Force le recalcul d'une étape et des suivantes (ex. après de nouvelles réponses)."""
        idx = STAGES.index(stage)
        for s in STAGES[idx:]:
            if s == "analyze":
                continue  # l'analyse image par image n'est jamais refaite implicitement
            self.meta.get("stages", {}).pop(s, None)
        self.save()

    def restart_analysis(self):
        """Recalcul explicite : archive les résultats dérivés et conserve les réponses."""
        backup = self.dir / "analysis_backups" / str(time.time_ns())
        backup.mkdir(parents=True)
        for name in ("frames.jsonl", "analyze_state.json", "frames_tracked.jsonl", "tracking_refine.json",
                     "identity.json", "ball.json", "events.json", "calibration.json", "session_stats.json"):
            src = self.dir / name
            if src.exists():
                shutil.move(str(src), str(backup / name))
        write_json(backup / "stages.json", self.meta.get("stages", {}))
        for stage in STAGES[STAGES.index("analyze"):]:
            self.meta.get("stages", {}).pop(stage, None)
        self.meta["engine_version"] = ENGINE_VERSION
        self.save()

    # ------------------------------------------------------------------ exécution
    def run(self, progress_cb=None, until: str | None = None) -> dict:
        self._stop.clear()
        video = VideoSource(self.meta["video_path"])
        if self.meta["identity"]["size"] != Path(self.meta["video_path"]).stat().st_size:
            raise RuntimeError("Le fichier a changé depuis la création de la session.")
        self.meta["status"] = "running"
        self.save()

        def prog(frac, info):
            self.progress = {"fraction": frac, **info}
            if progress_cb:
                progress_cb(frac, info)

        try:
            for stage in STAGES:
                if self._stop.is_set():
                    self.meta["status"] = "stopped"
                    break
                if self.done(stage):
                    continue
                self.progress = {"stage": stage, "fraction": 0.0}
                getattr(self, f"_stage_{stage}")(video, prog)
                if until and stage == until:
                    break
            else:
                self.meta["status"] = "done"
        except NeedHuman as e:
            self.meta["status"] = "waiting_human"
            self.meta["waiting"] = {"stage": e.stage, "task_ids": e.task_ids, "message": e.message, "at": now_iso()}
            self.save()
            return self.meta
        finally:
            video.close()
        self.meta.pop("waiting", None)
        self.save()
        return self.meta

    # ------------------------------------------------------------------ étapes
    def _stage_identify(self, video, prog):
        info = video.info()
        write_json(self.dir / "video_info.json", info)
        known = self.meta.get("known_video")
        self._mark("identify", "done", known_video=bool(known), duration_s=info["duration_seconds"],
                   resolution=[info["width"], info["height"]], fps=info["fps"])

    def _stage_import_prior(self, video, prog):
        known = self.meta.get("known_video")
        res = {"imported": 0}
        if known and known.get("corrections_file"):
            path = self.ws.cfg_dir / known["corrections_file"]
            if path.exists():
                corr = read_json(path)
                if corr["video"]["fingerprint"] != self.fp:
                    raise RuntimeError("Les corrections ne correspondent pas à ce fichier (empreinte différente).")
                res = self.ws.store.import_corrections(corr, self.ws.roster, self.meta["period"], path.name)
        self._mark("import_prior", "done", **res)

    def _reader(self):
        r = ScoreboardReader(self.ws.cfg.get("scoreboard", {}).get("layout", "arryadia_2026"))
        tpl = self.dir / "clock_templates.npz"
        if tpl.exists():
            r.load_templates(str(tpl))
        lay = read_json(self.dir / "scoreboard_layout.json")
        if lay:
            r.L.update(lay)
        return r

    def _stage_clock_templates(self, video, prog):
        r = ScoreboardReader(self.ws.cfg.get("scoreboard", {}).get("layout", "arryadia_2026"))
        shared = self.meta["options"].get("templates_file")
        if shared and Path(shared).exists():
            r.load_templates(shared)
        human = []
        for t in self.ws.store.resolved("clock_read"):
            if t.get("video") == self.fp:
                from .util import parse_clock
                secs, _ = parse_clock(str(t["response"].get("value")))
                if secs is not None:
                    human.append((t["time"], secs))
        info = bootstrap_templates(video, r, n=160, human_reads=human)
        if info["status"] == "needs_human":
            ids = []
            for k, tt in enumerate(np.linspace(0.2, 0.8, 3) * video.duration):
                f = video.frame_at(float(tt))
                if f is None:
                    continue
                tid = task_id(self.fp, "clock_read", float(tt))
                img = self.dir / "evidence" / f"{tid}.jpg"
                save_frame(f, str(img))
                task = {"id": tid, "kind": "clock_read", "video": self.fp, "period": self.meta["period"], "time": round(float(tt), 2),
                        "bbox": None, "match_clock": None, "clock_status": "unverified", "blocking": True, "priority": 100,
                        "reason": info.get("reason"), "question": "Quel chrono lisez-vous en haut à gauche (MM:SS) ? "
                        "Répondez « absent » s'il n'est pas affiché.", "choices": [], "allow_free_text": True,
                        "evidence": {"image": str(img.relative_to(self.ws.root))}, "origin": "engine"}
                ids.append(self.ws.store.add(task)[0]["id"])
            self.ws.store.save()
            raise NeedHuman("clock_templates", ids, "Le chrono n'a pas pu être appris automatiquement.")
        r.save_templates(str(self.dir / "clock_templates.npz"))
        write_json(self.dir / "scoreboard_layout.json", r.L)
        self._mark("clock_templates", "done", bootstrap={k: v for k, v in info.items() if k != "samples_per_digit"})

    def _stage_scan(self, video, prog):
        r = self._reader()
        res = scan(video, r, step=self.meta["options"]["scan_step"], progress=prog)
        write_json(self.dir / "scan.json", res)
        self._mark("scan", "done", samples=len(res["samples"]), elapsed_s=res["elapsed_s"],
                   speed_x=round(video.duration / max(res["elapsed_s"], 1e-6), 1))

    def _stage_timeline(self, video, prog):
        sc = read_json(self.dir / "scan.json")
        tl = Timeline.build(sc["samples"], video.duration, declared_period=self.meta["period"])
        write_json(self.dir / "timeline.json", tl.data)
        summ = tl.summary()
        periods = {s.get("period") for s in tl.segments if s["kind"] == "live"}
        warn = None
        if periods and self.meta["period"] not in periods:
            warn = f"Période déclarée {self.meta['period']} mais chronos lus : {sorted(p for p in periods if p)}"
        self._mark("timeline", "done", live_seconds=round(tl.live_seconds(), 1), warning=warn,
                   discontinuities=len(summ["discontinuities"]))

    def timeline(self):
        d = read_json(self.dir / "timeline.json")
        return None if d is None else Timeline(d)

    def _stage_analyze(self, video, prog):
        opts = self.meta["options"]
        model = Path(__file__).resolve().parents[1] / "models" / opts["model"]
        res = analyze_mod.run(video, self.timeline(), self.dir, str(model), det_fps=opts["det_fps"], tiles=opts["tiles"],
                              camera_fps=opts.get("camera_fps", 15.0), progress=prog, should_stop=self._stop.is_set)
        if res.get("status") == "stopped":
            self._mark("analyze", "partial", **{k: res[k] for k in ("done_until", "frames")})
            return
        self._mark("analyze", "done", frames=res["frames"], speed_x_vs_realtime=res.get("speed_x_vs_realtime"),
                   det_fps=opts["det_fps"], camera_fps=res["camera_fps"], tracking_version=res["tracking_version"],
                   tiles=opts["tiles"], model=opts["model"])

    def _frames(self):
        from .tracking_refine import VERSION, refine_file
        state = read_json(self.dir / "analyze_state.json", {})
        tm = read_json(self.ws.root / "teams.json")
        # Les anciennes sessions restent consultables sans transformation implicite.
        if state.get("tracking_version") == VERSION and tm and tm.get("centers"):
            path, _ = refine_file(self.dir, TeamModel.from_json(tm))
            return tracks_mod.load_frames(path)
        return tracks_mod.load_frames(self.dir / "frames.jsonl")

    def _stage_teams(self, video, prog):
        frames = tracks_mod.load_frames(self.dir / "frames.jsonl")
        tm_path = self.ws.root / "teams.json"
        model = TeamModel.from_json(read_json(tm_path)) if tm_path.exists() else None
        if model is None or model.centers is None:
            model, info = tracks_mod.fit_teams(frames)
            if info["status"] != "ok":
                self._mark("teams", "done", result="insufficient", fit=info)
                return
            self._team_mosaics(video, frames, model)
            write_json(tm_path, {**model.to_json(), "fit_info": info, "fitted_on_session": self.dir.name})
        # questions : une par groupe de couleur (bloquantes sauf acceptation automatique)
        tm = read_json(tm_path)
        ids, unresolved = [], []
        for c in range(len(model.centers)):
            tid = f"team_colors-{self.ws.cfg['match_key']}-cluster{c}"
            task = {"id": tid, "kind": "team_colors", "video": None, "period": None, "time": 0.0, "bbox": None,
                    "match_clock": None, "clock_status": None, "blocking": not self.meta["options"]["auto_accept_team_proposals"],
                    "priority": 90, "question": "À qui correspondent les maillots de ce groupe ?",
                    "reason": f"Proposition automatique : {LABEL_FR[model.label(c)]}.",
                    "choices": [{"value": k, "label": LABEL_FR[k]} for k in LABELS],
                    "evidence": {"image": f"evidence/team_cluster_{c}.jpg"}, "origin": "engine", "cluster": c,
                    "proposed": model.label(c)}
            t, _ = self.ws.store.add(task)
            ids.append(t["id"])
            if t["status"] == "resolved":
                model.cluster_labels[c] = t["response"]["value"]
                model.label_source[c] = "humain"
            elif self.meta["options"]["auto_accept_team_proposals"]:
                model.label_source[c] = "proposition_acceptee_automatiquement"
            else:
                unresolved.append(t["id"])
        self.ws.store.save()
        write_json(tm_path, {**tm, **model.to_json()})
        if unresolved:
            raise NeedHuman("teams", unresolved, "Confirmez les couleurs des équipes (une fois par match).")
        self._frames()  # construit le suivi corrigé une fois les couleurs confirmées
        self._mark("teams", "done", labels=model.cluster_labels, sources=model.label_source)

    def _team_mosaics(self, video, frames, model):
        rng = np.random.default_rng(0)
        picks = {c: [] for c in range(len(model.centers))}
        cand = [(fr, row) for fr in frames if fr["view"] in ("wide", "medium") for row in fr["p"] if row[6] is not None]
        for i in rng.permutation(len(cand))[:3000]:
            fr, row = cand[i]
            c, margin = model.predict(np.array(row[6], np.float32))
            if len(picks[c]) < 12 and margin > 0.2:
                picks[c].append((fr["t"], row[:4]))
        ev = self.ws.root / "evidence"
        ev.mkdir(parents=True, exist_ok=True)
        for c, items in picks.items():
            tiles = []
            for t, b in items:
                f = video.frame_at(t)
                if f is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b]
                crop = f[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4]
                if crop.size:
                    tiles.append(cv2.resize(crop, (48, 96)))
            if not tiles:
                tiles = [np.zeros((96, 48, 3), np.uint8)]
            while len(tiles) % 6:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
            cv2.imwrite(str(ev / f"team_cluster_{c}.jpg"), np.vstack(rows))

    def _stage_identity(self, video, prog):
        frames = self._frames()
        trs = tracks_mod.build_tracks(frames)
        tm = read_json(self.ws.root / "teams.json")
        model = TeamModel.from_json(tm) if tm else TeamModel()
        if model.centers is not None:
            tracks_mod.assign_teams(trs, model)
        tracks_mod.risk_points(trs)
        balls = ball_mod.ball_tracks(frames)
        prox = ball_mod.proximity(frames, balls)
        confirmed = all(v == "humain" for v in model.label_source.values()) if model.label_source else False
        W, H = video.width, video.height
        tl = self.timeline()
        res = identity_mod.resolve(trs, self.ws.store, self.ws.roster, self.fp, W, H, confirmed, timeline=tl)
        prox_by_track = {tid: v["near_s"] for tid, v in prox["per_track"].items()}
        created = identity_mod.propose_questions(trs, res, self.ws.store, self.ws.roster, video, tl, self.fp,
                                                 self.meta["period"], self.ws.root / "evidence",
                                                 budget=self.meta["options"]["identity_budget"], ball_prox=prox_by_track)
        for r in res.get("number_rejected", []):
            tr = trs[r["track_id"]]
            o = min(tr["obs"], key=lambda o: abs(o["t"] - r["t"]))
            x1, y1, x2, y2 = o["box"]
            bbox = [round(max(0, x1 / W), 5), round(max(0, y1 / H), 5), round(min(1, x2 / W), 5), round(min(1, y2 / H), 5)]
            tid = task_id(self.fp, "identity", o["t"], bbox)
            if tid not in self.ws.store.tasks:
                f = video.frame_at(o["t"])
                ev = {}
                if f is not None:
                    save_frame(f, str(self.ws.root / "evidence" / f"{tid}.jpg"), [o["box"]], [f"lu : {r['player']}"])
                    ev = {"image": f"evidence/{tid}.jpg", "clip_window": [max(0.0, o["t"] - 2.0), 4.0]}
                self.ws.store.add({"id": tid, "kind": "identity", "video": self.fp, "period": self.meta["period"],
                                   "time": round(o["t"], 3), "end": round(o["t"] + 4, 3), "bbox": bbox, "track_id": tr["id"],
                                   "match_clock": tl.at(o["t"]).get("clock"), "clock_status": "observed_engine" if tl.at(o["t"]).get("live") else "unverified",
                                   "reason": "Numéro lu automatiquement mais refusé : " + " ; ".join(r["rejected_reasons"]),
                                   "question": "Qui est le joueur encadré ?", "choices": self.ws.roster.choices(), "blocking": False,
                                   "priority": 70.0, "evidence": ev, "origin": "engine"})
        for c in res["conflicts"]:
            a, b = c["a"], c["b"]
            tid = task_id(self.fp, "conflict", a["anchor_t"], None, f"{a['track_id']}-{b['track_id']}-{a['player']}-{b['player']}")
            self.ws.store.add({"id": tid, "kind": "conflict", "video": self.fp, "period": self.meta["period"],
                               "time": a["anchor_t"], "bbox": None, "match_clock": tl.at(a["anchor_t"]).get("clock"),
                               "clock_status": "observed_engine", "blocking": False, "priority": 80,
                               "question": "Deux attributions se contredisent. Laquelle est juste ?",
                               "reason": f"{c['type']} : {a.get('player_name')} / {b.get('player_name')}",
                               "choices": [{"value": "a", "label": f"{a.get('player_name')} (t={a['anchor_t']:.1f})"},
                                           {"value": "b", "label": f"{b.get('player_name')} (t={b['anchor_t']:.1f})"},
                                           {"value": "none", "label": "Aucune des deux"}],
                               "evidence": {"clip_window": [max(0, a["anchor_t"] - 2), 5]}, "origin": "engine", "conflict": c})
            # Décision : la réponse à la question de conflit si elle existe. Sinon, une réponse humaine l'emporte sur
            # une réponse de Claude ; à égalité, aucune des deux attributions n'est comptée (l'une au moins est fausse).
            t = self.ws.store.tasks.get(tid)
            val = (t.get("response") or {}).get("value") if t and t["status"] == "resolved" else None
            human = {"human", "human_propagated"}
            if val in ("a", "b", "none"):
                drop, why = {"a": [b], "b": [a], "none": [a, b]}[val], "réponse à la question de conflit"
            elif a.get("source") in human and b.get("source") not in human:
                drop, why = [b], "en attente : la réponse humaine l'emporte sur celle de Claude"
            elif b.get("source") in human and a.get("source") not in human:
                drop, why = [a], "en attente : la réponse humaine l'emporte sur celle de Claude"
            else:
                drop, why = [a, b], "en attente : aucune des deux attributions n'est comptée"
            c["decision"] = {"value": val, "dropped": [(s["track_id"], s["player"]) for s in drop], "why": why}
            for s in drop:
                s["excluded_by_conflict"] = tid
        dropped = [s for s in res["segments"] if s.get("excluded_by_conflict")]
        res["segments"] = [s for s in res["segments"] if not s.get("excluded_by_conflict")]
        res["excluded_by_conflict"] = dropped
        self.ws.store.save()
        pcs = ball_mod.possession_chain(prox, trs)
        slim = {tid: {k: v for k, v in tr.items() if k != "obs"} | {"n_obs": tr["n_obs"]} for tid, tr in trs.items()}
        write_json(self.dir / "tracks.json", {"tracks": slim})
        write_json(self.dir / "identity.json", res)
        write_json(self.dir / "ball.json", {"coverage": prox["coverage"], "per_track": prox["per_track"],
                                            "pass_candidates": pcs, "ball_points": len(balls)})
        raja_s = sum(tr["n_obs"] for tr in trs.values() if tr.get("team") == "raja")
        # contrôle qualité automatique du suivi : pureté de couleur des pistes en plan large
        long_tr = [tr for tr in trs.values() if tr["views"].get("wide", 0) >= 10 and tr.get("team_conf") is not None]
        impure = [tr for tr in long_tr if tr["team_conf"] < 0.8]
        lens = sorted(tr["t_end"] - tr["t_start"] for tr in long_tr)
        qa = {"wide_tracks_10plus": len(long_tr), "tracks_team_purity_below_0_8": len(impure),
              "impure_fraction": round(len(impure) / max(1, len(long_tr)), 3),
              "median_track_duration_s": round(lens[len(lens) // 2], 2) if lens else None,
              "note": "piste impure = couleur majoritaire < 80 % des images : échange d'identité probable ou détection fusionnée"}
        attributed = sum(1 for s in res["segments"] for o in trs[s["track_id"]]["obs"] if s["t_from"] <= o["t"] <= s["t_to"])
        self._mark("identity", "done", tracks=len(trs), identified_segments=len(res["segments"]),
                   unmatched_answers=len(res["unmatched_answers"]), conflicts=len(res["conflicts"]),
                   new_questions=len(created), raja_track_frames=raja_s, attributed_frames=attributed,
                   ball_located_fraction=prox["coverage"]["ball_located_fraction"], tracking_qa=qa)
        pend = [t["id"] for t in self.ws.store.pending(kinds={"identity"}) if t.get("video") == self.fp]
        if created and self.meta["options"]["interactive_pause"] and not self.meta.get("skip_identity_pause"):
            # pause : l'étape est terminée, les questions sont prêtes. « Continuer sans répondre »
            # (skip_identity_pause) ou « Recalculer avec mes réponses » relancent la suite.
            raise NeedHuman("identity_review", pend[:50], f"{len(created)} nouvelles identités à confirmer.")

    def _stage_events(self, video, prog):
        sc = read_json(self.dir / "scan.json")
        tl = self.timeline()
        r = self._reader()
        banners = events_mod.banner_events(sc["samples"], video, r, self.ws.roster, tl,
                                           home_side=self.ws.cfg.get("scoreboard", {}).get("home_side", "raja"))
        goals = events_mod.goal_events(sc["score_events"], tl, self.ws.cfg.get("scoreboard", {}).get("home_side", "raja"))
        created = events_mod.replay_review_tasks(tl, self.fp, self.meta["period"], self.ws.store,
                                                 budget=self.meta["options"]["replay_review_budget"])
        # contrôle de cohérence avec l'état du terrain du Raja (périodes précédentes comprises)
        for b in banners:
            b["period"] = self.meta["period"]
        earlier = []
        for s in self.ws.sessions():
            if s.dir != self.dir and (s.meta.get("period") or 0) < (self.meta["period"] or 0):
                for b in read_json(s.dir / "events.json", {}).get("banners", []):
                    b["period"] = s.meta["period"]
                    earlier.append(b)
        events_mod.apply_banner_answers(banners, self.ws.store, self.fp, self.ws.roster)
        events_mod.check_raja_consistency(earlier + banners, self.ws.roster)
        wanted = set()
        for b in banners:
            if b["needs_review"]:
                task = events_mod.banner_check_task(self.fp, self.meta["period"], b)
                wanted.add(task["id"])
                self.ws.store.add(task)
        # une question « bandeau » devenue inutile (lecture améliorée) est retirée, jamais une question répondue
        for t in self.ws.store.tasks.values():
            if t["kind"] == "banner_check" and t.get("video") == self.fp and t["status"] == "pending" and t["id"] not in wanted:
                t["status"] = "obsolete"
                t["obsolete_reason"] = "bandeau relu automatiquement (équipe lue par la position de l'étiquette)"
        self.ws.store.save()
        write_json(self.dir / "events.json", {"banners": banners, "goals": goals})
        self._mark("events", "done", banners=len(banners), goals=len(goals), replay_questions=len(created))

    def _stage_calibration(self, video, prog):
        """Calibration seulement à partir de réponses humaines (points cliqués) ; sinon on
        demande quelques ancrages sur les plans larges les plus longs (non bloquant)."""
        frames = self._frames()
        L = self.ws.cfg.get("pitch", {}).get("length_m", 105.0)
        Wp = self.ws.cfg.get("pitch", {}).get("width_m", 68.0)
        answered = [t for t in self.ws.store.resolved("calibration") if t.get("video") == self.fp]
        Hs, logs = {}, []
        by_shot = {}
        for fr in frames:
            by_shot.setdefault(fr["shot"], []).append(fr)
        for t in answered:
            pts = (t["response"] or {}).get("points") or []
            fit = pitch_mod.fit_homography(pts, video.width, video.height, L, Wp)
            if not fit["ok"]:
                logs.append({"task_id": t["id"], "error": fit["reason"]})
                continue
            shot = t.get("shot")
            if shot not in by_shot:
                continue
            res = pitch_mod.propagate(by_shot[shot], t["time"], np.array(fit["H"]), get_frame=video.frame_at, L=L, Wp=Wp)
            for k, v in res["H"].items():
                Hs[k] = v.tolist()
            logs.append({"task_id": t["id"], "fit": {k: fit[k] for k in ("n_points", "fit_rmse_m", "leave_one_out_error_m", "checked")},
                         "propagation": res["log"]})
        # calibration automatique des autres plans larges (caméra principale à centre fixe)
        auto_log = {"enabled": bool(self.meta["options"].get("auto_calibration", False)), "experimental": True}
        auto_times = []
        if auto_log["enabled"]:
            cam = self._camera_model(video, answered, L, Wp)
            auto_log["camera"] = None if cam is None else {k: v for k, v in cam.items() if k != "C"} | {"C": cam["C"]}
            if cam is not None:
                from . import autocalib
                C = np.array(cam["C"])
                n_try = n_ok = 0
                for shot, frs in sorted(by_shot.items()):
                    wide = [f for f in frs if f["view"] == "wide"]
                    if len(wide) < 5 or any(f["t"] in Hs for f in wide):
                        continue
                    fr = max(wide[len(wide) // 4: 3 * len(wide) // 4 + 1] or wide, key=lambda f: len(f["p"]) * 0 + f.get("green", 0))
                    f = video.frame_at(fr["t"])
                    if f is None:
                        continue
                    n_try += 1
                    res = autocalib.calibrate_frame(f, [p[:4] for p in fr["p"]], C)
                    if not res.get("ok"):
                        continue
                    n_ok += 1
                    prop = pitch_mod.propagate(frs, fr["t"], res["H_img2pitch"], get_frame=video.frame_at, L=L, Wp=Wp)
                    for k, v in prop["H"].items():
                        if k not in Hs:
                            Hs[k] = v.tolist()
                            auto_times.append(k)
                    if prog:
                        prog("calibration auto", n_try, len(by_shot))
                auto_log.update(shots_tried=n_try, shots_calibrated=n_ok, frames=len(auto_times))
        # nouvelles demandes : plans larges les plus longs non calibrés
        shots = []
        for shot, frs in by_shot.items():
            wide = [f for f in frs if f["view"] == "wide"]
            if len(wide) < 10 or any(f["t"] in Hs for f in wide):
                continue
            shots.append((wide[-1]["t"] - wide[0]["t"], shot, wide))
        shots.sort(reverse=True)
        tl = self.timeline()
        n_new = 0
        # au plus N demandes en attente par vidéo : répondre ne doit pas en faire naître sans fin
        pending_cal = sum(1 for t in self.ws.store.tasks.values()
                          if t["kind"] == "calibration" and t.get("video") == self.fp and t["status"] == "pending")
        allowed = max(0, self.meta["options"]["calibration_requests"] - pending_cal)
        for dur, shot, wide in shots[:allowed]:
            fr = wide[len(wide) // 2]
            tid = task_id(self.fp, "calibration", fr["t"])
            if tid in self.ws.store.tasks:
                continue
            f = video.frame_at(fr["t"])
            if f is None:
                continue
            img = self.ws.root / "evidence" / f"{tid}.jpg"
            save_frame(f, str(img))
            self.ws.store.add({"id": tid, "kind": "calibration", "video": self.fp, "period": self.meta["period"], "time": fr["t"],
                               "bbox": None, "shot": shot, "match_clock": tl.at(fr["t"]).get("clock"), "clock_status": "observed_engine",
                               "blocking": False, "priority": 40 + dur,
                               "question": "Cliquez au moins 4 repères du terrain visibles (5 ou plus permettent un contrôle).",
                               "reason": f"Plan large de {dur:.0f} s : la calibration permet distances et vitesses sur ce plan.",
                               "choices": [{"value": k, "label": v} for k, v in pitch_mod.LANDMARK_LABELS.items()],
                               "evidence": {"image": f"evidence/{tid}.jpg"}, "origin": "engine"})
            n_new += 1
        self.ws.store.save()
        write_json(self.dir / "calibration.json", {"H": {str(k): v for k, v in Hs.items()}, "logs": logs,
                                                   "auto": auto_log, "auto_frames": [str(k) for k in auto_times],
                                                   "pitch_dimensions_m": [L, Wp],
                                                   "dimensions_verified": self.ws.cfg.get("pitch", {}).get("dimensions_verified", False)})
        self._mark("calibration", "done", calibrated_frames=len(Hs), anchors=len(answered), new_requests=n_new,
                   auto_calibrated_frames=len(auto_times), auto_shots=auto_log.get("shots_calibrated"))

    def _camera_model(self, video, answered, L, Wp):
        """Centre optique de la caméra principale, estimé sur les plans calibrés par réponse (humaine ou
        de Claude) et mis en cache au niveau du match ; recalculé si l'ensemble des réponses change."""
        from . import autocalib
        key = sorted(t["id"] for t in self.ws.store.resolved("calibration"))
        cache = read_json(self.ws.root / "camera_model.json")
        if cache and cache.get("answers") == key:
            return cache
        obs, ids = [], []
        for s in self.ws.sessions():
            vid = video if s.dir == self.dir else None
            for t in self.ws.store.resolved("calibration"):
                if t.get("video") != s.fp:
                    continue
                fit = pitch_mod.fit_homography((t["response"] or {}).get("points") or [], autocalib.IMG_W, autocalib.IMG_H, L, Wp)
                if not fit["ok"] or len(fit.get("landmarks", [])) < 5:
                    continue
                try:
                    src = vid or VideoSource(s.meta["video_path"])
                    f = src.frame_at(t["time"])
                except Exception:
                    f = None
                if f is None:
                    continue
                m, _ = autocalib.line_mask(f)
                ys, xs = np.nonzero(m)
                obs.append((np.stack([xs, ys], 1).astype(float), np.linalg.inv(np.array(fit["H"]))))
                ids.append(t["id"])
        C, rep = autocalib.fit_camera_center(obs)
        if C is None:
            return None
        rep["planes_used"] = [ids[i] for i in rep.pop("observations")]
        rep["planes_excluded"] = [ids[i] for i in rep.pop("excluded_observations")]
        out = {"C": [float(x) for x in C], "answers": key, **rep}
        write_json(self.ws.root / "camera_model.json", out)
        return out

    def _stage_session_stats(self, video, prog):
        frames = self._frames()
        trs = tracks_mod.build_tracks(frames)
        tm = read_json(self.ws.root / "teams.json")
        if tm:
            tracks_mod.assign_teams(trs, TeamModel.from_json(tm))
        ident = read_json(self.dir / "identity.json")
        ball = read_json(self.dir / "ball.json")
        cal = read_json(self.dir / "calibration.json", {"H": {}})
        calib = {"H": {float(k): np.array(v) for k, v in cal.get("H", {}).items()}}
        tl = self.timeline()
        step = 1.0 / self.meta["options"]["det_fps"]
        prox = {"per_track": {int(k): v for k, v in ball["per_track"].items()}} if ball else None
        data = stats_mod.session_player_data(self.meta["period"], tl, frames, trs, ident, prox, calib,
                                             ball.get("pass_candidates") if ball else None, step)
        cov = {"period": self.meta["period"], "video": self.meta["identity"]["name"], "duration_s": video.duration,
               "live_seconds": round(tl.live_seconds(), 1), "timeline": tl.summary(),
               "analysed_frames": len(frames), "det_fps": self.meta["options"]["det_fps"],
               "views": {v: sum(1 for f in frames if f["view"] == v) for v in ("wide", "medium", "close", "other")},
               "ball_located_fraction": (ball or {}).get("coverage", {}).get("ball_located_fraction"),
               "calibrated_frames": len(calib["H"]),
               "raja_track_frames": self.meta["stages"]["identity"].get("raja_track_frames"),
               "attributed_frames": self.meta["stages"]["identity"].get("attributed_frames")}
        write_json(self.dir / "session_stats.json", {"period": self.meta["period"], "data": data, "coverage": cov})
        self._mark("session_stats", "done")
