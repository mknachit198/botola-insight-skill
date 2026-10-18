"""Adaptateurs paramétrés des étapes historiques ; moteur importé tel quel."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from common import ENGINE, SKILL, engine_integrity, load_teams, match, read, record, rule_e, sha, slug, validate_readings, write


def setup(args):
    if sys.version_info < (3, 10):
        raise ValueError("Python 3.10 ou ultérieur est nécessaire.")
    if args.install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(SKILL / "scripts/requirements.txt")], check=True)
    if args.fetch_models:
        subprocess.run([sys.executable, str(ENGINE / "fetch_models.py"), "yolox_tiny.onnx", "yolox_s.onnx"], check=True)
    versions, missing = {}, []
    from importlib import metadata
    for package in ("numpy", "opencv-python-headless", "onnxruntime", "scipy", "scikit-learn", "pytesseract", "flask", "Pillow", "ortools"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    tools = {n: shutil.which(n) for n in ("ffmpeg", "ffprobe", "tesseract")}
    result = {"python": sys.version.split()[0], "dependencies": versions, "tools": tools,
              "engine_files_verified": engine_integrity(), "missing": missing + [k for k, v in tools.items() if not v]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["missing"]:
        raise ValueError("Installer les dépendances manquantes avant l'analyse.")


def prepare(args):
    cfg = match(args.match)
    dest = args.work / cfg["match_key"]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "match.json"
    if target.exists() and read(target) != cfg:
        raise ValueError("Configuration différente dans ce dossier ; choisir un nouveau dossier de travail.")
    write(target, cfg)
    print(json.dumps({"match": str(target), "next": "analyze", "colors_confirmation": "une fois par match, après visualisation des groupes"}, ensure_ascii=False))


def analyze(args):
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    from raja_engine import analyze as engine_analyze
    from raja_engine.pipeline import MatchWorkspace
    from raja_engine.teams import propose_label
    from raja_engine.timeline import Timeline
    from raja_engine.tracking_refine import refine_file
    from raja_engine.tracks import fit_teams, load_frames
    from raja_engine.util import read_json
    from raja_engine.videoio import VideoSource
    cfg = match(args.match)
    slug(args.part)
    root = args.work / cfg["match_key"]
    part = root / ("part_" + args.part)
    part.mkdir(parents=True, exist_ok=True)
    video = VideoSource(str(args.video))
    if video.duration > 330:
        video.close()
        raise ValueError("Fournir une partie d'environ 5 minutes (maximum 330 s). Aucun match complet avant le critère de l'étape 1.")
    model = ENGINE / "models/yolox_tiny.onnx"
    from moteur.fetch_models import MODELS
    if not model.is_file() or sha(model) != MODELS[model.name]:
        video.close()
        raise ValueError("Modèle absent/non conforme ; lancer setup --fetch-models.")
    inputs = {"video": sha(args.video), "match": sha(args.match), "model": sha(model),
              "timeline": sha(args.timeline) if args.timeline else None, "period": args.period,
              "det_fps": 5.0, "camera_fps": 15.0, "tiles": 2}
    seal = part / "analysis_inputs.json"
    if seal.exists() and read(seal) != inputs:
        video.close()
        raise ValueError("Entrées modifiées : utiliser une nouvelle partie/un nouveau dossier, sans mélanger les analyses.")
    write(seal, inputs)
    ws = MatchWorkspace(part / ".engine", args.match)
    session = ws.session_for(str(args.video.resolve()), args.period, {"det_fps": 5.0, "camera_fps": 15.0,
                               "tiles": 2, "identity_budget": 0, "interactive_pause": False})
    if args.clock_answers:
        for tid, value in read(args.clock_answers).items():
            if tid not in ws.store.tasks or ws.store.tasks[tid].get("kind") != "clock_read":
                raise ValueError(f"Question de chrono inconnue : {tid}")
            ws.store.answer(tid, str(value), label=str(value))
    analysis = session.dir
    start = time.perf_counter()
    try:
        if args.timeline:
            timeline = read(args.timeline)
            if not timeline.get("segments"):
                raise ValueError("Timeline vide.")
            write(analysis / "timeline.json", timeline)
            write(analysis / "video_info.json", video.info())
            engine_analyze.run(video, Timeline(timeline), analysis, str(model), det_fps=5.0, camera_fps=15.0, tiles=2)
        elif not session.done("analyze"):
            # S'arrêter à analyze ; ne jamais lancer implicitement l'ancien résolveur/les statistiques.
            status = session.run(until="analyze")
            if status.get("status") == "waiting_human":
                write(part / "questions.json", {"waiting": status["waiting"], "questions": [t for t in ws.store.tasks.values() if t.get("kind") == "clock_read"]})
                raise ValueError(f"Chrono à lire : {part / 'questions.json'} ; fournir --clock-answers puis reprendre.")
    finally:
        video.close()
    teams_file = root / "teams.json"
    confirmation = root / "colors_confirmation.json"
    if args.teams:
        load_teams(args.teams)
        if not args.confirmed_by:
            raise ValueError("L'import de couleurs exige --confirmed-by (source de la confirmation utilisateur).")
        if teams_file.exists() and read(teams_file) != read(args.teams):
            raise ValueError("Couleurs différentes de celles déjà confirmées pour ce match.")
        write(teams_file, read(args.teams))
        write(confirmation, {"confirmed_by": args.confirmed_by, "teams_sha256": sha(teams_file)})
    if not teams_file.exists() or not confirmation.exists():
        frames = load_frames(analysis / "frames.jsonl")
        model_teams, info = fit_teams(frames)
        if info["status"] != "ok":
            raise ValueError("Pas assez de détections pour proposer les couleurs ; aucune équipe inventée.")
        hints = cfg.get("kit_hints")
        if hints:
            for c, center in enumerate(model_teams.centers):
                model_teams.cluster_labels[c] = propose_label(center, hints)[0]
        write(root / "teams_proposed.json", model_teams.to_json())
        source = VideoSource(str(args.video))
        try:
            session._team_mosaics(source, frames, model_teams)
        finally:
            source.close()
        mosaic_dir = root / "color_examples"
        mosaic_dir.mkdir(exist_ok=True)
        for f in (ws.root / "evidence").glob("team_cluster_*.jpg"):
            shutil.copyfile(f, mosaic_dir / f.name)
        write(part / "part.json", {"part": args.part, "analysis_dir": str(analysis), "video": str(args.video.resolve()), "status": "needs_colors"})
        raise ValueError(f"Demander UNE confirmation des groupes de {mosaic_dir}, puis confirm-colors et reprendre analyze.")
    if read(confirmation)["teams_sha256"] != sha(teams_file):
        raise ValueError("Les couleurs ont changé depuis leur confirmation.")
    frames_path, refinement = refine_file(analysis, load_teams(teams_file))
    write(part / "part.json", {"part": args.part, "analysis_dir": str(analysis), "frames": str(frames_path),
         "timeline": str(analysis / "timeline.json"), "teams": str(teams_file), "video": str(args.video.resolve()),
         "status": "ready", "elapsed_this_call_s": round(time.perf_counter() - start, 3),
         "settings": inputs})
    print(json.dumps(read(part / "part.json"), ensure_ascii=False, indent=2))


def confirm_colors(args):
    root = args.work / match(args.match)["match_key"]
    proposed = root / "teams_proposed.json"
    data = read(proposed)
    mapping = read(args.mapping)
    if set(mapping) != {str(i) for i in range(len(data["centers"]))}:
        raise ValueError("Confirmer chaque groupe de couleurs dans le fichier de correspondance.")
    data["cluster_labels"] = mapping
    data["label_source"] = {k: "humain" for k in mapping}
    # Valider avant de remplacer la proposition.
    from raja_engine.teams import LABELS
    if not set(mapping.values()) <= set(LABELS):
        raise ValueError("Étiquette inconnue : utiliser les rôles documentés dans les formats.")
    write(root / "teams.json", data)
    write(root / "colors_confirmation.json", {"confirmed_by": args.confirmed_by, "teams_sha256": sha(root / "teams.json")})
    print("Couleurs confirmées et conservées pour le match.")


def planches(args):
    from raja_engine.perception import load_part
    from raja_engine.reading.service import build_planches
    from raja_engine import evidence as ev_mod
    frames, _, tracks = load_part(args.frames, args.teams)
    kwargs = {"team": args.team_label}
    inputs = [args.frames, args.teams, args.video]
    if args.pass_number == 2:
        if not args.first_meta or not args.evidence:
            raise ValueError("Seconde passe : --first-meta et --evidence requis.")
        ev = ev_mod.read(args.evidence)
        seen = {e["track"] for e in ev if e["confidence"] >= .7}
        first = {m["track"]: [c["t"] for c in m["crops"]] for m in read(args.first_meta)["planches"]}
        only = {i for i, tr in tracks.items() if tr["team"] == args.team_label and i not in seen and tr["t_end"] - tr["t_start"] >= 8.0}
        kwargs.update(min_dur=8.0, n=24, only=only, prefix="s", exclude=first, use_back=False)
        inputs += [args.first_meta, args.evidence]
    args.out.mkdir(parents=True, exist_ok=True)
    signature = {str(Path(p).resolve()): sha(p) for p in inputs}
    marker = args.out / "planches_inputs.json"
    params = {"pass": args.pass_number, "team": args.team_label, "inputs": signature}
    if marker.exists() and read(marker) != params:
        raise ValueError("Planches d'autres entrées dans ce dossier ; choisir un nouveau dossier.")
    if marker.exists() and (args.out / "planches_meta.json").exists():
        meta = read(args.out / "planches_meta.json")["planches"]
        if all((args.out / m["file"]).exists() for m in meta):
            print(f"{len(meta)} planches déjà présentes ; lectures existantes conservées.")
            return
    meta = build_planches(tracks, frames, str(args.video), args.out, **kwargs)
    write(marker, params)
    record(args.out, "planches", inputs, {"pass": args.pass_number, "team": args.team_label}, [args.out / "planches_meta.json"])
    print(json.dumps({"planches": len(meta), "crops": sum(len(m["crops"]) for m in meta), "out": str(args.out)}))


def build(args):
    from raja_engine import evidence as ev_mod
    from raja_engine.perception import load_part, tracklets, write_tracklets
    from raja_engine.reading.service import to_evidence
    slug(args.part)
    args.out.mkdir(parents=True, exist_ok=True)
    frames, _, tracks = load_part(args.frames, args.teams)
    items = tracklets(tracks, frames)
    inputs = [args.frames, args.teams]
    legacy = []
    if args.legacy_dataset:
        legacy = ev_mod.from_legacy_dataset(read(args.legacy_dataset))
        inputs.append(args.legacy_dataset)
    vlm = []
    specs = list(args.reading_set or [])
    if args.readings_dir:
        d = args.readings_dir
        specs.append([str(d / "planches_meta.json"), str(d / "lectures_vlm.json"), ""])
        sp = d / "seconde_passe"
        if (sp / "lectures_vlm.json").exists():
            specs.append([str(sp / "planches_meta.json"), str(sp / "lectures_vlm.json"), "seconde_passe"])
    if not specs:
        raise ValueError("Fournir --readings-dir ou --reading-set META LECTURES PREFIX (lectures vides explicites admises).")
    for meta_path, reads_path, prefix in specs:
        meta, lect = validate_readings(meta_path, reads_path)
        if prefix:
            p = Path(prefix)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Préfixe de provenance relatif attendu.")
            for m in meta["planches"]:
                m["file"] = p.as_posix().rstrip("/") + "/" + m["file"]
        reader = lect["reader"] + (" ; seconde passe" if prefix == "seconde_passe" else "")
        vlm += to_evidence(meta["planches"], lect["readings"], "vlm_plank", reader)
        inputs += [meta_path, reads_path]
    tp, ep = args.out / f"tracklets_part{args.part}.jsonl", args.out / f"evidence_part{args.part}.jsonl"
    write_tracklets(tp, items)
    ev_mod.write(ep, legacy + vlm)
    summary = {"tracklets": len(items), "by_team": dict(Counter(i["team"] for i in items)),
               "evidence": {"legacy_v7_annotation": len(legacy), "vlm_plank": len(vlm)},
               "tracklets_sha256": sha(tp), "evidence_sha256": sha(ep)}
    write(args.out / "build_summary.json", summary)
    record(args.out, "build", inputs, {"part": args.part}, [tp, ep])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve(args):
    from raja_engine import evidence as ev_mod
    from raja_engine.identity_resolver import resolve as solve, roster_allowed, observation_identities
    from raja_engine.perception import read_tracklets
    from raja_engine.roster import Roster
    from raja_engine.timeline import Timeline
    cfg = match(args.match)
    label = args.team_label
    roster_key = "roster" if label.startswith("raja") else "opponent_roster"
    if roster_key not in cfg:
        raise ValueError("Demander la composition adverse avant de résoudre cette équipe.")
    roster = Roster({"roster": cfg[roster_key]})
    tl = Timeline(read(args.timeline))
    tracks = read_tracklets(args.tracklets)
    selected = {i: t for i, t in tracks.items() if t["team"] == label}
    ids = set(selected)
    evidence = [e for e in ev_mod.read(args.evidence) if e["track"] in ids]
    allowed_base = roster_allowed(roster, tl, outfield=not label.endswith("_gk"))
    def allowed(unit):
        # Pas de fenêtre de participation utilisable sans chrono de direct.
        if any(tl.at(t).get("clock_s") is None for t in (unit["start"], unit["end"])):
            return []
        nums = allowed_base(unit)
        if label.endswith("_gk"):
            nums = [n for n in nums if roster.by_number[n]["role"] == "G"]
        return nums
    started = time.perf_counter()
    result = solve(tracks, evidence, allowed, rule_e(), team=label)
    result.update(team_label=label, match_key=cfg["match_key"], observations=observation_identities(result),
                  resolve_wall_s=round(time.perf_counter() - started, 6))
    write(args.out, result)
    record(args.out.parent, "resolve_" + label, [args.match, args.tracklets, args.evidence, args.timeline], {"rule": "E", "v8_1_options": False}, [args.out])
    print(json.dumps({"units": len(result["units"]), "out": str(args.out), "calibrated": False}))
