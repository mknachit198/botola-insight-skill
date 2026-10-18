"""Étape 2 : détection, plans, mouvement de caméra, suivi et indices d'identité.

Traite uniquement les segments de direct (et de temps additionnel) de la timeline.
Écrit une ligne JSON par image traitée dans frames.jsonl, avec point de reprise :
une interruption ne fait perdre que la fin du segment en cours.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .detector import BALL, PERSON, YoloxDetector
from .numbers import MIN_BOX_H_REL, read_number
from .teams import torso_feature
from .tracker import ShotTracker, Track
from .util import append_jsonl, read_json, write_json
from .views import CameraMotion, MotionComposer, classify_view, hist_signature, is_cut, pitch_region
from .tracking_refine import VERSION as TRACKING_VERSION, remove_short_tracks, write_frames


def _r(v, n=1):
    return float(round(float(v), n))


def run(video, timeline, work: Path, model_path: str, det_fps: float = 5.0, tiles: int = 2,
        number_every_s: float | None = None, progress=None, should_stop=None, include_replays: bool = False,
        camera_fps: float = 15.0) -> dict:
    if det_fps <= 0 or camera_fps <= 0:
        raise ValueError("Les cadences doivent être positives")
    camera_fps = max(camera_fps, det_fps)
    work = Path(work)
    frames_path = work / "frames.jsonl"
    state_path = work / "analyze_state.json"
    state = read_json(state_path, {"done_until": -1.0, "next_track_id": 1, "shot_id": 0, "frames": 0,
                                   "compute_s": 0.0, "video_s": 0.0})
    options = {"det_fps": det_fps, "camera_fps": camera_fps, "tiles": tiles, "model": Path(model_path).name}
    if state.get("frames", 0):
        if state.get("tracking_version") != TRACKING_VERSION:
            raise ValueError("Analyse d'une ancienne version : utilisez --restart-analysis pour la sauvegarder et la recalculer.")
        if state.get("analysis_options") != options:
            raise ValueError("Options d'analyse modifiées : utilisez --restart-analysis avant de recalculer.")
    state.update(tracking_version=TRACKING_VERSION, det_fps=det_fps, camera_fps=camera_fps,
                 analysis_options=options)
    # Une interruption entre l'ajout des lignes et le checkpoint ne doit pas
    # dupliquer les dernières images à la reprise.
    if frames_path.exists():
        from .util import read_jsonl
        committed = [f for f in read_jsonl(frames_path) if f["t"] <= state["done_until"] + 1e-6]
        write_frames(frames_path, committed)
    Track._next = max(Track._next, int(state["next_track_id"]))
    det = YoloxDetector(model_path)
    kinds = {"live", "stoppage", "live_banner", "live_interpolated"} | ({"no_scoreboard"} if include_replays else set())
    segs = [s for s in timeline.segments if s["kind"] in kinds]
    total = sum(s["file_end"] - s["file_start"] for s in segs) or 1.0
    step = 1.0 / det_fps
    shot_id = int(state["shot_id"])
    t_begin = time.time()
    processed_video_s = 0.0
    done_before = sum(max(0.0, min(s["file_end"], state["done_until"]) - s["file_start"]) for s in segs)
    last_number_t: dict[int, float] = {}

    for seg in segs:
        if seg["file_end"] <= state["done_until"]:
            continue
        start = max(seg["file_start"], state["done_until"] + step)
        tracker = ShotTracker()
        shot_id += 1
        tracker.new_shot(shot_id)
        cam = CameraMotion()
        composer = MotionComposer()
        prev_sig = None
        prev_t = None
        next_detection = 0.0
        last_view = "other"
        pending_cut, pending_cut_score = True, 1.0
        buffer = []
        for sample_idx, (t, fi, frame) in enumerate(video.iter_frames(1.0 / camera_fps, start, seg["file_end"])):
            if should_stop and should_stop():
                break
            H_img, W_img = frame.shape[:2]
            mask, green = pitch_region(frame)
            sig = hist_signature(frame)
            cut, cut_score = is_cut(prev_sig, sig)
            if prev_t is not None and t - prev_t > 2.5 / camera_fps:
                cut = True
            if cut and prev_sig is not None:
                shot_id += 1
                tracker.new_shot(shot_id)
                cam.reset()
                composer.reset()
                last_view = "other"
            pending_cut = pending_cut or cut
            pending_cut_score = max(pending_cut_score, cut_score)
            prev_sig, prev_t = sig, t
            detect_now = sample_idx / camera_fps + 1e-9 >= next_detection
            if not detect_now:
                # Les images intermédiaires servent uniquement au mouvement de caméra.
                # La dernière vue connue et la fraction verte excluent les gros plans.
                if last_view in ("wide", "medium") and green >= 0.20:
                    Hc, inl = cam.step(frame, mask, [])
                    composer.add(Hc, inl)
                else:
                    cam.reset()
                    composer.add(None)
                state["camera_frames"] = state.get("camera_frames", 0) + 1
                continue
            next_detection += step
            dets = det.detect(frame, tiles=tiles)
            persons = [d for d in dets if d.cls == PERSON]
            balls = [d for d in dets if d.cls == BALL]
            rel_h = [d.h / H_img for d in persons]
            view = classify_view(green, rel_h)
            if view in ("wide", "medium"):
                # garder les personnes dont les pieds sont sur le terrain (exclut public, stadiers)
                dil = mask
                keep = []
                for d in persons:
                    fx, fy = d.foot
                    yy = min(H_img - 1, int(fy))
                    xx = min(W_img - 1, max(0, int(fx)))
                    if dil[yy, xx] > 0 or dil[max(0, yy - 6), xx] > 0:
                        keep.append(d)
                persons = keep
                balls = [b for b in balls if mask[min(H_img - 1, int((b.y1 + b.y2) / 2)), min(W_img - 1, int((b.x1 + b.x2) / 2))] > 0
                         and (b.x2 - b.x1) < 0.03 * W_img]
            boxes = np.array([d.box for d in persons], np.float32).reshape(-1, 4)
            scores = np.array([d.score for d in persons], np.float32)
            Hc, inl = (None, 0)
            if view in ("wide", "medium"):
                Hc, inl = cam.step(frame, mask, boxes.tolist())
                composer.add(Hc, inl)
            else:
                cam.reset()
                composer.add(None)
            Hc, inl = composer.take()
            last_view = view
            state["camera_frames"] = state.get("camera_frames", 0) + 1
            assoc = dict(tracker.step(t, fi, boxes, scores, Hc)) if view != "other" else {}
            prow = []
            nums = []
            for i, d in enumerate(persons):
                tid = assoc.get(i)
                feat = torso_feature(frame, d.box) if view in ("wide", "medium", "close") else None
                prow.append([_r(d.x1), _r(d.y1), _r(d.x2), _r(d.y2), _r(d.score, 3), tid,
                             None if feat is None else [round(float(x), 3) for x in feat]])
                # v1 : Tesseract hors du chemin de production (13 lectures, toutes hors effectif, sur
                # 5 min) ; les numéros viennent du service reading/. Réactivable : number_every_s=1.0.
                if (number_every_s and tid is not None and d.h >= MIN_BOX_H_REL * H_img
                        and t - last_number_t.get(tid, -1e9) >= number_every_s):
                    n, conf, raw = read_number(frame, d.box)
                    last_number_t[tid] = t
                    if n is not None:
                        nums.append([tid, n, round(conf, 2)])
            rec = {"t": round(t, 3), "fi": fi, "shot": shot_id, "view": view, "green": round(green, 3),
                   "cut": bool(pending_cut), "cut_score": round(pending_cut_score, 3),
                   "H": None if Hc is None else [round(float(x), 6) for x in Hc.reshape(-1)], "inl": inl,
                   "p": prow, "b": [[_r(b.x1), _r(b.y1), _r(b.x2), _r(b.y2), _r(b.score, 3)] for b in balls],
                   "num": nums, "seg_kind": seg["kind"], "W": W_img, "Himg": H_img}
            buffer.append(rec)
            pending_cut, pending_cut_score = False, 0.0
            processed_video_s += step
            state["frames"] += 1
            if len(buffer) >= 50:
                for r in buffer:
                    append_jsonl(frames_path, r)
                buffer = []
                state.update(done_until=t, next_track_id=Track._next, shot_id=shot_id,
                             compute_s=state["compute_s"] + (time.time() - t_begin), video_s=state["video_s"] + processed_video_s)
                t_begin = time.time()
                processed_video_s = 0.0
                write_json(state_path, state)
                if progress:
                    done = done_before + sum(max(0.0, min(s["file_end"], t) - s["file_start"]) for s in segs if s["file_start"] <= t)
                    progress(min(1.0, done / total), {"stage": "detect_track", "t": t,
                                                      "speed_x": state["video_s"] / max(state["compute_s"], 1e-6)})
        for r in buffer:
            append_jsonl(frames_path, r)
        if should_stop and should_stop():
            last = buffer[-1]["t"] if buffer else state["done_until"]
            state.update(done_until=last, next_track_id=Track._next, shot_id=shot_id)
            state.update(compute_s=state["compute_s"] + (time.time() - t_begin),
                         video_s=state["video_s"] + processed_video_s)
            write_json(state_path, state)
            _filter_short_file(frames_path)
            return {"status": "stopped", **state}
        state.update(done_until=seg["file_end"], next_track_id=Track._next, shot_id=shot_id,
                     compute_s=state["compute_s"] + (time.time() - t_begin), video_s=state["video_s"] + processed_video_s)
        t_begin = time.time()
        processed_video_s = 0.0
        write_json(state_path, state)
    state["status"] = "done"
    _filter_short_file(frames_path)
    state["speed_x_vs_realtime"] = round(state["video_s"] / max(state["compute_s"], 1e-6), 3)
    state["det_fps"] = det_fps
    state["tiles"] = tiles
    state["model"] = Path(model_path).name
    write_json(state_path, state)
    return state


def _filter_short_file(path):
    from .util import read_jsonl
    frames = read_jsonl(path)
    remove_short_tracks(frames)
    write_frames(path, frames)
