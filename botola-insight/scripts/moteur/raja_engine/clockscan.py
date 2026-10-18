"""Étape 1 du moteur : balayage rapide du fichier (chrono, score, gels, temps additionnel).

Produit les échantillons utilisés par Timeline.build et la liste des changements de score.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .scoreboard import HAS_TESSERACT, ScoreboardReader
from .timeline import STOPPAGE_BASES, segment_reads


def _thumb(frame):
    g = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def bootstrap_templates(video, reader: ScoreboardReader, n: int = 160, human_reads: list | None = None,
                        log=print) -> dict:
    """Amorce les gabarits de chiffres. human_reads : [(t, secs)] fournis par l'utilisateur
    (question « quel chrono lisez-vous ? ») ; ils priment sur Tesseract."""
    times = np.linspace(0.5, max(0.6, video.duration - 0.5), n)
    frames = []
    for t in times:
        f = video.frame_at(float(t))
        if f is not None:
            frames.append((float(t), f))
    present = [(t, f) for t, f in frames if reader.present(f)[0]]
    info = {"sampled": len(frames), "scoreboard_present": len(present), "tesseract_available": HAS_TESSERACT}
    if len(present) < max(3, len(frames) // 20):
        if reader.relocate([f for _, f in frames]):
            present = [(t, f) for t, f in frames if reader.present(f)[0]]
            info["relocated_clock_box"] = reader.L["clock_box"]
        info["scoreboard_present"] = len(present)
    reads = []
    for t, f in present:
        r = reader.tesseract_clock(f)
        if r.seconds is not None:
            reads.append((t, r.seconds, f))
    info["tesseract_valid"] = len(reads)
    labeled = []
    if reads:
        segs, outl = segment_reads([(t, s) for t, s, _ in reads], tol=1.25, confirm=2)
        ok = {(t, s) for g in segs for t, s in g}
        for t, s, f in reads:
            if (t, s) in ok:
                g = reader.glyphs(f)
                if g is not None:
                    labeled.append((g, s))
        info["tesseract_consistent"] = len(labeled)
        info["tesseract_rejected_as_inconsistent"] = len(outl)
    if human_reads:
        for t, s in human_reads:
            f = video.frame_at(float(t))
            if f is not None and reader.present(f)[0]:
                g = reader.glyphs(f)
                if g is not None:
                    labeled.extend([(g, int(s))] * 3)  # poids supérieur à une lecture automatique
        info["human_reads_used"] = len(human_reads)
    if len(labeled) < 8:
        info["status"] = "needs_human"
        info["reason"] = ("Trop peu de lectures fiables du chrono pour apprendre ses chiffres "
                          f"({len(labeled)} images cohérentes).")
        return info
    info.update(reader.fit_templates(labeled))
    info["status"] = "ok" if not info["missing_digits"] else "partial"
    return info


def scan(video, reader: ScoreboardReader, step: float = 0.5, progress=None, sub_clock_every: float = 2.0,
         score_check_every: float = 1.0) -> dict:
    samples = []
    score_events = []
    prev_thumb = None
    last_sub_t = -1e9
    sig_ref = None
    sig_pending = 0
    last_score = None
    last_score_t = -1e9
    last_full_read = 0.0
    t0 = time.time()
    for t, idx, frame in video.iter_frames(step):
        th = _thumb(frame)
        diff = float(np.mean(np.abs(th - prev_thumb))) if prev_thumb is not None else None
        prev_thumb = th
        r = reader.read_clock(frame)
        s = {"t": round(t, 3), "present": r.present, "secs": r.seconds, "conf": round(r.confidence, 3),
             "diff": None if diff is None else round(diff, 3)}
        if not r.present and reader.banner(frame):
            s["banner"] = True
        if r.present and r.seconds in STOPPAGE_BASES:
            s["stoppage"] = reader.stoppage_indicator(frame)
            if t - last_sub_t >= sub_clock_every:
                add, _ = reader.read_sub_clock(frame)
                s["added"] = add
                last_sub_t = t
        samples.append(s)
        # score : relecture seulement si la zone change durablement
        if r.present and t - last_score_t >= score_check_every:
            sig = reader.score_signature(frame)
            if sig is not None:
                if sig_ref is None:
                    sig_ref = sig
                    sc, raw = reader.read_score(frame)
                    last_score = sc
                    score_events.append({"t": round(t, 2), "score": sc, "raw": raw, "kind": "initial"})
                else:
                    # un « 0 » qui devient « 1 » ne change que quelques % des pixels : seuil bas,
                    # confirmé sur 3 contrôles, plus une relecture de sécurité toutes les 60 s
                    changed = float(np.mean(np.abs(sig - sig_ref))) > 0.03
                    sig_pending = sig_pending + 1 if changed else 0
                    if sig_pending >= 3 or t - last_full_read >= 60.0:
                        last_full_read = t
                        sc, raw = reader.read_score(frame)
                        sig_ref = sig
                        sig_pending = 0
                        if sc is not None and sc != last_score:
                            score_events.append({"t": round(t, 2), "score": sc, "previous": last_score,
                                                 "raw": raw, "kind": "change"})
                            last_score = sc
                last_score_t = t
        if progress and len(samples) % 200 == 0:
            progress(t / max(video.duration, 1e-6), {"stage": "scan", "t": t,
                                                     "speed_x": t / max(time.time() - t0, 1e-6)})
    return {"samples": samples, "score_events": score_events, "elapsed_s": round(time.time() - t0, 1),
            "step_s": step}
