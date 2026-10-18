"""Extraits vidéo des questions (pour la page de questions ou l'interface locale).

- action / ralenti : de 6 s avant le passage jusqu'à sa fin (30 s au plus), son conservé
  (le commentaire nomme souvent les joueurs) ;
- identité : 4 s centrées sur l'image demandée, cadre jaune qui SUIT le joueur (interpolé entre
  les positions de la piste), sans son ;
- bandeau (carton, remplacement) : 2 s avant à 3 s après, pleine définition, sans son ;
- précision d'une réponse : même extrait que la question d'origine.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .tracks import build_tracks, load_frames


def _ffmpeg_cut(src, start, dur, out, audio=True, width=640):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start):.2f}", "-i", src, "-t", f"{dur:.2f}",
           "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart"]
    cmd += ["-c:a", "aac", "-b:a", "64k", "-ac", "1"] if audio else ["-an"]
    subprocess.run(cmd + [out], check=True, timeout=300)


def _identity_clip(src, t, track, bbox, out, width=640, half=2.0):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_out = int(round(H * width / W / 2) * 2)
    obs = [(o["t"], np.array(o["box"], float)) for o in (track["obs"] if track else [])]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{h_out}",
                             "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    start = max(0.0, t - half)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    n = int(round(2 * half * fps))
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        ft = start + i / fps
        box = None
        if obs:
            before = [o for o in obs if o[0] <= ft]
            after = [o for o in obs if o[0] >= ft]
            if before and after:
                (t0, b0), (t1, b1) = before[-1], after[0]
                if t1 - t0 <= 0.6:
                    a = 0.0 if t1 == t0 else (ft - t0) / (t1 - t0)
                    box = b0 + a * (b1 - b0)
        elif bbox and abs(ft - t) < 0.2:
            box = np.array([bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H])
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(f, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 2)
        f = cv2.resize(f, (width, h_out), interpolation=cv2.INTER_AREA)
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait(timeout=120)
    cap.release()


def make(ws, task_ids, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = {s.meta["identity"]["fingerprint"]: s for s in ws.sessions()}
    tracks_cache = {}
    made = {}
    for tid in task_ids:
        t = ws.store.tasks[tid]
        base = ws.store.tasks.get(t.get("refines")) if t.get("refines") else t
        s = sessions.get(base.get("video"))
        if s is None:
            continue
        src = s.meta["video_path"]
        out = out_dir / f"{tid}.mp4"
        if out.exists():
            made[tid] = str(out)
            continue
        if base["kind"] == "event_review":
            t0 = float(base["time"])
            end = float(base.get("end") or t0 + 10)
            start = max(0.0, t0 - 6.0)
            _ffmpeg_cut(src, start, min(36.0, end - start + 1.0), str(out), audio=True)
        elif base["kind"] == "banner_check":
            # bandeau : pleine définition (le texte doit rester lisible), 2 s avant, 3 s après
            t0 = float(base["time"])
            end = float(base.get("end") or t0 + 6)
            _ffmpeg_cut(src, t0 - 2.0, min(20.0, end - t0 + 5.0), str(out), audio=False, width=854)
        elif base["kind"] == "identity":
            trs = tracks_cache.get(s.dir)
            if trs is None:
                trs = build_tracks(load_frames(s.dir / "frames.jsonl"))
                tracks_cache[s.dir] = trs
            _identity_clip(src, float(base["time"]), trs.get(base.get("track_id")), base.get("bbox"), str(out))
        else:
            continue
        made[tid] = str(out)
    return made
