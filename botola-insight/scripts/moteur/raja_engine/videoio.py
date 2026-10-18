"""Lecture vidéo : parcours séquentiel échantillonné, accès aléatoire, extraits.

Le temps manipulé ici est TOUJOURS le temps interne du fichier (secondes depuis
le début du fichier). La conversion vers le chrono du match passe par timeline.py.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    def __init__(self, path: str):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"Vidéo illisible : {path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.duration = self._probe_duration() or (self.frame_count / self.fps)
        self._cap = None

    def _probe_duration(self):
        if not shutil.which("ffprobe"):
            return None
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                  "default=nw=1:nk=1", self.path], capture_output=True, text=True, timeout=60)
            return float(out.stdout.strip())
        except Exception:
            return None

    def info(self) -> dict:
        return {"fps": self.fps, "frame_count": self.frame_count, "width": self.width,
                "height": self.height, "duration_seconds": self.duration}

    def iter_frames(self, step_s: float, start_s: float = 0.0, end_s: float | None = None):
        """Itère (t, index_image, image) toutes les `step_s` secondes en lecture séquentielle.
        grab() saute le décodage complet des images intermédiaires : bien plus rapide
        qu'un positionnement aléatoire répété."""
        cap = cv2.VideoCapture(self.path)
        end_s = self.duration if end_s is None else min(end_s, self.duration)
        start_idx = int(round(start_s * self.fps))
        if start_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx
        next_t = start_s
        try:
            while True:
                t = idx / self.fps
                if t > end_s + 1e-6:
                    break
                if t + 0.5 / self.fps >= next_t:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield t, idx, frame
                    next_t += step_s
                    while next_t <= t:
                        next_t += step_s
                else:
                    if not cap.grab():
                        break
                idx += 1
        finally:
            cap.release()

    def frame_at(self, t: float):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * self.fps)))
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def extract_clip(src: str, start_s: float, dur_s: float, out: str, width: int = 640, crf: int = 30) -> str:
    """Extrait un clip H.264 léger (sans son) pour justifier une observation.
    -ss avant -i : positionnement rapide ; le temps est celui du fichier source."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start_s):.3f}", "-i", src, "-t", f"{dur_s:.3f}",
           "-an", "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True, timeout=300)
    return out


def save_frame(frame: np.ndarray, out: str, boxes=None, labels=None, quality: int = 85) -> str:
    """Image justificative, avec cadres (pixels) et étiquettes facultatifs."""
    img = frame.copy()
    for i, b in enumerate(boxes or []):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(img, str(labels[i]), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out
