"""Outils communs : JSON atomique, format des chronos, journal."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import tempfile
from pathlib import Path


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, data, indent=1):
    """Écriture atomique (fichier temporaire puis renommage) pour ne jamais corrompre
    un fichier de réponses en cas d'interruption."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent, default=_default)
    os.replace(tmp, p)


def append_jsonl(path, record):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"non sérialisable : {type(o)}")


# ---------------------------------------------------------------- chronos
# Représentation interne : secondes de match « réglementaires » (0–5400 et au-delà
# pour les prolongations) + secondes de temps additionnel séparées.

PERIOD_END = {1: 45 * 60, 2: 90 * 60, 3: 105 * 60, 4: 120 * 60}


def fmt_clock(seconds: float | None, added: float | None = None) -> str | None:
    """Format compatible avec l'API du portail : « MM:SS » ou « MM:SS +M:SS »."""
    if seconds is None:
        return None
    s = int(round(seconds))
    base = f"{s // 60:02d}:{s % 60:02d}"
    if added is not None and added >= 0:
        a = int(round(added))
        base += f" +{a // 60}:{a % 60:02d}"
    return base


_CLOCK_RE = re.compile(r"^\s*(\d{1,3}):(\d{2})(?:\s*\+\s*(\d{1,2}):(\d{2}))?\s*$")


def parse_clock(text: str | None):
    """« 90:00 +1:15 » -> (5400, 75). Renvoie (None, None) si illisible."""
    if not text:
        return None, None
    m = _CLOCK_RE.match(text.replace("–", "-"))
    if not m:
        return None, None
    base = int(m.group(1)) * 60 + int(m.group(2))
    add = None
    if m.group(3) is not None:
        add = int(m.group(3)) * 60 + int(m.group(4))
    return base, add


def clock_sort_key(seconds: float | None, added: float | None) -> float:
    """Clé d'ordre chronologique du match (le temps additionnel se place après la borne)."""
    if seconds is None:
        return float("inf")
    return float(seconds) + (float(added) if added else 0.0) * 1.0001
