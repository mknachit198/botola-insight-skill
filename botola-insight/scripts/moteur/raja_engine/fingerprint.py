"""Empreinte échantillonnée « sha256-3x64k-v1 », identique à celle du portail.

Elle reconnaît un fichier renommé ; elle ne certifie PAS l'intégrité complète
(seules trois fenêtres de 64 Kio sont lues). Le SHA-256 intégral peut être
calculé à part avec `full_sha256` si nécessaire.
"""
from __future__ import annotations

import hashlib
import os

WINDOW = 65536
METHOD = "sha256-3x64k-v1"


def sampled_fingerprint(path: str) -> dict:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(f"raja-sample-v1:{size}\n".encode("utf-8"))
    last = max(0, size - WINDOW)
    offsets = [0, last // 2, last]
    with open(path, "rb") as fh:
        for off in offsets:
            fh.seek(off)
            h.update(fh.read(WINDOW))
    return {"name": os.path.basename(path), "size": size, "fingerprint": h.hexdigest(),
            "fingerprint_method": METHOD}


def full_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
