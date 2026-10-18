#!/usr/bin/env python3
"""Télécharge les poids YOLOX (licence Apache-2.0) depuis les publications officielles GitHub
et vérifie leur SHA-256. Aucune autre communication réseau n'est faite par le moteur."""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
MODELS = {
    "yolox_tiny.onnx": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    "yolox_s.onnx": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
}


def main(names):
    out = Path(__file__).resolve().parent / "models"
    out.mkdir(exist_ok=True)
    for name in names or MODELS:
        dest = out / name
        if not dest.exists():
            print("téléchargement", name)
            urllib.request.urlretrieve(BASE + name, dest)
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if h != MODELS[name]:
            dest.unlink()
            raise SystemExit(f"{name} : empreinte inattendue, fichier supprimé.")
        print(name, "OK", h)


if __name__ == "__main__":
    main(sys.argv[1:])
