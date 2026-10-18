"""Envoi vers le portail WordPress Raja Insight (même contrat que engine_client.py).

Identifiants lus uniquement dans l'environnement : RI_WP_BASE_URL (https), RI_WP_USER,
RI_WP_APP_PASSWORD (mot de passe d'application WordPress). Aucun secret n'est écrit sur disque.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request


def configured() -> bool:
    return os.environ.get("RI_WP_BASE_URL", "").startswith("https://") and bool(os.environ.get("RI_WP_USER")) \
        and bool(os.environ.get("RI_WP_APP_PASSWORD"))


def post(match_id: int, kind: str, payload: dict, timeout: int = 30) -> dict:
    if kind not in ("video-session", "analysis", "corrections"):
        raise ValueError("type d'envoi inconnu")
    if not configured():
        raise RuntimeError("Portail non configuré : renseigner RI_WP_BASE_URL (https), RI_WP_USER, RI_WP_APP_PASSWORD.")
    base = os.environ["RI_WP_BASE_URL"].rstrip("/")
    auth = base64.b64encode(f"{os.environ['RI_WP_USER']}:{os.environ['RI_WP_APP_PASSWORD']}".encode()).decode()
    req = urllib.request.Request(f"{base}/wp-json/raja-insight/v1/matches/{int(match_id)}/{kind}",
                                 data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Basic " + auth})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("message", "Requête refusée")
        except Exception:
            msg = "Requête refusée"
        raise RuntimeError(f"HTTP {e.code} : {msg}")
