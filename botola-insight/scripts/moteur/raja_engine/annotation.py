"""Étape 0 — mode annotation (vérité terrain) et conversion en groundtruth.jsonl.

build_bundle : pour chaque chaîne Raja, 8 à 12 recadrages ÉQUIRÉPARTIS dans la chaîne (aucune
sélection « lisible » : la vérité terrain ne doit pas hériter des choix du lecteur), plus une
vue d'ensemble réduite avec la boîte entourée. Tout est rangé dans une bande JPEG par chaîne
(strips/<piste>.jpg) et décrit dans annotation_data.js. La page d'annotation ne montre AUCUNE
lecture ni proposition du système.

Réponse par chaîne : un numéro par défaut (ou « inconnu », ou « pas un joueur Raja »), et des
exceptions par recadrage quand la chaîne change de joueur. Conversion (to_groundtruth) :
une observation entre deux recadrages de même étiquette prend cette étiquette ; entre deux
étiquettes différentes, elle est dans une zone de bascule et reste null ; avant le premier /
après le dernier recadrage, elle prend l'étiquette du plus proche.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .reading import crops as crops_mod
from .reading.selection import evenly

VERSION = "annotation-1.0"
STRIP_H = 160
CTX_W = 480


def n_crops(tr, lo=8, hi=12, per_s=1.5):
    dur = tr["t_end"] - tr["t_start"]
    return int(min(hi, max(lo, round(dur / per_s))))


def chain_set(tr, test_end_local):
    return "test" if tr["t_start"] < test_end_local else "reglage"


def build_bundle(tracklets: dict, video, out, test_end_local: float, team="raja", min_obs=1):
    out = Path(out)
    (out / "strips").mkdir(parents=True, exist_ok=True)
    chosen, reqs = {}, []
    for tid in sorted(tracklets):
        tr = tracklets[tid]
        if tr.get("team") != team or tr["n_obs"] < min_obs:
            continue
        idx = evenly(tr, n_crops(tr))
        mid = idx[len(idx) // 2]
        chosen[tid] = (idx, mid)
        for k, i in enumerate(idx):
            o = tr["obs"][i]
            reqs.append({"key": f"{tid}:{k}", "fi": o["fi"], "box": o["box"]})
        o = tr["obs"][mid]
        reqs.append({"key": f"{tid}:ctx", "fi": o["fi"], "box": o["box"]})
    imgs = crops_mod.extract(video, reqs, context=True)
    chains = []
    for tid, (idx, mid) in chosen.items():
        tr = tracklets[tid]
        tiles, crops_meta, x = [], [], 0
        for k, i in enumerate(idx):
            c = imgs.get(f"{tid}:{k}")
            if c is None or c.size == 0:
                c = np.full((STRIP_H, 40, 3), 200, np.uint8)
            s = STRIP_H / c.shape[0]
            c = cv2.resize(c, (max(1, int(round(c.shape[1] * s))), STRIP_H),
                           interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
            tiles.append(c)
            crops_meta.append({"k": k + 1, "t": tr["obs"][i]["t"], "x": x, "w": c.shape[1]})
            x += c.shape[1] + 2
            tiles.append(np.full((STRIP_H, 2, 3), 255, np.uint8))
        ctx = imgs.get(f"{tid}:ctx#ctx")
        ctx_meta = None
        if ctx is not None:
            s = STRIP_H / ctx.shape[0]
            ctx = cv2.resize(ctx, (int(round(ctx.shape[1] * s)), STRIP_H), interpolation=cv2.INTER_AREA)
            tiles.append(ctx)
            ctx_meta = {"x": x, "w": ctx.shape[1], "t": tr["obs"][mid]["t"]}
            x += ctx.shape[1]
        strip = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(out / "strips" / f"{tid}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        chains.append({"id": tid, "set": chain_set(tr, test_end_local), "t_start": tr["t_start"],
                       "t_end": tr["t_end"], "n_obs": tr["n_obs"], "strip_w": int(strip.shape[1]),
                       "crops": crops_meta, "ctx": ctx_meta})
    chains.sort(key=lambda c: (c["set"] != "test", c["t_start"], c["id"]))
    data = {"version": VERSION, "strip_h": STRIP_H, "test_end_local": test_end_local,
            "video": Path(str(video)).name, "chains": chains}
    (out / "annotation_data.js").write_text("window.ANNOTATION_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n",
                                            encoding="utf-8")
    (out / "annotation_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
    return data


def _label(v):
    """'30' -> 30 ; 'unknown' -> None ; 'not_raja' -> 'not_raja'."""
    if v in (None, "", "unknown"):
        return None
    if v == "not_raja":
        return "not_raja"
    return int(v)


def to_groundtruth(answers: dict, data: dict, tracklets: dict):
    """answers : {"c<piste>": {"default": "30"|"unknown"|"not_raja", "crops": {"<k>": ...}}}.
    Renvoie les lignes {t, track, box, jersey, status, set} pour les chaînes répondues."""
    rows = []
    by_id = {c["id"]: c for c in data["chains"]}
    for key, ans in answers.items():
        tid = int(str(key).lstrip("c"))
        meta, tr = by_id.get(tid), tracklets.get(tid)
        if meta is None or tr is None or ans.get("default") in (None, ""):
            continue
        labels = [(c["t"], _label(ans.get("crops", {}).get(str(c["k"]), ans["default"]))) for c in meta["crops"]]
        for o in tr["obs"]:
            t = o["t"]
            prev = [lab for ct, lab in labels if ct <= t + 1e-6]
            nxt = [lab for ct, lab in labels if ct >= t - 1e-6]
            if prev and nxt:
                a, b = prev[-1], nxt[0]
                lab, status = (a, None) if a == b else (None, "switch_zone")
            else:
                lab, status = (prev[-1] if prev else nxt[0]), None
            if status is None:
                status = "not_raja" if lab == "not_raja" else ("unknown" if lab is None else "ok")
            rows.append({"t": t, "track": tid, "box": o["box"], "jersey": lab if isinstance(lab, int) else None,
                         "status": status, "set": meta["set"]})
    return rows


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
