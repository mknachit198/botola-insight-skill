"""Enchaînement du service de lecture : planches par chaîne → lecteur → preuves.

1. build_planches : pour chaque chaîne de l'équipe visée (durée ≥ min_dur), n instants
   choisis par selection.pick, recadrés dans la vidéo originale, composés en une planche ;
   les métadonnées (piste, instant, image, boîte) sont écrites dans planches_meta.json.
2. Un lecteur (readers.py) produit pour chaque recadrage : numéro lu ou rien, partie lue
   (dos, short, devant), confiance.
3. to_evidence : chaque lecture devient une preuve number_read rattachée à (piste, instant),
   avec la provenance planche#cK. Aucune agrégation ici.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from ..util import read_json, write_json
from . import crops as crops_mod
from . import planche as planche_mod
from . import selection

CONF = {"high": 0.9, "medium": 0.7, "low": 0.5}


def build_planches(tracks, frames, video, out, team="raja", min_dur=5.0, n=12, only=None, prefix="p",
                   exclude=None, use_back=True, cols=6, meta_name="planches_meta.json"):
    out = Path(out)
    (out / "planches").mkdir(parents=True, exist_ok=True)
    fidx = selection.frame_index(frames)
    chosen, reqs = {}, []
    for tid in sorted(tracks):
        tr = tracks[tid]
        if (tr.get("team_override") or tr.get("team")) != team:
            continue
        if only is not None and tid not in only:
            continue
        if tr["t_end"] - tr["t_start"] < min_dur:
            continue
        picks = selection.pick(tr, fidx, n=n, exclude=(exclude or {}).get(tid, ()), use_back=use_back)
        chosen[tid] = picks
        for k, r in enumerate(picks):
            o = tr["obs"][r["i"]]
            f = fidx[(tr["shot"], o["t"])]
            reqs.append({"key": f"{tid}:{k}", "fi": f["fi"], "box": o["box"]})
    imgs = crops_mod.extract(video, reqs)
    meta = []
    for tid, picks in chosen.items():
        tr = tracks[tid]
        cr = [imgs.get(f"{tid}:{k}") for k in range(len(picks))]
        labels = [f"c{k + 1} t={r['t']:.1f}" for k, r in enumerate(picks)]
        name = f"{prefix}{tid:03d}.jpg"
        img = planche_mod.compose(cr, labels, f"piste {tid}  ({tr['t_start']:.1f}-{tr['t_end']:.1f} s)", cols=cols)
        cv2.imwrite(str(out / "planches" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        meta.append({"track": tid, "file": f"planches/{name}", "t_start": tr["t_start"], "t_end": tr["t_end"],
                     "crops": [{"c": k + 1, "t": r["t"], "fi": fidx[(tr["shot"], r["t"])]["fi"],
                                "box": [round(v, 1) for v in tr["obs"][r["i"]]["box"]],
                                "score": r["score"], "back": r["back"], "size": r["size"], "iso": r["iso"]}
                               for k, r in enumerate(picks)]})
    write_json(out / meta_name, {"selection": {"n": n, "min_dur_s": min_dur, "team": team, "use_back": use_back,
                                               "second_pass": bool(exclude),
                                                          "weights": {"back": selection.W_BACK,
                                                                      "size": selection.W_SIZE,
                                                                      "iso": selection.W_ISO,
                                                                      "border": selection.W_BORDER}},
                                            "planches": meta})
    return meta


def to_evidence(meta, readings, source, reader_id, holdout_tracks=()):
    """readings : {"<track>": {"<c>": {"number": 30, "part": "back", "conf": "high"} | null}}.
    Une lecture ne vaut que pour l'instant de son recadrage."""
    from ..evidence import make
    out = []
    by_track = {str(m["track"]): m for m in meta}
    for tid, per_crop in readings.items():
        m = by_track.get(str(tid))
        if m is None:
            raise KeyError(f"planche inconnue pour la piste {tid}")
        crops_by_c = {str(c["c"]): c for c in m["crops"]}
        for c, r in per_crop.items():
            if r is None or r.get("number") is None:
                continue
            crop = crops_by_c[str(c)]
            out.append(make("number_read", int(tid), crop["t"], int(r["number"]), CONF[r.get("conf", "medium")],
                            source, f"{m['file']}#c{c}", holdout=int(tid) in set(holdout_tracks),
                            part=r.get("part", "back"), reader=reader_id, box=crop["box"]))
    return out


def load_readings(path):
    data = read_json(path)
    return data["readings"], data.get("reader", "unknown")
