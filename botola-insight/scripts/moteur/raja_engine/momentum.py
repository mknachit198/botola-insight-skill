"""Possession estimée minute par minute, à partir de la vidéo (ballon + joueurs + équipes).

Méthode (estimation, pas une possession officielle) :
- image par image (5 i/s, plans larges et moyens, direct uniquement) : le ballon détecté est
  « contrôlé » par le joueur dont les pieds sont le plus proches, si la distance est inférieure
  à 0,55 hauteur de joueur et qu'aucun adversaire n'est presque aussi proche (ballon disputé) ;
- l'équipe d'un joueur vient de la couleur de son maillot sur l'image ; si elle est ambiguë, de sa
  piste (vote ≥ 0,75) ; si les deux se contredisent, le joueur est ignoré ;
- la possession reste à la dernière équipe qui a contrôlé le ballon, au plus 6 s après le
  dernier contrôle vu ; au-delà, elle est « inconnue » (jamais attribuée par défaut) ;
- agrégation par minute de match (chrono du tableau), avec la couverture (secondes connues).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .teams import TeamModel
from .util import read_json

CONTROL_D = 0.55
CONTEST_MARGIN = 0.20
HOLD_S = 6.0


def _minute_label(st: dict) -> tuple[str, int, float] | None:
    """(libellé, période, clé de tri) de la minute de match ; None hors direct."""
    if not st.get("live") or st.get("clock_s") is None:
        return None
    per = st.get("period") or 1
    if st.get("added_s") is not None:
        base = int(st["clock_s"] // 60)
        n = int(max(0.0, st["added_s"]) // 60) + 1
        return f"{base}+{n}", per, base + n / 100.0
    m = int(st["clock_s"] // 60) + 1
    return str(m), per, float(m)


def controls(frames, tracks, team_model) -> list[dict]:
    out = []
    for fr in frames:
        if fr.get("view") not in ("wide", "medium") or not fr.get("b"):
            continue
        b = max(fr["b"], key=lambda x: x[4])
        if b[4] < 0.3:
            continue
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        cand = []
        for x1, y1, x2, y2, sc, tid, feat in fr["p"]:
            h = max(8.0, y2 - y1)
            team = None
            own = None
            if feat is not None and team_model is not None:
                c, margin = team_model.predict(np.array(feat, np.float32))
                lab = team_model.cluster_labels.get(c)
                if margin >= 0.25:
                    own = lab
            tr = tracks.get(str(tid)) if tid is not None else None
            trk = tr.get("team") if tr and (tr.get("team_conf") or 0) >= 0.75 else None
            if own in ("raja", "opponent") and trk in ("raja", "opponent") and own != trk:
                continue  # couleur de l'image et couleur de la piste en désaccord : joueur ignoré
            team = own if own in ("raja", "opponent") else (trk if trk in ("raja", "opponent") and own is None else None)
            if team is None:
                continue
            d = math.hypot(bx - (x1 + x2) / 2, by - y2) / h
            cand.append((d, team, tid))
        if not cand:
            continue
        cand.sort()
        d0, team0, tid0 = cand[0]
        if d0 > CONTROL_D:
            continue
        rival = next((c for c in cand[1:] if c[1] != team0), None)
        if rival is not None and rival[0] < d0 + CONTEST_MARGIN:
            continue
        out.append({"t": fr["t"], "team": team0, "track": tid0, "d": round(d0, 3)})
    return out


def possession_by_minute(frames, ctrl, timeline) -> dict:
    """{minute: {"raja": s, "opponent": s, "unknown": s, "period": p, "key": k}} sur les images de direct."""
    by_t = {round(c["t"], 3): c["team"] for c in ctrl}
    acc = defaultdict(lambda: {"raja": 0.0, "opponent": 0.0, "unknown": 0.0})
    meta = {}
    state, last = None, -1e9
    prev_t = None
    for fr in frames:
        t = fr["t"]
        dt = 0.2 if prev_t is None else min(0.6, max(0.0, t - prev_t))
        prev_t = t
        st = timeline.at(t)
        ml = _minute_label(st)
        if fr.get("cut") or st.get("kind") in ("no_scoreboard", "frozen", "discontinuity"):
            state = None if t - last > HOLD_S else state
        team = by_t.get(round(t, 3))
        if team is not None:
            state, last = team, t
        if ml is None:
            continue
        lab, per, key = ml
        meta[lab] = (per, key)
        if state is not None and t - last <= HOLD_S:
            acc[lab][state] += dt
        else:
            acc[lab]["unknown"] += dt
    out = {}
    for lab, v in acc.items():
        per, key = meta[lab]
        out[lab] = {**{k: round(x, 1) for k, x in v.items()}, "period": per, "key": key}
    return out


def session_possession(session) -> dict:
    frames = session._frames()
    tracks = (read_json(session.dir / "tracks.json", {}) or {}).get("tracks", {})
    tj = read_json(session.ws.root / "teams.json")
    tm = TeamModel.from_json(tj) if tj else None
    ctrl = controls(frames, tracks, tm)
    return {"controls": ctrl, "minutes": possession_by_minute(frames, ctrl, session.timeline())}


def _window(label: str) -> str:
    if "+" in label:
        return label.split("+")[0] + "+"
    m = int(label)
    a = (m - 1) // 5 * 5 + 1
    return f"{a}-{a + 4}"


def match_tables(ws, min_known_window_s: float = 30.0, min_known_rolling_s: float = 20.0) -> dict:
    """Tableaux du match : minute par minute (avec part lissée sur 5 minutes centrées, calculée sur
    les SECONDES connues et non en moyennant des pourcentages) et fenêtres de 5 minutes."""
    rows = []
    for s in ws.sessions():
        r = session_possession(s)
        for lab, v in r["minutes"].items():
            rows.append({"minute": lab, "periode": v["period"], "key": v["key"], "raja_s": v["raja"],
                         "zemamra_s": v["opponent"], "inconnu_s": v["unknown"]})
    rows.sort(key=lambda x: (x["periode"], x["key"]))
    for per in {x["periode"] for x in rows}:
        same = [x for x in rows if x["periode"] == per]
        for j, x in enumerate(same):
            w5 = same[max(0, j - 2): j + 3]
            R, O = sum(y["raja_s"] for y in w5), sum(y["zemamra_s"] for y in w5)
            x["part_raja_lissee_5min"] = round(R / (R + O), 3) if R + O >= min_known_rolling_s else None
            x["secondes_connues_5min"] = round(R + O, 1)
    windows = {}
    for x in rows:
        w = windows.setdefault((x["periode"], _window(x["minute"])), {"periode": x["periode"], "fenetre": _window(x["minute"]),
                                                                      "raja_s": 0.0, "zemamra_s": 0.0, "inconnu_s": 0.0, "key": x["key"]})
        for k in ("raja_s", "zemamra_s", "inconnu_s"):
            w[k] += x[k]
    wl = sorted(windows.values(), key=lambda w: (w["periode"], w["key"]))
    for w in wl:
        known = w["raja_s"] + w["zemamra_s"]
        w["part_raja"] = round(w["raja_s"] / known, 3) if known >= min_known_window_s else None
        w["secondes_connues"] = round(known, 1)
        w["secondes_direct"] = round(known + w["inconnu_s"], 1)
        for k in ("raja_s", "zemamra_s", "inconnu_s"):
            w[k] = round(w[k], 1)
        w.pop("key")
    for x in rows:
        x.pop("key")
    tot = {p: [sum(x["raja_s"] for x in rows if x["periode"] == p), sum(x["zemamra_s"] for x in rows if x["periode"] == p),
               sum(x["inconnu_s"] for x in rows if x["periode"] == p)] for p in sorted({x["periode"] for x in rows})}
    R = sum(v[0] for v in tot.values()); O = sum(v[1] for v in tot.values()); U = sum(v[2] for v in tot.values())
    summary = {"par_periode": {str(p): {"part_raja": round(v[0] / (v[0] + v[1]), 3) if v[0] + v[1] else None,
                                        "couverture": round((v[0] + v[1]) / max(1e-6, sum(v)), 3)} for p, v in tot.items()},
               "match": {"part_raja": round(R / (R + O), 3) if R + O else None, "couverture": round((R + O) / max(1e-6, R + O + U), 3)},
               "methode": __doc__.strip()}
    return {"minutes": rows, "fenetres": wl, "resume": summary}
