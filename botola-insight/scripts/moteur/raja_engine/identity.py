"""Identités : preuves, propagation prudente, conflits et questions à poser.

Règles :
- une réponse humaine sur une image identifie le joueur À CET INSTANT, puis se propage
  uniquement le long du même tronçon de piste continu (sans croisement avec un coéquipier
  ni trou de suivi > 0,6 s) ; elle ne valide jamais toute une trajectoire ni d'autres plans ;
- un numéro lu automatiquement doit être lu au moins deux fois sur le même tronçon ;
- un gros plan (ex. Barkok avant son entrée) prouve une apparition à l'écran, pas une
  présence sur le terrain : les statistiques de terrain n'utilisent que les plans larges/moyens ;
- la réponse humaine prime toujours ; une contradiction est consignée, pas écrasée.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .questions import task_id
from .tracker import iou_matrix
from .tracks import box_at
from .util import fmt_clock
from .videoio import save_frame

RELIABILITY = {
    "human": "élevée",
    "claude_review": "moyenne",  # numéro lu par Claude sur les images agrandies
    "claude_review_propagated": "moyenne",
    "human_propagated": "moyenne",
    "auto_number": "moyenne",
    "auto_role_gk": "moyenne",
    "auto_banner": "élevée",
}


def track_segments(tr: dict, gap: float = 0.6) -> list[tuple[float, float]]:
    times = [o["t"] for o in tr["obs"]]
    cuts = set(tr.get("risk_times", []))
    segs, start, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > gap or prev in cuts:
            segs.append((start, prev))
            start = t
        prev = t
    segs.append((start, prev))
    return segs


def _segment_containing(tr, t):
    for a, b in track_segments(tr):
        if a - 1e-6 <= t <= b + 1e-6:
            return a, b
    return None


def _overlap_score(ans_box, det_box):
    a = np.array([ans_box], np.float32)
    d = np.array([det_box], np.float32)
    iou = float(iou_matrix(a, d)[0, 0])
    x1, y1 = max(ans_box[0], det_box[0]), max(ans_box[1], det_box[1])
    x2, y2 = min(ans_box[2], det_box[2]), min(ans_box[3], det_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    contain = inter / max(1e-6, (det_box[2] - det_box[0]) * (det_box[3] - det_box[1]))
    return max(iou, 0.8 * contain if iou > 0.1 else 0.0)


def resolve(tracks: dict, store, roster, video_fp: str, W: int, H: int, team_labels_confirmed: bool, timeline=None) -> dict:
    """Renvoie {'segments': [...], 'anchors': [...], 'unmatched_answers': [...], 'conflicts': [...]}."""
    anchors, unmatched, number_rejected = [], [], []
    # 1) réponses humaines (identity / roster) de ce fichier
    # une question « groupée » porte plusieurs ancrages (même joueur sur plusieurs pistes) :
    # chaque ancrage est traité comme une réponse à part entière
    todo = []
    # une question de revérification (« replaces ») annule, dès qu'elle a reçu une réponse (même
    # « je ne sais pas »), les réponses qu'elle conteste, y compris leurs copies reportées d'un autre
    # enregistrement ; tant qu'elle est en attente, l'ancienne réponse reste en vigueur
    replaced = set()
    for t in store.tasks.values():
        if t.get("replaces") and t["status"] in ("resolved", "unknown"):
            replaced.update(t["replaces"])
    for t in store.resolved():
        if t["kind"] not in ("identity",) or t.get("video") not in (None, video_fp):
            continue
        if t["id"] in replaced or t.get("transferred_from") in replaced:
            continue
        if t.get("anchors"):
            todo += [(t, a["time"], a["bbox"]) for a in t["anchors"] if a.get("bbox")]
        elif t.get("bbox") is not None:
            todo.append((t, t["time"], t["bbox"]))
    for t, t_time, t_bbox in todo:
        resp = t.get("response") or {}
        val = resp.get("value")
        ab = [t_bbox[0] * W, t_bbox[1] * H, t_bbox[2] * W, t_bbox[3] * H]
        best = None
        for tr in tracks.values():
            o = box_at(tr, t_time, tol=0.25)
            if o is None:
                continue
            s = _overlap_score(ab, o["box"])
            if s >= 0.3 and (best is None or s > best[0]):
                best = (s, tr, o)
        player = roster.from_answer_value(val)
        rec = {"task_id": t["id"], "t": t_time, "value": val, "player": None if player is None else player["jersey_number"],
               "player_name": None if player is None else player["player_name"],
               "source": "claude_review" if str(resp.get("origin", "")).startswith("claude") else "human",
               "origin": t.get("origin", "engine"), "scope": resp.get("scope", "observation_only")}
        if best is None:
            rec["note"] = "aucune piste détectée ne correspond au cadre à cet instant (gros plan, ralenti ou détection manquée)"
            unmatched.append(rec)
            continue
        rec.update(track_id=best[1]["id"], match_score=round(best[0], 3), t_track=best[2]["t"])
        if val in ("opponent", "referee", "staff"):
            # pas un joueur du Raja : on corrige l'équipe de la piste, sans créer de segment nominatif
            best[1]["team_override"] = {"opponent": "opponent", "referee": "referee", "staff": "other"}[val]
            continue
        if rec["player"] is None:
            rec["note"] = "réponse sans joueur du Raja reconnu"
            unmatched.append(rec)
            continue
        anchors.append(rec)

    # 2) numéros lus automatiquement (≥ 2 lectures concordantes sur le même tronçon)
    for tr in tracks.values():
        if tr.get("team") not in ("raja", "raja_gk") or not tr["numbers"]:
            continue
        by_seg = defaultdict(list)
        for r in tr["numbers"]:
            seg = _segment_containing(tr, r["t"])
            if seg and r["conf"] >= 0.6:
                by_seg[seg].append(r)
        for seg, reads in by_seg.items():
            c = Counter(r["number"] for r in reads)
            n, k = c.most_common(1)[0]
            p = roster.by_number.get(n)
            if k >= 2 and p is not None and k / len(reads) >= 0.66:
                # garde-fous : un « 4 » peut être un « 14 » partiellement masqué ; un joueur non
                # entré en jeu ne peut pas être sur le terrain. Dans ces cas : question, pas d'attribution.
                problems = []
                others = [q["jersey_number"] for q in roster.players if q["jersey_number"] != n and str(n) in str(q["jersey_number"])]
                if others:
                    problems.append(f"numéro {n} lu, mais {', '.join(map(str, others))} contient ce chiffre (masquage partiel possible)")
                a_s, b_s = roster.declared_window(p)
                if a_s is None:
                    problems.append(f"{p['player_name']} n'est pas entré en jeu selon la composition déclarée")
                elif timeline is not None:
                    c = timeline.at(reads[0]["t"]).get("clock_s")
                    if c is not None and not (a_s - 120 <= c <= b_s + 120):
                        problems.append(f"chrono hors de la période de jeu déclarée de {p['player_name']}")
                rec = {"task_id": None, "t": reads[0]["t"], "player": n, "player_name": p["player_name"],
                       "source": "auto_number", "track_id": tr["id"], "reads": k, "scope": "observation_only"}
                if problems:
                    rec["rejected_reasons"] = problems
                    number_rejected.append(rec)
                else:
                    anchors.append(rec)

    # 3) gardien du Raja : un seul gardien Raja déclaré sur le terrain
    gks = [p for p in roster.players if p.get("role") == "G" and p.get("starter")]
    gk_subs = [p for p in roster.players if p.get("role") == "G" and not p.get("starter") and p.get("entered_minute_declared")]
    if team_labels_confirmed and len(gks) == 1 and not gk_subs:
        for tr in tracks.values():
            if tr.get("team") == "raja_gk" and tr.get("team_conf", 0) >= 0.7 and "team_override" not in tr:
                anchors.append({"task_id": None, "t": tr["t_start"], "player": gks[0]["jersey_number"],
                                "player_name": gks[0]["player_name"], "source": "auto_role_gk", "track_id": tr["id"],
                                "whole_track": True, "scope": "observation_only"})

    # 4) segments identifiés
    segments = []
    for a in anchors:
        tr = tracks.get(a["track_id"])
        if tr is None:
            continue
        if a.get("whole_track"):
            win = (tr["t_start"], tr["t_end"])
        else:
            ta = a.get("t_track", a["t"])  # instant de l'observation de piste la plus proche de la réponse
            win = _segment_containing(tr, ta) or (ta, ta)
            if tr.get("team_conf") is not None and tr.get("team_conf", 1.0) < 0.8 and tr["views"].get("wide", 0) >= 10:
                # couleur de maillot instable le long de la piste : échange probable entre deux
                # joueurs d'équipes différentes → aucune propagation au-delà de l'instant répondu
                win = (ta, ta)
        src = a["source"]
        if src == "human" and win[1] - win[0] > 0.5:
            rel_src = "human_propagated"
        elif src == "claude_review" and win[1] - win[0] > 0.5:
            rel_src = "claude_review_propagated"
        else:
            rel_src = src
        segments.append({"track_id": tr["id"], "shot": tr["shot"], "player": a["player"], "player_name": a["player_name"],
                         "t_from": win[0], "t_to": win[1], "anchor_t": a["t"], "source": src,
                         "reliability": RELIABILITY.get(rel_src, "faible"), "task_id": a.get("task_id"),
                         "views": tr["views"]})

    # 5) conflits : même piste → deux joueurs ; même joueur → deux pistes simultanées
    conflicts = []
    by_track = defaultdict(list)
    for s in segments:
        by_track[s["track_id"]].append(s)
    for tid, ss in by_track.items():
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                a, b = ss[i], ss[j]
                if a["player"] != b["player"] and min(a["t_to"], b["t_to"]) >= max(a["t_from"], b["t_from"]):
                    # la réponse humaine l'emporte sur l'automatique ; deux humains → conflit
                    if a["source"] == "human" and b["source"] != "human":
                        b["overridden_by_human"] = True
                    elif b["source"] == "human" and a["source"] != "human":
                        a["overridden_by_human"] = True
                    else:
                        conflicts.append({"type": "track_two_players", "track_id": tid, "a": a, "b": b})
    by_player = defaultdict(list)
    for s in segments:
        if not s.get("overridden_by_human") and s["player"] is not None:
            by_player[s["player"]].append(s)
    for p, ss in by_player.items():
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                a, b = ss[i], ss[j]
                if a["track_id"] != b["track_id"] and a["shot"] == b["shot"] and \
                        min(a["t_to"], b["t_to"]) - max(a["t_from"], b["t_from"]) > 0.3:
                    conflicts.append({"type": "player_two_tracks", "player": p, "a": a, "b": b})
    segments = [s for s in segments if not s.get("overridden_by_human")]
    return {"segments": segments, "anchors": anchors, "unmatched_answers": unmatched, "conflicts": conflicts,
            "number_rejected": number_rejected}


def propose_questions(tracks: dict, resolved: dict, store, roster, video, timeline, video_fp: str, period: int,
                      evidence_dir: Path, budget: int = 40, min_seg_s: float = 3.0, ball_prox: dict | None = None) -> list[dict]:
    """Choisit les tronçons Raja non identifiés les plus utiles et crée les demandes."""
    covered = defaultdict(list)
    for s in resolved["segments"]:
        covered[s["track_id"]].append((s["t_from"], s["t_to"]))
    cands = []
    for tr in tracks.values():
        team = tr.get("team_override") or tr.get("team")
        if team != "raja" or tr.get("team_conf", 0) < 0.6:
            continue
        if tr["views"].get("wide", 0) + tr["views"].get("medium", 0) < 0.6 * tr["n_obs"]:
            continue
        for a, b in track_segments(tr):
            if b - a < min_seg_s:
                continue
            if any(x <= a + 0.5 and y >= b - 0.5 for x, y in covered[tr["id"]]):
                continue
            obs = [o for o in tr["obs"] if a <= o["t"] <= b]
            # image la plus lisible : joueur le plus grand, au centre du tronçon de préférence
            o = max(obs, key=lambda o: (o["box"][3] - o["box"][1]) - 0.5 * abs(o["t"] - (a + b) / 2))
            prox = (ball_prox or {}).get(tr["id"], 0.0)
            cands.append({"tr": tr, "a": a, "b": b, "o": o, "value": (b - a) * (1 + 3 * prox)})
    cands.sort(key=lambda c: -c["value"])
    pending = len([t for t in store.pending(kinds={"identity"}) if t.get("video") == video_fp])
    room = max(0, budget - pending)
    created = []
    evidence_dir = Path(evidence_dir)
    for c in cands[:room]:
        tr, o = c["tr"], c["o"]
        Wd, Hd = o["W"] or video.width, o["Himg"] or video.height
        x1, y1, x2, y2 = o["box"]
        bbox = [max(0, x1 / Wd), max(0, y1 / Hd), min(1, x2 / Wd), min(1, y2 / Hd)]
        tid = task_id(video_fp, "identity", o["t"], bbox)
        if tid in store.tasks:
            continue
        st = timeline.at(o["t"])
        frame = video.frame_at(o["t"])
        ev = {}
        if frame is not None:
            img = evidence_dir / f"{tid}.jpg"
            save_frame(frame, str(img), [o["box"]], ["?"])
            pad = 0.6 * (y2 - y1)
            cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad * 0.5))
            cx2, cy2 = int(min(frame.shape[1], x2 + pad)), int(min(frame.shape[0], y2 + pad * 0.5))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size:
                crop = cv2.resize(crop, None, fx=max(1.0, 240 / crop.shape[0]), fy=max(1.0, 240 / crop.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(evidence_dir / f"{tid}_crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            ev = {"image": f"evidence/{tid}.jpg", "crop": f"evidence/{tid}_crop.jpg",
                  "clip_window": [max(0.0, o["t"] - 2.0), 4.0]}
        task = {"id": tid, "kind": "identity", "video": video_fp, "period": period, "time": round(o["t"], 3),
                "end": round(o["t"] + 4, 3), "bbox": [round(v, 5) for v in bbox], "track_id": tr["id"],
                "match_clock": st.get("clock"), "clock_status": "observed_engine" if st.get("live") else "unverified",
                "reason": (f"Joueur du Raja suivi {c['b'] - c['a']:.1f} s sans croisement ; numéro non lu. "
                           "La réponse vaudra pour ce tronçon de suivi seulement."),
                "question": "Qui est le joueur encadré ?", "choices": roster.choices(st.get("clock_s")), "blocking": False,
                "priority": round(c["value"], 2), "evidence": ev, "origin": "engine",
                "segment": [round(c["a"], 3), round(c["b"], 3)]}
        t, status = store.add(task)
        if status == "created":
            created.append(t)
    store.save()
    return created
