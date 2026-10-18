"""Page de questions publiable (artifact claude.ai) et réimport des réponses.

Quand l'utilisateur n'a pas accès à l'interface locale du moteur, les questions en attente sont
exportées dans une page HTML autonome (images intégrées) dont les réponses sont enregistrées
dans la base de l'artifact (`answers/<id question>`). `apply_answers` les réinjecte ensuite dans
la file du moteur, avec l'origine « page_de_questions » ; le journal en ajout seul les conserve.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .questions import KINDS
from .tracks import box_at, build_tracks, load_frames
from .util import now_iso

KIND_ORDER = ["team_colors", "clock_read", "banner_check", "conflict", "event_review", "identity", "calibration"]


def _jpeg(img, width=None, q=72):
    if width and img.shape[1] != width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _grid(frames, cols, w=320):
    tiles = []
    for f in frames:
        if f is None:
            continue
        tiles.append(cv2.resize(f, (w, int(f.shape[0] * w / f.shape[1]))))
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, img.shape[0] - 22), (min(img.shape[1], 12 + 9 * len(text)), img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text, (6, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _frames_at(video, times):
    out = []
    for t in times:
        f = video.frame_at(max(0.0, float(t)))
        out.append(f)
    return out


def _images_for(task, ws, sessions, tracks_cache):
    """Liste d'images (data URI) et légende pour une question."""
    ev = task.get("evidence") or {}
    kind = task["kind"]
    if kind == "team_colors":
        p = ws.root / (ev.get("image") or "")
        img = cv2.imread(str(p)) if p.is_file() else None
        return ([_jpeg(img, q=80)] if img is not None else []), "Échantillons de joueurs rangés dans ce groupe de couleur."
    if ev.get("v2_clip"):
        p = ws.root / ev["v2_clip"]
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = []
        for i in np.linspace(0, max(0, n - 2), 12).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            frames.append(_label(f, f"{i / fps:.1f} s") if ok else None)
        g = _grid(frames, 4, 300)
        return ([_jpeg(g, 1200)] if g is not None else []), "12 images régulières de l'extrait V2 (dans l'ordre)."
    s = sessions.get(task.get("video"))
    if s is None:
        return [], "Vidéo source non chargée."
    video = s["video"]
    t = float(task.get("time") or 0.0)
    if kind == "calibration":
        f = video.frame_at(t)
        resp = task.get("response") or {}
        if f is not None and str(resp.get("origin", "")).startswith("claude") and resp.get("points"):
            from .pitch import fit_homography, model_lines
            fit = fit_homography(resp["points"], f.shape[1], f.shape[0])
            if fit.get("ok"):
                Hi = np.linalg.inv(np.array(fit["H"]))
                g = f.copy()
                for (x1, y1), (x2, y2) in model_lines():
                    pts = np.array([[x1 + (x2 - x1) * k / 60, y1 + (y2 - y1) * k / 60] for k in range(61)])
                    q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hi).reshape(-1, 2)
                    cv2.polylines(g, [np.round(q).astype(np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)
                ang = np.linspace(0, 2 * np.pi, 120)
                for cx in (0.0,):
                    q = cv2.perspectiveTransform(np.stack([cx + 9.15 * np.cos(ang), 9.15 * np.sin(ang)], 1).reshape(-1, 1, 2), Hi).reshape(-1, 2)
                    cv2.polylines(g, [np.round(q).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
                return [_jpeg(g, q=85)], "Lignes du terrain (jaune) placées par Claude : elles doivent coller aux lignes blanches."
        return ([_jpeg(f, q=85)] if f is not None else []), "Image du plan à calibrer (pleine résolution)."
    if kind == "identity" and task.get("anchors"):
        # question groupée : une ligne par séquence (image entière réduite + 3 agrandissements)
        trs = tracks_cache.get(s["dir"])
        if trs is None:
            trs = build_tracks(load_frames(Path(s["dir"]) / "frames.jsonl"))
            tracks_cache[s["dir"]] = trs
        imgs = []
        for k, an in enumerate(task["anchors"][:8]):
            tr = trs.get(an.get("track_id"))
            ta = float(an["time"])
            f = video.frame_at(ta)
            if f is None:
                continue
            H, W = f.shape[:2]
            g = f.copy()
            x1, y1, x2, y2 = an["bbox"]
            cv2.rectangle(g, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 255, 255), 3)
            g = cv2.resize(g, (480, int(H * 480 / W)), interpolation=cv2.INTER_AREA)
            crops = []
            for dt in (-0.8, 0.0, 0.8):
                o = box_at(tr, ta + dt, 0.3) if tr else None
                ff = f if dt == 0.0 else (video.frame_at(ta + dt) if o is not None else None)
                if ff is None:
                    continue
                bx1, by1, bx2, by2 = o["box"] if o is not None else (x1 * W, y1 * H, x2 * W, y2 * H)
                ph = by2 - by1
                cx = (bx1 + bx2) / 2
                cy1, cy2 = int(max(0, by1 - 0.25 * ph)), int(min(H, by2 + 0.1 * ph))
                half = max(int(0.45 * (cy2 - cy1)), 12)
                c = ff[cy1:cy2, int(max(0, cx - half)):int(min(W, cx + half))]
                if c.size:
                    crops.append(cv2.resize(c, (int(g.shape[0] * 0.75), g.shape[0]), interpolation=cv2.INTER_CUBIC))
            row = np.hstack([g] + crops)
            row = cv2.copyMakeBorder(row, 30, 6, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(row, f"Sequence {k + 1}  ({an.get('clock') or '?'})" + (f"  {an['note']}" if an.get("note") else ""),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            imgs.append(_jpeg(row, 960, 80))
        return imgs, (f"{len(task['anchors'])} séquences regroupées (une ligne chacune : image entière, cadre jaune, puis 3 "
                      "agrandissements à −0,8 s, 0 et +0,8 s). Une seule réponse vaut pour toutes. Si l'une d'elles montre "
                      "un autre joueur, choisissez « Illisible / impossible à dire » et indiquez son numéro de séquence dans la note.")
    if kind == "identity":
        f = video.frame_at(t)
        imgs = []
        if f is not None:
            g = f.copy()
            H, W = g.shape[:2]
            if task.get("bbox"):
                x1, y1, x2, y2 = task["bbox"]
                cv2.rectangle(g, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 255, 255), 2)
            imgs.append(_jpeg(g, 854, 75))
            # bande de 5 agrandissements du même joueur suivi (−1,6 s … +1,6 s)
            tr = None
            if task.get("track_id") is not None:
                trs = tracks_cache.get(s["dir"])
                if trs is None:
                    trs = build_tracks(load_frames(Path(s["dir"]) / "frames.jsonl"))
                    tracks_cache[s["dir"]] = trs
                tr = trs.get(task["track_id"])
            crops = []
            for dt in (-1.6, -0.8, 0.0, 0.8, 1.6):
                o = box_at(tr, t + dt, 0.3) if tr else None
                if tr is not None and o is None:
                    continue
                ff = video.frame_at(t + dt) if dt else f
                if ff is None:
                    continue
                if o is not None:
                    bx1, by1, bx2, by2 = o["box"]
                else:
                    bx1, by1, bx2, by2 = task["bbox"][0] * W, task["bbox"][1] * H, task["bbox"][2] * W, task["bbox"][3] * H
                ph = (by2 - by1)
                cx = (bx1 + bx2) / 2
                cy1, cy2 = int(max(0, by1 - 0.35 * ph)), int(min(H, by2 + 0.15 * ph))
                half = max(int(0.55 * (cy2 - cy1)), 12)
                cx1, cx2 = int(max(0, cx - half)), int(min(W, cx + half))
                c = ff[cy1:cy2, cx1:cx2]
                if c.size:
                    c = cv2.resize(c, (180, 240), interpolation=cv2.INTER_CUBIC)
                    crops.append(_label(c, f"{dt:+.1f} s"))
            if crops:
                imgs.append(_jpeg(np.hstack(crops), q=78))
        return imgs, "Image entière (cadre jaune), puis le même joueur suivi de −1,6 s à +1,6 s."
    if kind == "conflict" and (task.get("conflict") or {}).get("type") == "player_two_tracks":
        # deux pistes attribuées au même joueur au même moment : A en jaune, B en cyan, et agrandissements
        c = task["conflict"]
        trs = tracks_cache.get(s["dir"])
        if trs is None:
            trs = build_tracks(load_frames(Path(s["dir"]) / "frames.jsonl"))
            tracks_cache[s["dir"]] = trs
        ta, tb = trs.get(c["a"]["track_id"]), trs.get(c["b"]["track_id"])
        t0, t1 = max(c["a"]["t_from"], c["b"]["t_from"]), min(c["a"]["t_to"], c["b"]["t_to"])
        times = list(np.linspace(t0 + 0.2, max(t0 + 0.4, t1 - 0.2), 6))
        frames, crops = [], {"A": [], "B": []}
        for f, tt in zip(_frames_at(video, times), times):
            if f is None:
                continue
            g = f.copy()
            for lab, tr, col in (("A", ta, (0, 255, 255)), ("B", tb, (255, 255, 0))):
                o = box_at(tr, tt, 0.3) if tr else None
                if o is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in o["box"]]
                cv2.rectangle(g, (x1, y1), (x2, y2), col, 2)
                cv2.putText(g, lab, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                ph = y2 - y1
                cc = f[max(0, int(y1 - 0.2 * ph)):int(y2 + 0.1 * ph), max(0, int((x1 + x2) / 2 - 0.6 * ph)):int((x1 + x2) / 2 + 0.6 * ph)]
                if cc.size:
                    crops[lab].append(_label(cv2.resize(cc, (150, 200), interpolation=cv2.INTER_CUBIC), f"{lab} {tt:.1f} s"))
            frames.append(_label(g, f"{tt:.1f} s"))
        imgs = []
        g = _grid(frames[:4], 2, 427)
        if g is not None:
            imgs.append(_jpeg(g, 854, 78))
        for lab in ("A", "B"):
            if crops[lab]:
                imgs.append(_jpeg(np.hstack(crops[lab][:6]), q=80))
        return imgs, ("Les deux joueurs ont reçu le même nom au même moment : A (cadre jaune) et B (cadre cyan). "
                      "Au moins une des deux réponses est fausse. Agrandissements de A, puis de B.")
    if kind in ("event_review", "banner_check", "conflict", "clock_read"):
        end = float(task.get("end") or t + 6)
        if kind == "event_review":
            times = list(np.linspace(max(0, t - 6), t - 0.5, 3)) + list(np.linspace(t + 0.3, max(t + 0.4, end - 0.3), 9))
            cap_txt = "3 images du direct avant le passage, puis 9 images du passage (ralenti probable)."
        elif kind == "banner_check":
            times = list(np.linspace(t, max(t + 0.5, end), 8))
            cap_txt = "Images du bandeau, dans l'ordre."
        else:
            times = list(np.linspace(max(0, t - 2), t + 3, 8))
            cap_txt = "Images autour de l'instant concerné."
        frames = [_label(f, f"{tt:.1f} s") if f is not None else None for f, tt in zip(_frames_at(video, times), times)]
        g = _grid(frames, 4, 300)
        imgs = [_jpeg(g, 1200)] if g is not None else []
        return imgs, cap_txt
    return [], ""


def _landmarks_payload():
    from .pitch import LANDMARK_LABELS, landmarks
    # seuls les repères à cliquer (les points de calcul sans libellé ne sont pas proposés)
    return [{"value": k, "label": LANDMARK_LABELS[k], "x": round(x, 2), "y": round(y, 2)}
            for k, (x, y) in landmarks().items() if k in LANDMARK_LABELS]


def collect(ws, max_bytes: int = 12_000_000, include_resolved: bool = False, kinds=None) -> list[dict]:
    from .videoio import VideoSource
    sessions = {}
    for s in ws.sessions(include_superseded=True):
        try:
            sessions[s.meta["identity"]["fingerprint"]] = {"video": VideoSource(s.meta["video_path"]), "dir": str(s.dir),
                                                           "period": s.meta["period"]}
        except Exception:
            continue
    def _by_claude(t):
        return t["status"] == "resolved" and str((t.get("response") or {}).get("origin", "")).startswith("claude")
    tasks = [t for t in ws.store.tasks.values() if (include_resolved or t["status"] == "pending" or _by_claude(t))
             and t["kind"] in KIND_ORDER and (kinds is None or t["kind"] in kinds)
             and t.get("origin") != "imported_corrections"]
    tasks.sort(key=lambda x: (KIND_ORDER.index(x["kind"]), not x.get("blocking"), -(x.get("priority") or 0),
                              x.get("period") or 0, x.get("time") or 0))
    out, total, cache = [], 0, {}
    from .util import read_json
    assets_map = read_json(ws.root / "page_assets.json", {})
    pos_maps = read_json(ws.root / "position_maps.json", {})
    for t in tasks:
        imgs, caption = _images_for(t, ws, sessions, cache)
        fallback = []
        if t.get("refines") and not assets_map.get(t["id"]) and assets_map.get(t["refines"]):
            assets_map[t["id"]] = assets_map[t["refines"]]  # une précision réutilise l'extrait de la question d'origine
        if assets_map.get(t["id"]) and t["kind"] == "event_review":
            # l'extrait vidéo remplace la planche ; une seule image est gardée, montrée seulement si la vidéo ne se lit pas
            fallback, imgs = imgs[:1], []
        size = sum(len(i) for i in imgs + fallback)
        if total + size > max_bytes:
            continue
        total += size
        choices = t.get("choices") or []
        if t["kind"] == "conflict" and (t.get("conflict") or {}).get("type") == "player_two_tracks":
            nm = t["conflict"]["a"].get("player_name")
            choices = [{"value": "a", "label": f"A (cadre jaune) est {nm}"}, {"value": "b", "label": f"B (cadre cyan) est {nm}"},
                       {"value": "none", "label": f"Aucun des deux n'est {nm}"}]
        if t["kind"] in ("identity", "roster") and not choices:
            choices = ws.roster.choices()
        video = assets_map.get(t["id"])
        if video and t["kind"] == "event_review":
            caption ="Extrait : 6 s de direct avant le passage, puis le passage lui-même (son conservé)."
        elif video and t["kind"] == "banner_check":
            caption = "Extrait du bandeau (2 s avant, 3 s après). En dessous, les images du bandeau dans l'ordre."
        elif video and t["kind"] == "identity":
            caption = "Extrait de 4 s : le cadre jaune suit le joueur. En dessous, l'image fixe et 5 agrandissements."
        out.append({"id": t["id"], "kind": t["kind"], "kindLabel": KINDS.get(t["kind"], t["kind"]), "video": video,
                    "blocking": bool(t.get("blocking")), "question": t.get("question") or t.get("reason"),
                    "reason": t.get("reason"), "clock": t.get("match_clock"), "clockStatus": t.get("clock_status"),
                    "period": t.get("period"), "fileTime": t.get("time"), "segment": t.get("segment"),
                    "choices": [{"value": c.get("value"), "label": c.get("label")} for c in choices],
                    "playerChoice": bool(t.get("player_choice")), "proposed": t.get("proposed"),
                    "landmarks": _landmarks_payload() if t["kind"] == "calibration" else None,
                    "positionMap": pos_maps.get(t["id"]) if t["kind"] == "identity" else None,
                    "images": imgs, "fallback": fallback, "caption": caption,
                    "claudeAnswer": None if not _by_claude(t) else {"label": t["response"].get("label") or t["response"].get("value"),
                                                                     "note": t["response"].get("note") or "",
                                                                     "confidence": t["response"].get("confidence") or ""},
                    "engineAnswer": None if not t.get("response") or _by_claude(t) else {"label": t["response"].get("label") or t["response"].get("value"),
                                                                         "status": t["status"]}})
    for s in sessions.values():
        s["video"].close()
    return out


def apply_answers(ws, rows: list[dict]) -> dict:
    """rows : documents de la base de la page ({id, value, label, player, outcome, note, status, points, answeredAt})."""
    applied = skipped = unknown_ids = 0
    for r in rows:
        tid = r.get("id")
        t = ws.store.tasks.get(tid)
        if t is None:
            unknown_ids += 1
            continue
        prev = t.get("response") or {}
        if prev.get("page_answered_at") == r.get("answeredAt"):
            skipped += 1
            continue
        status = r.get("status") if r.get("status") in ("resolved", "unknown", "skipped") else "resolved"
        extra = {k: r[k] for k in ("player", "outcome", "points") if r.get(k) not in (None, "", [])}
        extra["page_answered_at"] = r.get("answeredAt")
        ws.store.answer(tid, r.get("value"), label=r.get("label") or "", note=r.get("note") or "", status=status,
                        extra=extra, origin="human_page_de_questions")
        applied += 1
    return {"applied": applied, "already_applied": skipped, "unknown_ids": unknown_ids}


def render_html(ws, questions: list[dict], roster_choices: list[dict], title: str) -> str:
    data = {"match": ws.cfg.get("title") or ws.cfg["match_key"], "generatedAt": now_iso(), "questions": questions,
            "roster": [c for c in roster_choices if c.get("jersey_number") is not None]}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)


TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
<style>
/* Mise en page : barre d'état fixe (progression), puis une question à la fois — preuve à gauche, réponse à droite. */
:root{--bg:#f3f6f3;--panel:#ffffff;--ink:#14231b;--muted:#5a6b61;--line:#d6e0d9;--accent:#0a6b39;--accent-ink:#ffffff;--warn:#a35a00;--warn-bg:#fff1dc;--done:#0a6b39;--done-bg:#e3f2e8;
--display:"Barlow Condensed","Arial Narrow",sans-serif;--body:"Source Sans 3","Segoe UI",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e1612;--panel:#16201a;--ink:#e6efe8;--muted:#9bb0a3;--line:#2a3830;--accent:#3cc47a;--accent-ink:#06210f;--warn:#f0b35a;--warn-bg:#3a2a12;--done:#5fd795;--done-bg:#173022;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e1612;--panel:#16201a;--ink:#e6efe8;--muted:#9bb0a3;--line:#2a3830;--accent:#3cc47a;--accent-ink:#06210f;--warn:#f0b35a;--warn-bg:#3a2a12;--done:#5fd795;--done-bg:#173022;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:16px/1.45 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 40px}
.wrap{max-width:1180px;margin:0 auto}
header{position:sticky;top:env(safe-area-inset-top,0px);z-index:3;background:var(--bg);padding-block:12px 10px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;justify-content:space-between}
h1{font:700 26px/1.05 var(--display);letter-spacing:.01em;margin:0;text-wrap:balance}
.sub{color:var(--muted);font-size:13px}
.meter{display:flex;align-items:center;gap:10px;min-width:220px;flex:1;max-width:420px}
.bar{flex:1;height:8px;border-radius:4px;background:var(--line);overflow:hidden}.bar>i{display:block;height:100%;width:0;background:var(--accent);transition:width .3s}
.count{font:400 13px var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 10px}
.posbox{margin:10px 0;display:grid;gap:4px}
.posmap{width:100%;max-width:420px;height:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
.pm-field{fill:#2f7d3a;stroke:#e9f2ea;stroke-width:.5}
.pm-line{fill:none;stroke:#e9f2ea;stroke-width:.5}
.pm-raja{fill:#ffffff;stroke:#0a6b39;stroke-width:.8}
.pm-opp{fill:#3aa0e0;stroke:#0b3a5c;stroke-width:.6}
.pm-target{fill:#ffd400;stroke:#000;stroke-width:.8}
.pm-txt{font-size:3.2px;fill:var(--muted)}
.claude{border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:10px 12px;margin:10px 0;display:grid;gap:4px;background:var(--panel)}
.claude .hint{color:var(--muted);font-size:13px}
.chip{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:5px 12px;font:600 13px var(--body);cursor:pointer}
.chip[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.notice{background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:10px 12px;font-size:14px;margin:10px 0}
.q{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(0,1fr);gap:20px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
@media (max-width:860px){.q{grid-template-columns:1fr}}
.evidence{display:flex;flex-direction:column;gap:10px;min-width:0}.evidence img,.evidence video{display:block;width:100%;height:auto;border-radius:6px;border:1px solid var(--line);background:#000}
.cap{font-size:13px;color:var(--muted)}
.side{min-width:0;display:flex;flex-direction:column;gap:12px}
.kind{font:700 13px var(--display);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.kind .blk{color:var(--warn);margin-left:8px}
h2{font:700 24px/1.1 var(--display);margin:0;text-wrap:balance}
dl{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;margin:0;font-size:14px}dt{color:var(--muted)}dd{margin:0;min-width:0}
.mono{font-family:var(--mono);font-size:13px}
.choices{display:flex;flex-wrap:wrap;gap:6px}
.choices button{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:7px 10px;font:15px var(--body);cursor:pointer;text-align:left}
.choices button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--muted)}
select,input,textarea{font:15px var(--body);color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:7px 9px}
.actions{display:flex;flex-wrap:wrap;gap:8px}
.btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:9px 14px;font:600 15px var(--body);cursor:pointer}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.btn:disabled{opacity:.5;cursor:not-allowed}
button:focus-visible,select:focus-visible,input:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.nav{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 10px}
.status{border-radius:6px;padding:8px 10px;font-size:14px}.status.ok{background:var(--done-bg);color:var(--done)}
.calib{position:relative}.calib canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
ol.pts{margin:0;padding-left:20px;font-size:13px}
svg.pitch{width:100%;max-width:420px;height:auto;background:var(--accent);border-radius:6px}svg.pitch rect,svg.pitch line,svg.pitch circle:not(.lm){fill:none;stroke:#ffffff;stroke-width:.4;opacity:.85}
svg.pitch text{fill:#ffffff;font:2.6px var(--body)}svg.pitch circle.lm{fill:#ffffff;stroke:#0b2a17;stroke-width:.4;cursor:pointer}svg.pitch circle.lm.used{fill:#ffd400}svg.pitch circle.lm.cur{fill:#ff5a3c;r:2.6}
.empty{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:24px;font-size:16px}
@media (prefers-reduced-motion:reduce){.bar>i{transition:none}}
</style>

<div class="wrap">
<header>
  <div><h1 id="title"></h1><div class="sub" id="sub"></div></div>
  <div class="meter"><div class="bar"><i id="barfill"></i></div><span class="count" id="count">—</span></div>
</header>
<div id="dbnote" class="notice" hidden>Les réponses ne peuvent pas être enregistrées dans cette vue (connectez-vous à claude.ai et ouvrez la page depuis votre compte).</div>
<div class="filters" id="filters" role="group" aria-label="Filtrer les questions"></div>
<div class="nav"><button class="btn" id="prev" type="button">◀ Précédente</button><span class="count" id="pos"></span><button class="btn" id="next" type="button">Suivante ▶</button></div>
<main id="main"></main>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
(function(){
'use strict';
const D=JSON.parse(document.getElementById('data').textContent);
const Q=D.questions;
const answers={};
let db=null, view='todo', kindFilter='', idx=0, saving=false;
const $=s=>document.querySelector(s);
const el=(tag,attrs={},...kids)=>{const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs)){if(k==='class')e.className=v;else if(k.startsWith('on'))e.addEventListener(k.slice(2),v);else if(v!==undefined&&v!==null)e.setAttribute(k,v);}for(const k of kids){if(k==null)continue;e.append(k.nodeType?k:document.createTextNode(String(k)));}return e;};
$('#title').textContent='Questions — '+D.match;
$('#sub').textContent=Q.length+' questions préparées par le moteur · '+new Date(D.generatedAt).toLocaleString('fr-FR');
const kinds=[...new Set(Q.map(q=>q.kind))];
const mine=q=>!!answers[q.id];
const byClaude=q=>!!q.claudeAnswer&&!answers[q.id];
const isDone=q=>mine(q)||byClaude(q);
function filtered(){return Q.filter(q=>(!kindFilter||q.kind===kindFilter)&&(view==='all'||(view==='todo'?!isDone(q):view==='claude'?byClaude(q):mine(q))));}
function renderFilters(){
  const f=$('#filters');f.replaceChildren();
  for(const[v,l]of[['todo','À répondre ('+Q.filter(q=>!isDone(q)).length+')'],['claude','Réponses de Claude ('+Q.filter(byClaude).length+')'],['done','Mes réponses'],['all','Toutes']])f.append(el('button',{class:'chip',type:'button','aria-pressed':String(view===v),onclick:()=>{view=v;idx=0;render();}},l));
  f.append(el('span',{style:'width:12px'}));
  f.append(el('button',{class:'chip',type:'button','aria-pressed':String(!kindFilter),onclick:()=>{kindFilter='';idx=0;render();}},'Tous types'));
  for(const k of kinds){const lab=(Q.find(q=>q.kind===k)||{}).kindLabel||k;f.append(el('button',{class:'chip',type:'button','aria-pressed':String(kindFilter===k),onclick:()=>{kindFilter=k;idx=0;render();}},lab+' ('+Q.filter(q=>q.kind===k).length+')'));}
}
function renderMeter(){const n=Q.filter(isDone).length;$('#barfill').style.width=(Q.length?100*n/Q.length:0)+'%';$('#count').textContent=n+' / '+Q.length+' répondues';}
function render(){
  renderFilters();renderMeter();
  const list=filtered();
  if(idx>=list.length)idx=Math.max(0,list.length-1);
  $('#pos').textContent=list.length?(idx+1)+' / '+list.length:'';
  $('#prev').disabled=idx<=0;$('#next').disabled=idx>=list.length-1;
  const M=$('#main');M.replaceChildren();
  if(!list.length){M.append(el('div',{class:'empty'},view==='todo'?'Toutes les questions de cette sélection ont une réponse (de vous ou de Claude). Prévenez Claude dans la conversation : il réinjecte vos réponses et relance le calcul.':'Aucune question dans cette sélection.'));return;}
  M.append(card(list[idx]));
}
function card(q){
  const a=answers[q.id];
  let sel=a&&a.value!=null?{value:a.value,label:a.label}:null;
  const points=a&&Array.isArray(a.points)?a.points.slice():[];
  const ev=el('div',{class:'evidence'});
  let curLm=null;
  if(q.kind==='calibration'&&q.images[0]){
    const box=el('div',{class:'calib'});const img=el('img',{src:q.images[0],alt:'Image du plan à calibrer'});const cv=el('canvas');box.append(img,cv);ev.append(box);
    const lmLabel=v=>(q.landmarks.find(l=>l.value===v)||{}).label||v;
    const draw=()=>{const r=img.getBoundingClientRect();cv.width=r.width;cv.height=r.height;const c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);c.font='bold 13px sans-serif';points.forEach((p,i)=>{c.fillStyle='#ffd400';c.strokeStyle='#000';c.lineWidth=2;c.beginPath();c.arc(p.x*cv.width,p.y*cv.height,6,0,7);c.fill();c.stroke();c.fillStyle='#000';c.fillText(String(i+1),p.x*cv.width+8,p.y*cv.height-8);});const ol=document.getElementById('pts');if(ol){ol.replaceChildren(...points.map((p,i)=>el('li',{},lmLabel(p.landmark))));}card._svg&&card._svg();};
    img.addEventListener('load',draw);if(img.complete)setTimeout(draw,0);
    cv.addEventListener('click',e=>{if(!curLm){const st=document.getElementById('lmstate');if(st){st.textContent='Choisissez d’abord un repère sur le schéma (point blanc), puis cliquez l’endroit correspondant sur l’image.';st.className='status';}return;}const r=cv.getBoundingClientRect();const i=points.findIndex(p=>p.landmark===curLm);const pt={landmark:curLm,x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height};if(i>=0)points[i]=pt;else points.push(pt);const st=document.getElementById('lmstate');if(st){st.className='status';st.textContent='Point placé : '+lmLabel(curLm)+'. Choisissez le repère suivant sur le schéma ('+new Set(points.map(p=>p.landmark)).size+' placé(s), 4 minimum).';}curLm=null;draw();});
    card._draw=draw;
    card._pick=v=>{curLm=v;const st=document.getElementById('lmstate');if(st){st.className='status ok';st.textContent='Repère choisi : '+lmLabel(v)+'. Cliquez maintenant cet endroit sur l’image.';}card._svg&&card._svg();};
  }else{
    if(q.video){const v=el('video',{src:q.video,controls:'',playsinline:'',preload:'metadata',loop:''});v.addEventListener('error',()=>{const fb=q.fallback||[];v.replaceWith(el('div',{class:'cap'},fb.length||q.images.length?'Extrait vidéo indisponible dans cette vue ; utilisez les images.':'Extrait vidéo indisponible dans cette vue : ouvrez la page depuis votre compte claude.ai.'),...fb.map(src=>el('img',{src,alt:'Image du passage'})));});ev.append(v);}
    for(const src of q.images)ev.append(el('img',{src,alt:'Preuve image'}));}
  if(q.caption)ev.append(el('div',{class:'cap'},q.caption));
  const side=el('div',{class:'side'});
  side.append(el('div',{class:'kind'},q.kindLabel,q.blocking?el('span',{class:'blk'},'· bloque le calcul'):null));
  side.append(el('h2',{},q.question||''));
  const dl=el('dl');
  const row=(k,v)=>{if(v==null||v==='')return;dl.append(el('dt',{},k),el('dd',{},v));};
  row('Pourquoi',q.reason);
  if(q.kind!=='team_colors')row('Chrono',q.clock?(q.clock+(q.clockStatus?' ('+q.clockStatus.replace(/_/g,' ')+')':'')):'inconnu');
  if(q.period)row('Partie',String(q.period));
  if(q.fileTime!=null&&q.kind!=='team_colors'&&!q.id.startsWith('v2req-'))row('Temps fichier',Number(q.fileTime).toFixed(1)+' s');
  if(q.segment)row('Portée','réponse valable pour '+Number(q.segment[0]).toFixed(1)+'–'+Number(q.segment[1]).toFixed(1)+' s seulement');
  if(q.proposed)row('Proposition du moteur',q.proposed);
  if(q.engineAnswer)row('Réponse déjà connue',q.engineAnswer.label+' ('+q.engineAnswer.status+')');
  side.append(dl);
  if(q.positionMap){
    // carte de position : Raja attaque vers la droite ; en haut = côté gauche du Raja
    const NS='http://www.w3.org/2000/svg';const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','-56 -38 112 76');svg.setAttribute('class','posmap');svg.setAttribute('role','img');
    svg.setAttribute('aria-label','Position du joueur encadré sur le terrain à cet instant');
    const add=(tag,at)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(at))e.setAttribute(k,v);svg.append(e);return e;};
    add('rect',{x:-52.5,y:-34,width:105,height:68,class:'pm-field'});add('line',{x1:0,y1:-34,x2:0,y2:34,class:'pm-line'});add('circle',{cx:0,cy:0,r:9.15,class:'pm-line'});
    add('rect',{x:-52.5,y:-20.16,width:16.5,height:40.32,class:'pm-line'});add('rect',{x:36,y:-20.16,width:16.5,height:40.32,class:'pm-line'});
    const P=q.positionMap;const Y=v=>-v;
    for(const[x,l]of P.opp)add('circle',{cx:x,cy:Y(l),r:1.6,class:'pm-opp'});
    for(const[x,l]of P.raja)add('circle',{cx:x,cy:Y(l),r:1.6,class:'pm-raja'});
    add('circle',{cx:P.target[0],cy:Y(P.target[1]),r:2.6,class:'pm-target'});
    const t1=add('text',{x:50,y:-35.5,class:'pm-txt','text-anchor':'end'});t1.textContent='attaque du Raja →';
    side.append(el('div',{class:'posbox'},el('div',{class:'cap'},'Position à cet instant (calibration automatique vérifiée à l’œil). Jaune : joueur encadré ; blanc : autres joueurs du Raja visibles ; bleu : Zemamra. En haut : côté gauche du Raja. Simple repère : la position seule ne suffit pas à identifier un joueur.'),svg));
  }
  if(q.claudeAnswer&&!answers[q.id]){side.append(el('div',{class:'claude'},el('strong',{},'Réponse de Claude : '+q.claudeAnswer.label+(q.claudeAnswer.confidence?' (confiance '+q.claudeAnswer.confidence+')':'')),el('div',{},q.claudeAnswer.note),el('div',{class:'hint'},'Juste : rien à faire. Fausse : répondez ci-dessous, votre réponse remplace la sienne.')));}
  if(q.kind==='clock_read'){side.append(el('label',{},'Chrono lu (MM:SS)',el('input',{id:'free',placeholder:'27:41',value:a&&a.value||''})));}
  else if(q.kind==='calibration'){
    // schéma du terrain vu de la caméra principale : tribune opposée en haut, but gauche à gauche
    const NS='http://www.w3.org/2000/svg';const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','-58 -40 116 80');svg.setAttribute('class','pitch');svg.setAttribute('role','group');svg.setAttribute('aria-label','Schéma du terrain : choisissez un repère');
    const line=(x1,y1,x2,y2)=>{const l=document.createElementNS(NS,'line');l.setAttribute('x1',x1);l.setAttribute('y1',y1);l.setAttribute('x2',x2);l.setAttribute('y2',y2);svg.append(l);};
    const rect=(x,y,w,h)=>{const r=document.createElementNS(NS,'rect');r.setAttribute('x',x);r.setAttribute('y',y);r.setAttribute('width',w);r.setAttribute('height',h);svg.append(r);};
    rect(-52.5,-34,105,68);line(0,-34,0,34);const cc=document.createElementNS(NS,'circle');cc.setAttribute('cx',0);cc.setAttribute('cy',0);cc.setAttribute('r',9.15);svg.append(cc);
    rect(-52.5,-20.16,16.5,40.32);rect(36,-20.16,16.5,40.32);rect(-52.5,-9.16,5.5,18.32);rect(47,-9.16,5.5,18.32);
    const tx=(x,y,t)=>{const e=document.createElementNS(NS,'text');e.setAttribute('x',x);e.setAttribute('y',y);e.textContent=t;svg.append(e);};
    tx(-8,-36,'tribune opposée (loin)');tx(-6,39,'côté caméra (près)');
    const dots=[];
    for(const lm of q.landmarks){const d=document.createElementNS(NS,'circle');d.setAttribute('cx',lm.x);d.setAttribute('cy',lm.y);d.setAttribute('r',1.8);d.setAttribute('tabindex','0');d.setAttribute('class','lm');const tt=document.createElementNS(NS,'title');tt.textContent=lm.label;d.append(tt);d.addEventListener('click',()=>card._pick(lm.value));d.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();card._pick(lm.value);}});svg.append(d);dots.push([lm.value,d]);}
    card._svg=()=>{for(const[v,d]of dots){const used=points.some(p=>p.landmark===v);d.setAttribute('class','lm'+(used?' used':'')+(v===curLm?' cur':''));}};
    side.append(el('div',{class:'cap'},'1. Cliquez un repère sur le schéma.  2. Cliquez le même endroit sur l’image.  Recommencez pour au moins 4 repères DIFFÉRENTS et bien répartis (5 ou plus permettent un contrôle).'),svg,
      el('div',{class:'status',id:'lmstate'},'Aucun repère choisi.'),el('ol',{class:'pts',id:'pts'}),
      el('button',{class:'btn',type:'button',onclick:()=>{points.pop();card._draw&&card._draw();}},'Retirer le dernier point'));
    setTimeout(()=>card._svg&&card._svg(),0);
  }else{
    const ch=el('div',{class:'choices',role:'group','aria-label':'Réponses possibles'});
    q.choices.forEach(c=>{const b=el('button',{type:'button','aria-pressed':String(!!sel&&sel.value===c.value),onclick:()=>{sel=c;ch.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');}},c.label);ch.append(b);});
    side.append(ch);
  }
  if(q.playerChoice){
    const ps=el('select',{id:'player'},el('option',{value:''},'— aucun / adverse / inconnu —'));for(const c of D.roster)ps.append(el('option',{value:c.value},c.label));if(a&&a.player)ps.value=a.player;
    side.append(el('label',{},'Joueur du Raja concerné',ps),el('label',{},'Issue (facultatif)',el('input',{id:'outcome',placeholder:'ex. capté, corner, hors cadre',value:a&&a.outcome||''})));
  }
  side.append(el('label',{},'Note (facultatif)',el('textarea',{id:'note',rows:'2'},a&&a.note||'')));
  const msg=el('div',{class:'status'});
  if(a){msg.classList.add('ok');msg.textContent='Réponse enregistrée : '+(a.status==='resolved'?(a.label||a.value):a.status==='unknown'?'illisible':'passée')+'. Vous pouvez la modifier.';}
  const save=async status=>{
    if(!db||saving)return;
    const body={id:q.id,status,note:document.getElementById('note').value.trim(),answeredAt:new Date().toISOString(),value:null,label:''};
    if(status==='resolved'){
      if(q.kind==='clock_read'){body.value=document.getElementById('free').value.trim();body.label=body.value;if(!/^\d{1,3}:\d{2}$/.test(body.value)&&body.value!=='absent'){msg.className='status';msg.textContent='Écrivez le chrono au format MM:SS (ou « absent »).';return;}}
      else if(q.kind==='calibration'){
        const distinct=new Set(points.map(p=>p.landmark)).size;
        if(distinct<4){msg.className='status';msg.textContent='Il faut au moins 4 repères DIFFÉRENTS (choisis sur le schéma).';return;}
        // aire du polygone des points cliqués : des points alignés ne permettent pas de calibrer
        const pts=points.map(p=>[p.x,p.y]);let area=0;const hull=pts.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]);const cr=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);const lo=[],up=[];for(const p of hull){while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],p)<=0)lo.pop();lo.push(p);}for(const p of hull.slice().reverse()){while(up.length>=2&&cr(up[up.length-2],up[up.length-1],p)<=0)up.pop();up.push(p);}const h=lo.slice(0,-1).concat(up.slice(0,-1));for(let i=0;i<h.length;i++){const a=h[i],b=h[(i+1)%h.length];area+=a[0]*b[1]-b[0]*a[1];}area=Math.abs(area)/2;
        if(area<0.01){msg.className='status';msg.textContent='Les points sont presque alignés : choisissez des repères mieux répartis (par exemple les 4 coins d’une surface).';return;}
        body.value='points';body.points=points;body.label=points.length+' repères';}
      else{if(!sel){msg.className='status';msg.textContent='Choisissez une réponse, ou « Illisible ».';return;}body.value=sel.value;body.label=sel.label;}
      if(q.playerChoice){body.player=document.getElementById('player').value||null;body.outcome=document.getElementById('outcome').value.trim();}
    }
    saving=true;msg.className='status';msg.textContent='Enregistrement…';
    try{await db.collection('answers').doc(q.id).set(body);answers[q.id]=body;msg.className='status ok';msg.textContent='Enregistré.';setTimeout(()=>{if(view==='todo')render();else{idx=Math.min(idx+1,filtered().length-1);render();}},350);}
    catch(e){msg.className='status';msg.textContent='Non enregistré ('+(e&&e.code||'erreur')+'). Réessayez dans un instant.';}
    finally{saving=false;}
  };
  const acts=el('div',{class:'actions'},
    el('button',{class:'btn primary',type:'button',onclick:()=>save('resolved'),disabled:db?null:'disabled'},'Enregistrer'),
    el('button',{class:'btn',type:'button',onclick:()=>save('unknown'),disabled:db?null:'disabled'},'Illisible / impossible à dire'),
    el('button',{class:'btn',type:'button',onclick:()=>save('skipped'),disabled:db?null:'disabled'},'Passer'));
  side.append(acts,msg);
  return el('section',{class:'q','aria-label':'Question'},ev,side);
}
$('#prev').addEventListener('click',()=>{idx=Math.max(0,idx-1);render();});
$('#next').addEventListener('click',()=>{idx=Math.min(filtered().length-1,idx+1);render();});
render();
if(window.claude&&window.claude.use){
  window.claude.use('db').then(d=>{
    db=d;
    if(!d){$('#dbnote').hidden=false;render();return;}
    d.collection('answers').onSnapshot(s=>{for(const doc of s.docs){if(doc.exists)answers[doc.id]=doc.data();}render();},()=>{$('#dbnote').hidden=false;});
    render();
  });
}else{$('#dbnote').hidden=false;}
})();
</script>
"""
