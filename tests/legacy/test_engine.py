#!/usr/bin/env python3
"""Tests unitaires du moteur (sans vidéo intégrale). Lancement : python3 tests/test_engine.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raja_engine.fingerprint import sampled_fingerprint  # noqa: E402
from raja_engine.identity import track_segments  # noqa: E402
from raja_engine.pitch import fit_homography, kinematics, landmarks  # noqa: E402
from raja_engine.questions import QuestionStore, task_id  # noqa: E402
from raja_engine.roster import Roster  # noqa: E402
from raja_engine.timeline import Timeline  # noqa: E402
from raja_engine.tracks import safe_window  # noqa: E402
from raja_engine.xg import xg  # noqa: E402

PASSED = []


def check(cond, msg):
    if not cond:
        raise AssertionError("ÉCHEC : " + msg)
    PASSED.append(msg)
    print("OK  ", msg)


def test_fingerprint_matches_portal_js():
    js = ROOT / "tests" / "file-identity.js"
    tmp = Path(tempfile.mkdtemp())
    for size in (10, 65536, 200_000, 1_000_003):
        f = tmp / f"v{size}.bin"
        f.write_bytes(os.urandom(size))
        py = sampled_fingerprint(str(f))["fingerprint"]
        if shutil.which("node") and js.exists():
            code = (f"const {{identify}}=require({json.dumps(str(js))});const fs=require('fs');"
                    f"const b=fs.readFileSync({json.dumps(str(f))});"
                    "identify(new File([b],'x.bin')).then(r=>console.log(r.fingerprint));")
            out = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=60).stdout.strip()
            check(out == py, f"empreinte sha256-3x64k-v1 identique au portail (taille {size})")
    shutil.rmtree(tmp)


def test_question_store_persistence():
    tmp = Path(tempfile.mkdtemp())
    st = QuestionStore(tmp)
    tid = task_id("abcdef1234", "identity", 12.5, [0.1, 0.2, 0.3, 0.4])
    st.add({"id": tid, "kind": "identity", "time": 12.5, "bbox": [0.1, 0.2, 0.3, 0.4], "video": "abcdef1234"})
    st.answer(tid, "19", label="19 · Othmane Chraibi")
    st.answer(tid, "14", label="14 · Younes Bakiz", note="correction")
    (tmp / "questions.json").unlink()  # perte de l'état courant
    st2 = QuestionStore(tmp)
    check(tid not in st2.tasks, "questions.json supprimé : la demande disparaît de l'état…")
    st2.add({"id": tid, "kind": "identity", "time": 12.5, "bbox": [0.1, 0.2, 0.3, 0.4], "video": "abcdef1234"})
    st3 = QuestionStore(tmp)
    st3.add({"id": tid, "kind": "identity", "time": 12.5, "bbox": [0.1, 0.2, 0.3, 0.4], "video": "abcdef1234"})
    st3._replay_log()
    check(st3.tasks[tid]["response"]["value"] == "14", "…mais la réponse est restaurée depuis le journal en ajout seul")
    check(len(st3.tasks[tid].get("history", [])) >= 1, "l'ancienne réponse reste dans l'historique")
    _, status = st3.add({"id": tid, "kind": "identity", "time": 13.0, "bbox": [0.1, 0.2, 0.3, 0.4]})
    check(status == "conflict", "une demande existante dont le temps change est refusée (conflit)")
    # anciennes questions d'action : la liste « Faute » seule est remplacée, la réponse donnée est gardée
    from raja_engine.events import EVENT_CHOICES, refresh_event_choices
    old = [{"value": "tir_cadre", "label": "Tir cadré"}, {"value": "faute", "label": "Faute"}]
    eid = task_id("abcdef1234", "event_review", 40.0, None)
    st3.add({"id": eid, "kind": "event_review", "time": 40.0, "video": "abcdef1234", "choices": old})
    st3.answer(eid, "faute", label="Faute")
    pid = "precision-" + eid
    prec = [{"value": "faute_commise", "label": "commise"}, {"value": "faute_subie", "label": "subie"}]
    st3.add({"id": pid, "kind": "event_review", "time": 40.0, "video": "abcdef1234", "refines": eid, "choices": prec})
    refresh_event_choices(st3)
    st4 = QuestionStore(tmp)
    vals = [c["value"] for c in st4.tasks[eid]["choices"]]
    check(st4.tasks[eid]["choices"] == EVENT_CHOICES and "faute_commise" in vals and "faute_subie" in vals and "faute" not in vals,
          "anciennes questions d'action : choix « faute commise / subie » proposés (y compris pour corriger)")
    check(st4.tasks[eid]["response"]["value"] == "faute" and st4.tasks[pid]["choices"] == prec,
          "la réponse déjà donnée et les questions de précision ne sont pas modifiées")
    corr = json.load(open(ROOT / "matches/raja-zemamra/corrections/raja-corrections_1.json"))
    r = st3.import_corrections(corr, None, 1, "raja-corrections_1.json")
    imp = st3.tasks["raja-zemamra-h1_0610_3"]
    orig = next(t for t in corr["tasks"] if t["id"] == "raja-zemamra-h1_0610_3")
    check(r["imported"] == 18 and imp["response"] == orig["response"], "les 18 réponses V1 sont importées à l'identique")
    check(imp["match_clock"] is None and imp["clock_status"] == "unverified", "un chrono V1 nul reste nul (non converti)")
    r2 = st3.import_corrections(corr, None, 1, "raja-corrections_1.json")
    check(r2["imported"] == 0, "un second import ne duplique ni n'écrase rien")
    shutil.rmtree(tmp)


def _samples(spec, step=0.5):
    """spec : liste de (durée, type, paramètre)."""
    out, t = [], 0.0
    for dur, kind, p in spec:
        end = t + dur
        while t < end - 1e-9:
            s = {"t": round(t, 3), "present": False, "secs": None, "conf": 0.3, "diff": 5.0}
            if kind == "live":
                s.update(present=True, secs=int(t + p))
            elif kind == "freeze":
                s.update(present=True, secs=p, diff=0.0)
            elif kind == "stoppage":
                s.update(present=True, secs=5400, stoppage=True, added=int(t + p))
            elif kind == "banner":
                s.update(banner=True)
            out.append(s)
            t += step
    return out, t


def test_timeline_rules():
    spec = [(60, "live", 100.0), (12, "replay", None), (30, "live", 100.0), (10, "freeze", 230), (40, "live", 300.0),
            (8, "banner", None), (20, "live", 300.0)]
    samples, dur = _samples(spec)
    samples[30]["secs"] += 60  # erreur OCR isolée (un chiffre des minutes)
    for i in range(160, 166):    # erreur OCR persistante sur 3 s (« 68 » lu pour « 69 »)
        samples[i]["secs"] -= 60
    tl = Timeline.build(samples, dur)
    kinds = [s["kind"] for s in tl.segments]
    check("no_scoreboard" in kinds, "un passage sans tableau devient « sans tableau », pas du direct")
    check(tl.at(65)["live"] is False, "pendant un ralenti, aucun chrono de direct n'est attribué")
    check(abs(tl.at(10)["clock_s"] - 110.0) <= 0.5, "chrono du direct = temps fichier + décalage du segment")
    check(abs(tl.at(85)["clock_s"] - 185.0) <= 0.5, "même décalage après un ralenti sans coupure")
    check(any(s["kind"] in ("frozen", "discontinuity") for s in tl.segments), "le gel d'image est isolé")
    check(abs(tl.at(130)["clock_s"] - 430.0) <= 0.5, "après le gel, un NOUVEAU décalage est estimé (pas de report constant)")
    check(len(tl.data["ocr_corrections"]) >= 1, "une erreur de lecture persistante encadrée est rejetée (A–B–C)")
    check(tl.at(156)["live"] and tl.at(156).get("interpolated"), "bandeau pendant le direct : chrono interpolé et signalé")
    s2, d2 = _samples([(30, "live", 5370.0), (20, "stoppage", 30.0)])
    tl2 = Timeline.build(s2, d2)
    st = tl2.at(40)
    check(st["clock"] == "90:00 +1:10" and st["period"] == 2, "temps additionnel représenté « 90:00 +1:10 »")
    check(tl2.to_file(5400, 70) and abs(tl2.to_file(5400, 70)[0] - 40) < 0.6, "conversion inverse chrono additionnel → fichier")


def test_identity_window():
    tr = {"obs": [{"t": round(0.2 * i, 2)} for i in range(50)] + [{"t": round(12 + 0.2 * i, 2)} for i in range(10)],
          "risk_times": [4.0], "t_start": 0.0, "t_end": 13.8}
    a, b = safe_window(tr, 6.0)
    check(a == 4.0 and b == 9.8, "une réponse à t=6 s ne vaut qu'entre le croisement (4 s) et le trou de suivi (9,8 s)")
    segs = track_segments(tr)
    check(len(segs) == 3, "tronçons d'une piste coupés aux croisements et aux trous")


def test_kinematics():
    s, x = [], 0.0
    for i in range(30):
        t = i * 0.2
        x += (7.2 if t < 3 else 2.0) * 0.2
        s.append((t, x, 0.0))
    k = kinematics(s)
    check(abs(k["distance_m"] - 26.16) < 0.3, "distance d'un trajet connu (26,16 m) retrouvée")
    check(k["sprints"] == 1 and abs(k["vmax_kmh"] - 25.9) < 0.5, "sprint ≥ 25,2 km/h pendant ≥ 1 s détecté")
    k2 = kinematics(s[:10] + [(t + 5, x, y) for t, x, y in s[10:]])
    check(k2["covered_s"] < k["covered_s"], "un trou de suivi n'est pas interpolé (distance non inventée)")


def test_homography():
    lm = landmarks()
    import numpy as np
    import cv2
    Htrue = np.array([[8.0, 1.0, 400], [0.2, 4.0, 300], [0.0004, 0.004, 1.0]])
    names = ["surface_gauche_ligne_but_haut", "surface_gauche_ligne_but_bas", "surface_gauche_angle_haut",
             "surface_gauche_angle_bas", "point_penalty_gauche", "six_metres_gauche_angle_haut"]
    pts = []
    for n in names:
        X, Y = lm[n]
        u, v, w = Htrue @ np.array([X + 52.5, Y, 1.0])
        pts.append({"landmark": n, "x": (u / w) / 854, "y": (v / w) / 480})
    fit = fit_homography(pts, 854, 480)
    check(fit["ok"] and fit["fit_rmse_m"] < 0.05 and fit["checked"], "homographie retrouvée et contrôlée (en laisser un)")
    # repères dégénérés : 3 des 4 repères sur la même ligne du terrain (cas réel de la page, 42:41)
    by = {p["landmark"]: p for p in pts}
    deg = [by["surface_gauche_angle_haut"], by["surface_gauche_angle_bas"], dict(by["point_penalty_gauche"], landmark="arc_gauche_haut"),
           by["surface_gauche_ligne_but_haut"]]
    f2 = fit_homography(deg, 854, 480)
    check(not f2["ok"] and "dégénérés" in f2["reason"], "4 repères dont 3 alignés : calibration refusée")
    # repères incohérents : un clic sous le mauvais nom
    bad = [dict(p) for p in pts]
    bad[0]["landmark"], bad[1]["landmark"] = bad[1]["landmark"], bad[0]["landmark"]
    f3 = fit_homography(bad, 854, 480)
    check(not f3["ok"] and "incohérents" in f3["reason"], "repères contradictoires : calibration refusée (écart moyen > 1 m)")


def test_banner_matching_and_xg():
    cfg = json.load(open(ROOT / "matches/raja-zemamra/match.json"))
    r = Roster(cfg)
    for num, name, exp in [(19, "CHRAMI", "Othmane Chraibi"), (30, "AGBEXPORNU", "Michael Agbekpornu"),
                           (6, "SARHOK", "Aymen Barkok"), (84, "BAKIZ", "Younes Bakiz"), (9, "KHAR", "Ismail Khafi")]:
        p, conf, how = r.match_banner_line(num, name)
        check(p is not None and p["player_name"] == exp, f"bandeau « {num}. {name} » rapproché de {exp} ({how})")
    p = r.from_answer_value("20")
    check(p["player_name"] == "Youness Dahmani" and p["jersey_number"] == 11, "réponse V1 « 20 » = Dahmani, n°11 sur ce match")
    # bandeaux de l'adversaire rapprochés du Raja par le seul numéro : la cohérence du terrain les écarte
    from raja_engine.events import check_raja_consistency

    def ban(clock, typ, start, players, review):
        return {"type": typ, "file_start": start, "file_end": start + 5, "period": 2, "match_clock_banner_start": clock,
                "needs_review": review, "players": [{"team": "raja", "jersey_number": n, "player_name": nm,
                                                     "match_method": how} for n, nm, how in players]}
    bs = [ban("55:29", "substitution_out", 1278, [(11, "Youness Dahmani", "numero_et_nom")], False),
          ban("55:35", "substitution_in", 1284, [(77, "Adam Ennafati", "numero_et_nom")], False),
          ban("74:14", "substitution_out", 2376, [(14, "Younes Bakiz", "numero_et_nom"), (19, "Othmane Chraibi", "numero_et_nom")], False),
          ban("74:20", "substitution_in", 2382, [(6, "Aymen Barkok", "numero_et_nom"), (9, "Ismail Khafi", "numero_et_nom")], False),
          ban("76:23", "substitution_out", 2504, [(14, "Younes Bakiz", "numero_seul_nom_divergent")], True),
          ban("83:32", "substitution_out", 2934, [(10, "Felipinho", "numero_seul_nom_divergent"), (7, "Mohamed Boulacsoute", "numero_seul_nom_divergent")], True),
          ban("83:38", "substitution_in", 2940, [(11, "Youness Dahmani", "numero_et_nom"), (17, "Ayoub El Amloud", "numero_et_nom")], False)]
    check_raja_consistency(bs, r)
    flagged = [b["match_clock_banner_start"] for b in bs if b.get("team_probable") == "opponent"]
    check(flagged == ["76:23", "83:32", "83:38"] and all(bs[i]["needs_review"] is False for i in range(4)),
          "bandeaux adverses (retour de Dahmani, Bakiz déjà sorti, paire sortie/entrée) écartés ; les vrais remplacements restent")
    check(bs[6]["needs_review"] and any("pas de retour" in x for x in bs[6]["raja_check"]["reasons"]),
          "le motif est expliqué : un joueur remplacé ne revient pas")
    check_raja_consistency(bs, r)
    check([b["match_clock_banner_start"] for b in bs if b.get("team_probable") == "opponent"] == flagged, "contrôle idempotent")
    # équipe d'un bandeau par la position de son étiquette (gauche = domicile, droite = extérieur)
    import numpy as np
    from raja_engine.scoreboard import ScoreboardReader
    lay = {"banner_text_box": [0.085, 0.126, 0.236, 0.192], "banner_team_box": [0.085, 0.19, 0.14, 0.225]}
    rd = ScoreboardReader(lay)
    teal, green = (170, 150, 20), (160, 200, 40)  # BGR : étiquette, lignes du bandeau
    sides = []
    for tag_x, n_lines in ((0.085, 1), (0.185, 1), (0.085, 2), (0.185, 2)):
        f = np.full((480, 854, 3), 30, np.uint8)
        y = 0.126
        for _ in range(n_lines):
            f[int(y * 480):int((y + 0.035) * 480), int(0.085 * 854):int(0.236 * 854)] = green
            y += 0.04
        f[int(y * 480):int((y + 0.035) * 480), int(tag_x * 854):int((tag_x + 0.052) * 854)] = teal
        sides.append(rd.banner_side(f)[0])
    empty = rd.banner_side(np.full((480, 854, 3), 30, np.uint8))[0]
    check(sides == ["home", "away", "home", "away"] and empty is None,
          "équipe du bandeau lue par la position de l'étiquette, quel que soit le nombre de lignes")
    # possession estimée : ballon aux pieds d'un joueur ; ballon disputé ou loin = aucun contrôle
    from raja_engine.momentum import controls
    trk = {"1": {"team": "raja", "team_conf": 1.0}, "2": {"team": "opponent", "team_conf": 1.0}}

    def fr(t, ball, players):
        return {"t": t, "view": "wide", "b": [ball], "p": [[*bx, 0.9, tid, None] for bx, tid in players]}
    fs = [fr(1.0, [100, 196, 106, 202, 0.8], [((90, 150, 110, 200), 1), ((300, 150, 320, 200), 2)]),
          fr(1.2, [100, 196, 106, 202, 0.8], [((90, 150, 110, 200), 1), ((96, 150, 116, 200), 2)]),
          fr(1.4, [400, 196, 406, 202, 0.8], [((90, 150, 110, 200), 1), ((300, 150, 320, 200), 2)])]
    cs = controls(fs, trk, None)
    check([(c["t"], c["team"]) for c in cs] == [(1.0, "raja")],
          "possession : ballon aux pieds attribué ; ballon disputé ou éloigné, rien n'est attribué")
    # calibration automatique : image synthétique rendue avec une caméra connue, puis retrouvée
    import cv2
    from raja_engine import autocalib as ac
    Cc = np.array([0.0, -96.0, 17.0])
    truth = (np.radians(-6.0), np.radians(9.5), np.radians(-0.6), 2400.0)
    Ht = ac.H_camera(truth, Cc)
    img = np.zeros((480, 854, 3), np.uint8); img[:] = (40, 140, 40)
    img[:120] = (60, 60, 60)  # tribunes
    for k in range(len(ac.FEATS)):
        q = ac.project(Ht, ac.SP[ac.SID == k])
        ok = np.isfinite(q).all(1) & (np.abs(q) < 5000).all(1)
        if ok.sum() > 1:
            cv2.polylines(img, [np.round(q[ok]).astype(np.int32)], False, (235, 235, 235), 2, cv2.LINE_AA)
    img[:120] = (60, 60, 60)
    res = ac.calibrate_frame(img, [], Cc)
    grid = np.array([[x, y] for x in np.arange(-40, 41, 10.0) for y in np.arange(-30, 31, 10.0)])
    a, b = ac.project(res["H_pitch2img"], grid), ac.project(Ht, grid)
    vis = (b[:, 0] > 0) & (b[:, 0] < 854) & (b[:, 1] > 120) & (b[:, 1] < 480)
    err = float(np.median(np.linalg.norm(a[vis] - b[vis], axis=1)))
    check(res["ok"] and err < 3.0, f"calibration automatique retrouvée sur image synthétique (écart médian {err:.1f} px)")
    m = json.load(open(ROOT / "matches/raja-zemamra/xg_model.json"))
    check(abs(xg(m, 10.393343375489899, -3.639522818161372) - 0.14383328375555465) < 1e-12, "xG du but V2 recalculé à l'identique")


def test_camera_center():
    """Centre optique retrouvé sur des plans synthétiques ; un plan d'une autre caméra est écarté."""
    import numpy as np
    from raja_engine import autocalib as ac
    rng = np.random.default_rng(1)
    C_true = np.array([0.5, -97.0, 17.5])
    planes = [(np.radians(p), np.radians(t), np.radians(-0.6), f)
              for p, t, f in ((-14, 9.0, 2300), (-4, 10.0, 2600), (5, 9.5, 2400), (15, 8.5, 2200))]
    obs = []
    for pr, C in [(p, C_true) for p in planes] + [((np.radians(18), np.radians(15), np.radians(-0.6), 1800), np.array([-30.0, -80.0, 25.0]))]:
        H = ac.H_camera(pr, C)
        q = ac.project(H, ac.SP)
        ok = np.isfinite(q).all(1) & (q[:, 0] > 0) & (q[:, 0] < 854) & (q[:, 1] > 120) & (q[:, 1] < 480)
        obs.append((q[ok] + rng.normal(0, 0.5, (ok.sum(), 2)), H))
    C, rep = ac.fit_camera_center(obs, max_px=200)
    err = float(np.linalg.norm(np.asarray(C) - C_true))
    check(err < 1.5 and rep["excluded_observations"] == [4],
          f"centre de la caméra retrouvé (écart {err:.2f} m), plan d'une autre caméra écarté")


def test_site_bundle_rules():
    """Fichier du site : sources en langage public, contrôle des valeurs et des fiabilités."""
    from raja_engine.site_export import FORMAT, VERSION, _pub_src, validate
    check(_pub_src("relecture V2, votre réponse, votre réponse") == "relecture vidéo, relecture humaine",
          "site : sources réécrites pour le public, sans doublon")
    ok = {"format": FORMAT, "version": VERSION, "match_key": "m", "title": "Raja — X",
          "players": [{"number": 7, "name": "A", "metrics": [{"key": "shots", "value": 2, "reliability": "élevée"}]}]}
    bad = {**ok, "players": [{"number": 7, "name": "A", "metrics": [{"key": "shots", "value": -1, "reliability": "sûre"}]}]}
    check(validate(ok) == [] and len(validate(bad)) == 2, "site : valeur négative et fiabilité inconnue refusées")
    from raja_engine.publish import site_bundle_v1
    full = {**ok, "possession": {"windows": [], "raja": 0.6},
            "players": [{"number": 7, "name": "A", "metrics": [
                {"key": k, "value": 1, "reliability": "faible"} for k in ("shots", "xg", "xa", "vmax_kmh", "distance_observed_m", "passes_completed")],
                "actions": [{"clock": "1:00", "label": "Passes réussies"}, {"clock": "2:00", "label": "Tirs, Passes réussies"}]}]}
    v1 = site_bundle_v1(full)
    check("possession" not in v1 and [m["key"] for m in v1["players"][0]["metrics"]] == ["shots"]
          and [a["label"] for a in v1["players"][0]["actions"]] == ["Tirs"] and "possession" in full,
          "site v1 : possession, distances, vitesses, xG/xA et passes retirées ; le reste gardé ; entrée intacte")
    from raja_engine.publish import site_bundle_experimental
    full["players"][0]["metrics"][-1]["reliability"] = "moyenne"
    ex = site_bundle_experimental(full)
    ms = {m["key"]: m for m in ex["players"][0]["metrics"]}
    check(len(ms) == 6 and ms["shots"]["reliability"] == "faible" and not ms["shots"].get("experimental")
          and all(ms[k]["reliability"] == "faible" and ms[k]["experimental"] for k in ("xg", "xa", "vmax_kmh", "distance_observed_m", "passes_completed"))
          and "moyenne" in ms["passes_completed"]["reason"] and ex["possession"]["experimental"]
          and len(ex["players"][0]["actions"]) == 2 and full["players"][0]["metrics"][-1]["reliability"] == "moyenne",
          "site (décision du 18/10) : estimations expérimentales gardées en fiabilité faible, possession marquée, entrée intacte")


def test_align_offset():
    """Recalage de deux enregistrements : décalage de 5 images retrouvé (30 i/s)."""
    import numpy as np
    import cv2
    from raja_engine.align import offset_at
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(3)
    frames = [cv2.GaussianBlur((rng.random((108, 192, 3)) * 255).astype(np.uint8), (0, 0), 3) for _ in range(240)]
    for name, lag, size in (("a.avi", 0, (192, 108)), ("b.avi", 5, (384, 216))):
        w = cv2.VideoWriter(str(tmp / name), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
        for f in [frames[0]] * lag + frames:
            w.write(cv2.resize(f, size))
        w.release()
    r = offset_at(str(tmp / "a.avi"), str(tmp / "b.avi"), 2.0, guess=0.0, search=1.0, win=2.0)
    shutil.rmtree(tmp)
    check(r["ok"] and abs(r["offset"] - 5 / 30) < 0.02, f"recalage de deux enregistrements : décalage {r['offset']:.3f} s (attendu 0,167)")


def test_identity_grouped_and_replaces():
    from raja_engine.identity import resolve
    cfg = json.load(open(ROOT / "matches/raja-zemamra/match.json"))
    roster = Roster(cfg)

    def track(tid, t0, t1, x):
        obs = [{"t": round(t0 + 0.2 * i, 3), "box": [x, 100, x + 40, 200], "score": 0.9, "view": "wide", "feat": None,
                "W": 1000, "Himg": 1000} for i in range(int((t1 - t0) / 0.2) + 1)]
        return {"id": tid, "shot": 1, "obs": obs, "views": {"wide": len(obs)}, "numbers": [], "t_start": t0, "t_end": t1,
                "n_obs": len(obs), "team": "raja", "team_conf": 1.0, "median_h": 100.0, "max_gap_s": 0.2, "risk_times": []}
    d = Path(tempfile.mkdtemp())
    st = QuestionStore(d)
    fp = "f" * 64
    trs = {1: track(1, 0.0, 10.0, 100), 2: track(2, 20.0, 30.0, 500), 3: track(3, 40.0, 50.0, 700)}
    bb = lambda x: [x / 1000, 0.1, (x + 40) / 1000, 0.2]  # noqa: E731
    old = {"id": "old", "kind": "identity", "video": fp, "period": 1, "time": 5.0, "bbox": bb(100), "question": "?"}
    st.add(old); st.answer("old", "19", label="19", origin="human_page_de_questions")
    grp = {"id": "grp", "kind": "identity", "video": fp, "period": 1, "time": 25.0, "bbox": None, "question": "?",
           "anchors": [{"time": 25.0, "bbox": bb(500), "track_id": 2}, {"time": 45.0, "bbox": bb(700), "track_id": 3}]}
    st.add(grp); st.answer("grp", "14", label="14", origin="claude_review")
    res = resolve(trs, st, roster, fp, 1000, 1000, True)
    by = {s["track_id"]: s["player"] for s in res["segments"]}
    check(by == {1: 19, 2: 14, 3: 14}, "question groupée : une réponse vaut pour chaque séquence (pistes 2 et 3)")
    rc = {"id": "rc", "kind": "identity", "video": fp, "period": 1, "time": 5.0, "bbox": None, "question": "?",
          "replaces": ["old"], "anchors": [{"time": 5.0, "bbox": bb(100), "track_id": 1}]}
    st.add(rc)
    by = {s["track_id"]: s["player"] for s in resolve(trs, st, roster, fp, 1000, 1000, True)["segments"]}
    check(by.get(1) == 19, "revérification en attente : l'ancienne réponse reste en vigueur")
    st.answer("rc", "14", label="14", origin="human_page_de_questions")
    by = {s["track_id"]: s["player"] for s in resolve(trs, st, roster, fp, 1000, 1000, True)["segments"]}
    check(by.get(1) == 14 and list(by.values()).count(19) == 0, "revérification répondue : elle remplace l'ancienne réponse")
    st.answer("rc", None, status="unknown", origin="human_page_de_questions")
    by = {s["track_id"]: s["player"] for s in resolve(trs, st, roster, fp, 1000, 1000, True)["segments"]}
    check(1 not in by, "revérification « je ne sais pas » : l'ancienne réponse n'est plus utilisée")
    shutil.rmtree(d)


def test_close_pending():
    d = Path(tempfile.mkdtemp())
    work = d / "work"
    st_dir = work / "raja-zemamra-reference" / "questions"
    st = QuestionStore(st_dir)
    for i, blk in ((1, False), (2, False), (3, True)):
        st.add({"id": f"q{i}", "kind": "identity", "video": "f" * 64, "period": 1, "time": float(i), "bbox": None,
                "question": "?", "blocking": blk})
    st.save()
    cfg = ROOT / "matches/raja-zemamra/match.json"
    out = subprocess.run([sys.executable, "-m", "raja_engine", "close-pending", "--match", str(cfg), "--work", str(work),
                          "--keep", "q2"], cwd=ROOT, capture_output=True, text=True)
    st2 = QuestionStore(st_dir)
    check(out.returncode == 0 and st2.tasks["q1"]["status"] == "skipped" and st2.tasks["q2"]["status"] == "pending"
          and st2.tasks["q3"]["status"] == "pending",
          "mode sans questions : les questions ouvertes sont fermées, sauf celles gardées et les bloquantes")
    shutil.rmtree(d)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\n{len(PASSED)} vérifications réussies.")
