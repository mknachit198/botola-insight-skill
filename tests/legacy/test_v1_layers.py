"""Tests des trois couches v1 (perception, résolution, publication), de l'annotation et de l'évaluation.

Depuis moteur/ :  python tests/test_v1_layers.py
Données synthétiques uniquement (aucune vidéo) ; le dernier test relit les fichiers de l'étape 1
s'ils sont présents.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from raja_engine import evidence as ev  # noqa: E402
from raja_engine.annotation import to_groundtruth  # noqa: E402
from raja_engine.identity_resolver import (enforce_simultaneity, observation_identities, posterior,  # noqa: E402
                                           resolve, split_units)
from raja_engine.perception import crossing_cuts, describe_crossings, gaps  # noqa: E402
from raja_engine.publish import published_identity, strip_frozen  # noqa: E402
from raja_engine.tracks import risk_points  # noqa: E402


def obs(t, x, kit="raja", y=500, h=100, w=40):
    return {"t": round(t, 3), "box": [x, y - h, x + w, y], "view": "wide", "kit": kit}


def track(tid, xs, t0=0.0, team="raja", kits=None, step=0.2):
    o = [obs(t0 + i * step, x, (kits or {}).get(i, "raja")) for i, x in enumerate(xs)]
    return {"id": tid, "shot": 1, "team": team, "obs": o, "t_start": o[0]["t"], "t_end": o[-1]["t"], "n_obs": len(o)}


def reading(tid, t, n, c=0.9, src="vlm_plank"):
    return ev.make("number_read", tid, t, n, c, src, f"p{tid}.jpg#c", part="back")


class Perception(unittest.TestCase):
    def test_crossing_described_not_cut(self):
        a = track(1, [100, 110, 120, 130, 140])
        b = track(2, [200, 150, 120, 90, 60])          # croise a vers t=0,4
        trs = {1: a, 2: b}
        describe_crossings(trs)
        self.assertTrue(a["crossings"])
        c = [x for x in a["crossings"] if x["t"] == 0.4][0]
        self.assertEqual(c["with"], 2)
        self.assertGreater(c["iou"], 0.9)
        self.assertTrue(c["same_team"])
        self.assertEqual(crossing_cuts(a, 0.4), [0.4])
        self.assertEqual(crossing_cuts(a, 1.1), [])       # « aucune coupure »

    def test_other_team_never_cuts(self):
        a, b = track(1, [100, 120, 140]), track(2, [100, 120, 140], team="opponent")
        describe_crossings({1: a, 2: b})
        self.assertTrue(a["crossings"])
        self.assertEqual(crossing_cuts(a, 0.15, 0.6), [])

    def test_risk_points_is_rule_over_description(self):
        a = track(1, [100, 110, 120, 130, 140])
        b = track(2, [200, 150, 120, 90, 60])
        risk_points({1: a, 2: b})
        self.assertIn(0.4, a["risk_times"])
        self.assertIn("crossings", a)

    def test_gaps(self):
        a = track(1, [100, 101, 102])
        a["obs"][2]["t"] = 2.0
        self.assertEqual(gaps(a), [{"from": 0.2, "to": 2.0, "s": 1.8}])


class Evidence(unittest.TestCase):
    def test_make_validate(self):
        e = reading(5, 1.234, 30)
        self.assertEqual(e["id"], "vlm_plank:5:1.23:back")
        with self.assertRaises(ev.EvidenceError):
            ev.make("number_read", 5, 1.0, 30, 1.5, "x", "y")
        with self.assertRaises(ev.EvidenceError):
            ev.make("commentary_mention", 5, 1.0, 30, 0.5, "x", "y")   # un nom = des candidats, pas une étiquette
        with self.assertRaises(ev.EvidenceError):
            ev.make("number_read", 5, 1.0, 30, 0.5, "x", "y", physical_traits_used=True)

    def test_write_rejects_duplicates(self):
        import tempfile
        e = reading(5, 1.0, 30)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ev.EvidenceError):
                ev.write(Path(d) / "e.jsonl", [e, dict(e)])


class Resolver(unittest.TestCase):
    RULE = {"cut_iou": 0.4, "cut_foot": 0.0, "max_gap": 3.0, "kit_cut": "foreign"}

    def test_units_gap_and_kit(self):
        a = track(1, [100] * 10, kits={4: "referee"})
        a["obs"][7]["t"] = 5.0
        for o in a["obs"][8:]:
            o["t"] += 4.0
        describe_crossings({1: a})
        us = split_units(a, self.RULE)
        self.assertEqual([len(u["obs"]) for u in us], [4, 1, 2, 3])   # arbitre isolé, puis trou de 3,8 s
        self.assertFalse(us[1]["attributable"])

    def test_posterior(self):
        p = posterior([reading(1, 0, 30), reading(1, 1, 30)], [3, 30])
        self.assertGreater(p["30"], 0.95)
        q = posterior([reading(1, 0, 30), reading(1, 1, 3)], [3, 30])
        self.assertLess(max(q.values()), 0.6)            # contradiction : rien de publiable
        self.assertAlmostEqual(max(posterior([], [3, 30]).values()), 0.45)
        # v8.1 : mode "confiance" (option) : une lecture unique de confiance 0,90 donne p = 0,90 ;
        # mode "v8" (défaut) : 0,8608, inchangé
        roster = [2, 3, 7, 11, 13, 14, 19, 21, 30, 45]
        s_cal = posterior([reading(1, 0, 30, c=0.9)], roster, mode="confiance")
        self.assertAlmostEqual(s_cal["30"], 0.90, places=5)
        s_def = posterior([reading(1, 0, 30, c=0.9)], roster)
        self.assertAlmostEqual(s_def["30"], 0.86079, places=4)
        with self.assertRaises(ValueError):
            posterior([reading(1, 0, 30)], roster, mode="inconnu")

    def test_handoff_trimming(self):
        a = track(1, [100 + i for i in range(10)], t0=0.0)   # 0,0 .. 1,8
        b = track(2, [300 + i for i in range(10)], t0=1.6)   # 1,6 .. 3,4 (chevauche a sur 1,6..1,8)
        tl = {1: a, 2: b}
        describe_crossings(tl)
        evs = [reading(1, 0.2, 19, 0.9), reading(1, 1.0, 19, 0.9),
               reading(2, 2.0, 19, 0.9), reading(2, 2.8, 19, 0.9)]
        r = resolve(tl, evs, [11, 19], {**self.RULE, "cut_iou": 1.1, "trim_handoff": True})
        u = {x["track"]: x for x in r["units"]}
        self.assertEqual(u[1]["best"], 19)
        self.assertEqual(u[2]["best"], 19)
        self.assertLess(u[1]["end"], u[2]["start"])
        self.assertEqual(len(r["handoffs"]), 1)

    def test_resolve_propagates_within_unit_only(self):
        a = track(1, [100 + i for i in range(20)])
        b = track(2, [400 + i for i in range(20)])
        tl = {1: a, 2: b}
        describe_crossings(tl)
        r = resolve(tl, [reading(1, 0.4, 30), reading(1, 2.0, 30)], [3, 30], self.RULE)
        u = {x["track"]: x for x in r["units"]}
        self.assertEqual(u[1]["best"], 30)
        self.assertIsNone(u[2]["best"])
        rows = observation_identities(r)
        self.assertEqual(sum(1 for x in rows if x["player"] == 30), 20)

    def test_simultaneity(self):
        units = [{"unit": "a", "start": 0, "end": 5, "p": 0.97, "probs": {"30": 0.97, "?": 0.03}},
                 {"unit": "b", "start": 2, "end": 6, "p": 0.90, "probs": {"30": 0.90, "3": 0.06, "?": 0.04}}]
        conf = enforce_simultaneity(units)
        self.assertEqual(units[0]["best"], 30)
        self.assertEqual(units[1]["best"], 3)
        self.assertEqual(conf[0]["unit"], "b")

    def test_split_on_conflict(self):
        a = track(1, [100 + i for i in range(30)])
        tl = {1: a}
        describe_crossings(tl)
        evs = [reading(1, 0.2, 19), reading(1, 0.6, 19), reading(1, 4.0, 11), reading(1, 5.0, 11)]
        r = resolve(tl, evs, [11, 19], {**self.RULE, "split_on_conflict": True})
        bests = [(u["best"], u["attributable"]) for u in r["units"]]
        self.assertEqual(r["conflict_zones"], 1)
        self.assertIn((19, True), bests)
        self.assertIn((11, True), bests)
        self.assertIn((None, False), bests)               # bascule inconnue : zone non attribuée


class Annotation(unittest.TestCase):
    def test_groundtruth_switch_zone(self):
        a = track(7, [100 + i for i in range(11)])         # t = 0 .. 2
        data = {"chains": [{"id": 7, "set": "test", "crops": [{"k": 1, "t": 0.0}, {"k": 2, "t": 1.0}, {"k": 3, "t": 2.0}]}]}
        ans = {"c7": {"default": "30", "crops": {"3": "3"}}}
        rows = to_groundtruth(ans, data, {7: a})
        st = {r["t"]: (r["jersey"], r["status"]) for r in rows}
        self.assertEqual(st[0.4], (30, "ok"))
        self.assertEqual(st[1.0], (30, "ok"))
        self.assertEqual(st[1.4], (None, "switch_zone"))
        self.assertEqual(st[2.0], (3, "ok"))
        rows = to_groundtruth({"c7": {"default": "not_raja"}}, data, {7: a})
        self.assertTrue(all(r["status"] == "not_raja" for r in rows))


class Publication(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(published_identity({"player": 30, "p": 0.91}), 30)
        self.assertIsNone(published_identity({"player": 30, "p": 0.89}))
        self.assertIsNone(published_identity({"player": 30, "p": 0.93}, "events"))

    def test_frozen_metrics_removed(self):
        rec = {"player": 30, "distance_km": 9.1, "xg": 0.2, "visible_s": 120, "nested": [{"vmax_kmh": 31}]}
        self.assertEqual(strip_frozen(rec), {"player": 30, "visible_s": 120, "nested": [{}]})


class Evaluation(unittest.TestCase):
    def test_metrics(self):
        from eval_groundtruth import evaluate
        gt = [{"track": 1, "t": 0.0, "jersey": 30, "status": "ok"}, {"track": 1, "t": 0.2, "jersey": 30, "status": "ok"},
              {"track": 2, "t": 0.0, "jersey": 3, "status": "ok"}, {"track": 3, "t": 0.0, "jersey": None, "status": "not_raja"},
              {"track": 4, "t": 0.0, "jersey": None, "status": "unknown"}]
        ident = [{"track": 1, "t": 0.0, "player": 30, "p": 0.95}, {"track": 1, "t": 0.2, "player": 30, "p": 0.5},
                 {"track": 2, "t": 0.0, "player": 30, "p": 0.99}, {"track": 3, "t": 0.0, "player": 3, "p": 0.99},
                 {"track": 4, "t": 0.0, "player": 3, "p": 0.99}]
        m = evaluate(gt, ident, 0.9, [], 0.0)
        self.assertEqual((m["correct"], m["wrong"], m["published_unverifiable"]), (1, 2, 1))
        self.assertAlmostEqual(m["precision_pct"], 33.33, places=2)
        self.assertAlmostEqual(m["coverage_pct"], 66.67, places=2)     # 2 publiées / 3 annotées d'un numéro
        self.assertAlmostEqual(m["correct_coverage_pct"], 33.33, places=2)


class RealFiles(unittest.TestCase):
    D = ROOT.parent / "etape1_lecture_dense"

    @unittest.skipUnless((D / "evidence_part7.jsonl").exists(), "fichiers de l'étape 1 absents")
    def test_step1_files(self):
        from raja_engine.perception import read_tracklets
        tl = read_tracklets(self.D / "tracklets_part7.jsonl")
        evs = ev.read(self.D / "evidence_part7.jsonl")
        self.assertEqual(sum(t["n_obs"] for t in tl.values() if t["team"] == "raja"), 8373)
        self.assertEqual(sum(1 for e in evs if e["source"] == "vlm_plank"), 274)
        r = resolve(tl, evs, [2, 3, 7, 11, 13, 14, 19, 21, 30, 45],
                    {"cut_iou": 0.15, "cut_foot": 0.6, "max_gap": 0.6, "kit_cut": "none"})
        self.assertFalse(r["calibrated"])
        self.assertFalse(r["physical_traits_used_for_identity"])
        cfg = json.loads((ROOT / "tests/configs_v8.json").read_text())
        self.assertIn("P0_point_zero", cfg["configs"])


if __name__ == "__main__":
    unittest.main()
