"""Contrats des adaptateurs, avec données exclusivement synthétiques."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "botola-insight/scripts"
sys.path.insert(0, str(SCRIPTS))
import common
import evaluation
import publication
import stages


class Commands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="botola synthétique ")
        self.d = Path(self.tmp.name)
        self.cfg = {"match_key": "club-a-club-b", "team_name": "Club A", "opponent": "Club B", "focus_venue": "away",
                    "roster": {"team": "Club A", "players": [
                        {"jersey_number": 7, "player_name": "Joueur synthétique", "role": "M", "starter": True},
                        {"jersey_number": 1, "player_name": "Gardien synthétique", "role": "G", "starter": True}]}}
        self.match = self.d / "match.json"
        self.timeline = self.d / "timeline.json"
        common.write(self.match, self.cfg)
        common.write(self.timeline, {"segments": [
            {"file_start": 0, "file_end": 2, "kind": "live", "offset": 100, "offset_uncertainty_s": .1, "period": 1},
            {"file_start": 2.01, "file_end": 3, "kind": "replay", "period": 1}]})

    def tearDown(self):
        self.tmp.cleanup()

    def quiet(self, fn, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(args)

    def published(self, rows, events=None, experimental=None):
        resolution = self.d / "resolution.json"
        common.write(resolution, {"match_key": self.cfg["match_key"], "team_label": "raja", "observations": rows})
        ep = self.d / "events.json" if events is not None else None
        xp = self.d / "experimental.json" if experimental is not None else None
        if ep: common.write(ep, {"events": events})
        if xp: common.write(xp, experimental)
        out = self.d / "publication"
        self.quiet(publication.publish, Namespace(match=self.match, timeline=self.timeline, resolution=resolution,
                                                  part="23b", events=ep, experimental=xp, out=out))
        return out, common.read(out / "site_match.json")

    @staticmethod
    def row(t, p, player=7):
        return {"track": 5, "t": t, "player": player, "p": p, "unit": "5.0"}

    def test_engine_untouched(self):
        self.assertGreater(common.engine_integrity(), 50)

    def test_missing_roster_stops(self):
        self.cfg["roster"]["players"] = []
        common.write(self.match, self.cfg)
        with self.assertRaises(ValueError): common.match(self.match)

    def test_same_number_allowed_on_different_teams(self):
        self.cfg["opponent_roster"] = self.cfg["roster"]
        common.write(self.match, self.cfg)
        self.assertEqual(common.match(self.match)["team_name"], "Club A")

    def test_duplicate_number_in_one_team_stops(self):
        self.cfg["roster"]["players"].append(self.cfg["roster"]["players"][0])
        common.write(self.match, self.cfg)
        with self.assertRaises(ValueError): common.match(self.match)

    def test_duplicate_json_key_stops(self):
        p = self.d / "bad.json"
        p.write_text('{"readings":{},"readings":{}}')
        with self.assertRaises(ValueError): common.read(p)

    def reading_pair(self, r):
        meta, reads = self.d / "meta.json", self.d / "reads.json"
        common.write(meta, {"planches": [{"track": 3, "file": "planches/p003.jpg", "crops": [{"c": 1}]}]})
        common.write(reads, {"reader": "session synthétique", "readings": {"3": {"1": r}}})
        return meta, reads

    def test_unreadable_null_and_confidence_mapping(self):
        from raja_engine.reading.service import CONF
        common.validate_readings(*self.reading_pair(None))
        self.assertEqual(CONF, {"high": .9, "medium": .7, "low": .5})

    def test_appearance_and_arbitrary_confidence_rejected(self):
        for r in ({"number": 7, "part": "hair", "conf": "high"},
                  {"number": 7, "part": "back", "conf": .9},
                  {"number": 7, "part": "back", "conf": "high", "face": "same"}):
            with self.assertRaises(ValueError): common.validate_readings(*self.reading_pair(r))

    def test_unknown_crop_rejected(self):
        m, r = self.reading_pair(None)
        common.write(r, {"reader": "x", "readings": {"3": {"2": None}}})
        with self.assertRaises(ValueError): common.validate_readings(m, r)

    def test_threshold_boundaries_and_no_guess_leak(self):
        rows = [self.row(.0, .89999), self.row(.2, .9), self.row(.4, .95), self.row(.6, 1, None), self.row(2.4, .99)]
        out, site = self.published(rows)
        public = common.read(out / "identities_published.json")["observations"]
        self.assertEqual([r["player"] for r in public], [None, 7, 7, None])
        summary = common.read(out / "summary.json")
        self.assertEqual((summary["unknown"], summary["observations"], summary["coverage_pct"]), (2, 4, 50))
        self.assertFalse(summary["step1_passed"])
        self.assertEqual(site["home"], "Club A")  # colonne historique de l'équipe suivie
        self.assertEqual(site["actual_home"], "Club B")
        self.assertEqual(site["focus_team"], "Club A")
        self.assertEqual(site["players"][0]["metrics"][0]["value"], .4)

    def test_event_identity_requires_095(self):
        events = [{"t": t, "track": 5, "type": "goal", "source": "lecture synthétique", "reviewed": True} for t in (.2, .4)]
        _, site = self.published([self.row(.2, .94999), self.row(.4, .95)], events=events)
        self.assertIsNone(site["facts"][0]["scorer"])
        self.assertEqual(site["facts"][1]["scorer"]["number"], 7)
        self.assertEqual(len(site["players"][0]["actions"]), 1)

    def test_event_player_cannot_bypass_resolver(self):
        with self.assertRaises(ValueError):
            self.published([self.row(.2, .4)], events=[{"t": .2, "track": 5, "type": "goal", "number": 7,
                                                      "source": "synthétique", "reviewed": True}])

    def test_experimental_values_forced_low(self):
        x = {"players": [{"number": 7, "metrics": [{"key": "xg", "value": .3, "reliability": "élevée",
                                                     "sources": ["synthétique"], "method": "simulation"}]}]}
        _, site = self.published([self.row(.2, .9)], experimental=x)
        metric = site["players"][0]["metrics"][1]
        self.assertEqual(metric["value"], .3)
        self.assertEqual(metric["reliability"], "faible")
        self.assertTrue(metric["experimental"])
        self.assertTrue(site["possession"]["experimental"])
        self.assertIsNone(site["possession"]["raja"])

    def test_nonfinite_and_duplicate_observations_rejected(self):
        for rows in ([self.row(.2, 1.1)], [self.row(.2, .95), self.row(.2, .95)]):
            with self.assertRaises(ValueError): self.published(rows)

    def test_empty_resolution_stays_unknown(self):
        out, site = self.published([])
        self.assertIsNone(common.read(out / "summary.json")["coverage_pct"])
        self.assertTrue(all("non identifié" in p["status"] for p in site["players"]))
        self.assertFalse(any(p["metrics"] for p in site["players"]))

    def test_comparison_does_not_hide_score_regression(self):
        a = {"mesure": "historique", "results": {"E": {"precision_pct": 95, "resolve_wall_s": 3}}}
        b = {"results": {"E": {"precision_pct": 95, "resolve_wall_s": 1, "unit_contradictions": {"n": 2}}}}
        self.assertTrue(evaluation.deterministic_comparison(a, b)["deterministic_equal"])
        b["results"]["E"]["precision_pct"] = 94
        self.assertFalse(evaluation.deterministic_comparison(a, b)["deterministic_equal"])

    def test_conflicting_match_config_not_overwritten(self):
        args = Namespace(match=self.match, work=self.d / "work")
        self.quiet(stages.prepare, args)
        self.cfg["team_name"] = "Club C"
        common.write(self.match, self.cfg)
        with self.assertRaises(ValueError): stages.prepare(args)

    def test_cli_from_other_working_directory(self):
        proc = subprocess.run([sys.executable, str(SCRIPTS / "botola.py"), "prepare", "--match", str(self.match),
                               "--work", str(self.d / "autre dossier")], cwd=self.d, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_csv_formula_is_text(self):
        f = self.d / "formula.csv"
        publication.write_csv(f, ["nom"], [["=1+1"]])
        self.assertIn("'=1+1", f.read_text(encoding="utf-8-sig"))

    def resolve_fixture(self, label="raja"):
        from raja_engine.evidence import make, write as write_evidence
        from raja_engine.perception import write_tracklets
        tracks, evidence, out = self.d / "tracks.jsonl", self.d / "evidence.jsonl", self.d / "resolved.json"
        write_tracklets(tracks, [{"id": 5, "shot": 1, "team": label, "obs": [{"t": t, "kit": label} for t in (0, .2, .4)], "crossings": []}])
        write_evidence(evidence, [make("number_read", 5, t, 7, .9, "vlm_plank", f"synthetic#c{k}") for k, t in enumerate((0, .2, .4))])
        self.quiet(stages.resolve, Namespace(match=self.match, team_label=label, timeline=self.timeline,
                                            tracklets=tracks, evidence=evidence, out=out))
        return common.read(out)

    def test_new_club_resolves_and_unknown_clock_abstains(self):
        r = self.resolve_fixture()
        self.assertEqual(r["units"][0]["best"], 7)
        self.assertFalse(r["rule"].get("trim_handoff", False))
        self.assertEqual(r["rule"].get("posterior", "v8"), "v8")
        common.write(self.timeline, {"segments": [{"file_start": 0, "file_end": 3, "kind": "replay", "period": 1}]})
        self.assertIsNone(self.resolve_fixture()["units"][0]["best"])

    def test_opponent_requires_own_roster(self):
        with self.assertRaises(ValueError): self.resolve_fixture("opponent")
        self.cfg["opponent_roster"] = self.cfg["roster"]
        common.write(self.match, self.cfg)
        r = self.resolve_fixture("opponent")
        self.assertEqual(r["team_label"], "opponent")
        self.assertEqual(r["units"][0]["best"], 7)

    def test_all_color_clusters_must_be_confirmed(self):
        root = self.d / "work" / self.cfg["match_key"]
        common.write(root / "teams_proposed.json", {"k": 2, "centers": [[0] * 13, [1] * 13],
                                                    "cluster_labels": {"0": "other", "1": "other"}})
        mapping = self.d / "mapping.json"
        common.write(mapping, {"0": "raja"})
        args = Namespace(work=self.d / "work", match=self.match, mapping=mapping, confirmed_by="utilisateur synthétique")
        with self.assertRaises(ValueError): stages.confirm_colors(args)
        self.assertFalse((root / "teams.json").exists())
        common.write(mapping, {"0": "raja", "1": "opponent"})
        self.quiet(stages.confirm_colors, args)
        seal = common.read(root / "colors_confirmation.json")
        self.assertEqual(seal["teams_sha256"], common.sha(root / "teams.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
