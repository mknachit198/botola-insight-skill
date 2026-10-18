"""Contrôles de l'intégration : composition, frontières, reprise et compatibilité."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raja_engine import analyze
from raja_engine.detector import Detection, PERSON
from raja_engine.identity import track_segments
from raja_engine.pipeline import Session
from raja_engine.teams import TeamModel
from raja_engine.tracking_refine import VERSION, refine_file, refine_frames, short_links, split_by_team
from raja_engine.tracks import build_tracks, risk_points, assign_teams
from raja_engine.util import read_json, read_jsonl, write_json
from raja_engine.views import MotionComposer


def model():
    return TeamModel.from_json({"centers": [[0.0], [1.0]], "cluster_labels": {"0": "raja", "1": "opponent"}})


def frame(t, rows, shot=1, H=True):
    return {"t": t, "fi": int(30 * t), "shot": shot, "view": "wide", "p": rows, "num": [], "b": [],
            "H": np.eye(3).reshape(-1).tolist() if H else None, "W": 100, "Himg": 100}


def row(tid, feat=0.0, x=10):
    return [x, 10, x + 10, 30, 0.9, tid, [feat]]


class TrackingTests(unittest.TestCase):
    def test_motion_order_and_missing(self):
        c = MotionComposer()
        self.assertIsNone(c.take()[0])
        translate = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]])
        scale = np.diag([2, 2, 1])
        c.add(translate, 40); c.add(scale, 30)
        H, n = c.take()
        np.testing.assert_allclose(H, scale @ translate)
        self.assertEqual(n, 30)
        c.add(None); c.add(np.eye(3))
        self.assertIsNone(c.take()[0])
        c.add(np.eye(3))
        self.assertIsNotNone(c.take()[0])
        c.reset(); c.add(np.eye(3))
        self.assertIsNone(c.take()[0])

    def test_split_remaps_numbers_and_leading_unknown(self):
        frames = [frame(i * .2, [row(5, feat)]) for i, feat in enumerate([0, 0, 1, 1, 1])]
        frames[0]["p"][0][6] = None
        frames[3]["num"] = [[5, 19, .9]]
        self.assertEqual(split_by_team(frames, model()), 1)
        self.assertEqual(frames[0]["p"][0][5], 5)
        self.assertNotEqual(frames[3]["p"][0][5], 5)
        self.assertEqual(frames[3]["num"][0][0], frames[3]["p"][0][5])

    def test_short_fragments_after_split(self):
        frames = [frame(i * .2, [row(1, feat)]) for i, feat in enumerate([0, 0, 0, 1, 1])]
        refined, _ = refine_frames(frames, model())
        self.assertIsNotNone(refined[0]["p"][0][5])
        self.assertIsNone(refined[3]["p"][0][5])
        self.assertEqual(frames[3]["p"][0][5], 1, "Les observations brutes doivent rester intactes")

    def test_isolated_opponent_frame_is_not_attributed(self):
        frames = [frame(i * .2, [row(1, f)]) for i, f in enumerate([0, 0, 0, 1, 0, 0, 0])]
        frames[3]["num"] = [[1, 30, .9]]
        refined, report = refine_frames(frames, model())
        self.assertIsNone(refined[3]["p"][0][5])
        self.assertEqual(refined[3]["num"], [])
        self.assertEqual(refined[0]["p"][0][5], refined[-1]["p"][0][5])
        self.assertEqual(report["opposite_kit_observations_removed"], 1)

    def test_short_link_and_forbidden_boundaries(self):
        frames = [frame(t, [row(1)]) for t in [0., .2, .4]] + [frame(t, [row(2)]) for t in [1., 1.2, 1.4]]
        self.assertEqual(short_links(frames, model()), {1: 2})
        frames[3]["H"] = None
        self.assertEqual(short_links(frames, model()), {})
        frames[3]["H"] = np.eye(3).reshape(-1).tolist()
        for f in frames[3:]:
            f["shot"] = 2
        self.assertEqual(short_links(frames, model()), {})

    def test_overlap_and_ambiguity_abstain(self):
        frames = [frame(t, [row(1)] + ([row(2)] if t >= .2 else [])) for t in [0., .2, .4]]
        frames += [frame(t, [row(2)]) for t in [.6, .8]]
        self.assertEqual(short_links(frames, model()), {})
        frames = [frame(t, [row(1)]) for t in [0., .2, .4]]
        frames += [frame(t, [row(2), row(3, x=11)]) for t in [1., 1.2, 1.4]]
        self.assertEqual(short_links(frames, model()), {})

    def test_merged_tracks_keep_identity_boundaries(self):
        frames = [frame(t, [row(1)]) for t in [0., .2, .4]] + [frame(t, [row(2)]) for t in [1.2, 1.4, 1.6]]
        refined, _ = refine_frames(frames, model())
        tracks = build_tracks(refined)
        self.assertEqual(len(tracks), 1)
        tr = next(iter(tracks.values()))
        tr["risk_times"] = [.2]
        self.assertEqual(track_segments(tr), [(0., .2), (.4, .4), (1.2, 1.6)])

    def test_refine_cache_rebuilds_after_team_change(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            fs = [frame(t, [row(1)]) for t in [0., .2, .4]]
            (d / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in fs))
            path, report = refine_file(d, model())
            first = path.stat().st_mtime_ns
            _, repeated = refine_file(d, model())
            self.assertEqual(path.stat().st_mtime_ns, first)
            changed = model(); changed.cluster_labels[0] = "opponent"
            _, updated = refine_file(d, changed)
            self.assertNotEqual(report["input"], updated["input"])
            self.assertEqual(repeated, report)

    def test_restart_archives_without_touching_questions(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            ws = SimpleNamespace(root=d)
            s = Session(ws, d / "session")
            s.dir.mkdir()
            s.meta = {"stages": {"timeline": {"status": "done"}, "analyze": {"status": "done"}}}
            (s.dir / "frames.jsonl").write_text("old")
            (d / "questions.json").write_text("answers")
            s.restart_analysis()
            self.assertFalse((s.dir / "frames.jsonl").exists())
            self.assertEqual(len(list((s.dir / "analysis_backups").glob("*/frames.jsonl"))), 1)
            self.assertEqual((d / "questions.json").read_text(), "answers")
            self.assertIn("timeline", s.meta["stages"])
            self.assertNotIn("analyze", s.meta["stages"])

    def test_analyze_fifteen_motion_five_detections_and_resume(self):
        class Video:
            def iter_frames(self, step, start, end):
                for i in range(round(start * 15), round(end * 15)):
                    yield i / 15, i, np.zeros((100, 100, 3), np.uint8)
        class Detector:
            calls = 0
            def __init__(self, path): pass
            def detect(self, image, tiles):
                Detector.calls += 1
                return [Detection(10, 10, 20, 30, .9, PERSON)]
        class Motion:
            calls = 0
            def __init__(self): self.reset()
            def reset(self): self.first = True
            def step(self, *args):
                Motion.calls += 1
                if self.first:
                    self.first = False
                    return None, 0
                return np.array([[1., 0, 1], [0, 1, 0], [0, 0, 1]]), 30
        tl = SimpleNamespace(segments=[{"kind": "live", "file_start": 0., "file_end": 1.}])
        patches = {"YoloxDetector": Detector, "CameraMotion": Motion,
                   "pitch_region": lambda f: (np.full((100, 100), 255, np.uint8), .5),
                   "classify_view": lambda *a: "wide", "hist_signature": lambda f: 0,
                   "is_cut": lambda a, b: (a is None, float(a is None)),
                   "torso_feature": lambda *a: np.asarray([0.])}
        with tempfile.TemporaryDirectory() as td, patch.multiple(analyze, **patches):
            d = Path(td)
            result = analyze.run(Video(), tl, d, "fake.onnx")
            self.assertEqual(Detector.calls, 5)
            self.assertEqual(Motion.calls, 15)
            frames = read_jsonl(d / "frames.jsonl")
            self.assertEqual(len(frames), 5)
            self.assertEqual([f["H"][2] for f in frames[1:]], [3.] * 4)
            # Une ligne non validée par le checkpoint doit disparaître à la reprise.
            with (d / "frames.jsonl").open("a") as f:
                f.write(json.dumps(frame(2., [row(999)])) + "\n")
            analyze.run(Video(), tl, d, "fake.onnx")
            self.assertEqual(len(read_jsonl(d / "frames.jsonl")), 5)
            with self.assertRaisesRegex(ValueError, "restart-analysis"):
                analyze.run(Video(), tl, d, "fake.onnx", camera_fps=10)

    def test_old_partial_analysis_is_not_mixed(self):
        with tempfile.TemporaryDirectory() as td:
            write_json(Path(td) / "analyze_state.json", {"frames": 100})
            with self.assertRaisesRegex(ValueError, "ancienne version"):
                analyze.run(None, None, Path(td), "fake.onnx")


if __name__ == "__main__":
    unittest.main(verbosity=2)
