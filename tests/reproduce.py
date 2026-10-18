#!/usr/bin/env python3
"""Reproduction figée : aucune donnée d'annotation embarquée, aucun réglage."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "botola-insight/scripts/botola.py"
EXPECTED = {"tracklets_part7.jsonl": "773c4f5b19da672fabe9f32787797339b0ab0d527aaacdad32ee1f5eb076cb03",
            "evidence_part7.jsonl": "f29af3da57f41dcb2b06fea2f26bd3bcb85e923f602616ad1ddabc6d41222977"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    ref, out = a.reference.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    def call(name, argv):
        p = subprocess.run([sys.executable, str(CLI), *map(str, argv)], capture_output=True, text=True)
        (out / (name + ".log")).write_text(p.stdout + p.stderr, encoding="utf-8")
        if p.returncode:
            raise SystemExit(p.stdout + p.stderr)
    call("build", ["build", "--part", "7", "--frames", ref / "essais_video_partie7/camera15_final/frames_tracked.jsonl",
                   "--teams", ref / "espace_travail/raja-zemamra-reference/teams.json", "--readings-dir", ref / "etape1_lecture_dense",
                   "--legacy-dataset", ref / "essai_solveur_global_partie7/dataset.json", "--out", out / "build"])
    hashes = {}
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((out / "build" / name).read_bytes()).hexdigest()
        original = hashlib.sha256((ref / "etape1_lecture_dense" / name).read_bytes()).hexdigest()
        hashes[name] = {"expected": expected, "actual": actual, "reference": original, "equal": actual == expected == original}
    if not all(v["equal"] for v in hashes.values()):
        raise SystemExit("Empreinte différente : arrêt, sans modifier les références.")
    call("evaluate", ["evaluate", "--tracklets", out / "build/tracklets_part7.jsonl", "--evidence", out / "build/evidence_part7.jsonl",
                      "--gt", ref / "groundtruth/part7_3min.jsonl", "--match", ref / "moteur/matches/raja-zemamra/match.json",
                      "--timeline", ref / "essais_video_partie7/camera15_final/timeline_local.json",
                      "--expected", ref / "groundtruth/eval_test.json", "--out", out / "eval_test.json"])
    comparison = json.loads((out / "eval_test.comparison.json").read_text())
    report = {"purpose": "reproduction seule", "hashes": hashes, "evaluation_deterministic_equal": comparison["deterministic_equal"],
              "metadata_exception_paths": [d["path"] for d in comparison["documented_metadata_differences"]],
              "evaluation_byte_identical": False}
    (out / "reproduction.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
