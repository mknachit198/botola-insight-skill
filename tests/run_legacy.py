#!/usr/bin/env python3
"""Exécuter les tests historiques sans publier leurs données privées."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "botola-insight"
TESTS = ["test_engine.py", "test_v1_layers.py", "test_global_identity.py", "test_tracking_step1.py",
         "controle_non_regression_v8_1.py"]


def run(reference, out):
    reference, out = reference.resolve(), out.resolve()
    required = ["moteur/matches/raja-zemamra", "etape1_lecture_dense/tracklets_part7.jsonl", "etape1_lecture_dense/evidence_part7.jsonl"]
    missing = [p for p in required if not (reference / p).exists()]
    if missing:
        raise ValueError("Relais externe incomplet : " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="botola-tests-") as td:
        base = Path(td)
        engine = base / "moteur"
        shutil.copytree(SKILL / "scripts/moteur/raja_engine", engine / "raja_engine", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tests/legacy", engine / "tests")
        shutil.copy(SKILL / "assets/configs_v8.json", engine / "tests/configs_v8.json")
        shutil.copytree(reference / "moteur/matches/raja-zemamra", engine / "matches/raja-zemamra")
        (base / "etape1_lecture_dense").mkdir()
        for name in ("tracklets_part7.jsonl", "evidence_part7.jsonl"):
            shutil.copy(reference / "etape1_lecture_dense" / name, base / "etape1_lecture_dense" / name)
        (base / "archive/resolver_0_1").mkdir(parents=True)
        shutil.copy(ROOT / "tests/legacy/resolver_0_1.py", base / "archive/resolver_0_1/identity_resolver.py")
        for name in TESTS:
            proc = subprocess.run([sys.executable, "tests/" + name], cwd=engine, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env={**os.environ, "ORT_DISABLE_TELEMETRY": "1"})
            (out / (name + ".log")).write_text(proc.stdout, encoding="utf-8")
            results[name] = {"exit_code": proc.returncode, "output": proc.stdout}
            print(f"{name}: {'OK' if proc.returncode == 0 else 'ÉCHEC'}", flush=True)
    (out / "legacy-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(r["exit_code"] == 0 for r in results.values())
    ok = ok and "45 identiques, 0 différents" in results[TESTS[-1]]["output"]
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(0 if run(a.reference, a.out) else 1)
