#!/usr/bin/env python3
"""Contrôle du contenu distribuable ; n'inspecte pas les dossiers de travail privés."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("botola-insight", "tests", "tools", "reports", ".github")
FILES = ("README.md", "LICENSE", "NOTICE.md", "A_FAIRE.md", ".gitignore", "requirements.txt", "requirements-reproduction.txt")
SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".js", ".html", ".css", ".log"}
SKIP_DIRS = {"__pycache__", "models", "node_modules", ".git", ".venv", "work", "data", "output", "outputs", "planches", "groundtruth", "annotations"}
PRIVATE_NAMES = {"lectures_vlm.json", "planches_meta.json", "teams.json", "match.json", "resolution.json", "site_match.json", "dataset.json", "eval_test.json"}


def files(root=ROOT):
    for name in FILES:
        p = root / name
        if p.is_file(): yield p, name
    for dirname in DIRS:
        base = (root / dirname).resolve()
        if not base.is_dir(): continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if any(x in SKIP_DIRS for x in rel.parts): continue
            if p.is_file() and (p.suffix in SUFFIXES or p.name in (".gitignore", "LICENSE")):
                yield p, dirname + "/" + rel.as_posix()


def check(root=ROOT):
    selected = list(files(root))
    bad = []
    for p, rel in selected:
        if p.name in PRIVATE_NAMES:
            bad.append(rel + " : donnée privée")
        text = p.read_text(encoding="utf-8")
        # Formats de secrets courants ; les noms de variables et les SHA ne sont pas des clés.
        if re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{30,}", text):
            bad.append(rel + " : secret potentiel")
        if rel.endswith("SKILL.md") and ("TODO" in text or "name: botola-insight" not in text):
            bad.append(rel + " : compétence incomplète")
    if bad: raise ValueError("\n".join(bad))
    return selected


if __name__ == "__main__":
    selected = check()
    print(json.dumps({"files": len(selected), "bytes": sum(p.stat().st_size for p, _ in selected),
                      "private_data_included": False, "secret_patterns_found": False}, indent=2))
