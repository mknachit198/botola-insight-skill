#!/usr/bin/env python3
"""Archives publiques à partir de la liste de code/documentation autorisée."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from check_package import check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    selected = check()
    packages = [("botola-insight-skill.zip", [(f, "botola-insight-skill/" + n) for f, n in selected]),
                ("botola-insight.zip", [(f, n) for f, n in selected if n.startswith("botola-insight/")])]
    result = {}
    for filename, items in packages:
        dest = a.out / filename
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f, n in items:
                info = zipfile.ZipInfo(n, date_time=(2026, 10, 18, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, f.read_bytes())
        result[filename] = {"files": len(items), "bytes": dest.stat().st_size,
                            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (a.out / "SHA256SUMS.txt").write_text("\n".join(v["sha256"] + "  " + k for k, v in result.items()) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
