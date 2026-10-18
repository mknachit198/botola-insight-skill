"""Contrôle de non-régression v8.1 : sans options, resolver-0.2 = resolver-0.1.

Depuis moteur/ :  python tests/controle_non_regression_v8_1.py

Compare, sur toute la partie 7 (test + réglage, aucune vérité terrain lue), les sorties de
resolver-0.1 (copie gelée : archive/resolver_0_1/identity_resolver.py) et de resolver-0.2 pour :
- les configurations figées de configs_v8.json (mesure officielle) ;
- les 13 règles de mesure_regles_v8.py × 3 jeux de preuves.
Champs comparés : unités (bornes, probabilités, meilleur numéro, p, lectures), conflits de
simultanéité, preuves orphelines, zones de conflit, et identité de chaque observation.
Les champs ajoutés en 0.2 (sure_numbers_read, handoffs, posterior_mode) sont ignorés.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from raja_engine import evidence as ev_mod  # noqa: E402
from raja_engine import identity_resolver as new  # noqa: E402
from raja_engine.perception import read_tracklets  # noqa: E402
from mesure_regles_v8 import ALLOWED, RULES  # noqa: E402

B = ROOT.parent
UNIT_KEYS = ("unit", "track", "start", "end", "n_obs", "attributable", "end_reason", "probs", "best", "p",
             "n_readings", "numbers_read", "support", "obs_t")


def load_old():
    src = (B / "archive/resolver_0_1/identity_resolver.py").read_text(encoding="utf-8")
    src = src.replace("from .perception import crossing_cuts", "from raja_engine.perception import crossing_cuts")
    mod = types.ModuleType("resolver_0_1")
    exec(compile(src, "resolver_0_1", "exec"), mod.__dict__)
    assert mod.VERSION == "resolver-0.1"
    return mod


def norm(res, mod):
    units = [{k: (list(u[k]) if k == "obs_t" else u[k]) for k in UNIT_KEYS} for u in res["units"]]
    obs = mod.observation_identities(res)
    return {"units": units, "conflicts": res["simultaneity_conflicts"], "orphans": res["orphan_evidence"],
            "zones": res["conflict_zones"], "obs": obs}


def main():
    old = load_old()
    tl = read_tracklets(B / "etape1_lecture_dense/tracklets_part7.jsonl")
    ev = ev_mod.read(B / "etape1_lecture_dense/evidence_part7.jsonl")
    cases = []
    cfg = json.loads((ROOT / "tests/configs_v8.json").read_text(encoding="utf-8"))
    for name, c in cfg["configs"].items():
        cases.append((f"configs_v8:{name}", [e for e in ev if e["source"] in c["sources"]], c["rule"]))
    sets = {"v7": [e for e in ev if e["source"] == "legacy_v7_annotation"],
            "VLM": [e for e in ev if e["source"] == "vlm_plank"], "union": ev}
    for rname, rule in RULES:
        for sname, evs in sets.items():
            cases.append((f"{rname.split(' ')[0]}:{sname}", evs, rule))
    raja = {i: t for i, t in tl.items() if t["team"] == "raja"}
    bad = []
    for name, evs, rule in cases:
        a = norm(old.resolve(raja, evs, ALLOWED, rule), old)
        b = norm(new.resolve(raja, evs, ALLOWED, rule), new)
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            bad.append(name)
    print(f"{len(cases)} cas comparés ({old.VERSION} contre {new.VERSION}, options v8.1 désactivées) : "
          f"{len(cases) - len(bad)} identiques, {len(bad)} différents {bad}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
