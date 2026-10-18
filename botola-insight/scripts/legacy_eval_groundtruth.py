"""Évaluation officielle contre la vérité terrain (cahier v1, §4) — un seul script, un seul dénominateur.

Depuis moteur/ :
    python tests/eval_groundtruth.py                       # jeu de test, configurations figées
    python tests/eval_groundtruth.py --set reglage         # jeu de réglage (le seul pour choisir)
    python tests/eval_groundtruth.py --gt ../groundtruth/part7_3min.jsonl --json ../groundtruth/eval_test.json
    python tests/eval_groundtruth.py --gt ../groundtruth/part7_3min_arbitre_v8_1.jsonl --configs tests/configs_v8_1.json

Cinq métriques, sur les observations Raja annotées du jeu choisi :
1. précision      : parmi les observations publiées avec une identité (p ≥ seuil) et vérifiables,
                    part correcte (une identité publiée sur « pas un joueur Raja » compte fausse) ;
2. couverture     : part des observations annotées d'un numéro qui sont publiées avec une identité
                    (et couverture correcte : publiées ET justes) ;
3. couverture par joueur et plancher (minimum sur les joueurs ayant ≥ 30 observations annotées) ;
4. contradictions internes :
   - par unité résolue (couche 2) : unités attribuables avec ≥ 2 lectures sûres (confiance ≥ 0,7)
     portant ≥ 2 numéros distincts ;
   - par chaîne brute (couche 1, rappel) : chaînes du suivi avec ≥ 2 lectures sûres portant ≥ 2 numéros ;
5. temps de calcul de la résolution.
Critère de l'étape 1 : la contradiction qui compte est celle des CHAÎNES (santé du suivi). Avec une
règle qui coupe les unités entre lectures contradictoires (K), la contradiction par unité est nulle
par construction : elle ne prouve rien à elle seule.
Configurations : configs_v8.json (figées avant la vérité terrain, mesure officielle) ;
configs_v8_1.json (candidates, conçues APRÈS lecture du test : le test ne peut pas les valider).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from raja_engine import evidence as ev_mod  # noqa: E402
from raja_engine.identity_resolver import observation_identities, resolve  # noqa: E402
from raja_engine.perception import read_tracklets  # noqa: E402
from raja_engine.publish import published_identity  # noqa: E402

BUNDLE = ROOT.parent
ALLOWED = [2, 3, 7, 11, 13, 14, 19, 21, 30, 45]
MIN_PLAYER_OBS = 30


def read_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def chain_contradictions(evidence, tracks):
    per, cnt = defaultdict(set), Counter()
    for e in evidence:
        if e["track"] in tracks and e["confidence"] >= 0.7 and e["player"] is not None:
            per[e["track"]].add(e["player"])
            cnt[e["track"]] += 1
    multi = [t for t in per if cnt[t] >= 2]
    two = sorted(t for t in multi if len(per[t]) > 1)
    return {"chains_two_numbers": len(two), "chains_2plus_reads": len(multi),
            "pct": None if not multi else round(100 * len(two) / len(multi), 1), "tracks": two}


def unit_contradictions(units, evidence, tracks):
    """Contradictions internes sur les unités attribuables produites par la couche 2 (lectures ≥ 0,7)."""
    ev_by_id = {e["id"]: e for e in evidence}
    multi_u, two_u = [], []
    for u in units:
        if not u.get("attributable", True) or u.get("track") not in tracks:
            continue
        sure_evs = [ev_by_id[eid] for eid in u.get("support", [])
                    if eid in ev_by_id and ev_by_id[eid]["confidence"] >= 0.7 and ev_by_id[eid]["player"] is not None]
        if len(sure_evs) >= 2:
            multi_u.append(u["unit"])
            if len({e["player"] for e in sure_evs}) > 1:
                two_u.append(u["unit"])
    return {"units_two_numbers": len(two_u), "units_2plus_reads": len(multi_u),
            "pct": None if not multi_u else round(100 * len(two_u) / len(multi_u), 1), "units": two_u}


def evaluate(gt_rows, identities, threshold, evidence, wall_s, units=None):
    sysmap = {(r["track"], round(r["t"], 3)): r for r in identities}
    n_ok = correct = wrong = unverifiable = published_on_ok = 0
    per_gt, per_correct = Counter(), Counter()
    errors = Counter()
    for g in gt_rows:
        s = sysmap.get((g["track"], round(g["t"], 3)), {"player": None, "p": 0.0})
        pub = published_identity({**s}, "visible_time", {"visible_time": threshold})
        if g["status"] == "ok":
            n_ok += 1
            per_gt[g["jersey"]] += 1
            if pub is not None:
                published_on_ok += 1
                if pub == g["jersey"]:
                    correct += 1
                    per_correct[g["jersey"]] += 1
                else:
                    wrong += 1
                    errors[(g["jersey"], pub)] += 1
        elif g["status"] == "not_raja":
            if pub is not None:
                wrong += 1
                errors[("pas Raja", pub)] += 1
        elif pub is not None:
            unverifiable += 1
    per_player = {j: {"annotated": per_gt[j], "correct": per_correct[j],
                      "coverage_pct": round(100 * per_correct[j] / per_gt[j], 1)} for j in sorted(per_gt)}
    eligible = [v["coverage_pct"] for v in per_player.values() if v["annotated"] >= MIN_PLAYER_OBS]
    tracks = {g["track"] for g in gt_rows}
    chain_c = chain_contradictions(evidence, tracks)
    unit_c = unit_contradictions(units, evidence, tracks) if units is not None else None
    return {
        "gt_obs": len(gt_rows), "gt_obs_with_number": n_ok, "gt_chains": len(tracks),
        "published_verifiable": correct + wrong, "correct": correct, "wrong": wrong,
        "published_unverifiable": unverifiable,
        "precision_pct": None if correct + wrong == 0 else round(100 * correct / (correct + wrong), 2),
        "precision_ci95": wilson(correct, correct + wrong),
        "coverage_pct": None if n_ok == 0 else round(100 * published_on_ok / n_ok, 2),
        "correct_coverage_pct": None if n_ok == 0 else round(100 * correct / n_ok, 2),
        "per_player": per_player, "floor_pct": min(eligible) if eligible else None,
        "contradictions": chain_c,
        "unit_contradictions": unit_c,
        "resolve_wall_s": round(wall_s, 3), "threshold": threshold,
        "top_errors": [{"truth": str(a), "published": b, "obs": n} for (a, b), n in errors.most_common(8)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="test", choices=["test", "reglage"])
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--dir", type=Path, default=BUNDLE / "etape1_lecture_dense")
    ap.add_argument("--configs", type=Path, default=Path(__file__).with_name("configs_v8.json"))
    ap.add_argument("--only", help="liste de configurations séparées par des virgules")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    gt_path = a.gt or (BUNDLE / "groundtruth" / ("part7_3min.jsonl" if a.set == "test" else "part7_reglage.jsonl"))
    if not gt_path.exists():
        print(f"Vérité terrain absente : {gt_path}. Annoter (étape 0) puis lancer import_groundtruth.py.")
        sys.exit(2)
    gt = [g for g in read_jsonl(gt_path) if g.get("set", a.set) == a.set]
    cfg = json.loads(a.configs.read_text(encoding="utf-8"))
    tl = read_tracklets(a.dir / "tracklets_part7.jsonl")
    ev = ev_mod.read(a.dir / "evidence_part7.jsonl")
    names = a.only.split(",") if a.only else list(cfg["configs"])
    out = {"set": a.set, "gt_file": gt_path.name, "configs_frozen_at": cfg["frozen_at"], "results": {}}
    print(f"Jeu {a.set} ({gt_path.name}) : {len({g['track'] for g in gt})} chaînes annotées, {len(gt)} observations, seuil p ≥ {cfg['threshold_visible_time']}")
    print(f"{'configuration':22s} {'précision':>18s} {'couverture':>11s} {'couv. juste':>11s} {'plancher':>9s} {'contra(u)':>10s} {'contra(c)':>10s} {'temps':>7s}")
    for name in names:
        c = cfg["configs"][name]
        evs = [e for e in ev if e["source"] in c["sources"]]
        t0 = time.perf_counter()
        res = resolve(tl, evs, ALLOWED, c["rule"])
        wall = time.perf_counter() - t0
        m = evaluate(gt, observation_identities(res), cfg["threshold_visible_time"], evs, wall, units=res["units"])
        out["results"][name] = {"label": c["label"], **m}
        prec = "—" if m["precision_pct"] is None else f"{m['precision_pct']:.2f}% ({m['correct']}/{m['published_verifiable']})"
        cov = "—" if m["coverage_pct"] is None else f"{m['coverage_pct']:.1f}%"
        cj = "—" if m["correct_coverage_pct"] is None else f"{m['correct_coverage_pct']:.1f}%"
        fl = "—" if m["floor_pct"] is None else f"{m['floor_pct']:.1f}%"
        ct = m["contradictions"]
        ut = m["unit_contradictions"]
        ut_s = "—" if ut is None else f"{ut['units_two_numbers']:>3d}/{ut['units_2plus_reads']:<4d}"
        ct_s = f"{ct['chains_two_numbers']:>3d}/{ct['chains_2plus_reads']:<4d}"
        print(f"{name:22s} {prec:>18s} {cov:>11s} {cj:>11s} {fl:>9s} {ut_s:>10s} {ct_s:>10s} {m['resolve_wall_s']:6.2f}s")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
