"""Règles de propagation × jeux de preuves, mesurées SANS vérité terrain (étape 1.2).

Depuis moteur/ :
    python tests/mesure_regles_v8.py [--dir ../etape1_lecture_dense] [--split reglage|test|tout]

Pour chaque règle et chaque jeu de preuves (v7, VLM, union) :
- couverte      : observations Raja dans une unité portant ≥ 1 lecture (définition de l'annexe A) ;
- publiée≥0,90  : observations dont l'identité résolue a p ≥ 0,90 (score NON calibré) ;
- contradictions: unités à ≥ 2 numéros lus / unités à ≥ 2 lectures ;
- lecture-contre-lecture (LOO) : chaque lecture est comparée au vote des AUTRES lectures de
  son unité (instant différent) ; un désaccord révèle une propagation à travers un changement
  de joueur ou une erreur de lecture. Estimation de précision sans vérité terrain, biaisée vers
  les instants lisibles ; elle ne remplace pas eval_groundtruth.py.
Partage : chaînes commençant avant t_local = 208,0 s (3 premières minutes de direct) = test ;
après = réglage. Le choix d'une règle ne doit regarder que la colonne réglage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from raja_engine import evidence as ev_mod  # noqa: E402
from raja_engine.identity_resolver import resolve  # noqa: E402
from raja_engine.perception import read_tracklets  # noqa: E402

TEST_END_LOCAL = 208.0
ALLOWED = [2, 3, 7, 11, 13, 14, 19, 21, 30, 45]  # titulaires de champ, partie 7 (28'–35')
RULES = [
    ("A strict moteur : IoU>0,15 ou pieds<0,6 h ; trou 0,6 s", dict(cut_iou=.15, cut_foot=.6, max_gap=.6, kit_cut="none")),
    ("B vrai recouvrement IoU>0,15 ; trou 0,6 s", dict(cut_iou=.15, cut_foot=0, max_gap=.6, kit_cut="none")),
    ("C fort recouvrement IoU>0,4 ; trou 0,6 s", dict(cut_iou=.4, cut_foot=0, max_gap=.6, kit_cut="none")),
    ("D aucune coupure ; trou 0,6 s", dict(cut_iou=1.1, cut_foot=0, max_gap=.6, kit_cut="none")),
    ("E aucune coupure ; trou 3 s", dict(cut_iou=1.1, cut_foot=0, max_gap=3., kit_cut="none")),
    ("F IoU>0,4 ; trou 3 s", dict(cut_iou=.4, cut_foot=0, max_gap=3., kit_cut="none")),
    ("G IoU>0,4 ; trou 3 s ; tenue étrangère", dict(cut_iou=.4, cut_foot=0, max_gap=3., kit_cut="foreign")),
    ("H IoU>0,4 ; trou 3 s ; toute tenue ≠ Raja", dict(cut_iou=.4, cut_foot=0, max_gap=3., kit_cut="any")),
    ("I aucune coupure ; trou 3 s ; toute tenue ≠ Raja", dict(cut_iou=1.1, cut_foot=0, max_gap=3., kit_cut="any")),
    ("J IoU>0,15 ; trou 3 s ; toute tenue ≠ Raja", dict(cut_iou=.15, cut_foot=0, max_gap=3., kit_cut="any")),
    ("K = E + coupure sur lectures contradictoires", dict(cut_iou=1.1, cut_foot=0, max_gap=3., kit_cut="none",
                                                        split_on_conflict=True)),
    ("M IoU>0,2 toutes équipes ; trou 3 s (candidat, diagnostic 157/174)", dict(cut_iou=.2, cut_foot=0, max_gap=3.,
                                                        kit_cut="none", cut_any_team=True)),
    ("L = G + coupure sur lectures contradictoires", dict(cut_iou=.4, cut_foot=0, max_gap=3., kit_cut="foreign",
                                                        split_on_conflict=True)),
]


def split_of(tr):
    return "test" if tr["t_start"] < TEST_END_LOCAL else "reglage"


def measure(tracklets, evidence, rule, split):
    keep = {i: tr for i, tr in tracklets.items() if tr["team"] == "raja" and (split == "tout" or split_of(tr) == split)}
    res = resolve(keep, evidence, ALLOWED, rule)
    total = sum(tr["n_obs"] for tr in keep.values())
    by_unit_ev = defaultdict(list)
    ev_by_id = {e["id"]: e for e in evidence}
    for u in res["units"]:
        for eid in u["support"]:
            by_unit_ev[u["unit"]].append(ev_by_id[eid])
    covered = sum(u["n_obs"] for u in res["units"] if u["n_readings"] > 0)
    published = sum(u["n_obs"] for u in res["units"] if u["best"] is not None and u["p"] >= .90)
    multi = [u for u in res["units"] if u["n_readings"] >= 2]
    contra = [u for u in multi if len(u["numbers_read"]) > 1]
    agree = disagree = 0
    disagreements = []
    for uid, evs in by_unit_ev.items():
        for e in evs:
            others = [o for o in evs if abs(o["t"] - e["t"]) > 0.1]
            if not others:
                continue
            votes = Counter()
            for o in others:
                votes[o["player"]] += o["confidence"]
            pred = votes.most_common(1)[0][0]
            if pred == e["player"]:
                agree += 1
            else:
                disagree += 1
                disagreements.append((uid, e["player"], pred))
    tracks_pub = defaultdict(int)
    for u in res["units"]:
        if u["best"] is not None and u["p"] >= .90:
            tracks_pub[u["best"]] += u["n_obs"]
    return {"obs": total, "covered": covered, "covered_pct": round(100 * covered / max(total, 1), 1),
            "published": published, "published_pct": round(100 * published / max(total, 1), 1),
            "units": len(res["units"]), "units_multi": len(multi), "units_contra": len(contra),
            "loo_agree": agree, "loo_disagree": disagree,
            "loo_pct": None if agree + disagree == 0 else round(100 * agree / (agree + disagree), 1),
            "conflicts": len(res["simultaneity_conflicts"]), "orphans": len(res["orphan_evidence"]),
            "per_player_obs": dict(sorted(tracks_pub.items())), "disagreements": disagreements}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ROOT.parent / "etape1_lecture_dense")
    ap.add_argument("--split", default="reglage", choices=["reglage", "test", "tout"])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    tl = read_tracklets(a.dir / "tracklets_part7.jsonl")
    ev = ev_mod.read(a.dir / "evidence_part7.jsonl")
    sets = {"v7": [e for e in ev if e["source"] == "legacy_v7_annotation"],
            "VLM": [e for e in ev if e["source"] == "vlm_plank"], "union": ev}
    out = {}
    for sname, evs in sets.items():
        keep = {i for i, tr in tl.items() if tr["team"] == "raja" and (a.split == "tout" or split_of(tr) == a.split)}
        per = defaultdict(set)
        cnt = Counter()
        for e in evs:
            if e["track"] in keep and e["confidence"] >= 0.7:
                per[e["track"]].add(e["player"])
                cnt[e["track"]] += 1
        multi = [t for t in per if cnt[t] >= 2]
        two = [t for t in multi if len(per[t]) > 1]
        out[f"chaines | {sname}"] = {"chains_read": len(per), "chains_2plus_reads": len(multi), "chains_two_numbers": sorted(two)}
        print(f"Contradictions internes par CHAÎNE ({sname}, lectures ≥ 0,7) : {len(two)}/{len(multi)} "
              f"chaînes à ≥ 2 lectures ; chaînes lues : {len(per)}/{len(keep)} ; {sorted(two)}")
    print(f"Jeu : {a.split}  (score p non calibré ; LOO = lecture contre les autres lectures de son unité)")
    print(f"{'règle':52s} {'preuves':6s} {'couv.':>6s} {'pub≥.9':>7s} {'contra':>9s} {'LOO':>12s}")
    for name, rule in RULES:
        for sname, evs in sets.items():
            m = measure(tl, evs, rule, a.split)
            out[f"{name} | {sname}"] = m
            loo = f"{m['loo_agree']}/{m['loo_agree'] + m['loo_disagree']}"
            print(f"{name:52s} {sname:6s} {m['covered_pct']:5.1f}% {m['published_pct']:6.1f}% "
                  f"{m['units_contra']:3d}/{m['units_multi']:<4d} {loo:>12s}  (obs {m['obs']})")
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
