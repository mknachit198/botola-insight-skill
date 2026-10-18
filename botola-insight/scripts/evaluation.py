"""Audit reproductible ; aucune recherche d'hyperparamètres."""
import json
import time
from common import match, read, record, validate_readings, write


def readings_check(args):
    meta, data = validate_readings(args.meta, args.readings)
    n = sum(r is not None for cs in data["readings"].values() for r in cs.values())
    print(json.dumps({"valid": True, "readings": n, "planches": len(meta["planches"])}))


def deterministic_comparison(expected, actual):
    """Exceptions NOMMÉES seulement : pas de suppression générale des différences."""
    import copy
    e, a = copy.deepcopy(expected), copy.deepcopy(actual)
    differences = []
    def pop_pair(e_obj, a_obj, key, path):
        sentinel = "<absent>"
        ev, av = e_obj.pop(key, sentinel), a_obj.pop(key, sentinel)
        if ev != av:
            differences.append({"path": path, "expected": ev, "actual": av})
    pop_pair(e, a, "mesure", "/mesure")
    for name in set(e.get("results", {})) & set(a.get("results", {})):
        for k in ("resolve_wall_s", "unit_contradictions"):
            # unit_contradictions n'est une exception que si absent de l'historique.
            if k == "unit_contradictions" and k in e["results"][name]:
                continue
            pop_pair(e["results"][name], a["results"][name], k, f"/results/{name}/{k}")
    return {"deterministic_equal": e == a, "byte_identical_claimed": False,
            "documented_metadata_differences": differences,
            "policy": "Temps mesurés conservés ; champs nouveaux consignés ; aucun score exclu."}


def evaluate(args):
    from raja_engine import evidence as ev_mod
    from raja_engine.identity_resolver import resolve, observation_identities, roster_allowed
    from raja_engine.perception import read_tracklets
    from raja_engine.roster import Roster
    from raja_engine.timeline import Timeline
    from legacy_eval_groundtruth import evaluate as measure, read_jsonl
    cfg, configurations = match(args.match), read(args.configs)
    key = "roster" if args.team_label.startswith("raja") else "opponent_roster"
    if key not in cfg:
        raise ValueError("Composition de l'équipe évaluée absente.")
    allowed = roster_allowed(Roster({"roster": cfg[key]}), Timeline(read(args.timeline)),
                             outfield=not args.team_label.endswith("_gk"))
    gt = [g for g in read_jsonl(args.gt) if g.get("set", args.set) == args.set]
    if not gt:
        raise ValueError("Aucune observation annotée pour le jeu demandé.")
    tracks, evidence = read_tracklets(args.tracklets), ev_mod.read(args.evidence)
    out = {"set": args.set, "gt_file": args.gt.name, "configs_frozen_at": configurations["frozen_at"], "results": {}}
    for name, c in configurations["configs"].items():
        evs = [e for e in evidence if e["source"] in c["sources"]]
        start = time.perf_counter()
        res = resolve(tracks, evs, allowed, c["rule"], team=args.team_label)
        wall = time.perf_counter() - start
        metrics = measure(gt, observation_identities(res), configurations["threshold_visible_time"], evs, wall, units=res["units"])
        out["results"][name] = {"label": c["label"], **metrics}
    write(args.out, out)
    record(args.out.parent, "evaluate", [args.tracklets, args.evidence, args.gt, args.match, args.timeline, args.configs],
           {"set": args.set, "purpose": "reproduction seule ; aucun réglage"}, [args.out])
    if args.expected:
        comparison = deterministic_comparison(read(args.expected), read(args.out))
        write(args.out.with_suffix(".comparison.json"), comparison)
        if not comparison["deterministic_equal"]:
            raise ValueError("L'évaluation ne reproduit pas tous les champs déterministes attendus.")
    print(json.dumps({"set": args.set, "configurations": len(out["results"]), "out": str(args.out)}, ensure_ascii=False))
