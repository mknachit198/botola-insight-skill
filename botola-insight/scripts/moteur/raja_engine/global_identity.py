"""Attribution expérimentale par numéros et continuité spatiale et temporelle.

Entrées : tronçons sans croisement, lectures de maillots, composition et
raccords géométriques. Aucun profil physique. Les poids sont des heuristiques,
pas des probabilités. Ce module ne modifie ni identity.json ni les statistiques.
OR-Tools reste une dépendance facultative, importée uniquement au calcul.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

VERSION = "global-0.2"


def split_safe_segments(track, max_gap=.6):
    """Isole chaque observation de croisement et coupe tout trou > max_gap."""
    risks = set(track.get("risk_times", []))
    result, current = [], []
    for observation in track["obs"]:
        crossing = observation["t"] in risks
        if current and (crossing or observation["t"] - current[-1]["t"] > max_gap + 1e-6):
            result.append({"obs": current, "crossing_observation": False})
            current = []
        if crossing:
            result.append({"obs": [observation], "crossing_observation": True})
        else:
            current.append(observation)
    if current:
        result.append({"obs": current, "crossing_observation": False})
    return result


def anchor_seeds(anchors, conflict_groups=()):
    """Écarte les lectures contradictoires et les doublons simultanés."""
    votes = defaultdict(set)
    for anchor in anchors:
        votes[anchor["node"]].add(anchor["number"])
    seeds = {i: next(iter(v)) for i, v in votes.items() if len(v) == 1}
    conflicts = {i: {"reason": "conflicting_number_reads", "numbers": sorted(v)}
                 for i, v in votes.items() if len(v) > 1}
    collided = defaultdict(set)
    for group in conflict_groups:
        by_number = defaultdict(list)
        for nid in group:
            if nid in seeds:
                by_number[seeds[nid]].append(nid)
        for ids in by_number.values():
            if len(ids) > 1:
                for nid in ids:
                    collided[nid].update(set(ids) - {nid})
    for nid, rivals in collided.items():
        conflicts[nid] = {"reason": "simultaneous_number_reads", "numbers": [seeds.pop(nid)],
                          "other_nodes": sorted(rivals)}
    return seeds, conflicts


def motion_support(nodes, edges, seeds, blocked=(), max_hops=3):
    """Meilleur produit des poids, limité à trois raccords depuis un numéro lu."""
    valid = {n["id"] for n in nodes} - set(blocked)
    adjacency = defaultdict(list)
    for edge in edges:
        a, b, weight = edge["a"], edge["b"], edge["continuity_weight"]
        if a not in valid or b not in valid:
            continue
        if not 0 < weight < 1:
            raise ValueError("Un poids de continuité doit être compris entre 0 et 1")
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))
    support = defaultdict(dict)
    for source, number in seeds.items():
        queue, visited = [(-1., source, 0)], {}
        while queue:
            negative, nid, hops = heapq.heappop(queue)
            weight = -negative
            if weight <= visited.get((nid, hops), 0):
                continue
            visited[(nid, hops)] = weight
            support[nid][number] = max(weight, support[nid].get(number, 0))
            if hops >= max_hops:
                continue
            for target, quality in adjacency[nid]:
                # Une lecture directe contraire est une barrière, pas un relais.
                if target in seeds and seeds[target] != number:
                    continue
                product = weight * quality
                if product > .5:
                    heapq.heappush(queue, (-product, target, hops + 1))
    return dict(support)


def make_scores(nodes, edges, conflict_groups, anchors, continuity_floor=.8):
    seeds, conflicts = anchor_seeds(anchors, conflict_groups)
    allowed = {n["id"]: set(n["allowed_players"]) for n in nodes}
    for nid, number in list(seeds.items()):
        if nid not in allowed or number not in allowed[nid]:
            conflicts[nid] = {"reason": "number_outside_declared_roster", "numbers": [seeds.pop(nid)]}
    support = motion_support(nodes, edges, seeds, conflicts)
    scores, evidence = {}, {}
    for node in nodes:
        nid = node["id"]
        utilities, details = {None: 0.}, {}
        if nid not in conflicts:
            for number in node["allowed_players"]:
                weight = support.get(nid, {}).get(number, 0.)
                if seeds.get(nid) == number:
                    utility, source = 8., "number_read"
                elif nid not in seeds and weight > .5:
                    utility, source = 10 * (weight - continuity_floor), "motion_from_number_read"
                else:
                    continue
                utilities[number] = utility
                details[number] = {"source": source, "continuity_weight": round(weight, 5)}
        scores[nid], evidence[nid] = utilities, details
    return scores, evidence, seeds, conflicts


def optimize(nodes, edges, conflict_groups, scores, seconds=3., forbidden=None):
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise RuntimeError("Installer requirements-global.txt pour ce prototype facultatif") from error
    model = cp_model.CpModel()
    variables, objective = {}, []
    for node in nodes:
        nid, local = node["id"], []
        for number, score in scores[nid].items():
            variable = model.new_bool_var(f"x_{nid}_{number}")
            variables[(nid, number)] = variable
            local.append(variable)
            objective.append(round(1000 * score) * variable)
        model.add_exactly_one(local)
    for group in conflict_groups:
        numbers = set().union(*(set(scores[i]) for i in group)) - {None}
        for number in numbers:
            simultaneous = [variables[(i, number)] for i in group if (i, number) in variables]
            if len(simultaneous) > 1:
                model.add_at_most_one(simultaneous)
    # La cohérence ne crée aucun candidat et ne peut réunir des observations simultanées.
    for edge in edges:
        a, b = edge["a"], edge["b"]
        for number in (set(scores[a]) & set(scores[b])) - {None}:
            if min(scores[a][number], scores[b][number]) < -1:
                continue
            same = model.new_bool_var(f"edge_{a}_{b}_{number}")
            model.add(same <= variables[(a, number)])
            model.add(same <= variables[(b, number)])
            model.add(same >= variables[(a, number)] + variables[(b, number)] - 1)
            objective.append(round(100 * edge["continuity_weight"]) * same)
    if forbidden is not None:
        model.add(variables[forbidden] == 0)
    model.maximize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 17
    status = solver.solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assignments = {}
    if feasible:
        for node in nodes:
            nid = node["id"]
            assignments[nid] = next(n for n in scores[nid] if solver.value(variables[(nid, n)]))
    return {"status": solver.status_name(status), "optimal": status == cp_model.OPTIMAL,
            "objective": solver.objective_value / 1000 if feasible else None,
            "assignments": assignments, "wall_s": solver.wall_time}


def solve(nodes, edges, conflict_groups, anchors, min_margin=.25, seconds=3., continuity_floor=.8):
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Identifiants de tronçons dupliqués")
    for edge in edges:
        a, b = by_id[edge["a"]], by_id[edge["b"]]
        if a["shot"] != b["shot"] or a["end"] >= b["start"]:
            raise ValueError("Raccord entre plans distincts ou tronçons qui se chevauchent")
        if a.get("crossing_observation") or b.get("crossing_observation"):
            raise ValueError("Raccord sur une observation de croisement")
    scores, evidence, seeds, conflicts = make_scores(nodes, edges, conflict_groups, anchors, continuity_floor)
    best = optimize(nodes, edges, conflict_groups, scores, seconds)
    accepted, records, counterfactual_s = {}, [], 0.
    for node in nodes:
        nid = node["id"]
        number = best["assignments"].get(nid)
        margin, reason = None, "unknown"
        if number is not None:
            if seeds.get(nid) == number:
                accepted[nid], reason = number, "number_read"
            elif best["optimal"]:
                alternative = optimize(nodes, edges, conflict_groups, scores, seconds, (nid, number))
                counterfactual_s += alternative["wall_s"]
                if alternative["optimal"]:
                    margin = best["objective"] - alternative["objective"]
                    if margin >= min_margin:
                        accepted[nid], reason = number, "candidate_margin_passed"
                    else:
                        reason = "global_margin_too_small"
                else:
                    reason = "alternative_not_proven_optimal"
            else:
                reason = "solution_not_proven_optimal"
        if nid in conflicts:
            reason = conflicts[nid]["reason"]
        records.append({"node": nid, "parent_track": node.get("parent_track"),
                        "start": node.get("start"), "end": node.get("end"),
                        "proposed": number, "accepted": accepted.get(nid), "reason": reason,
                        "global_margin": None if margin is None else round(margin, 5),
                        "n_obs": node["n_obs"], "evidence": evidence[nid].get(number),
                        "top_candidates": sorted([{"number": n, "utility": round(v, 5)}
                                                  for n, v in scores[nid].items() if n is not None],
                                                 key=lambda r: -r["utility"])[:3]})
    for group in conflict_groups:
        counts = Counter(accepted[i] for i in group if i in accepted)
        if any(count > 1 for count in counts.values()):
            raise AssertionError("Attributions simultanées incompatibles")
    total = sum(n["n_obs"] for n in nodes)
    covered = sum(n["n_obs"] for n in nodes if n["id"] in accepted)
    return {"version": VERSION, "optimal": best["optimal"], "status": best["status"],
            "objective": best["objective"], "solver_wall_s": best["wall_s"],
            "counterfactual_wall_s": counterfactual_s, "records": records,
            "accepted": accepted, "anchor_conflicts": conflicts, "simultaneity_violations": 0,
            "candidate_coverage": {"observations": total, "attributed": covered,
                                   "pct": round(100 * covered / max(total, 1), 2)},
            "mode": "numbers_and_spatiotemporal_continuity", "min_global_margin": min_margin,
            "continuity_floor": continuity_floor, "weights_are_calibrated_probabilities": False,
            "physical_traits_used_for_identity": False, "production_approved": False}
