"""Couche 2 — résolution d'identité : une distribution de probabilité par unité de piste.

Entrées : tracklets (couche 1, croisements décrits, tenue par observation), preuves
(evidence.jsonl), numéros autorisés (composition et fenêtres de participation), RÈGLE de
propagation. Sortie : pour chaque unité, {"probs", "best", "p", "support", "cuts"}.

Différences avec global_identity (global-0.2) :
- la règle de coupure est un PARAMÈTRE mesuré, plus une constante d'entrée : coupure aux
  croisements de même équipe au-delà d'un recouvrement, aux trous au-delà d'une durée, et
  (option) aux observations dont la tenue n'est pas celle de l'équipe de la piste ;
- toutes les lectures d'une unité votent ; des lectures contradictoires étalent la
  probabilité au lieu d'être écartées en silence ;
- l'option « ? » (inconnu / unité impure / hors composition) est toujours présente ;
- contraintes dures : un numéro par unité, jamais le même numéro sur deux unités
  simultanées ; numéros hors composition refusés.

Options v8.1 (resolver-0.2), reprises d'un correctif externe après revue. TOUTES DÉSACTIVÉES par
défaut : sans elles, unités, probabilités et identités par observation sont identiques à resolver-0.1
(contrôlé par tests/controle_non_regression_v8_1.py) et les configurations figées du v8 se rejouent
à l'identique.
- règle["trim_handoff"] = True : quand deux unités simultanées revendiquent le même joueur et que
  leurs lectures de ce joueur sont disjointes dans le temps (relais d'une piste à l'autre), seule
  la zone de recouvrement non ancrée est élaguée, au lieu d'annuler toute l'unité perdante ;
- règle["posterior"] = "confiance" : le taux de faux positif est ramené à l'a priori pour qu'une
  lecture unique de confiance c donne exactement p = c (au lieu de 0,8608 pour c = 0,90 en mode
  "v8"). Ce n'est PAS une calibration : les confiances 0,9 / 0,7 / 0,5 sont des étiquettes du
  lecteur. Effet mesuré : avec la règle stricte du point zéro, la précision sur le test tombe de
  96,2 % à 84,7 %. À ne retenir qu'après calibration sur le réglage annoté (annexe B) ;
- `sure_numbers_read` consigne les numéros lus avec confiance ≥ 0,7 par unité.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .perception import crossing_cuts

VERSION = "resolver-0.2"

# Règle de départ recommandée par le cahier v1 (§5, 1.2) ; à confirmer sur le jeu de réglage.
DEFAULT_RULE = {"cut_iou": 0.4, "cut_foot": 0.0, "max_gap": 3.0, "kit_cut": "foreign"}
STRICT_RULE = {"cut_iou": 0.15, "cut_foot": 0.6, "max_gap": 0.6, "kit_cut": "none"}
FOREIGN_KITS = {"referee", "opponent", "opponent_gk"}
PRIOR_UNKNOWN = 0.10      # part a priori d'unités impures / non identifiables (non calibrée)
UNKNOWN_READ_P = 1 / 20   # probabilité historique v8 qu'une unité « ? » produise une lecture donnée


def _foreign(kit, team, mode):
    if mode == "none" or kit == team:
        return False
    if mode == "foreign":
        return kit in FOREIGN_KITS
    if mode == "any":
        return True
    raise ValueError(f"kit_cut inconnu : {mode}")


def split_units(tr: dict, rule: dict) -> list[dict]:
    """Découpe une piste en unités. Une observation de croisement coupant ou de tenue
    étrangère devient une unité isolée « non attribuable » (jamais étiquetée)."""
    cuts = set(crossing_cuts(tr, rule["cut_iou"], rule.get("cut_foot", 0.0), any_team=rule.get("cut_any_team", False)))
    team = tr.get("team")
    out, cur = [], []

    def close(reason):
        nonlocal cur
        if cur:
            out.append({"obs": cur, "attributable": True, "end_reason": reason})
            cur = []

    for o in tr["obs"]:
        t = o["t"]
        isolated = None
        if t in cuts:
            isolated = "crossing"
        elif _foreign(o.get("kit"), team, rule.get("kit_cut", "none")):
            isolated = f"kit:{o.get('kit')}"
        if cur and t - cur[-1]["t"] > rule["max_gap"] + 1e-6:
            close("gap")
        if isolated:
            close(isolated)
            out.append({"obs": [o], "attributable": False, "end_reason": isolated})
        else:
            cur.append(o)
    close("end")
    return out


def build_units(tracklets: dict, rule: dict, team: str = "raja") -> list[dict]:
    units = []
    for tid in sorted(tracklets):
        tr = tracklets[tid]
        if tr.get("team") != team:
            continue
        for k, u in enumerate(split_units(tr, rule)):
            units.append({"id": f"{tid}.{k}", "track": tid, "shot": tr["shot"], "start": u["obs"][0]["t"],
                          "end": u["obs"][-1]["t"], "n_obs": len(u["obs"]), "attributable": u["attributable"],
                          "end_reason": u["end_reason"], "obs_t": [o["t"] for o in u["obs"]]})
    return units


def attach_evidence(units, evidence, tol=0.11):
    """Rattache chaque preuve à l'unité contenant (piste, instant). Une preuve tombant sur
    une observation non attribuable n'est rattachée à rien (consignée à part)."""
    by_track = defaultdict(list)
    for u in units:
        by_track[u["track"]].append(u)
    attached, orphans = defaultdict(list), []
    for ev in evidence:
        best = None
        for u in by_track.get(ev["track"], []):
            for t in u["obs_t"]:
                d = abs(t - ev["t"])
                if d <= tol and (best is None or d < best[0]):
                    best = (d, u)
        if best is None or not best[1]["attributable"]:
            orphans.append(ev["id"])
        else:
            attached[best[1]["id"]].append(ev)
    return attached, orphans


def split_on_conflicts(units, attached, min_conf=0.7):
    """Correction à l'exécution, sans vérité terrain ni paramètre réglé : si une unité porte
    des lectures sûres (confiance ≥ min_conf) de numéros différents, le changement de joueur
    a eu lieu entre deux lectures consécutives discordantes. L'unité est coupée : la partie
    jusqu'à la dernière lecture du premier numéro, une ZONE DE CONFLIT non attribuable (on ne
    sait pas où se situe la bascule), puis la partie à partir de la lecture suivante."""
    out, new_attached, zones = [], {}, 0
    for u in units:
        evs = sorted(attached.get(u["id"], []), key=lambda e: e["t"])
        sure = [e for e in evs if e["confidence"] >= min_conf and e["player"] is not None]
        bounds = []
        for a, b in zip(sure, sure[1:]):
            if a["player"] != b["player"]:
                bounds.append((a["t"], b["t"]))
        if not u["attributable"] or not bounds:
            out.append(u)
            new_attached[u["id"]] = attached.get(u["id"], [])
            continue
        pieces = []
        ts = u["obs_t"]
        edges = [(-1e9, None)] + bounds + [(None, 1e9)]
        for k in range(len(edges) - 1):
            lo = edges[k][1] if edges[k][1] is not None else -1e9   # début : lecture b précédente
            hi = edges[k + 1][0] if edges[k + 1][0] is not None else 1e9  # fin : lecture a suivante
            seg = [t for t in ts if lo - 1e-6 <= t <= hi + 1e-6]
            if seg:
                pieces.append((seg, True))
            if k < len(edges) - 2:
                a_t, b_t = edges[k + 1]
                zone = [t for t in ts if a_t + 1e-6 < t < b_t - 1e-6]
                if zone:
                    pieces.append((zone, False))
                    zones += 1
        for j, (seg, attributable) in enumerate(pieces):
            nid = f"{u['id']}c{j}"
            out.append({**u, "id": nid, "start": seg[0], "end": seg[-1], "n_obs": len(seg), "obs_t": seg,
                        "attributable": attributable,
                        "end_reason": "conflict_split" if attributable else "conflict_zone"})
            new_attached[nid] = [e for e in evs if seg[0] - 1e-6 <= e["t"] <= seg[-1] + 1e-6] if attributable else []
    return out, new_attached, zones


POSTERIOR_MODES = ("v8", "confiance")


def posterior(readings, allowed, prior_unknown=PRIOR_UNKNOWN, mode="v8"):
    """Distribution sur allowed ∪ {'?'} à partir de lectures indépendantes (vote bayésien naïf).
    Une lecture d'un numéro hors composition pèse contre tous les candidats et pour « ? ».
    mode "v8" (défaut, resolver-0.1) ; mode "confiance" : taux de faux positif ramené à l'a priori,
    une lecture unique de confiance c donne p = c (option v8.1, non calibrée)."""
    if mode not in POSTERIOR_MODES:
        raise ValueError(f"mode de calcul de p inconnu : {mode!r}")
    legacy_v8 = mode == "v8"
    allowed = sorted(set(allowed))
    if not allowed:
        return {"?": 1.0}
    K = len(allowed)
    p0 = (1.0 - prior_unknown) / K
    logp = {n: math.log(p0) for n in allowed}
    logp["?"] = math.log(prior_unknown)
    for ev in readings:
        c = min(max(float(ev["confidence"]), 0.05), 0.99)
        if legacy_v8:
            for n in allowed:
                logp[n] += math.log(c) if ev["player"] == n else math.log((1 - c) / max(K, 2))
            logp["?"] += math.log(UNKNOWN_READ_P)
        else:
            false_p = (1.0 - c) * p0 / max(1.0 - p0, 1e-9)
            if ev["player"] in allowed:
                for n in allowed:
                    logp[n] += math.log(c) if ev["player"] == n else math.log(false_p)
                logp["?"] += math.log(false_p)
            else:
                for n in allowed:
                    logp[n] += math.log(false_p)
                logp["?"] += math.log(c)
    m = max(logp.values())
    z = sum(math.exp(v - m) for v in logp.values())
    return {str(k): math.exp(v - m) / z for k, v in logp.items()}


def resolve(tracklets: dict, evidence: list[dict], allowed, rule: dict | None = None, team="raja",
            types=("number_read",), exclude_holdout=True):
    """allowed : liste de numéros, ou fonction (unité) -> liste (fenêtres de participation)."""
    rule = {**DEFAULT_RULE, **(rule or {})}
    mode = rule.get("posterior", "v8")
    trim_handoff = bool(rule.get("trim_handoff", False))
    evs = [e for e in evidence if e["type"] in types and not (exclude_holdout and e.get("holdout"))]
    units = build_units(tracklets, rule, team)
    attached, orphans = attach_evidence(units, evs)
    conflict_zones = 0
    if rule.get("split_on_conflict"):
        units, attached, conflict_zones = split_on_conflicts(units, attached)
    out = []
    for u in units:
        al = allowed(u) if callable(allowed) else allowed
        rd = attached.get(u["id"], [])
        if not u["attributable"] or not rd:
            probs = {"?": 1.0}  # aucune preuve dans l'unité : inconnu (pas d'a priori publié)
        else:
            probs = posterior(rd, al, mode=mode)
        best = max(probs, key=probs.get)
        sure_nums = sorted({e["player"] for e in rd if e["player"] is not None and e["confidence"] >= 0.7})
        out.append({"unit": u["id"], "track": u["track"], "start": u["start"], "end": u["end"],
                    "n_obs": u["n_obs"], "attributable": u["attributable"], "end_reason": u["end_reason"],
                    "probs": {k: round(v, 5) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:4]},
                    "best": None if best == "?" else int(best), "p": round(probs[best], 5),
                    "n_readings": len(rd),
                    "numbers_read": sorted({e["player"] for e in rd if e["player"] is not None}),
                    "sure_numbers_read": sure_nums,
                    "support": [e["id"] for e in rd], "obs_t": list(u["obs_t"]),
                    "_rd": rd})
    conflicts, handoffs = enforce_simultaneity(out, trim_handoff=trim_handoff, return_handoffs=True)
    for u in out:
        u.pop("_rd", None)
    return {"version": VERSION, "rule": rule, "units": out, "orphan_evidence": orphans,
            "simultaneity_conflicts": conflicts, "handoffs": handoffs, "conflict_zones": conflict_zones,
            "calibrated": False, "posterior_mode": mode, "physical_traits_used_for_identity": False}


def _try_trim_handoff(u, clashes, n):
    """Si les lectures du joueur n dans u et dans chaque unité en collision sont disjointes
    dans le temps (relais d'une piste à une autre), élague uniquement la queue ou la tête
    non ancrée qui déborde sur l'autre unité."""
    u_ts = [e["t"] for e in u.get("_rd", []) if e.get("player") == n]
    if not u_ts or "obs_t" not in u:
        return False
    new_obs = list(u["obs_t"])
    h_updates = []
    for h in clashes:
        h_ts = [e["t"] for e in h.get("_rd", []) if e.get("player") == n]
        if not h_ts or "obs_t" not in h:
            return False
        if max(u_ts) < min(h_ts) - 1e-6:
            cut_t = max(max(u_ts), h["start"] - 1e-4)
            if cut_t >= h["start"] - 1e-6:
                if cut_t < min(h_ts) - 1e-6:
                    h_obs = [t for t in h["obs_t"] if t > cut_t + 1e-6]
                    if not h_obs:
                        return False
                    h_updates.append((h, h_obs))
                else:
                    return False
            new_obs = [t for t in new_obs if t <= cut_t + 1e-6]
        elif max(h_ts) < min(u_ts) - 1e-6:
            cut_t = min(min(u_ts), h["end"] + 1e-4)
            if cut_t <= h["end"] + 1e-6:
                if max(h_ts) < cut_t - 1e-6:
                    h_obs = [t for t in h["obs_t"] if t < cut_t - 1e-6]
                    if not h_obs:
                        return False
                    h_updates.append((h, h_obs))
                else:
                    return False
            new_obs = [t for t in new_obs if t >= cut_t - 1e-6]
        else:
            return False
    if not new_obs:
        return False
    for h, h_obs in h_updates:
        h["obs_t"] = h_obs
        h["start"], h["end"], h["n_obs"] = h_obs[0], h_obs[-1], len(h_obs)
    u["obs_t"] = new_obs
    u["start"], u["end"], u["n_obs"] = new_obs[0], new_obs[-1], len(new_obs)
    return True


def enforce_simultaneity(units, trim_handoff=False, return_handoffs=False):
    """Glouton par p décroissant : un numéro déjà tenu par une unité simultanée plus sûre est
    retiré de l'unité moins sûre, qui se rabat sur son candidat suivant (ou « ? »).
    Lorsque trim_handoff=True, un chevauchement purement non ancré entre deux unités dont les
    lectures de ce joueur sont disjointes dans le temps est résolu par élagage du relais."""
    held = defaultdict(list)  # numéro -> [unit_dict]
    conflicts = []
    handoffs = []
    order = sorted(units, key=lambda u: (-u["p"], -len(u.get("support", [])))) if trim_handoff else sorted(units, key=lambda u: -u["p"])
    for u in order:
        probs = {k: v for k, v in u["probs"].items()}
        while True:
            best = max(probs, key=probs.get)
            if best == "?":
                break
            n = int(best)
            clash = [h for h in held[n] if not (u["end"] < h["start"] - 1e-6 or u["start"] > h["end"] + 1e-6)]
            if not clash:
                break
            if trim_handoff and _try_trim_handoff(u, clash, n):
                handoffs.append({"unit": u["unit"], "number": n, "against": [h["unit"] for h in clash]})
                break
            conflicts.append({"unit": u["unit"], "number": n, "against": [h["unit"] for h in clash]})
            probs.pop(best)
            if not probs:
                probs = {"?": 1.0}
        best = max(probs, key=probs.get)
        if best != "?":
            held[int(best)].append(u)
            u["best"], u["p"] = int(best), round(probs[best], 5) if best in u["probs"] else u["p"]
        else:
            u["best"], u["p"] = None, round(probs.get("?", 1.0), 5)
    return (conflicts, handoffs) if return_handoffs else conflicts


def observation_identities(result):
    """Développe les unités en une ligne par observation (piste, instant)."""
    rows = []
    for u in result["units"]:
        for t in u["obs_t"]:
            rows.append({"track": u["track"], "t": t, "player": u["best"], "p": u["p"], "unit": u["unit"]})
    return rows


def roster_allowed(roster, timeline=None, outfield=True):
    """Numéros autorisés pour une unité selon les fenêtres de participation déclarées."""
    def f(unit):
        clocks = []
        if timeline is not None:
            for t in (unit["start"], unit["end"]):
                c = timeline.at(t).get("clock_s")
                if c is not None:
                    clocks.append(c)
        out = []
        for p in roster.players:
            if outfield and p["role"] == "G":
                continue
            a, b = roster.declared_window(p)
            if a is not None and (not clocks or (min(clocks) >= a and max(clocks) <= b)):
                out.append(p["jersey_number"])
        return sorted(out)
    return f
