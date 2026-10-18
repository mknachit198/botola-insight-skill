"""Couche 3 — publication : seuil de probabilité par statistique. La prudence vit ICI.

Une observation n'est publiée avec une identité que si p ≥ seuil de la statistique visée ;
sinon l'identité reste null (jamais 0, jamais devinée). Provenance et couverture sont
conservées à côté de chaque valeur. Métriques gelées en v1 : jamais exportées, même vides —
sauf sur le site, par décision de l'utilisateur du 18/10/2026 : elles y sont affichées comme
estimations expérimentales, toujours en fiabilité faible (site_bundle_experimental).
"""
from __future__ import annotations

VERSION = "publish-0.2"

THRESHOLDS = {"events": 0.95, "visible_time": 0.90}
FROZEN_V1 = {"distance_m", "distance_km", "vmax_kmh", "sprints", "xg", "xa", "possession_certaine", "passes"}
# Fichier du site (site_match.json, extension WordPress) : mêmes métriques gelées, sous leurs clés
# de site (cahier v1 §1 et étape 2.4 : « retirer les colonnes gelées » des exports et du site).
FROZEN_SITE_METRICS = {"distance_observed_m", "distance_m", "distance_km", "vmax_kmh", "sprints", "xg", "xa",
                       "passes_completed", "passes"}
FROZEN_SITE_ACTION_LABELS = {"Passes réussies"}
FROZEN_SITE_BLOCKS = {"possession"}  # possession estimée : ballon trop rarement localisé (cahier v1 §1)


def site_bundle_v1(bundle: dict) -> dict:
    """Retire du fichier du site les métriques gelées en v1, ainsi que les actions qui ne servaient
    qu'à elles. Le reste (faits, événements par joueur, temps visible, fiabilités) est inchangé."""
    import copy
    b = copy.deepcopy(bundle)
    removed = sorted(k for k in FROZEN_SITE_BLOCKS if k in b)
    for k in removed:
        b.pop(k)
    gone = set()
    for p in b.get("players", []):
        keep = []
        for m in p.get("metrics", []):
            if m.get("key") in FROZEN_SITE_METRICS:
                gone.add(m["key"])
            else:
                keep.append(m)
        p["metrics"] = keep
        actions = []
        for a in p.get("actions", []):
            labels = [x.strip() for x in str(a.get("label", "")).split(",") if x.strip()]
            kept = [x for x in labels if x not in FROZEN_SITE_ACTION_LABELS]
            if kept:
                actions.append({**a, "label": ", ".join(kept)})
        p["actions"] = actions
    b["publication"] = {"version": VERSION, "gelees_retirees": removed + sorted(gone)}
    return b


EXPERIMENTAL_GROUP = "Estimations expérimentales"


def site_bundle_experimental(bundle: dict) -> dict:
    """Décision de l'utilisateur du 18/10/2026 (écart assumé au cahier v1 §1, pour le site seulement) :
    les métriques gelées sont AFFICHÉES comme estimations expérimentales, toujours en fiabilité
    « faible », regroupées et expliquées sur la page. La possession estimée est gardée et marquée
    expérimentale. Rien n'est inventé : seules la fiabilité et le groupe d'affichage changent."""
    import copy
    b = copy.deepcopy(bundle)
    marked = set()
    for p in b.get("players", []):
        for m in p.get("metrics", []):
            if m.get("key") in FROZEN_SITE_METRICS:
                if m.get("reliability") != "faible":
                    m["reason"] = (m.get("reason") or "").rstrip() + (" " if m.get("reason") else "") + \
                        f"Fiabilité ramenée à « faible » (estimation expérimentale ; relevé d'origine : {m.get('reliability')})."
                m["reliability"] = "faible"
                m["group"] = EXPERIMENTAL_GROUP
                m["experimental"] = True
                marked.add(m["key"])
    if isinstance(b.get("possession"), dict):
        b["possession"]["experimental"] = True
        marked.add("possession")
    b["publication"] = {"version": VERSION, "mode": "experimental_affiche",
                        "estimations_experimentales": sorted(marked),
                        "decision": "utilisateur, 18/10/2026 : afficher en fiabilité faible avec une section d'explication"}
    return b


def published_identity(row: dict, stat: str = "visible_time", thresholds=THRESHOLDS):
    thr = thresholds[stat]
    if row.get("player") is None or row.get("p") is None or row["p"] < thr:
        return None
    return row["player"]


def publish_rows(rows, stat="visible_time", thresholds=THRESHOLDS):
    return [{**r, "published": published_identity(r, stat, thresholds), "threshold": thresholds[stat]} for r in rows]


def strip_frozen(record: dict) -> dict:
    """Retire récursivement les métriques gelées d'un export v1."""
    if isinstance(record, dict):
        return {k: strip_frozen(v) for k, v in record.items() if k not in FROZEN_V1}
    if isinstance(record, list):
        return [strip_frozen(v) for v in record]
    return record
