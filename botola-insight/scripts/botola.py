#!/usr/bin/env python3
"""Une entrée par étape, chemins explicites, moteur Raja Insight inchangé."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SKILL


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    def command(name, help, function):
        s = sub.add_parser(name, help=help, description=help)
        s.set_defaults(function=function)
        return s
    def path(s, name, required=True, **kw):
        s.add_argument("--" + name, type=Path, required=required, **kw)
    def team(s):
        s.add_argument("--team-label", choices=["raja", "opponent", "raja_gk", "opponent_gk"], default="raja",
                       help="raja = équipe suivie configurée, quel que soit son club")
    s = command("setup", "0. Vérifier/installer l'environnement et les modèles vérifiés.", "stages.setup")
    s.add_argument("--install", action="store_true")
    s.add_argument("--fetch-models", action="store_true")
    s = command("prepare", "1. Valider la composition d'un match.", "stages.prepare")
    path(s, "match"); path(s, "work")
    s = command("analyze", "2. Analyser une partie originale (~5 min), 5 détections/s et caméra 15 i/s.", "stages.analyze")
    for k in ("match", "work", "video"): path(s, k)
    s.add_argument("--part", required=True)
    s.add_argument("--period", type=int, choices=[1, 2], required=True)
    for k in ("timeline", "teams", "clock-answers"): path(s, k, False)
    s.add_argument("--confirmed-by", help="Provenance de la confirmation utilisateur des couleurs importées")
    s = command("confirm-colors", "Confirmer les groupes après une seule question à l'utilisateur par match.", "stages.confirm_colors")
    for k in ("match", "work", "mapping"): path(s, k)
    s.add_argument("--confirmed-by", required=True)
    s = command("planches", "3. Produire les recadrages de la vidéo originale.", "stages.planches")
    for k in ("frames", "teams", "video", "out"): path(s, k)
    team(s)
    s.add_argument("--pass-number", type=int, choices=[1, 2], default=1)
    path(s, "first-meta", False); path(s, "evidence", False)
    s = command("readings-check", "4. Valider les lectures faites visuellement par l'IA (aucun appel API).", "evaluation.readings_check")
    path(s, "meta"); path(s, "readings")
    s = command("build", "5a. Convertir suivi et lectures en tracklets et preuves, sans inférer de numéros.", "stages.build")
    for k in ("frames", "teams", "out"): path(s, k)
    s.add_argument("--part", required=True)
    path(s, "legacy-dataset", False); path(s, "readings-dir", False)
    s.add_argument("--reading-set", nargs=3, action="append", metavar=("META", "LECTURES", "PREFIX"))
    s = command("resolve", "5b. Résoudre les identités : règle E figée, options v8.1 désactivées.", "stages.resolve")
    for k in ("match", "timeline", "tracklets", "evidence", "out"): path(s, k)
    team(s)
    s = command("publish", "6. Produire site_match.json et le paquet d'analyse, seuils 0,90/0,95.", "publication.publish")
    for k in ("match", "timeline", "resolution", "out"): path(s, k)
    s.add_argument("--part", required=True)
    path(s, "events", False, help="Événements relus, avec piste et temps fichier (voir formats)")
    path(s, "experimental", False, help="Estimations déjà calculées, sourcées ; aucune valeur inventée")
    s = command("packet", "Recréer le paquet d'analyse depuis un site_match.json publié.", "publication.packet_command")
    path(s, "site"); path(s, "out")
    s = command("report", "7. Lire le bilan honnête de la publication.", "publication.report")
    path(s, "publication")
    s = command("evaluate", "Audit : reproduction des configurations figées, jamais un réglage sur le test.", "evaluation.evaluate")
    for k in ("tracklets", "evidence", "gt", "match", "timeline", "out"): path(s, k)
    path(s, "configs", False, default=SKILL / "assets/configs_v8.json")
    path(s, "expected", False, help="Comparer les champs déterministes à une évaluation historique")
    s.add_argument("--set", choices=["test", "reglage"], default="test")
    team(s)
    return p


def main():
    args = parser().parse_args()
    module, name = args.function.split(".")
    import importlib
    try:
        getattr(importlib.import_module(module), name)(args)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Arrêt : {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
