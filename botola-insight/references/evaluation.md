# Évaluation, reproduction et améliorations

## Référence figée

Les données privées viennent du relais v8.2 ; elles ne sont pas distribuées. Les tests publics synthétiques contrôlent les interfaces et les invariants. Le rejeu complet exige le relais externe : voir `tests/run_legacy.py` et `tests/reproduce.py` à la racine du dépôt.

La nouvelle commande `build` doit reproduire octet pour octet :

- tracklets_part7.jsonl : `773c4f5b19da672fabe9f32787797339b0ab0d527aaacdad32ee1f5eb076cb03` ;
- evidence_part7.jsonl : `f29af3da57f41dcb2b06fea2f26bd3bcb85e923f602616ad1ddabc6d41222977`.

Elle utilise les frames, les couleurs et les lectures historiques ; pas une nouvelle détection ni une nouvelle lecture VLM. Le rejeu des détections vidéo est un contrôle d’intégration distinct, sensible aux bibliothèques natives et aux arrondis.

L’évaluateur appelle les fonctions originales d’`eval_groundtruth.py`, avec les configurations figées de `configs_v8.json`. La composition et la timeline remplacent la liste de numéros codée pour la partie 7. Les résultats déterministes de toutes les configurations doivent être identiques à `groundtruth/eval_test.json`.

Exception explicitement acceptée par l’utilisateur : le fichier historique contient un champ `mesure` absent de l’évaluateur actuel ; `resolve_wall_s` est un temps réellement mesuré ; `unit_contradictions` a été ajouté depuis. Ces différences sont consignées, jamais remplacées par des valeurs historiques. Si un autre champ diffère, la reproduction échoue. Ne pas annoncer une identité d’empreinte pour le JSON d’évaluation.

## Verdict conservé

Référence officielle E_union : précision 91,48 % (3 113/3 403), couverture 64,3 %, couverture correcte 58,9 %, plancher 10,6 %, contradictions 5/35 chaînes à plusieurs lectures. Le test contient 82 chaînes et 5 949 observations, dont 5 290 annotées d’un numéro. **Critère de l’étape 1 non atteint.**

Les seuils de passage sont couverture ≥ 60 %, précision ≥ 95 %, plancher ≥ 40 % pour les joueurs avec ≥ 30 observations annotées, contradictions des chaînes < 2 %. Le bilan historique signale des désaccords d’annotation à arbitrer par l’utilisateur. Ne pas supprimer les cas litigieux pour présenter la performance restante comme mesure officielle.

## Recommandations pour améliorer la performance

1. Conserver d’abord cette référence reproductible et ses empreintes. Mesurer séparément précision, couverture, inconnu, temps de calcul et coût de relecture ; optimiser une seule dimension sans les autres donne un résultat trompeur.
2. Faire arbitrer les annotations contestées par l’utilisateur, sans modifier le test silencieusement. Conserver les versions et leur provenance.
3. Annoter le jeu de réglage et ajouter une autre partie indépendante. Les 57 chaînes de réglage actuelles sont trop peu diverses pour valider les choix. Prévoir une évaluation sur d’autres clubs, couleurs, stades et diffusions avant de revendiquer une portée Botola.
4. Comparer sur le réglage des améliorations de suivi et de calibration des scores, puis geler une variante avant son évaluation finale. Les options v8.1 restent désactivées dans cette distribution.
5. Optimiser les entrées/sorties et la reprise par empreinte avant de changer modèle ou cadence. La lecture séquentielle des recadrages et la réutilisation de résultats valides évitent du travail sans changer les algorithmes.
6. Ne pas promettre qu’un GPU, 30 i/s ou un modèle plus gros réglera la couverture des numéros. Le bilan v8 rapporte qu’une cadence accrue ne suffit pas à résoudre les défauts du suivi. Les accélérations futures doivent être mesurées sur une charge distincte du test officiel, avec coût mémoire et reproductibilité.

La présente version compare seulement la reproduction des configurations historiques. Elle ne désigne pas un nouveau « meilleur modèle » : aucun benchmark indépendant supplémentaire n’a été fourni pour le justifier.
