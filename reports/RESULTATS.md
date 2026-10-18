# Résultats exécutés — 18 octobre 2026

## Verdict

Les commandes généralisées reproduisent les deux JSONL demandés octet pour octet. Les champs déterministes des six évaluations figées concordent avec la référence. Le moteur embarqué reste inchangé. **Le critère de l’étape 1 n’est pas atteint ; aucune amélioration de précision n’est revendiquée par le conditionnement en compétence.**

## Intégrité et tests

| Vérification | Résultat |
|---|---|
| VERIFIER_RELAIS.py avec vidéo originale | 508 fichiers et vidéo conformes ; 271 fichiers complémentaires facultatifs absents |
| Fichiers embarqués du moteur | 57 empreintes conformes |
| test_engine.py | 55 vérifications réussies |
| test_v1_layers.py | 17 tests réussis |
| test_global_identity.py | 13 tests réussis |
| test_tracking_step1.py | 11 tests réussis |
| Tests des nouvelles commandes | 21 tests réussis |
| controle_non_regression_v8_1.py | **45 identiques, 0 différents** |
| Validation du SKILL.md | valide |

Les tests historiques ont été exécutés dans un dossier temporaire contenant **le moteur de la compétence**, avec les données du relais externe. Il ne s’agit pas seulement de tests lancés sur une copie source indépendante. Les tests nouveaux utilisent des données synthétiques ; aucune annotation réelle n’est embarquée.

Les 117 vérifications/tests, ainsi que les 45 comparaisons de non-régression, ont réussi. Journaux détaillés et environnement exact sont joints à ce dossier. La CI GitHub fournie n’a pas encore été exécutée sur GitHub.

## Reproduction exacte de la couche 1 et des preuves

| Fichier | SHA-256 obtenu et attendu |
|---|---|
| tracklets_part7.jsonl | 773c4f5b19da672fabe9f32787797339b0ab0d527aaacdad32ee1f5eb076cb03 |
| evidence_part7.jsonl | f29af3da57f41dcb2b06fea2f26bd3bcb85e923f602616ad1ddabc6d41222977 |

Résultat : 382 tracklets, dont 139 de l’équipe Raja, et 440 preuves (166 historiques et 274 VLM). Ces empreintes concordent pour la construction à partir du suivi fourni, puis pour la construction utilisant les métadonnées des planches régénérées, et enfin pour la construction à partir du suivi vidéo recalculé. Dans tous les cas, les lectures de référence sont réutilisées : une nouvelle lecture VLM n’est pas prétendue identique.

## Planches depuis la vidéo originale

- Première passe : 83 planches, 994 recadrages ; **83 JPEG sur 83 identiques par SHA-256**.
- Seconde passe : 23 planches, 381 recadrages ; **23 JPEG sur 23 identiques par SHA-256**.
- Métadonnées des recadrages : identiques. La première passe écrit deux champs descriptifs supplémentaires déjà présents dans le service actuel du relais (`selection.use_back`, `selection.second_pass`). Le JSON complet de ses métadonnées n’est donc pas présenté comme identique octet pour octet à l’ancien fichier.
- La seconde passe doit prendre les preuves de la première passe. Lui fournir le fichier final des deux passes retire logiquement des chaînes déjà lues : ce n’est pas le bon état intermédiaire pour reproduire les 23 planches historiques.

## Évaluation figée

| Configuration | Précision | Couverture | Couverture correcte | Plancher | Contradictions chaînes |
|---|---:|---:|---:|---:|---:|
| P0_point_zero | 96.25 % | 24.67 % | 23.74 % | 0.0 % | 0/19 |
| E_union | 91.48 % | 64.33 % | 58.85 % | 10.6 % | 5/35 |
| F_union | 89.57 % | 66.69 % | 59.74 % | 10.6 % | 5/35 |
| K_union | 90.21 % | 66.84 % | 60.3 % | 10.6 % | 5/35 |
| E_vlm_seul | 91.17 % | 62.12 % | 56.64 % | 10.6 % | 5/33 |
| E_v7_seul | 89.43 % | 45.26 % | 40.47 % | 0.0 % | 0/19 |

L’évaluation appelle les fonctions originales de l’évaluateur et les configurations figées de configs_v8.json. Les mêmes dénominateurs et tous les champs déterministes concordent. Aucun choix ni réglage n’a été effectué à partir de ces scores.

Exception acceptée dans la demande : `mesure` n’est plus produit, `resolve_wall_s` est mesuré à nouveau, `unit_contradictions` a été ajouté à l’évaluateur depuis l’archive. Le rapport `reproduction.json` liste précisément les chemins concernés. **Le fichier d’évaluation n’a pas la même empreinte**, et aucune durée historique n’a été copiée pour faire croire le contraire.

## Intégration vidéo et limites de calcul

La vidéo originale 1920×1080 à 30 i/s, durée 303,767 s, a été analysée avec les modèles vérifiés. Timeline et couleurs validées du relais ont été réutilisées. Traitement : 257,8 s de direct, 1 289 images de détection et 3 867 images de caméra, soit 5/15 i/s. Calcul mesuré : 844,363 s (environ 14 minutes), puis 1,979 s de raffinement. Des travaux de planches/tests ont coexisté : ce temps n’est pas un benchmark matériel isolé ni une promesse de débit.

Un premier essai a été bloqué par le contrôle automatique à cause de la télémétrie ONNX. La relance désactive cette télémétrie avant initialisation et termine correctement. Les poids téléchargés ne sont pas inclus dans les archives.

Le fichier brut frames_tracked.jsonl n’est pas identique octet pour octet : arrondis de matrices de caméra et champs de numéros historiques diffèrent. Les pistes/cadres utiles et les deux JSONL exigés sont néanmoins reproduits exactement. Voir video-integration.json. Cette distribution ne prétend pas garantir la reproductibilité bit à bit de toutes les bibliothèques natives sur chaque plateforme.

## Parcours indépendant et publication

Un agent en contexte neuf a utilisé le SKILL.md sur des entrées exclusivement synthétiques : 8/12 observations publiées, 4 inconnues ; trois faits, un seul attribué à un joueur à p ≥ 0,95. Les fichiers, CSV et empreintes ont été vérifiés. Aucun changement du code ou des entrées n’a été nécessaire. Ce contrôle valide le parcours de publication, pas la précision vidéo.

Le contrat JSON WordPress 2.2.0 a été confronté au code du nettoyeur et du rendu fourni. Aucun WordPress réel ni compte Claude n’a été utilisé. Les libellés de l’extension restent centrés sur le Raja. Les événements et estimations réelles du match n’ont pas été inventés à partir des seules identités.

## Généralisation réalisée

| Ancien script / accès | Nouvelle commande |
|---|---|
| planches_premiere_passe.py | planches |
| planches_seconde_passe.py | planches --pass-number 2 |
| build_etape1_part7.py | build --part IDENTIFIANT et chemins explicites |
| appel direct à identity_resolver.py | resolve avec composition et timeline |
| publication/export propres au match | publish et report |
| paquet_chatgpt.py | packet, aussi appelé par publish |
| eval_groundtruth.py | evaluate avec fichiers, match et timeline explicites |

Aucune donnée de test ne sert à ajuster ces commandes. Les étapes restantes sont détaillées dans A_FAIRE.md à la racine.
