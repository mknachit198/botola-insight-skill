# Publication et prudence

| Sortie | Seuil | Si le seuil n’est pas atteint |
|---|---:|---|
| Identité pour temps visible | 0,90 | null, observation inconnue |
| Identité pour événement | 0,95 | fait sans joueur, à vérifier |

Les seuils sont ceux de `moteur/raja_engine/publish.py`, inchangé. Ils ne constituent pas une garantie de précision : les scores sont non calibrés et la règle E n’atteint pas le critère de l’étape 1 sur le test de référence.

Le temps visible est la somme des observations publiées à cinq i/s, soit 0,2 s par observation. Il est limité aux images traitées en direct. Ce n’est ni le temps de jeu, ni une mesure de toutes les apparitions. La couverture opérationnelle est le nombre d’observations publiées divisé par le nombre d’observations de l’équipe détectées ; elle ne se compare pas directement au dénominateur annoté de l’évaluation officielle.

La décision du 18/10/2026 impose « expérimental, fiabilité faible » pour possession, passes, distances, vitesses, sprints, xG et xA. Le moteur `site_bundle_experimental` est appelé à l’export ; l’adaptateur marque aussi explicitement la fiabilité de la possession. Une valeur importée conserve sa valeur, ses sources et sa méthode, mais jamais une fiabilité supérieure à faible. Aucun zéro n’est créé pour une statistique inconnue.

Le parcours n’exécute pas automatiquement les anciens modules de statistiques ou de calibration géométrique. Ces algorithmes restent conservés dans le moteur, mais une valeur expérimentale doit être produite et sourcée séparément avant son import. Une identité seule ne permet pas de calculer une passe ou un xG.

## WordPress

Importer le fichier par l’écran de l’extension 2.2.0, uniquement sur demande de l’utilisateur. Le dépôt ne contient pas de clé Sportmonks et le parcours n’appelle pas cette API. Le fichier est compatible avec le contrat JSON de l’extension ; son rendu serveur n’est pas modifié.

Limites connues de l’extension 2.2.0 : libellés « Raja » en dur, un seul effectif indexé par numéro, un enregistrement par match_key. **Ne pas publier un autre club dans cette interface sans adapter ses libellés.** Ne pas importer successivement des parties sous la même clé en pensant qu’elles sont fusionnées : elles peuvent remplacer le match stocké. La compétence produit des analyses de partie ; la consolidation inter-parties et le match complet restent à traiter après franchissement du critère.

Le rapport doit toujours distinguer « non identifié », « absent de la vidéo », « non sélectionné » et « pas entré en jeu ». Seul le premier est déduit de l’absence d’identité publiée. Une participation déclarée reste présentée comme déclaration.
