# Ce qui reste à faire

## Recherche et qualité

- Faire arbitrer par l’utilisateur les annotations contestées du relais, en conservant la version officielle et la provenance des changements.
- Annoter le jeu de réglage et ajouter une partie indépendante ; geler les choix avant toute nouvelle évaluation finale.
- Atteindre le critère de l’étape 1 : couverture ≥ 60 %, précision ≥ 95 %, plancher par joueur ≥ 40 %, contradictions de chaînes < 2 %. Il n’est pas atteint aujourd’hui.
- Calibrer les scores sur le réglage et confirmer sur une autre partie. Les variantes v8.1 restent désactivées.
- Évaluer d’autres clubs de Botola, d’autres tenues, diffuseurs et stades. Les tests réels actuels restent centrés sur le Raja.

## Produit et diffusion

- Créer/pousser le dépôt public `mcsanchez198/botola-insight-skill` sur GitHub. Le ZIP est prêt à cette fin ; la création d’un dépôt GitHub n’est pas attestée par cette livraison.
- Essayer l’installation dans les comptes cibles Codex/ChatGPT et Claude, en particulier les restrictions des environnements hébergés. La compatibilité de structure est vérifiée ; l’exécution dans un compte Claude ne l’est pas.
- Faire tourner la CI fournie sur GitHub. La validation locale est en Linux/Python 3.12 ; la matrice 3.10/3.12 n’a pas encore tourné sur GitHub.
- Valider l’acquisition automatique du chrono et la proposition de couleurs sur un nouvel habillage de diffusion ; l’essai réel livré réutilise la timeline et les couleurs confirmées du relais.
- Vérifier l’import et le rendu dans un WordPress de préproduction avec l’extension 2.2.0. Le contrat de fichiers est contrôlé, aucun site de production n’a été modifié.
- Généraliser les libellés Raja de l’extension WordPress pour les autres clubs, puis tester cette interface. Le moteur et le paquet de cette compétence sont déjà paramétrés par club.
- Après franchissement du critère, développer une consolidation explicite des parties, sans doublons et sans écraser les analyses existantes. Le parcours actuel traite et exporte une partie à la fois.

## Fonctions volontairement non inventées

Le parcours ne déduit pas possession, passes, distances, vitesses, xG ou xA à partir des seules identités. Il peut importer des estimations sourcées et les marque toujours « expérimental, fiabilité faible ». Il ne réencode pas la vidéo, n’invente pas des lectures et n’utilise pas le physique ou l’équipement pour identifier les joueurs. Les limites de la référence restent visibles après son conditionnement en compétence.
