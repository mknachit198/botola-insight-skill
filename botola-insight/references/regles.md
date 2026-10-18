# Règles du cahier et décisions applicables

Sources faisant foi : `RAJA_INSIGHT_RESTRUCTURATION_V1.md`, puis `COMMENCER_ICI_REPRISE_IA.md`, `BILAN_V8.md` (§3 et §8), `COMMANDES_REPRISE.md`, `REVUE_CORRECTIF_V8_1.md`, fournis dans le relais v8.2. Ce document en conserve les contraintes opérationnelles ; il ne remplace pas la vérité terrain ni les arbitrages humains.

## Périmètre

- Cible : présence visible, identités sur preuves lisibles, faits/événements relus, provenance et couverture.
- Trois couches séparées : perception → pistes/preuves ; résolution → distributions et inconnu ; publication → seuils par statistique.
- Conserver les algorithmes existants. La compétence est une couche de commandes paramétrées. Les empreintes des fichiers du moteur sont dans `assets/engine_sha256.json`.
- Garder les vidéos originales. Pas de réencodage, de changement de résolution ni de remplacement silencieux d’un fichier.
- Ne pas traiter un match complet tant que le critère de l’étape 1 n’est pas franchi. Procéder sur des parties d’environ cinq minutes.
- Une identité n’est jamais prouvée par le visage, les cheveux, le teint, la morphologie. L’équipement (chaussures, accessoires, etc.) est interdit par défaut. Les couleurs servent à distinguer les équipes après confirmation, pas les personnes.
- Inconnu = `null`, jamais un numéro deviné ni la valeur 0 utilisée comme substitut à l’inconnu.
- Toute ambiguïté indispensable se soumet à l’utilisateur. Ne pas arbitrer la vérité terrain à sa place.

## Paramètres conservés

Détection YOLOX-tiny, deux tuiles, cinq détections/s, mouvement de caméra 15 i/s. Pas de passage à 30 i/s ni de substitution de modèle au nom d’une performance supposée. Première passe de lecture : chaînes ≥ 5 s, jusqu’à 12 recadrages. Seconde : chaînes ≥ 8 s sans lecture ≥ 0,7, jusqu’à 24 recadrages, instants déjà vus exclus.

Règle E : `cut_iou=1.1`, `cut_foot=0`, `max_gap=3`, `kit_cut="none"`. `trim_handoff` reste faux, le posterior reste `v8` et `split_on_conflict` n’est pas activé. Ne pas confondre le solveur historique CP-SAT, également conservé dans le moteur, avec la couche 2 effectivement appelée par cette compétence.

## Publication et évaluation

Temps visible : p ≥ 0,90. Événements : p ≥ 0,95. Les scores ne sont pas calibrés. Garder les preuves et les identités candidates dans les fichiers privés de résolution ; ne publier que les identités passant le seuil.

Décision utilisateur du **18 octobre 2026** : possession, passes, distances, vitesses, sprints, xG et xA peuvent être affichés seulement comme **estimations expérimentales de fiabilité faible**, avec méthode/source. Cette décision adapte la publication, elle ne valide pas scientifiquement ces métriques.

Critère de l’étape 1 : couverture ≥ 60 %, précision ≥ 95 %, aucun joueur évaluable sous 40 % de couverture correcte, contradictions de chaînes < 2 %. Le critère n’est pas atteint. Les contradictions par unité peuvent être nulles par construction si l’on coupe aux conflits ; elles ne remplacent pas le diagnostic des chaînes brutes.

Le jeu de test est figé. Son usage dans ce dépôt est uniquement la reproduction, jamais la sélection d’un modèle, d’une règle ou d’un seuil. Les variantes futures se choisissent sur le réglage annoté et se confirment sur une autre partie tenue à l’écart.

## Extension à la Botola

Les noms des clubs, compositions et couleurs deviennent des entrées. Les étiquettes internes `raja`/`opponent` sont conservées pour préserver les sorties historiques. Cela rend l’exécution configurable ; cela ne constitue pas une validation multi-clubs. Focus et tests réels de référence : Raja–Zemamra, partie 7. Les cas synthétiques vérifient seulement les interfaces génériques.
