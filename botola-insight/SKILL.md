---
name: botola-insight
description: Analyser des parties vidéo originales d’environ cinq minutes de matchs de Botola, préparer la composition et les couleurs, produire des planches, lire visuellement les numéros, résoudre les identités et exporter un bilan traçable. Développement et validation de référence centrés sur les matchs du Raja. Utiliser pour une analyse vidéo, une reprise de données Raja Insight ou un export WordPress/paquet d’analyse. Ne pas utiliser pour des pronostics, des statistiques officielles, reconnaître quelqu’un par son physique, analyser automatiquement un match complet ou annoncer une fiabilité validée pour toute la Botola.
---

# Botola Insight

Lire [les règles du cahier](references/regles.md) avant de traiter un match. Préserver le moteur : ne modifier ni ses fichiers, ni les configurations figées, ni les seuils. Conserver l’inconnu. **Le critère de l’étape 1 n’est pas atteint.**

Exécuter les commandes avec l’environnement Python du projet et des chemins absolus. Placer vidéos, modèles téléchargés et résultats hors du dépôt public. Les modèles restent dans `scripts/moteur/models/`, répertoire ignoré. Pour les commandes ci-dessous, définir :

- `BI` : chemin absolu du dossier installé `botola-insight` ;
- `PY` : Python du venv ; `MATCH` : configuration remplie ; `WORK` : dossier privé de résultats ;
- `VIDEO` : partie vidéo originale ; `PART` : identifiant de partie, par exemple `7` ; `PERIOD` : 1 ou 2.

Ne pas transposer les chemins, numéros, couleurs, décalages temporels ou compositions de la partie 7 à un autre match. Le nom interne `raja` désigne l’équipe suivie de `match.json`, même pour un autre club. Expliquer que la validation empirique reste centrée sur le Raja.

## 0. Vérifier l’environnement

Vérifier Python ≥ 3.10, créer un venv si nécessaire, puis :

```bash
"$PY" "$BI/scripts/botola.py" setup --install --fetch-models
```

Cette commande installe les dépendances Python, télécharge via le `moteur/fetch_models.py` d’origine les modèles YOLOX dont les SHA-256 sont vérifiés, et contrôle ffmpeg, ffprobe, tesseract ainsi que l’intégrité du moteur. Installer les outils système manquants avec le gestionnaire de la machine. Lire [l’environnement](references/environnement.md) pour la reproduction et les limites de portabilité. La télémétrie ONNX est désactivée avant son initialisation. Aucun service payant ni clé API n’est nécessaire au parcours normal.

Si le terminal, les bibliothèques natives, les téléchargements ou l’accès visuel aux images sont indisponibles, arrêter l’étape concernée et expliquer ce qui manque. Ne pas simuler une exécution.

## 1. Préparer le match

Copier `assets/match.template.json` vers `MATCH`, hors du dépôt. Lire [les formats](references/formats.md). Demander les informations absentes : clubs, composition, numéros réellement portés, titulaires, remplaçants et entrées/sorties connues, couleurs des équipes et gardiens, période de la vidéo. Demander date, score et domicile/extérieur seulement si nécessaires à l’affichage ; à défaut, laisser inconnus. Une couleur annoncée prépare la proposition, elle ne remplace pas sa confirmation sur images.

Ne pas compléter une composition à partir de suppositions. Puis :

```bash
"$PY" "$BI/scripts/botola.py" prepare --match "$MATCH" --work "$WORK"
```

## 2. Analyser la partie et confirmer les couleurs

```bash
"$PY" "$BI/scripts/botola.py" analyze --match "$MATCH" --work "$WORK" --video "$VIDEO" --part "$PART" --period "$PERIOD"
```

La commande conserve le traitement original : chrono, classification direct/ralenti/absence de tableau, détection à **5 i/s**, mouvement de caméra à **15 i/s**, suivi et couleurs. Elle refuse les fichiers de plus de 330 s. Ne pas réencoder ni redimensionner la vidéo. Les JPEG de planches sont des images dérivées, pas un réencodage de la vidéo.

Si le chrono demande une lecture, ouvrir les images indiquées dans `questions.json`. Lire ce qui est visible, écrire les réponses au format documenté, puis reprendre la même commande avec `--clock-answers CHEMIN`. Demander à l’utilisateur si le chrono reste ambigu. Ne jamais inventer un décalage constant ni attribuer le chrono du direct à un ralenti. Une timeline déjà validée peut être fournie avec `--timeline CHEMIN`.

Au premier passage, ouvrir les mosaïques de `WORK/match_key/color_examples/`. Poser **une seule question groupée par match**, portant sur tous les groupes : équipe suivie, adversaire, gardiens, arbitre, autre. Écrire la réponse dans `mapping.json`, puis :

```bash
"$PY" "$BI/scripts/botola.py" confirm-colors --match "$MATCH" --work "$WORK" --mapping /chemin/mapping.json --confirmed-by "confirmation utilisateur dans cette session"
```

Relancer `analyze`. Réutiliser la confirmation pour les autres parties du match. Ne pas auto-accepter la proposition. Un import de couleurs déjà confirmées exige `--teams CHEMIN --confirmed-by SOURCE` ; ne pas utiliser cette option pour inventer une confirmation.

Lire le `part.json` obtenu. Affecter ses chemins `frames`, `teams`, `timeline` aux variables `FRAMES`, `TEAMS`, `TIMELINE`. Définir `READINGS` et `BUILT` comme deux dossiers privés pour cette partie. Relancer avec les mêmes entrées reprend le travail ; si elles changent, utiliser un autre dossier et ne pas mélanger les résultats.

## 3. Produire les planches

```bash
"$PY" "$BI/scripts/botola.py" planches --frames "$FRAMES" --teams "$TEAMS" --video "$VIDEO" --out "$READINGS"
```

Première passe inchangée : jusqu’à 12 recadrages par chaîne de l’équipe suivie d’au moins 5 s. Lire `planches_meta.json`, puis ouvrir les planches à une taille permettant de distinguer les chiffres. Le service `moteur/raja_engine/reading/` sélectionne et extrait les recadrages.

## 4. Lire soi-même les chiffres et écrire les preuves

Lire intégralement [le protocole des numéros](references/lecture-numeros.md) avant toute lecture. **L’IA de cette session lit les planches elle-même**, avec sa capacité visuelle ; aucun appel API externe n’est ajouté. Si elle ne peut pas voir les images, arrêter et le dire.

Lire seulement les chiffres du maillot ou du short. Ne pas utiliser visage, cheveux, teint, morphologie ou équipement. Ne consulter ni ancienne lecture ni vérité terrain pendant une nouvelle lecture. Confiances : `high` = 0,9, `medium` = 0,7, `low` = 0,5. Un chiffre illisible/ambigu ne produit aucune lecture. Ne pas forcer le numéro dans la composition.

Écrire `READINGS/lectures_vlm.json` selon `assets/lectures.schema.json` et le format historique `etape1_lecture_dense/lectures_vlm.json`. Identifier le lecteur réel et la session ; ne pas s’attribuer le nom d’un autre modèle. Exemple purement synthétique :

```json
{"reader":"assistant de cette session, date et modèle si connu","protocol":"chiffres uniquement ; aucune ancienne lecture consultée","readings":{"12":{"1":{"number":7,"part":"back","conf":"high"},"2":null}}}
```

Puis construire les fichiers de couche 1 et les preuves (la commande valide les lectures) :

```bash
"$PY" "$BI/scripts/botola.py" build --part "$PART" --frames "$FRAMES" --teams "$TEAMS" --readings-dir "$READINGS" --out "$BUILT"
```

Une seconde passe est possible pour les chaînes ≥ 8 s sans lecture ≥ 0,7 : 24 recadrages, instants précédents exclus, indice de dos ignoré. Utiliser **les preuves de la première passe**, avant d’ajouter les nouvelles lectures :

```bash
"$PY" "$BI/scripts/botola.py" planches --pass-number 2 --frames "$FRAMES" --teams "$TEAMS" --video "$VIDEO" --first-meta "$READINGS/planches_meta.json" --evidence "$BUILT/evidence_part$PART.jsonl" --out "$READINGS/seconde_passe"
```

Lire ces images et écrire `seconde_passe/lectures_vlm.json`, puis relancer `build` : il inclut les deux passes. Ne pas lancer une troisième passe pour combler les inconnus. `--legacy-dataset` sert uniquement à reprendre des preuves historiques fournies ; leur provenance est conservée. Pour reproduire une référence, réutiliser ses lectures archivées : une nouvelle lecture VLM n’est pas déterministe.

## 5. Résoudre les identités

Définir `RESOLUTION` comme chemin privé de sortie :

```bash
"$PY" "$BI/scripts/botola.py" resolve --match "$MATCH" --timeline "$TIMELINE" --tracklets "$BUILT/tracklets_part$PART.jsonl" --evidence "$BUILT/evidence_part$PART.jsonl" --out "$RESOLUTION"
```

La couche 2 applique `identity_resolver.py`, règle **E_union de configs_v8.json**, options v8.1 désactivées. Fenêtres de participation issues du match. Aucun numéro autorisé n’est codé pour le Raja dans l’adaptateur. Une unité sans chrono utilisable s’abstient. Les scores restent non calibrés.

Pour l’adversaire ou un gardien, utiliser le même rôle `--team-label` aux étapes planches et résolution et des dossiers séparés ; composition adverse requise pour l’adversaire. Ne pas fusionner les numéros des deux équipes. L’export WordPress porte sur une équipe suivie à la fois ; voir les formats.

## 6. Produire les exports

Définir `PUBLISHED` comme dossier de sortie de cette partie :

```bash
"$PY" "$BI/scripts/botola.py" publish --match "$MATCH" --timeline "$TIMELINE" --resolution "$RESOLUTION" --part "$PART" --out "$PUBLISHED"
```

La couche 3 appelle les seuils du moteur : **0,90 pour le temps visible, 0,95 pour les événements**. Elle produit `site_match.json`, les identités publiées, le bilan et `paquet_analyse/` (JSON et CSV issus de la logique de `paquet_chatgpt.py`).

Lire [les règles de publication](references/publication.md). Les événements relus peuvent venir de `--events CHEMIN`, avec piste et instant. Les estimations déjà calculées et sourcées peuvent venir de `--experimental CHEMIN`. **Possession, passes, distances, vitesses, xG et xA restent toujours “expérimental, fiabilité faible”**, décision du 18/10/2026. Une donnée absente reste absente, jamais zéro. La commande ne prétend pas calculer ces métriques depuis les seules identités.

Produire les fichiers n’envoie rien au site. Ne pas importer dans WordPress sans demande de l’utilisateur. L’extension 2.2.0 utilise encore des libellés Raja : son interface n’est pas généralisée aux autres clubs. Une analyse de partie ne doit pas écraser silencieusement une analyse de match complet.

## 7. Faire un compte rendu honnête

```bash
"$PY" "$BI/scripts/botola.py" report --publication "$PUBLISHED"
```

Indiquer la partie réellement traitée, les observations couvertes, les observations inconnues, les joueurs non identifiés, les sources et les limites. Ne pas confondre couverture opérationnelle, précision mesurée sur annotations et temps de jeu. Mentionner explicitement : **“Le critère de l’étape 1 n’est pas encore atteint.”** Ne pas annoncer une performance validée sur les autres clubs de Botola.

Pour une reproduction ou une demande d’amélioration, lire [l’évaluation et les prochaines étapes](references/evaluation.md). Ne jamais consulter le jeu de test pour choisir un seuil, une règle, un modèle ou une correction. Si une information indispensable manque ou si une ambiguïté subsiste, arrêter l’étape et demander à l’utilisateur.
