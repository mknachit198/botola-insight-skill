# Formats et chemins

## Configuration du match

Copier `assets/match.template.json` hors du dépôt. Il est volontairement incomplet : aucune composition n’est inventée. `match_key` : minuscules/chiffres/tirets, 3 à 80 caractères pour WordPress. `team_name` et `roster.team` nomment le club suivi ; `opponent` nomme l’adversaire. `focus_venue` : `home`, `away`, `neutral` ou `null`. Date inconnue : `null`, `date_verified=false`. Score facultatif : `{"raja":1,"opponent":0,"source":"source effectivement consultée"}` ; les nombres sont ceux de l’équipe suivie puis de l’adversaire.

Chaque entrée de `roster.players` suit cette structure (exemple synthétique à remplacer) :

```json
{"jersey_number":7,"player_name":"Nom fourni par l’utilisateur","role":"M","starter":true,"entered_minute_declared":null,"left_minute_declared":null,"source":"composition fournie"}
```

Postes : G, D, M, A. Numéros entiers 0 à 99, uniques dans une équipe ; le même numéro peut exister dans l’autre équipe. `opponent_roster` facultatif, même structure ; le demander pour analyser ses identités. Déclarer les changements connus, ne pas inventer des minutes d’entrée/sortie. Les fenêtres ont la tolérance historique du moteur ; elles restreignent une identité mais ne prouvent pas sa présence.

`kit_hints` utilise les rôles internes : `raja`, `raja_gk`, `opponent`, `opponent_gk`, `referee`. Couleurs reconnues par la proposition historique : blanc, noir, rouge, orange, jaune, vert, turquoise, bleu clair, bleu, violet, rose. La proposition reste à confirmer, notamment lorsque l’habillage ou les tenues diffèrent de la référence.

## Temps, couleurs et suivi

- `part.json` : chemins effectifs de la vidéo, des images suivies, de la timeline et des couleurs ; état `ready` ou `needs_colors`.
- `timeline.json` : segments avec `file_start`, `file_end`, `kind`, `period`. Les segments de direct ont `offset`, `offset_uncertainty_s`. Chrono = temps local + offset de ce segment. Les ralentis/absences de tableau n’ont pas de chrono de direct. Ne pas réutiliser le décalage de la partie 7.
- `questions.json` : demandes de chrono, leurs identifiants et les chemins des images. Le fichier `--clock-answers` est un objet `{"identifiant_exact":"MM:SS"}` ou `"absent"`. Ouvrir les images avant de répondre. Les chemins d’images du moteur sont relatifs au dossier `.engine/match_key/` indiqué par le parcours.
- `teams_proposed.json` : centres de couleurs et étiquettes proposées. `mapping.json` : objet `{"0":"raja","1":"opponent",…}` avec **tous** les groupes affichés, sans supposer leur nombre. Étiquettes permises : `raja`, `raja_gk`, `opponent`, `opponent_gk`, `referee`, `other`.
- `teams.json` et `colors_confirmation.json` : couleurs confirmées, source de la réponse et empreinte. Une confirmation par match ; ne pas réutiliser celle d’un autre match.
- `frames_tracked.jsonl` : fichier historique du suivi, une ligne par image traitée ; temps et cadres restent locaux à cette vidéo.

## Lecture et identité

- `planches_meta.json` : `selection`, puis `planches` ; chacune contient `track`, `file`, `t_start`, `t_end`, `crops`. Chaque crop a `c` (à partir de 1), `t`, `fi`, `box` et les scores de sélection.
- `lectures_vlm.json` : `reader`, `protocol` et `notes` facultatifs (notes jamais utilisées comme preuve), `readings` indexé par numéros de piste puis de case **sous forme de chaînes**. Schéma : `assets/lectures.schema.json`. Les clés dupliquées sont refusées.
- `tracklets_partIDENTIFIANT.jsonl` : format original de `perception.write_tracklets`, avec pistes, observations, tenues et croisements. Aucun numéro de partie fixe dans le constructeur.
- `evidence_partIDENTIFIANT.jsonl` : format original de `evidence.write`, preuves `number_read`, piste, temps, numéro lu, confiance et provenance `planches/fichier.jpg#cN`. Seconde passe préfixée `seconde_passe/`. L’ordre des lectures/provenances est conservé pour le rejeu exact.
- `resolution.json` : unités, distributions, meilleur candidat et score, supports, identités par observation ; sortie privée de couche 2, **pas une identité publiable en soi**.
- `identities_published.json` : observations en direct après seuil 0,90, candidat sous seuil remplacé par `null`.

Pour combiner plusieurs sources : `build --reading-set META LECTURES PREFIX` répétable ; `PREFIX` peut être la chaîne vide. Ne pas fournir deux fois le même lot. `--legacy-dataset` prend un dataset historique explicitement fourni. Laisser cette option absente pour un nouveau match sans lectures antérieures.

## Événements et estimations facultatives

`--events` : objet `{"events":[...]}`. Chaque entrée comporte `track`, `t` local exact d’une observation, `type`, `label` facultatif, `detail` neutre, `source`, `reviewed:true`. Utiliser `goal` ou `But` pour un but. Ne pas mettre `player`, `scorer` ou `number` : l’identité vient du résolveur à 0,95. Un événement non rattaché reste sans joueur ; un événement hors direct est exclu. Le texte libre ne doit pas réintroduire une identité non publiée.

`--experimental` : objet avec `players:[{"number":7,"metrics":[...]}]` et éventuellement `possession`. Chaque métrique exige `key`, `value` numérique positive ou nulle, `sources` et `method`. Clés admises : distance_observed_m, distance_m, distance_km, vmax_kmh, sprints, xg, xa, passes_completed, passes. Aucune estimation absente n’est calculée par l’exporteur.

Possession : `raja` (proportion 0–1 ou null), `coverage` (0–1 ou null), `windows` (period, label, raja, known_s), `source`, `method`. Toujours expérimental/faible. Les deux équipes ne doivent jamais être mélangées dans une liste de joueurs indexée seulement par numéro.

## Export compatible WordPress 2.2.0

`site_match.json` garde `format="raja-insight-site"`, `version=1`, les clés score.raja et coverage.raja_frames. **Ce sont des clés de compatibilité.** `home` désigne la colonne historique du club suivi, `away` l’adversaire. Le domicile réel, s’il est connu, est consigné dans `actual_home`, `actual_away`, `focus_venue`. Le lecteur WordPress 2.2 ignore les champs supplémentaires.

`played=false` est affiché par WordPress comme « sur le banc, sans entrer en jeu ». Pour ne jamais transformer un inconnu en absent, les fiches exportées concernent les joueurs identifiés ou à participation déclarée. Les autres restent dans `summary.json` comme non identifiés. Une déclaration de titularisation ne crée aucune preuve visuelle.

Le paquet contient 00_CONSIGNES_A_COLLER.md, 01_match.json, CSV 02 à 05, points en suspens 06 et possession 07. CSV UTF-8 avec BOM, séparateur point-virgule, texte de formule neutralisé. Le vocabulaire des CSV est générique ; le contrat JSON reste compatible avec le site historique.
