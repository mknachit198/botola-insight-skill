# Botola Insight Skill

Compétence au format ouvert `SKILL.md` pour analyser des parties vidéo de matchs de Botola. **Développement, tests réels et focus actuels : Raja et matchs du Raja.** Les autres clubs sont configurables ; leur performance n’est pas encore validée.

Le moteur Raja Insight v8.2 est conservé sans modification. Les nouvelles commandes paramètrent les entrées, enchaînent les couches et produisent les exports. Le critère de l’étape 1 n’est pas atteint : ne pas présenter cette version comme un système fiable de statistiques complètes de match.

## Contenu

- `botola-insight/SKILL.md` : déroulé 0 à 7 pour l’IA.
- `botola-insight/scripts/` : commande `botola.py`, adaptateurs et moteur d’origine.
- `botola-insight/references/` : cahier, lecture des numéros, formats, publication, évaluation.
- `botola-insight/assets/` : modèle de match, schéma des lectures, configurations figées, empreintes du moteur.
- `tests/` : tests publics synthétiques et tests historiques nécessitant le relais privé externe.
- `reports/` : résultats exécutés et limites de validation.

Aucune vidéo, aucun modèle ONNX, aucune planche, lecture réelle ou annotation de vérité terrain n’est distribuée. Les métadonnées de résultats et empreintes ne contiennent pas ces annotations.

## Récupérer le dépôt

Nom prévu : `mcsanchez198/botola-insight-skill`, public, licence MIT. Le ZIP livré constitue le dépôt prêt à publier ; sa présence ne signifie pas qu’un dépôt GitHub a déjà été créé.

Après sa publication :

```bash
git clone https://github.com/mcsanchez198/botola-insight-skill.git
cd botola-insight-skill
```

Sinon, décompresser le ZIP livré et ouvrir son dossier. Le sous-dossier `botola-insight` est la compétence installable. Le ZIP complet du dépôt et le ZIP de la compétence seule sont deux formats différents.

## Installer dans Codex

Depuis la racine du dépôt, installation personnelle (ne pas écraser une installation existante sans en conserver les changements) :

```bash
mkdir -p "$HOME/.agents/skills"
cp -R botola-insight "$HOME/.agents/skills/botola-insight"
```

Pour un projet seulement, copier dans `.agents/skills/botola-insight/` à la racine de ce projet. Redémarrer la session si nécessaire et invoquer `$botola-insight` ou la sélectionner dans la liste des skills. Exemple : « Utilise $botola-insight pour préparer l’analyse de cette partie de cinq minutes ; demande-moi la composition manquante. »

L’installation depuis GitHub peut aussi être demandée à l’installateur de skills de Codex en donnant l’URL du sous-dossier `botola-insight` du dépôt publié. Ne pas installer tout le dépôt comme une compétence unique.

## Installer dans ChatGPT

Sur les comptes proposant cette fonction : **Plugins → Skills → Create → Upload from your computer**. Utiliser l’archive de la compétence seule, avec `botola-insight/SKILL.md` et ses ressources. Dans l’application de bureau, la page Skills peut être directement dans la barre latérale. Les intitulés et disponibilités dépendent du compte et du déploiement.

L’analyse complète requiert un environnement Work/Codex disposant d’un terminal Python, des bibliothèques natives, de l’accès aux fichiers vidéo et de la lecture d’images. Une conversation dépourvue de ces capacités peut suivre le protocole ou lire un paquet d’analyse, mais ne peut pas exécuter le moteur. L’import d’un SKILL.md ne fournit pas ces capacités à lui seul.

La distribution universelle via le catalogue de plugins est une étape de diffusion distincte, non incluse dans ce dépôt de compétence autonome.

## Installer dans Claude

**Claude Code**, depuis la racine du dépôt :

```bash
mkdir -p "$HOME/.claude/skills"
cp -R botola-insight "$HOME/.claude/skills/botola-insight"
```

Pour un projet seulement : `.claude/skills/botola-insight/`. Invoquer `/botola-insight` et fournir la vidéo et la composition.

**Claude web/desktop** : activer l’exécution de code si disponible, ouvrir **Customize → Skills → + Create skill → Upload a skill**, puis envoyer le ZIP de la compétence seule. Les environnements hébergés peuvent limiter les paquets natifs, le réseau ou la taille des vidéos ; Claude Code local est la voie la plus prévisible pour le moteur complet. L’import Claude n’a pas été testé dans un compte Claude durant cette livraison.

## Préparer l’environnement

Linux/macOS, Python ≥ 3.10 ; Python 3.12 recommandé pour reproduire cet environnement :

```bash
python3 -m venv .venv
.venv/bin/python botola-insight/scripts/botola.py setup --install --fetch-models
```

Sur Windows utiliser `.venv\Scripts\python.exe`. Installer aussi ffmpeg/ffprobe et tesseract ; Node.js est nécessaire au test historique d’empreinte JavaScript. Exemple Debian/Ubuntu : `sudo apt-get install ffmpeg tesseract-ocr nodejs`.

Pour les versions exactes du rejeu Linux/Python 3.12 :

```bash
.venv/bin/python -m pip install -r requirements-reproduction.txt
```

Les téléchargements YOLOX sont vérifiés par SHA-256. Le pipeline désactive la télémétrie ONNX avant l’initialisation. Les vidéos restent originales. Les images de recadrages sont extraites localement. Pas de clé API nécessaire au parcours ; l’IA lit elle-même les chiffres avec la vision disponible dans son environnement.

## Utiliser la compétence

L’IA doit lire `botola-insight/SKILL.md` et respecter ses arrêts pour données manquantes. Commandes disponibles :

| Étape | Commande |
|---|---|
| 0 — Environnement | `botola.py setup` |
| 1 — Match | `botola.py prepare` |
| 2 — Vidéo | `botola.py analyze` ; `confirm-colors` après réponse utilisateur |
| 3 — Recadrages | `botola.py planches` |
| 4 — Lecture visuelle de l’IA et preuves | écriture de lectures_vlm.json, puis `botola.py build` |
| 5 — Identités | `botola.py resolve` |
| 6 — Site et paquet | `botola.py publish` |
| 7 — Bilan | `botola.py report` |

Chaque commande propose `--help`. Les exemples complets sont dans le SKILL.md. Garder les résultats dans un dossier privé extérieur au dépôt. `--part` et les chemins sont explicites : aucun réglage pour la partie 7 n’est appliqué à un autre match.

Les deux seuils de publication sont 0,90 et 0,95. Possession, passes, distances, vitesses, xG/xA restent expérimentaux, de fiabilité faible. Une donnée absente n’est jamais remplacée par zéro. L’équipement et le physique ne sont pas utilisés pour reconnaître un joueur.

`site_match.json` respecte le format de l’extension WordPress 2.2.0. Son interface contient encore des libellés Raja et ne fusionne pas les parties : lire `references/publication.md` avant import. Aucun envoi au site n’est automatique.

## Vérifier

Tests sans données privées :

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests historiques et preuve de reproduction, avec le relais fourni décompressé **hors du dépôt** :

```bash
.venv/bin/python tests/run_legacy.py --reference /chemin/raja_insight_relais --out /chemin/prive/tests
.venv/bin/python tests/reproduce.py --reference /chemin/raja_insight_relais --out /chemin/prive/reproduction
```

Le second vérifie les deux SHA-256 et tous les champs déterministes de l’évaluation figée. Il consigne les différences de métadonnées acceptées ; il ne falsifie pas les durées mesurées. Le contrôle historique doit afficher « 45 identiques, 0 différents ». Le test sert à reproduire, jamais à régler.

Les fichiers `reports/RESULTATS.md` et `A_FAIRE.md` indiquent exactement ce qui a été exécuté et ce qui reste à faire.

## Fabriquer les archives

```bash
python3 tools/package.py --out /chemin/distribution
```

Produit le ZIP du dépôt et le ZIP de la compétence seule, en excluant les données privées et modèles. Avant publication GitHub, exécuter `python3 tools/check_package.py` et examiner `git status`. Le script d’archive utilise une liste de chemins de code/documentation autorisés ; le .gitignore seul n’est pas une preuve d’absence de données.

Pour mettre en ligne : créer le dépôt public vide `botola-insight-skill` sur votre compte, puis initialiser Git dans ce dossier, ajouter les seuls fichiers du dépôt, créer un commit et pousser vers l’URL choisie. Ne pas ajouter le relais privé ou les sorties d’analyse.

## Licence et sources

MIT pour ce dépôt et le code du projet fourni par son titulaire. Conserver le copyright et la licence lors de la redistribution. Les dépendances, modèles et images/vidéos gardent leurs propres licences et droits ; voir `NOTICE.md`.

Procédures d’installation vérifiées le 18/10/2026, sources officielles :

- Standard Agent Skills : https://agentskills.io/specification
- OpenAI/Codex : https://learn.chatgpt.com/docs/build-skills
- ChatGPT : https://help.openai.com/en/articles/20001066-skills-in-chatgpt
- Claude Code : https://code.claude.com/docs/en/skills
- Claude Skills : https://support.claude.com/en/articles/12512180-use-skills-in-claude
