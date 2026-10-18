# Protocole de lecture visuelle — BILAN_V8 §3

1. Ouvrir la planche et ses métadonnées. Identifier le numéro de piste et la case `c1`, `c2`, etc. Le temps est local au fichier.
2. Lire uniquement les chiffres visibles du maillot (dos/devant) ou du short. Ne pas nommer le joueur. Un numéro hors composition reste le numéro lu ; le résolveur décidera de son admissibilité.
3. Ne pas utiliser visage, cheveux, peau, morphologie, position habituelle, réputation, équipement ou mémoire du match. Ne pas faire correspondre des personnes par ressemblance.
4. Ne consulter ni anciennes lectures, ni sorties du résolveur, ni vérité terrain pour une lecture nouvelle. Ne pas propager mentalement le numéro d’une case aux autres. Une preuve est attachée à un recadrage et son instant.
5. Choisir `part` : `back`, `front` ou `short`. Choisir la confiance ci-dessous. En cas d’illisibilité ou d’ambiguïté sans lecture défendable : omettre la case ou écrire `null`.
6. Consigner le lecteur réellement utilisé, la session et le protocole. Ne pas affirmer que les anciennes lectures étaient aveugles sans vérifier leur provenance : le bilan historique mentionne une exception.
7. Exécuter `readings-check --meta … --readings …` si nécessaire ; `build` effectue ce contrôle automatiquement. Ce contrôle vérifie le format et le rattachement, pas la justesse visuelle du chiffre.

| Valeur dans le JSON | Confiance numérique | Usage |
|---|---:|---|
| `high` | 0,9 | Chiffres lisibles clairement dans ce recadrage |
| `medium` | 0,7 | Lecture défendable, moins nette |
| `low` | 0,5 | Lecture faible mais réellement visible ; jamais un nombre inventé |
| `null` / case omise | aucune preuve | Illisible, ambigu ou absent |

Ce sont des étiquettes de lecture, **pas des probabilités calibrées**. Ne pas écrire directement 0,9 dans `conf` : le format historique attend une chaîne.

Une composition de planches conserve les chiffres de piste/case/temps imprimés en marge : ne jamais les prendre pour le numéro du maillot. Ne pas utiliser un agrandissement génératif pour reconstruire des chiffres absents. L’affichage à une taille confortable est permis, sans inventer de détails.

Après la première passe, seules les chaînes ≥ 8 s sans lecture ≥ 0,7 sont candidates à la seconde passe. Une troisième passe n’est pas un moyen de forcer une identité. Les chaînes courtes ou filmées de face peuvent rester inconnues.

Le rejeu de non-régression réutilise les lectures historiques fournies, dans leur ordre d’origine. Il ne relit pas les images pour obtenir artificiellement les mêmes réponses.
