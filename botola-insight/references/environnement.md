# Environnement et portabilité

Python ≥ 3.10. Environnement de validation : Linux x86_64, Python 3.12.14, venv dédié. Les versions exactes utilisées sont consignées dans `reports/environment.txt` à la racine du dépôt. `requirements.txt` reste compatible avec les versions de Python à partir de 3.10 ; ce minimum déclaré n’équivaut pas à une validation exécutée sur chaque système.

Outils système : ffmpeg/ffprobe, tesseract-ocr ; Node.js pour le test historique de compatibilité des empreintes JavaScript. Exemple Debian/Ubuntu : `sudo apt-get install ffmpeg tesseract-ocr nodejs`. Installer seulement avec les droits disponibles sur la machine. Les fichiers vidéo restent originaux.

Modèles téléchargés par `scripts/moteur/fetch_models.py` depuis la publication officielle YOLOX :

| Modèle | SHA-256 |
|---|---|
| yolox_tiny.onnx | 427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7 |
| yolox_s.onnx | c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063 |

Le parcours normal utilise tiny, comme la référence. Le téléchargement de s ne l’active pas. Les modèles sont exclus du dépôt et soumis à leurs propres licences.

NumPy est borné à `<2.4` : un échec natif a été constaté avec 2.5.4 dans cet environnement ; 2.3.5 a permis les tests. Cette borne stabilise l’environnement sans changer le moteur. Pour un rejeu strict, utiliser Python 3.12 et `requirements-reproduction.txt`.

La commande configure `ORT_DISABLE_TELEMETRY=1` **avant** d’importer ONNX Runtime et appelle `disable_telemetry_events()` avant l’inférence. L’API seule peut être trop tardive pour l’événement initial des versions récentes. Sources primaires :

- https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
- https://onnxruntime.ai/docs/api/python/api_summary.html

Téléchargements et installation nécessitent internet. L’analyse, les planches, la résolution et les exports n’ont pas besoin d’un service réseau. L’accès aux images doit être fourni par l’environnement de l’IA ; le SKILL.md ne crée pas de capacité visuelle ni de terminal dans un produit qui ne les propose pas.

Calculer sur CPU peut prendre plusieurs minutes par partie de cinq minutes, selon machine et définition. Ne pas annoncer un temps sans le mesurer. Éviter plusieurs analyses vidéo simultanées : le moteur gère ses threads et la mémoire native peut être importante. Les manifestes évitent de réutiliser des résultats issus d’autres entrées.
