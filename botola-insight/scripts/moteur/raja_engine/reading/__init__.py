"""Service de lecture des numéros (remplace numbers.py dans le chemin de production v1).

selection  : choix des instants à recadrer (dos probable, isolé, grand, sans croisement)
crops      : extraction des recadrages dans la vidéo originale (décodage séquentiel, sans réencodage)
planche    : composition des planches (grille de recadrages agrandis, étiquetés c1..cN)
readers    : lecteurs interchangeables (fichier de lectures VLM, API VLM, Tesseract hors production)
service    : enchaînement et conversion des lectures en preuves (evidence.jsonl)
"""
