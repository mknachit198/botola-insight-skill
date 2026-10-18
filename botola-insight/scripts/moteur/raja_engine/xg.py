"""xG pilote (régression logistique du dossier V2, entraînée sur StatsBomb Open Data).

Entrées : position du tir en mètres (ligne de but / axe), tête, coup franc direct.
Limites conservées : modèle de tournois internationaux, non validé sur la Botola ; la
position vient d'une calibration dont l'erreur s'ajoute ; la hauteur du ballon, la pression
et la position du gardien ne sont pas modélisées. Une valeur sans position localisée reste
inconnue : aucun xG forfaitaire n'est attribué.
"""
from __future__ import annotations

import math


def features(dx_goal_line_m: float, dy_centre_m: float):
    """dx : distance à la ligne de but (m), dy : écart latéral à l'axe du but (m)."""
    dist = math.hypot(dx_goal_line_m, dy_centre_m)
    angle = math.atan2(7.32 * dx_goal_line_m, dx_goal_line_m ** 2 + dy_centre_m ** 2 - (7.32 / 2) ** 2)
    if angle < 0:
        angle += math.pi
    return dist, angle


def xg(model: dict, dx: float, dy: float, header: bool = False, direct_free_kick: bool = False) -> float:
    dist, angle = features(dx, dy)
    x = [dist, angle, 1.0 if header else 0.0, 1.0 if direct_free_kick else 0.0]
    z = model["intercept"] + sum(c * (v - m) / s for c, v, m, s in zip(model["coef"], x, model["mean"], model["scale"]))
    return 1.0 / (1.0 + math.exp(-z))


def sensitivity(model: dict, dx: float, dy: float, sigma_m: float = 0.8, n: int = 2000, seed: int = 0, **kw):
    import random
    rnd = random.Random(seed)
    vals = sorted(xg(model, max(0.1, dx + rnd.gauss(0, sigma_m)), dy + rnd.gauss(0, sigma_m), **kw) for _ in range(n))
    return vals[int(0.05 * n)], vals[int(0.95 * n)]
