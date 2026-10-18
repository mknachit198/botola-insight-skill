"""Couleur des maillots : regroupement automatique puis étiquetage (humain ou règles).

1. Pour chaque joueur détecté en plan large/moyen, on mesure la couleur du haut du corps
   (pixels verts du terrain exclus).
2. Un regroupement k-moyennes (k=5 par défaut) est appris sur un échantillon du match.
3. Chaque groupe reçoit une étiquette : Raja, Raja gardien, adversaire, adversaire gardien,
   arbitre, autre. Une proposition automatique est faite à partir des couleurs déclarées du
   match ; une question humaine unique (planche d'exemples) la confirme ou la corrige.
4. Une piste reçoit l'étiquette majoritaire de ses images, avec la part de votes en confiance.
"""
from __future__ import annotations

import cv2
import numpy as np

LABELS = ["raja", "raja_gk", "opponent", "opponent_gk", "referee", "other"]
LABEL_FR = {"raja": "Raja (joueur de champ)", "raja_gk": "Raja (gardien)", "opponent": "Adversaire (joueur de champ)",
            "opponent_gk": "Adversaire (gardien)", "referee": "Arbitre", "other": "Autre / indéterminé"}


def torso_feature(frame: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < 14 or w < 5:
        return None
    ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.50 * h)
    tx1, tx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    crop = frame[max(0, ty1):max(0, ty2), max(0, tx1):max(0, tx2)]
    if crop.size < 12:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grass = (H >= 28) & (H <= 80) & (S >= 60) & (V >= 40) & (V <= 200)
    keep = ~grass
    if keep.sum() < 6:
        return None
    H, S, V = H[keep], S[keep], V[keep]
    white = ((S < 50) & (V > 150)).mean()
    black = (V < 60).mean()
    sat = S >= 60
    hist = np.zeros(9, np.float32)
    # teintes : rouge, orange, jaune, vert, turquoise, bleu clair, bleu, violet, rose
    edges = [0, 8, 20, 35, 80, 95, 105, 125, 150, 170, 181]
    hh = H[sat]
    if len(hh):
        counts, _ = np.histogram(hh, bins=edges)
        counts[0] += counts[-1]
        hist = counts[:9].astype(np.float32) / len(H)
    return np.concatenate([[white, black, float(np.median(V)) / 255.0, float(sat.mean())], hist]).astype(np.float32)


def propose_label(center: np.ndarray, kit_hints: dict | None = None) -> tuple[str, str]:
    """Proposition automatique pour un groupe de couleurs (à confirmer par l'utilisateur)."""
    white, black, v, satf = center[:4]
    hist = center[4:]
    names = ["rouge", "orange", "jaune", "vert", "turquoise", "bleu clair", "bleu", "violet", "rose"]
    dom = names[int(np.argmax(hist))] if hist.max() > 0.08 else None
    desc = f"blanc {white:.0%}, noir {black:.0%}, saturé {satf:.0%}" + (f", dominante {dom}" if dom else "")
    hints = kit_hints or {"raja": ["blanc", "vert"], "opponent": ["bleu clair", "bleu"], "referee": ["noir"],
                          "raja_gk": ["orange", "jaune"], "opponent_gk": []}
    if black > 0.45:
        return "referee", desc
    if white > 0.30 and "blanc" in hints.get("raja", []):
        return "raja", desc
    if dom and dom in hints.get("opponent", []):
        return "opponent", desc
    if dom and dom in hints.get("raja_gk", []):
        return "raja_gk", desc
    if dom and dom in hints.get("raja", []):
        return "raja", desc
    return "other", desc


class TeamModel:
    def __init__(self, k: int = 7):
        self.k = k
        self.centers = None
        self.cluster_labels: dict[int, str] = {}
        self.label_source: dict[int, str] = {}

    def fit(self, feats: np.ndarray, seed: int = 0):
        from sklearn.cluster import KMeans
        k = min(self.k, max(2, len(feats) // 20))
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)
        self.centers = km.cluster_centers_.astype(np.float32)
        counts = np.bincount(km.labels_, minlength=k)
        for c in range(k):
            lab, _ = propose_label(self.centers[c])
            self.cluster_labels[c] = lab
            self.label_source[c] = "proposition_automatique"
        return km.labels_, counts

    def predict(self, feat: np.ndarray) -> tuple[int, float]:
        d = np.linalg.norm(self.centers - feat, axis=1)
        order = np.argsort(d)
        margin = float((d[order[1]] - d[order[0]]) / max(d[order[1]], 1e-6)) if len(d) > 1 else 1.0
        return int(order[0]), margin

    def label(self, cluster: int) -> str:
        return self.cluster_labels.get(cluster, "other")

    def to_json(self):
        return {"k": self.k, "centers": None if self.centers is None else self.centers.tolist(),
                "cluster_labels": {str(k): v for k, v in self.cluster_labels.items()},
                "label_source": {str(k): v for k, v in self.label_source.items()}}

    @classmethod
    def from_json(cls, d):
        m = cls(d.get("k", 5))
        if d.get("centers") is not None:
            m.centers = np.array(d["centers"], np.float32)
        m.cluster_labels = {int(k): v for k, v in d.get("cluster_labels", {}).items()}
        m.label_source = {int(k): v for k, v in d.get("label_source", {}).items()}
        return m
