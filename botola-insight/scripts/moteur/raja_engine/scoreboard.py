"""Lecture du tableau d'affichage incrusté (chrono en haut à gauche, score, temps additionnel).

Principe (auto-calibré pour chaque fichier) :
1. Le cadre du chrono est cherché à une position normalisée (gabarit de diffusion) puis
   vérifié par sa couleur ; s'il est absent, l'image est classée « sans tableau »
   (ralenti, graphique, pause…).
2. Amorçage : Tesseract lit le chrono sur un échantillon d'images ; un ajustement robuste
   « chrono = temps fichier + décalage » ne garde que les lectures cohérentes.
3. Les chiffres de ces images deviennent des gabarits ; toutes les images sont ensuite lues
   par plus-proches-voisins sur ces gabarits (< 1 ms par image).
4. Si Tesseract est absent ou échoue, le moteur pose une question humaine
   (« quel chrono lisez-vous ? ») et amorce les gabarits à partir de la réponse.

Le score est lu de façon parcimonieuse (Tesseract seulement quand la zone change) et
le temps additionnel via le sous-chrono affiché sous le chrono principal.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import pytesseract
    HAS_TESSERACT = shutil.which("tesseract") is not None
except Exception:  # pragma: no cover
    pytesseract = None
    HAS_TESSERACT = False

# Gabarit mesuré sur la diffusion Arryadia 2026 (Raja–Zemamra) ; les coordonnées sont
# normalisées, donc valables en 854×480 comme en 640×360.
LAYOUTS = {
    "arryadia_2026": {
        "clock_box": [0.1375, 0.1306, 0.1969, 0.1639],
        "clock_norm_size": [120, 36],
        # colonnes des 4 chiffres dans le cadre normalisé de 120 px (profil d'encre moyen)
        "clock_slots": [[12, 32], [32, 52], [63, 84], [84, 104]],
        "teal_hsv": [[74, 90, 140], [96, 255, 255]],
        "score_box": [0.134, 0.075, 0.203, 0.128],
        "sub_clock_box": [0.140, 0.165, 0.176, 0.193],
        "plus_box": [0.176, 0.165, 0.199, 0.193],
        "search_region": [0.05, 0.08, 0.30, 0.24],
        # bandeaux (carton, remplacement, nom de joueur) : la zone turquoise s'étend à gauche
        # sur l'étiquette « RCA » (normalement violette) et remplace le chrono
        "banner_probe": [0.092, 0.137, 0.132, 0.158],
        "banner_text_box": [0.085, 0.126, 0.236, 0.192],
        "banner_icon_box": [0.222, 0.128, 0.256, 0.190],
        "banner_team_box": [0.085, 0.190, 0.140, 0.225],
    }
}


@dataclass
class ClockRead:
    present: bool
    seconds: int | None = None
    text: str | None = None
    confidence: float = 0.0
    method: str = ""
    digits: list = field(default_factory=list)


def _tess(img: np.ndarray, whitelist: str, psm: int = 7) -> str:
    if not HAS_TESSERACT:
        return ""
    cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(img, config=cfg).strip()
    except Exception:
        return ""


class ScoreboardReader:
    def __init__(self, layout: str | dict = "arryadia_2026"):
        self.L = dict(LAYOUTS[layout]) if isinstance(layout, str) else dict(layout)
        self.templates: dict[int, np.ndarray] | None = None  # digit -> (n, d) échantillons
        self.template_info: dict = {}
        self._last_score_sig = None

    # ------------------------------------------------------------ géométrie
    @staticmethod
    def _crop(frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        return frame[int(round(y1 * h)):int(round(y2 * h)), int(round(x1 * w)):int(round(x2 * w))]

    def clock_patch(self, frame):
        W, H = self.L["clock_norm_size"]
        c = self._crop(frame, self.L["clock_box"])
        if c.size == 0:
            return None
        return cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)

    def _teal_frac(self, img) -> float:
        if img is None or img.size == 0:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lo, hi = self.L["teal_hsv"]
        m = ((hsv[..., 0] >= lo[0]) & (hsv[..., 0] <= hi[0]) & (hsv[..., 1] >= lo[1]) & (hsv[..., 2] >= lo[2]))
        return float(m.mean())

    def banner(self, frame) -> bool:
        """Bandeau (carton / remplacement / nom) affiché à la place du chrono."""
        if "banner_probe" not in self.L:
            return False
        return self._teal_frac(self._crop(frame, self.L["banner_probe"])) >= 0.5

    def present(self, frame) -> tuple[bool, float]:
        """Le chrono est-il affiché (cadre turquoise seul, sans bandeau) ?
        Renvoie (présent, fraction turquoise)."""
        p = self.clock_patch(frame)
        if p is None:
            return False, 0.0
        frac = self._teal_frac(p)
        if frac >= 0.40 and self.banner(frame):
            return False, frac
        return frac >= 0.40, frac

    def relocate(self, frames: list[np.ndarray]) -> bool:
        """Recherche automatique du cadre si le gabarit ne correspond pas (autre diffusion,
        autre recadrage). Renvoie True si un cadre stable a été trouvé et adopté."""
        boxes = []
        lo, hi = self.L["teal_hsv"]
        for f in frames:
            h, w = f.shape[:2]
            sx1, sy1, sx2, sy2 = self.L["search_region"]
            x0, y0 = int(sx1 * w), int(sy1 * h)
            roi = f[y0:int(sy2 * h), x0:int(sx2 * w)]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            m = ((hsv[..., 0] >= lo[0]) & (hsv[..., 0] <= hi[0]) & (hsv[..., 1] >= lo[1]) & (hsv[..., 2] >= lo[2])).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
            n, _, st, _ = cv2.connectedComponentsWithStats(m)
            if n < 2:
                continue
            i = 1 + int(np.argmax(st[1:, 4]))
            x, y, bw, bh, a = st[i]
            if bh < 4 or not (2.2 < bw / bh < 4.5) or a / (bw * bh) < 0.6:
                continue
            boxes.append([(x + x0) / w, (y + y0) / h, (x + x0 + bw) / w, (y + y0 + bh) / h])
        if len(boxes) < max(3, len(frames) // 10):
            return False
        med = np.median(np.array(boxes), 0)
        dx, dy = med[0] - self.L["clock_box"][0], med[1] - self.L["clock_box"][1]
        self.L["clock_box"] = [float(v) for v in med]
        for k in ("score_box", "sub_clock_box", "plus_box"):
            b = self.L[k]
            self.L[k] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
        return True

    # ------------------------------------------------------------ chiffres du chrono
    def glyphs(self, frame) -> np.ndarray | None:
        """4 vecteurs (encre normalisée 0..1) pour MM:SS, ou None si cadre absent."""
        p = self.clock_patch(frame)
        if p is None:
            return None
        v = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
        bg = np.percentile(v, 90)
        dark = np.percentile(v, 3)
        ink = np.clip((bg - v) / max(bg - dark, 1.0), 0, 1)
        out = []
        for a, b in self.L["clock_slots"]:
            g = ink[3:-3, a:b]
            g = cv2.resize(g, (10, 16), interpolation=cv2.INTER_AREA)
            g = g - g.mean()
            n = np.linalg.norm(g)
            out.append((g / n if n > 1e-6 else g).reshape(-1))
        return np.stack(out)

    def tesseract_clock(self, frame) -> ClockRead:
        p = self.clock_patch(frame)
        if p is None:
            return ClockRead(False)
        g = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)[..., 2]
        g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (b > 0).mean() < 0.5:
            b = 255 - b
        b = cv2.copyMakeBorder(b[6:-6, 12:-12], 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        txt = _tess(b, "0123456789:")
        m = re.search(r"(\d{2}):?(\d{2})$", txt.replace(" ", ""))
        if not m:
            return ClockRead(True, None, txt, 0.0, "tesseract")
        mm, ss = int(m.group(1)), int(m.group(2))
        if ss > 59:
            return ClockRead(True, None, txt, 0.0, "tesseract")
        return ClockRead(True, mm * 60 + ss, f"{mm:02d}:{ss:02d}", 0.5, "tesseract", [mm // 10, mm % 10, ss // 10, ss % 10])

    def save_templates(self, path: str):
        if self.templates is None:
            return
        np.savez_compressed(path, **{f"d{d}": v for d, v in self.templates.items()},
                            layout=np.array([str(self.L)]))

    def load_templates(self, path: str) -> bool:
        try:
            z = np.load(path, allow_pickle=False)
        except Exception:
            return False
        self.templates = {int(k[1:]): z[k] for k in z.files if k.startswith("d")}
        self.template_info = {"samples_per_digit": {d: int(len(v)) for d, v in self.templates.items()},
                              "missing_digits": [d for d in range(10) if d not in self.templates],
                              "loaded_from": str(path)}
        return True

    def fit_templates(self, labeled: list[tuple[np.ndarray, int]], keep_existing: bool = True):
        """labeled : liste de (glyphes 4×d, secondes de chrono exactes)."""
        by_digit: dict[int, list] = {d: [] for d in range(10)}
        if keep_existing and self.templates:
            for d, arr in self.templates.items():
                by_digit[d].extend(list(arr))
        for g, secs in labeled:
            mm, ss = divmod(int(secs), 60)
            for gi, d in zip(g, [mm // 10, mm % 10, ss // 10, ss % 10]):
                by_digit[d].append(gi)
        self.templates = {d: np.stack(v) for d, v in by_digit.items() if v}
        self.template_info = {"samples_per_digit": {d: len(v) for d, v in by_digit.items()},
                              "missing_digits": [d for d, v in by_digit.items() if not v]}
        return self.template_info

    def read_clock(self, frame, k: int = 3) -> ClockRead:
        ok, frac = self.present(frame)
        if not ok:
            return ClockRead(False)
        if self.templates is None:
            return self.tesseract_clock(frame)
        g = self.glyphs(frame)
        digits, margins = [], []
        for pos, vec in enumerate(g):
            allowed = range(6) if pos == 2 else range(10)
            scores = []
            for d in allowed:
                T = self.templates.get(d)
                if T is None:
                    continue
                dist = np.sort(np.linalg.norm(T - vec, axis=1))[:k]
                scores.append((float(dist.mean()), d))
            if not scores:
                return ClockRead(True, None, None, 0.0, "templates")
            scores.sort()
            best = scores[0]
            second = scores[1][0] if len(scores) > 1 else best[0] + 1
            digits.append(best[1])
            margins.append((second - best[0]) / max(second, 1e-6))
        secs = (digits[0] * 10 + digits[1]) * 60 + digits[2] * 10 + digits[3]
        conf = float(min(margins))
        return ClockRead(True, secs, f"{digits[0]}{digits[1]}:{digits[2]}{digits[3]}", conf, "templates", digits)

    # ------------------------------------------------------------ temps additionnel
    def stoppage_indicator(self, frame) -> bool:
        """Cadre rouge « +N » sous le chrono = temps additionnel annoncé/en cours."""
        c = self._crop(frame, self.L["plus_box"])
        if c.size == 0:
            return False
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        red = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 90))
        return float(red.mean()) >= 0.18

    def read_sub_clock(self, frame) -> tuple[int | None, str]:
        c = self._crop(frame, self.L["sub_clock_box"])
        if c.size == 0:
            return None, ""
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (b > 0).mean() < 0.5:
            b = 255 - b
        b = cv2.copyMakeBorder(b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        txt = _tess(b, "0123456789:")
        m = re.search(r"(\d{1,2}):?(\d{2})", txt.replace(" ", ""))
        if not m or int(m.group(2)) > 59:
            return None, txt
        return int(m.group(1)) * 60 + int(m.group(2)), txt

    # ------------------------------------------------------------ score
    def score_signature(self, frame) -> np.ndarray | None:
        c = self._crop(frame, self.L["score_box"])
        if c.size == 0:
            return None
        g = cv2.cvtColor(cv2.resize(c, (64, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (g > 170).astype(np.float32)  # chiffres blancs

    def read_score(self, frame) -> tuple[tuple[int, int] | None, str]:
        c = self._crop(frame, self.L["score_box"])
        if c.size == 0:
            return None, ""
        votes = {}
        raw = []
        for fx in (5, 7):
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
            _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if (b > 0).mean() < 0.5:
                b = 255 - b
            b = cv2.copyMakeBorder(b, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
            txt = _tess(b, "0123456789-")
            raw.append(txt)
            m = re.search(r"(\d{1,2})\s*-+\s*(\d{1,2})", txt)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                votes[key] = votes.get(key, 0) + 1
        if not votes:
            return None, "|".join(raw)
        best = max(votes.items(), key=lambda kv: kv[1])
        return best[0], "|".join(raw)


    # ------------------------------------------------------------ bandeaux
    def read_banner(self, frame) -> dict:
        """Lit un bandeau : lignes « N. NOM », équipe (RCA/RCAZ) et icône (couleur).
        Les noms lus sont bruités en basse définition : ils sont ensuite rapprochés de la
        composition (numéro + similarité du nom) dans events.py, jamais pris tels quels."""
        out = {"lines": [], "team": None, "icon": None, "raw": ""}
        c = self._crop(frame, self.L["banner_text_box"])
        if c.size == 0:
            return out
        g = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2]
        g = cv2.resize(g, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = b == 0  # texte sombre sur turquoise
        if ink.mean() > 0.5:
            ink = ~ink
        rows = ink.mean(1)
        ink[rows > 0.75] = False  # bordures horizontales
        cols = ink.mean(0)
        ink[:, cols > 0.85] = False
        rows = ink.mean(1)
        text_rows = rows > 0.04
        lines, start = [], None
        for i, v in enumerate(list(text_rows) + [False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= 12:
                    lines.append((start, i))
                start = None
        raws = []
        for a, z in lines:
            img = np.where(ink[max(0, a - 4):z + 4], 0, 255).astype(np.uint8)
            img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
            txt = _tess(img, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-", psm=7)
            raws.append(txt)
            m = re.search(r"(\d{1,2})\s*[.,]?\s*([A-Z][A-Z' -]{1,})", txt.strip())
            if m:
                out["lines"].append({"jersey_number": int(m.group(1)), "name": m.group(2).strip()})
            elif txt.strip():
                out["lines"].append({"jersey_number": None, "name": txt.strip()})
        out["raw"] = " | ".join(raws)
        t = self._crop(frame, self.L["banner_team_box"])
        if t.size:
            g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
            _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if (b > 0).mean() < 0.5:
                b = 255 - b
            tt = _tess(cv2.copyMakeBorder(b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255), "ABCRZ")
            out["team"] = "RCAZ" if "RCAZ" in tt else ("RCA" if "RCA" in tt else None)
            out["team_raw"] = tt
        out["side"], out["side_rows"] = self.banner_side(frame)
        out.update(self.banner_icon(frame))
        return out

    def banner_side(self, frame):
        """Équipe du bandeau par la POSITION de l'étiquette sous les lignes : à gauche (sous
        l'équipe à domicile, « RCA ») ou à droite (sous l'équipe extérieure, « RCAZ »).
        Mesure géométrique, indépendante de l'OCR et du nombre de lignes du bandeau."""
        tb, team = self.L.get("banner_text_box"), self.L.get("banner_team_box")
        if not tb or not team:
            return None, [0, 0]
        H, W = frame.shape[:2]
        tw = team[2] - team[0]
        y0, y1 = int(tb[1] * H), int(min(1.0, tb[3] + 0.12) * H)

        def zone(xa, xb):
            c = frame[y0:y1, max(0, int(xa * W)):int(xb * W)]
            if c.size == 0:
                return np.zeros(max(1, y1 - y0))
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            m = (hsv[..., 0] >= 75) & (hsv[..., 0] <= 100) & (hsv[..., 1] > 90) & (hsv[..., 2] > 90)
            return m.mean(1)
        Lz = zone(team[0] + 0.1 * tw, team[2] - 0.1 * tw)
        Rz = zone(tb[2] - 0.9 * tw, tb[2] - 0.1 * tw)
        left, right = int(((Lz > 0.6) & (Rz < 0.1)).sum()), int(((Rz > 0.6) & (Lz < 0.1)).sum())
        if left >= 2 and right == 0:
            return "home", [left, right]
        if right >= 2 and left == 0:
            return "away", [left, right]
        return None, [left, right]

    def banner_icon(self, frame) -> dict:
        """Couleur de l'icône du bandeau (lecture rapide, sans OCR).
        - flèche rouge/verte de remplacement : grande pastille à droite du bandeau (zone fixe) ;
        - carton : petite icône placée JUSTE APRÈS le nom, donc à une position variable :
          on la cherche sur toute la largeur du bandeau (le fond turquoise et le texte sombre
          n'ont ni jaune ni rouge saturés)."""
        out = {"icon": None}
        tb = self.L["banner_text_box"]
        wide = self._crop(frame, [tb[0], tb[1], self.L["banner_icon_box"][2], tb[3]])
        card = None
        if wide.size:
            hsv = cv2.cvtColor(wide, cv2.COLOR_BGR2HSV)
            sat = (hsv[..., 1] >= 120) & (hsv[..., 2] >= 120)
            hch = hsv[..., 0]
            ny = int((sat & (hch >= 18) & (hch <= 35)).sum())
            nr = int((sat & ((hch <= 8) | (hch >= 170))).sum())
            area = wide.shape[0] * wide.shape[1]
            out["card_pixels"] = {"yellow": ny, "red": nr, "area": area}
            # une icône de carton couvre ~0,3–1,5 % de la zone ; une flèche de remplacement bien plus
            if ny >= max(6, 0.002 * area) and ny > nr:
                card = "yellow"
            elif max(6, 0.002 * area) <= nr <= 0.03 * area:
                card = "red_small"
        ic = self._crop(frame, self.L["banner_icon_box"])
        if ic.size:
            hsv = cv2.cvtColor(ic, cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1] >= 120
            hch = hsv[..., 0]
            red = float((sat & ((hch <= 8) | (hch >= 170))).mean())
            green = float((sat & (hch >= 45) & (hch <= 70)).mean())
            yellow = float((sat & (hch >= 18) & (hch <= 35)).mean())
            out["icon_fractions"] = {"red": round(red, 3), "green": round(green, 3), "yellow": round(yellow, 3)}
            # flèche rouge/verte = remplacement (grande pastille) ; carton = petite icône
            if red >= 0.08 and red >= green:
                out["icon"] = "red"
            elif green >= 0.08:
                out["icon"] = "green"
            elif card is not None:
                out["icon"] = card
            elif yellow >= 0.02 and yellow > red:
                out["icon"] = "yellow"
            elif red >= 0.02:
                out["icon"] = "red_small"
        elif card is not None:
            out["icon"] = card
        return out
