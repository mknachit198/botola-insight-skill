"""Correspondance temps fichier ↔ chrono du match.

Règles appliquées (exigées par le cahier des charges) :
- la référence est le chrono incrusté en haut à gauche ; le temps interne du fichier sert
  seulement à retrouver les images ;
- le décalage chrono − temps fichier est estimé PAR SEGMENT : après une coupure ou un gel,
  un nouveau segment commence, on n'applique jamais un décalage constant à tout le fichier ;
- les passages sans tableau (ralentis, graphiques, pauses) ne reçoivent pas de chrono de
  direct et n'alimentent pas les décomptes ;
- le temps additionnel est représenté séparément (« 90:00 +1:15 »).
"""
from __future__ import annotations

import numpy as np

from .util import fmt_clock

STOPPAGE_BASES = (2700, 5400, 6300, 7200)


def _interval_offset(reads):
    """reads : liste de (t, secs). Le chrono affiché vaut floor(t + d) ; chaque lecture
    contraint d ∈ [secs − t, secs + 1 − t). On prend l'intersection robuste
    (percentiles 10/90 pour tolérer le flou des transitions d'images)."""
    lo = np.array([s - t for t, s in reads], float)
    hi = lo + 1.0
    L = float(np.percentile(lo, 90)) if len(lo) >= 5 else float(lo.max())
    U = float(np.percentile(hi, 10)) if len(hi) >= 5 else float(hi.min())
    if L <= U:
        return (L + U) / 2.0, (U - L) / 2.0
    med = float(np.median(lo)) + 0.5
    return med, 0.5


def segment_reads(reads, tol=1.25, confirm=3, max_gap=None):
    """Regroupe des lectures (t, secs) en segments de décalage constant.
    Une lecture isolée incompatible est traitée comme une erreur de lecture, pas comme
    une coupure : il faut `confirm` lectures concordantes pour ouvrir un nouveau segment."""
    segs = []
    cur = None
    pending = []
    outliers = []
    for t, s in reads:
        o = s - t
        if cur is not None and abs(o - np.median([x[1] - x[0] for x in cur[-7:]])) <= tol and \
                (max_gap is None or t - cur[-1][0] <= max_gap):
            cur.append((t, s))
            outliers.extend(pending)
            pending = []
            continue
        if pending and abs(o - (pending[0][1] - pending[0][0])) <= tol:
            pending.append((t, s))
            if len(pending) >= confirm:
                if cur:
                    segs.append(cur)
                cur = pending
                pending = []
            continue
        outliers.extend(pending)
        pending = [(t, s)]
        if cur is None and confirm <= 1:
            cur, pending = pending, []
    if cur:
        segs.append(cur)
    outliers.extend(pending)
    return segs, outliers


class Timeline:
    def __init__(self, data: dict):
        self.data = data
        self.segments = data["segments"]

    # -------------------------------------------------------------- construction
    @classmethod
    def build(cls, samples: list[dict], duration: float, declared_period: int | None = None,
              min_conf: float = 0.08, freeze_diff: float = 0.35, freeze_min_s: float = 1.5):
        """samples : une entrée par image échantillonnée :
        {t, present, secs, conf, stoppage, added, diff}"""
        samples = sorted(samples, key=lambda s: s["t"])
        step = float(np.median(np.diff([s["t"] for s in samples]))) if len(samples) > 2 else 0.5

        # 1) gels d'image (diffusion figée) : différence quasi nulle entre échantillons
        frozen = []
        run = []
        for s in samples:
            if s.get("diff") is not None and s["diff"] < freeze_diff:
                run.append(s["t"])
            else:
                if run and run[-1] - run[0] + step >= freeze_min_s:
                    frozen.append([run[0] - step, run[-1]])
                run = []
        if run and run[-1] - run[0] + step >= freeze_min_s:
            frozen.append([run[0] - step, run[-1]])

        def in_frozen(t):
            return any(a <= t <= b for a, b in frozen)

        # 2) lectures du chrono principal (hors temps additionnel figé et hors gel)
        main, stop, stop_unread = [], [], []
        for s in samples:
            if not s.get("present") or s.get("secs") is None or in_frozen(s["t"]):
                continue
            if s["secs"] in STOPPAGE_BASES:
                # chrono figé à 45:00/90:00 : temps additionnel (ou fin de période). Ces
                # lectures ne servent jamais à estimer un décalage de direct.
                if s.get("added") is not None:
                    stop.append((s["t"], s["added"], s["secs"]))
                else:
                    stop_unread.append(s["t"])
                continue
            if s.get("conf", 1.0) < min_conf:
                continue
            main.append((s["t"], s["secs"]))

        segs, outliers = segment_reads(main)
        # A–B–C : si A et C ont le même décalage et B est court, B est une erreur de lecture
        # (ex. « 68:01 » lu pour « 69:01 ») : le chrono ne peut pas reculer puis revenir.
        corrected = []
        changed = True
        while changed and len(segs) >= 3:
            changed = False
            for i in range(1, len(segs) - 1):
                A, B, C = segs[i - 1], segs[i], segs[i + 1]
                oa = np.median([x[1] - x[0] for x in A])
                ob = np.median([x[1] - x[0] for x in B])
                oc = np.median([x[1] - x[0] for x in C])
                if abs(oa - oc) <= 1.5 and B[-1][0] - B[0][0] < 30 and abs(ob - oa) > 1.5:
                    corrected.append({"file_start": B[0][0], "file_end": B[-1][0], "offset_rejected": float(ob),
                                      "offset_kept": float(oa), "n_reads": len(B)})
                    outliers.extend(B)
                    segs[i - 1:i + 2] = [A + C]
                    changed = True
                    break
        # un trou de lectures > 2 s à l'intérieur d'un même décalage devient un intervalle
        # qualifié (bandeau, ralenti…) au lieu d'être absorbé par le segment de direct
        split = []
        for g in segs:
            cur = [g[0]]
            for x in g[1:]:
                if x[0] - cur[-1][0] > 2.0:
                    split.append(cur)
                    cur = []
                cur.append(x)
            split.append(cur)
        segs = split
        live = []
        for g in segs:
            off, half = _interval_offset(g)
            live.append({"kind": "live", "file_start": g[0][0], "file_end": g[-1][0],
                         "offset": round(off, 3), "offset_uncertainty_s": round(half, 3),
                         "n_reads": len(g), "clock_start": g[0][1], "clock_end": g[-1][1]})

        # temps additionnel : même logique sur le sous-chrono, par base (45:00, 90:00…)
        stop_segs = []
        for base in sorted({b for _, _, b in stop}):
            rr = [(t, a) for t, a, b in stop if b == base]
            gs, _ = segment_reads(rr, confirm=2)
            for g in gs:
                off, half = _interval_offset(g)
                stop_segs.append({"kind": "stoppage", "base": base, "file_start": g[0][0], "file_end": g[-1][0],
                                  "offset": round(off, 3), "offset_uncertainty_s": round(half, 3),
                                  "n_reads": len(g), "added_start": g[0][1], "added_end": g[-1][1]})

        # 3) fusion chronologique + qualification des intervalles
        observed = sorted(live + stop_segs, key=lambda x: x["file_start"])
        out = []
        cuts = []
        cursor = 0.0
        prev = None
        for seg in observed:
            # étendre le segment observé d'un demi-pas d'échantillonnage de chaque côté
            seg = dict(seg)
            seg["file_start"] = max(cursor, seg["file_start"] - step / 2)
            seg["file_end"] = seg["file_end"] + step / 2
            if seg["file_start"] - cursor > step * 0.75:
                out.append(cls._gap(cursor, seg["file_start"], prev, seg, samples, frozen))
            elif prev is not None and prev["kind"] == "live" and seg["kind"] == "live" and \
                    abs(seg["offset"] - prev["offset"]) > 1.5:
                # coupure franche entre deux images consécutives (montage, changement de source)
                d = seg["offset"] - prev["offset"]
                cuts.append({"file_t": round(seg["file_start"], 3), "offset_change_s": round(d, 2),
                             "clock_before": fmt_clock(prev["file_end"] + prev["offset"]),
                             "clock_after": fmt_clock(seg["file_start"] + seg["offset"]),
                             "match_seconds_missing": round(max(0.0, d), 1),
                             "note": "temps de match non diffusé" if d > 0 else "retour en arrière du chrono (rediffusion ou recouvrement)"})
            out.append(seg)
            cursor = seg["file_end"]
            prev = seg
        if duration - cursor > step:
            out.append(cls._gap(cursor, duration, prev, None, samples, frozen))

        for seg in out:
            seg["period"] = cls._period(seg, declared_period)
            seg["file_start"] = round(seg["file_start"], 3)
            seg["file_end"] = round(seg["file_end"], 3)
        data = {
            "version": 1,
            "sampling_step_s": step,
            "duration_s": duration,
            "segments": out,
            "frozen_intervals": [[round(a, 2), round(b, 2)] for a, b in frozen],
            "hard_cuts": cuts,
            "rejected_reads": len(outliers),
            "ocr_corrections": corrected,
            "accepted_reads": sum(len(g) for g in segs),
            "stoppage_reads": len(stop),
            "stoppage_frames_without_subclock": len(stop_unread),
            "declared_period": declared_period,
            "rules": [
                "chrono de direct = lecture du tableau en haut à gauche, décalage estimé par segment",
                "aucun décalage n'est propagé à travers une coupure, un gel ou un passage sans tableau",
                "les passages sans tableau ne sont pas comptés comme du direct",
            ],
        }
        return cls(data)

    @staticmethod
    def _gap(a, b, prev, nxt, samples, frozen):
        inside = [s for s in samples if a <= s["t"] <= b]
        n = max(1, len(inside))
        absent = sum(1 for s in inside if not s.get("present")) / n
        froz = sum(min(b, fb) - max(a, fa) for fa, fb in frozen if fb > a and fa < b)
        seg = {"file_start": a, "file_end": b, "n_samples": len(inside),
               "scoreboard_absent_fraction": round(absent, 3), "frozen_seconds": round(max(0.0, froz), 2)}
        if prev is not None and nxt is not None and prev["kind"] == "live" and nxt["kind"] == "live":
            delta = nxt["offset"] - prev["offset"]
            seg["offset_change_s"] = round(delta, 2)
            seg["clock_before"] = fmt_clock(prev["file_end"] + prev["offset"])
            seg["clock_after"] = fmt_clock(nxt["file_start"] + nxt["offset"])
        else:
            delta = None
        banner = sum(1 for s in inside if s.get("banner")) / n
        seg["banner_fraction"] = round(banner, 3)
        if froz > 0.5 * (b - a):
            seg["kind"] = "frozen"
        elif banner >= 0.5 and prev is not None and prev["kind"] == "live" and (delta is None or abs(delta) <= 1.5):
            # bandeau (carton/remplacement) affiché à la place du chrono pendant le direct :
            # chrono interpolé avec le décalage du segment précédent (et suivant s'il concorde)
            seg["kind"] = "live_banner"
            seg["offset"] = prev["offset"]
            seg["offset_uncertainty_s"] = prev["offset_uncertainty_s"] + (0.0 if delta is not None else 1.0)
            seg["interpolated"] = True
            seg["bounded_both_sides"] = delta is not None
            seg["note"] = "bandeau affiché à la place du chrono ; chrono interpolé"
        elif delta is not None and abs(delta) > 1.5:
            seg["kind"] = "discontinuity"
            seg["note"] = ("temps de match non diffusé" if delta > 0 else
                           "le fichier avance sans que le chrono avance (gel, rediffusion ou changement de période)")
            seg["match_seconds_missing"] = round(max(0.0, delta), 1)
        elif absent >= 0.5:
            seg["kind"] = "no_scoreboard"
            seg["note"] = "ralenti, graphique ou plan sans tableau : pas de chrono de direct"
            seg["clock_continuous_across"] = delta is not None and abs(delta) <= 1.5
        elif delta is not None and abs(delta) <= 1.5:
            # tableau présent, lectures rejetées, mais même décalage avant et après
            seg["kind"] = "live_interpolated"
            seg["offset"] = prev["offset"]
            seg["offset_uncertainty_s"] = max(prev["offset_uncertainty_s"], nxt["offset_uncertainty_s"])
            seg["interpolated"] = True
            seg["note"] = "chrono présent mais mal lu ; interpolé entre deux lectures concordantes"
        else:
            seg["kind"] = "unreadable"
            seg["note"] = "tableau présent mais chrono non lu de façon fiable"
        if prev is None:
            seg["position"] = "before_first_read"
        if nxt is None:
            seg["position"] = "after_last_read"
        return seg

    @staticmethod
    def _period(seg, declared):
        base = seg.get("base")
        if base is not None:
            return {2700: 1, 5400: 2, 6300: 3, 7200: 4}[base]
        if seg.get("kind") in ("live_banner", "live_interpolated"):
            return None if seg.get("offset") is None else (1 if seg["file_start"] + seg["offset"] < 2700 else 2)
        if seg.get("kind") == "live":
            c = seg["clock_start"]
            return 1 if c < 2700 else 2 if c < 5400 else 3 if c < 6300 else 4
        return declared

    # -------------------------------------------------------------- conversions
    def at(self, t: float) -> dict:
        """État du fichier au temps t : type de segment et chrono si observable."""
        for seg in self.segments:
            if seg["file_start"] <= t <= seg["file_end"]:
                k = seg["kind"]
                if k in ("live", "live_banner", "live_interpolated"):
                    secs = t + seg["offset"]
                    return {"kind": k, "live": True, "clock_s": secs, "added_s": None,
                            "clock": fmt_clock(int(secs)), "period": seg["period"],
                            "uncertainty_s": seg["offset_uncertainty_s"], "interpolated": k != "live"}
                if k == "stoppage":
                    add = t + seg["offset"]
                    return {"kind": k, "live": True, "clock_s": seg["base"], "added_s": add,
                            "clock": fmt_clock(seg["base"], int(add)), "period": seg["period"],
                            "uncertainty_s": seg["offset_uncertainty_s"]}
                return {"kind": k, "live": False, "clock_s": None, "added_s": None, "clock": None,
                        "period": seg.get("period")}
        return {"kind": "outside", "live": False, "clock_s": None, "added_s": None, "clock": None, "period": None}

    def to_file(self, clock_s: float, added_s: float | None = None) -> list[float]:
        """Chrono -> temps(s) fichier dans les segments de direct (liste vide si non diffusé)."""
        res = []
        for seg in self.segments:
            if added_s is None and seg["kind"] == "live":
                t = clock_s - seg["offset"]
                if seg["file_start"] - 0.5 <= t <= seg["file_end"] + 0.5:
                    res.append(round(t, 3))
            elif added_s is not None and seg["kind"] == "stoppage" and seg["base"] == clock_s:
                t = added_s - seg["offset"]
                if seg["file_start"] - 0.5 <= t <= seg["file_end"] + 0.5:
                    res.append(round(t, 3))
        return res

    def live_seconds(self) -> float:
        return sum(s["file_end"] - s["file_start"] for s in self.segments if s["kind"] in ("live", "stoppage", "live_banner", "live_interpolated"))

    def summary(self) -> dict:
        kinds = {}
        for s in self.segments:
            kinds[s["kind"]] = kinds.get(s["kind"], 0.0) + s["file_end"] - s["file_start"]
        disc = [s for s in self.segments if s["kind"] in ("discontinuity", "frozen")]
        return {"seconds_by_kind": {k: round(v, 1) for k, v in kinds.items()},
                "hard_cuts": self.data.get("hard_cuts", []),
                "discontinuities": [{k: s.get(k) for k in ("file_start", "file_end", "kind", "offset_change_s",
                                                          "clock_before", "clock_after", "match_seconds_missing",
                                                          "frozen_seconds")} for s in disc],
                "accepted_reads": self.data.get("accepted_reads"),
                "rejected_reads": self.data.get("rejected_reads")}
