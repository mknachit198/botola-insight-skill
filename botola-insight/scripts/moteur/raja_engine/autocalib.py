"""Calibration automatique des plans larges de la caméra principale.

Principe (aucun clic) :
1. Modèle de caméra : la caméra principale ne se déplace pas ; seuls l'orientation (pan, tilt,
   roulis) et le zoom changent. Son centre optique C est estimé une fois, à partir des plans déjà
   calibrés et vérifiés (réponses humaines ou de Claude), par ajustement conjoint sur les pixels
   des lignes blanches.
2. Pour une image : masque des lignes blanches (joueurs retirés), recherche grossière → fine de
   (pan, tilt, zoom) en maximisant l'accord lignes du modèle / lignes de l'image (F1 à tolérance
   décroissante : 22, 8 puis 3 px), puis ajustement des 4 paramètres sur les pixels associés aux
   éléments du terrain (lignes, rond central, arcs), perte robuste, rayon décroissant.
3. Acceptation stricte : F1 ≥ 0,40, écart médian ≤ 2 px, ≥ 800 pixels de ligne expliqués.
   Sinon : pas de calibration (jamais de valeur par défaut).

EXPÉRIMENTAL — désactivé par défaut (option `auto_calibration`, drapeau
`--auto-calibration-experimental`). Validation en laissant-un-de-côté (docs/EVALUATION.md, § 13) :
11 plans acceptés sur 16 ; aux pieds des joueurs, 5 justes à moins de 1 m, 3 décalés de 1,6 à
2,9 m et 3 complètement faux (19 à 27 m) avec un score d'acceptation aussi bon que les justes.
Contrôle visuel de 51 images acceptées : 11 fausses. Le critère d'acceptation ne suffit donc pas."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

L, W = 105.0, 68.0
IMG_W, IMG_H = 854, 480
ROLL0 = math.radians(-0.6)


# ------------------------------------------------------------------ masque des lignes
def line_mask(frame, boxes=()):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 50)).astype(np.uint8)
    field = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    field = cv2.erode(field, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    white = (th > 25) & (hsv[..., 1] < 90) & (hsv[..., 2] > 140)
    m = (white & (field > 0)).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(m, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), 0, -1)
    return m, field


# ------------------------------------------------------------------ éléments du terrain
def _features():
    h, w = L / 2, W / 2
    F = [("y", -w, (-h, h)), ("y", w, (-h, h)), ("x", -h, (-w, w)), ("x", h, (-w, w)), ("x", 0.0, (-w, w))]
    for s in (-1, 1):
        gx = s * h
        a, b = sorted((gx, gx - s * 16.5))
        F += [("y", -20.16, (a, b)), ("y", 20.16, (a, b)), ("x", gx - s * 16.5, (-20.16, 20.16))]
        a, b = sorted((gx, gx - s * 5.5))
        F += [("y", -9.16, (a, b)), ("y", 9.16, (a, b)), ("x", gx - s * 5.5, (-9.16, 9.16))]
        F.append(("arc", (s * (h - 11.0), 0.0, 9.15, s), None))
    F.append(("circle", (0.0, 0.0, 9.15), None))
    return F


FEATS = _features()


def _samples(step=0.1):
    pts, ids = [], []
    for k, (kind, p, ext) in enumerate(FEATS):
        if kind in ("x", "y"):
            n = max(2, int((ext[1] - ext[0]) / step))
            t = np.linspace(ext[0], ext[1], n)
            q = np.stack([np.full(n, p), t], 1) if kind == "x" else np.stack([t, np.full(n, p)], 1)
        else:
            cx, cy, r = p[0], p[1], p[2]
            ang = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / step))
            q = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1)
            if kind == "arc":
                s = p[3]
                q = q[s * (q[:, 0] - (s * L / 2 - s * 16.5)) < 0]
        pts.append(q)
        ids.append(np.full(len(q), k))
    return np.concatenate(pts), np.concatenate(ids)


SP, SID = _samples()
SP_COARSE = SP[::5]


def project(Hm, pts):
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(Hm).T
    return p[:, :2] / p[:, 2:3]


def _residuals_m(G, pix, fid):
    P = np.c_[pix, np.ones(len(pix))] @ G.T
    P = P[:, :2] / P[:, 2:3]
    r = np.zeros(len(pix))
    for k in np.unique(fid):
        m = fid == k
        kind, p, _ = FEATS[k]
        if kind == "x":
            r[m] = P[m, 0] - p
        elif kind == "y":
            r[m] = P[m, 1] - p
        else:
            r[m] = np.hypot(P[m, 0] - p[0], P[m, 1] - p[1]) - p[2]
    return r, P


def _local_scale(H, P, fid, eps=0.05):
    sc = np.ones(len(P))
    for k in np.unique(fid):
        m = fid == k
        kind, p, _ = FEATS[k]
        Q = P[m]
        if kind == "x":
            n = np.tile([1.0, 0.0], (len(Q), 1))
        elif kind == "y":
            n = np.tile([0.0, 1.0], (len(Q), 1))
        else:
            d = Q - np.array(p[:2])
            n = d / np.maximum(1e-6, np.linalg.norm(d, axis=1, keepdims=True))
        sc[m] = np.linalg.norm(project(H, Q + eps * n) - project(H, Q), axis=1) / eps
    return sc


# ------------------------------------------------------------------ modèle de caméra
def _rot(p, t, r):
    f = np.array([math.sin(p) * math.cos(t), math.cos(p) * math.cos(t), -math.sin(t)])
    right = np.array([math.cos(p), -math.sin(p), 0.0])
    down = np.cross(f, right)
    cr, sr = math.cos(r), math.sin(r)
    return np.stack([cr * right + sr * down, -sr * right + cr * down, f])


def H_camera(params, C):
    """Homographie terrain (x, y) → image pour la caméra de centre C (repère X = x, Y = −y, Z haut)."""
    p, t, r, f = params
    A = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1.0]]) @ _rot(p, t, r)
    Hm = np.stack([A @ np.array([1.0, 0, 0]), A @ np.array([0, -1.0, 0]), -A @ np.asarray(C, float)], 1)
    return Hm / Hm[2, 2]


def _associate(H, pix, R):
    proj = project(H, SP)
    ok = np.isfinite(proj).all(1) & (np.abs(proj) < 4000).all(1)
    d, j = cKDTree(proj[ok]).query(pix, distance_upper_bound=R)
    sel = np.isfinite(d)
    return pix[sel], SID[ok][j[sel]]


def fit_camera_center(observations: list[tuple[np.ndarray, np.ndarray]], C0=(0.0, -95.0, 17.0), max_px=400,
                      max_plane_px=3.0):
    """observations : [(pixels de ligne, H terrain→image vérifiée)]. Renvoie (C, rapport).

    Deux passes : un plan dont l'écart médian dépasse `max_plane_px` avec le centre commun (autre
    caméra, ou calibration de référence imprécise) est écarté et C est réestimé sans lui."""
    sets, inits, used = [], [], []
    rng = np.random.default_rng(0)
    for j, (pix, Hm) in enumerate(observations):
        apix, afid = _associate(np.asarray(Hm), pix, 3)
        if len(apix) < 200:
            continue
        # au plus `max_px` pixels par plan, répartis entre les éléments du terrain
        if len(apix) > max_px:
            keep = np.zeros(len(apix), bool)
            ks = np.unique(afid)
            for k in ks:
                idx = np.flatnonzero(afid == k)
                keep[rng.choice(idx, min(len(idx), max(20, max_px // len(ks))), replace=False)] = True
            apix, afid = apix[keep], afid[keep]
        sets.append((apix, afid))
        used.append(j)
        G = np.linalg.inv(Hm)
        c = G @ np.array([IMG_W / 2, IMG_H * 0.62, 1.0])
        c = c[:2] / c[2]
        inits.append([math.atan2(c[0], -c[1] - C0[1]), math.radians(9.5), ROLL0, 2400.0])
    if len(sets) < 3:
        return None, {"reason": "moins de 3 plans vérifiés utilisables"}
    C, params, med = _solve_center(sets, inits, C0)
    excluded = [i for i, m in enumerate(med) if m > max_plane_px]
    if excluded and len(sets) - len(excluded) >= 3:
        keep = [i for i in range(len(sets)) if i not in excluded]
        C, p2, m2 = _solve_center([sets[i] for i in keep], [params[i] for i in keep], C)
        med = [None] * len(sets)
        for i, m in zip(keep, m2):
            med[i] = m
    else:
        excluded = []
    return C, {"planes": len(sets) - len(excluded), "observations": [used[i] for i in range(len(sets)) if i not in excluded],
               "excluded_observations": [used[i] for i in excluded],
               "median_px_per_plane": [None if m is None else round(m, 2) for m in med]}


def _solve_center(sets, inits, C0):
    """Ajustement conjoint de C et des (pan, tilt, roulis, focale) de chaque plan."""
    from scipy.sparse import lil_matrix

    def fun(v):
        C = v[:3]
        out = []
        for i, (apix, afid) in enumerate(sets):
            Hm = H_camera(v[3 + 4 * i: 7 + 4 * i], C)
            r, P = _residuals_m(np.linalg.inv(Hm), apix, afid)
            out.append(r * _local_scale(Hm, P, afid) / math.sqrt(len(apix) / 300))
        return np.concatenate(out)
    # Jacobienne creuse : les résidus d'un plan ne dépendent que de C et de ses 4 paramètres
    # (7 évaluations groupées par itération au lieu de 3 + 4 × nombre de plans).
    sizes = [len(a) for a, _ in sets]
    S = lil_matrix((sum(sizes), 3 + 4 * len(sets)), dtype=np.int8)
    row = 0
    for i, n in enumerate(sizes):
        S[row:row + n, :3] = 1
        S[row:row + n, 3 + 4 * i: 7 + 4 * i] = 1
        row += n
    x0 = np.r_[np.asarray(C0, float), np.concatenate([np.asarray(p, float) for p in inits])]
    res = least_squares(fun, x0, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=600, jac_sparsity=S)
    C = res.x[:3]
    params, med = [], []
    for i, (apix, afid) in enumerate(sets):
        pr = res.x[3 + 4 * i: 7 + 4 * i]
        Hm = H_camera(pr, C)
        r, P = _residuals_m(np.linalg.inv(Hm), apix, afid)
        params.append(pr.tolist())
        med.append(float(np.median(np.abs(r * _local_scale(Hm, P, afid)))))
    return C, params, med


# ------------------------------------------------------------------ recherche + ajustement
def _f1(Hm, dt, field, mpts, tol):
    q = project(Hm, SP_COARSE)
    ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < IMG_W - 1) & (q[:, 1] >= 0) & (q[:, 1] < IMG_H - 1)
    q = q[ok]
    if len(q) < 40:
        return -1.0
    q = q[field[q[:, 1].astype(int), q[:, 0].astype(int)] > 0]
    if len(q) < 40:
        return -1.0
    prec = (dt[q[:, 1].astype(int), q[:, 0].astype(int)] <= tol).mean()
    d, _ = cKDTree(q).query(mpts, distance_upper_bound=tol + 1)
    rec = np.isfinite(d).mean()
    return float(2 * prec * rec / max(1e-6, prec + rec))


def _search(mask, field, C):
    dt = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(mask)
    idx = np.random.default_rng(0).choice(len(xs), size=min(800, len(xs)), replace=False)
    mpts = np.stack([xs[idx], ys[idx]], 1).astype(float)
    cands = []
    for p in np.radians(np.arange(-36, 36.1, 2.0)):
        for t in np.radians(np.arange(6.0, 14.01, 1.0)):
            for f in 1400 * (1.12 ** np.arange(0, 10)):
                cands.append((_f1(H_camera((p, t, ROLL0, f), C), dt, field, mpts, 22.0), p, t, f))
    cands.sort(key=lambda x: -x[0])
    mid = []
    for _, p0, t0, f0 in cands[:20]:
        for dp in np.radians(np.arange(-1.5, 1.51, 0.5)):
            for dtl in np.radians(np.arange(-0.75, 0.76, 0.25)):
                for ff in (0.91, 0.94, 0.97, 1.0, 1.03, 1.06, 1.09):
                    prm = (p0 + dp, t0 + dtl, ROLL0, f0 * ff)
                    mid.append((_f1(H_camera(prm, C), dt, field, mpts, 8.0), *prm))
    mid.sort(key=lambda x: -x[0])
    fine, seen = [], []
    for _, p0, t0, _r, f0 in mid:
        if any(abs(p0 - a) < math.radians(1.0) and abs(f0 / b - 1) < 0.05 for a, b in seen):
            continue
        seen.append((p0, f0))
        for dp in np.radians(np.arange(-0.4, 0.41, 0.2)):
            for dtl in np.radians(np.arange(-0.2, 0.21, 0.1)):
                for ff in (0.98, 1.0, 1.02):
                    prm = (p0 + dp, t0 + dtl, ROLL0, f0 * ff)
                    fine.append((_f1(H_camera(prm, C), dt, field, mpts, 3.0), *prm))
        if len(seen) >= 5:
            break
    fine.sort(key=lambda x: -x[0])
    return fine[:4], dt, mpts


def _fit_params(pix, params, C, radii=(14, 9, 6, 4, 3)):
    params = np.array(params, float)
    last = {}
    for R in radii:
        Hm = H_camera(params, C)
        apix, afid = _associate(Hm, pix, R)
        if len(apix) < 30:
            return params, {"n": int(len(apix))}
        sc = _local_scale(Hm, project(np.linalg.inv(Hm), apix), afid)
        cnt = {k: (afid == k).sum() for k in np.unique(afid)}
        wts = np.array([1.0 / math.sqrt(max(1.0, cnt[k] / 200.0)) for k in afid])

        def fun(v):
            r, _ = _residuals_m(np.linalg.inv(H_camera(v, C)), apix, afid)
            return r * sc * wts
        params = least_squares(fun, params, loss="soft_l1", f_scale=2.0, x_scale=[0.01, 0.01, 0.01, 50.0]).x
        Hm = H_camera(params, C)
        r, P = _residuals_m(np.linalg.inv(Hm), apix, afid)
        rp = np.abs(r * _local_scale(Hm, P, afid))
        last = {"n": int(len(apix)), "median_px": round(float(np.median(rp)), 2), "p90_px": round(float(np.percentile(rp, 90)), 2),
                "features": {int(k): int(v) for k, v in cnt.items()}}
    return params, last


def calibrate_frame(frame, boxes, C, n_try=3) -> dict:
    """Renvoie {'ok': bool, 'H_pitch2img', 'H_img2pitch', 'f1', 'median_px', 'n', 'params'}."""
    mask, field = line_mask(frame, boxes)
    if int(mask.sum()) < 300:
        return {"ok": False, "reason": "trop peu de lignes visibles"}
    top, dt, _ = _search(mask, field, C)
    ys, xs = np.nonzero(mask)
    pix = np.stack([xs, ys], 1).astype(float)
    idx = np.random.default_rng(1).choice(len(xs), size=min(1500, len(xs)), replace=False)
    best = None
    for _, p, t, r, f in top[:n_try]:
        prm, info = _fit_params(pix, (p, t, r, f), C)
        Hm = H_camera(prm, C)
        f1 = _f1(Hm, dt, field, pix[idx], 3.0)
        cand = {"f1": round(f1, 3), "params": [float(x) for x in prm], **info, "H_pitch2img": Hm}
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    if best is None:
        return {"ok": False, "reason": "aucune solution"}
    best["ok"] = best["f1"] >= 0.40 and best.get("median_px", 99) <= 2.0 and best.get("n", 0) >= 800
    best["H_img2pitch"] = np.linalg.inv(best["H_pitch2img"])
    return best
