"""Hessian guesses and quasi-Newton updates (Hartree/bohr^2)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import primitives as prim
from .geometry import Array

# Lindh, Bernhardsson, Karlstrom and Malmqvist, Chem. Phys. Lett. 241, 423 (1995):
# alpha (bohr^-2) and reference distances (bohr) by the periods of the two atoms
# (row 0: H and He, row 1: Li-Ne, row 2: everything heavier).
_LINDH_ALPHA = np.array(
    [[1.0000, 0.3949, 0.3949], [0.3949, 0.2800, 0.2800], [0.3949, 0.2800, 0.2800]]
)
_LINDH_R = np.array([[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]])
_LINDH_K = {"stretch": 0.45, "bend": 0.15, "torsion": 0.005}
_PERIOD_STARTS = (3, 11)  # atomic numbers starting periods 2 and 3


def _lindh_rows(numbers: Sequence[int]) -> np.ndarray:
    z = np.asarray(numbers)
    return np.searchsorted(_PERIOD_STARTS, z, side="right")


def lindh_rho(numbers: Sequence[int], positions: Array) -> Array:
    """Lindh's pair weights rho_ij = exp(alpha_ij (r_ref,ij^2 - r_ij^2))."""
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    rows = _lindh_rows(numbers)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    alpha = _LINDH_ALPHA[rows[:, None], rows[None, :]]
    r0 = _LINDH_R[rows[:, None], rows[None, :]]
    rho = np.exp(alpha * (r0 * r0 - d2))
    np.fill_diagonal(rho, 0.0)
    return rho


def lindh(
    numbers: Sequence[int],
    positions: Array,
    *,
    cutoff: float = 1e-4,
    neighbor_cutoff: float = 1e-2,
    linear_degrees: float = 175.0,
) -> Array:
    """Lindh's model Hessian in Cartesian coordinates.

    Sums ``k rho...`` weighted outer products of stretch, bend, and torsion B-matrix
    rows over all atom pairs, triples, and chains whose weight passes ``cutoff``.
    Bends within ``180 - linear_degrees`` of linear, and torsions through them, are
    left out (their derivatives are singular). The result is positive semidefinite
    and invariant to rigid motion.

    Args:
        numbers: Atomic numbers.
        positions: ``(n, 3)`` positions in bohr.
        cutoff: Smallest weight (product of rho) kept.
        neighbor_cutoff: Smallest rho for an atom to count as a neighbor when
            building bends and torsions.
        linear_degrees: Bends at or above this angle are treated as linear.
    """
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(x)
    h = np.zeros((3 * n, 3 * n))
    if n < 2:
        return h
    rho = lindh_rho(numbers, x)

    i, j = np.nonzero(np.triu(rho > cutoff, 1))
    pairs = np.stack([i, j], axis=1)
    prim.add_outer(
        h,
        prim.stretch_derivatives(x, pairs),
        pairs,
        _LINDH_K["stretch"] * rho[i, j],
    )

    neighbors = [np.nonzero(rho[a] > neighbor_cutoff)[0] for a in range(n)]
    bends = []
    for b in range(n):
        nb = neighbors[b]
        for p in range(len(nb)):
            for q in range(p + 1, len(nb)):
                bends.append((nb[p], b, nb[q]))
    if bends:
        tri = np.array(bends, dtype=np.intp)
        w = _LINDH_K["bend"] * rho[tri[:, 0], tri[:, 1]] * rho[tri[:, 1], tri[:, 2]]
        angle = np.degrees(prim.bend_values(x, tri))
        ok = (w > cutoff) & (angle < linear_degrees) & (angle > 180.0 - linear_degrees)
        prim.add_outer(h, prim.bend_derivatives(x, tri[ok]), tri[ok], w[ok])

    tors = []
    for j_ in range(n):
        for k_ in neighbors[j_]:
            if k_ <= j_:
                continue
            for i_ in neighbors[j_]:
                if i_ == k_:
                    continue
                for l_ in neighbors[k_]:
                    if l_ in (j_, i_):
                        continue
                    tors.append((i_, j_, k_, l_))
    if tors:
        quad = np.array(tors, dtype=np.intp)
        w = (
            _LINDH_K["torsion"]
            * rho[quad[:, 0], quad[:, 1]]
            * rho[quad[:, 1], quad[:, 2]]
            * rho[quad[:, 2], quad[:, 3]]
        )
        a1 = np.degrees(prim.bend_values(x, quad[:, :3]))
        a2 = np.degrees(prim.bend_values(x, quad[:, 1:]))
        lo = 180.0 - linear_degrees
        ok = (
            (w > cutoff)
            & (a1 < linear_degrees)
            & (a2 < linear_degrees)
            & (a1 > lo)
            & (a2 > lo)
        )
        prim.add_outer(h, prim.torsion_derivatives(x, quad[ok]), quad[ok], w[ok])
    return 0.5 * (h + h.T)


# ------------------------------------------------------------------ updates


def bfgs(h: Array, s: Array, y: Array, *, damped: bool = True) -> Array:
    """BFGS update; with ``damped``, Powell's damping keeps it positive definite."""
    hs = h @ s
    shs = float(s @ hs)
    ys = float(y @ s)
    if shs <= 1e-14:
        return h
    if damped and ys < 0.2 * shs:
        theta = 0.8 * shs / (shs - ys)
        y = theta * y + (1.0 - theta) * hs
        ys = float(y @ s)
    if ys <= 1e-14:
        return h
    return h + np.outer(y, y) / ys - np.outer(hs, hs) / shs


def sr1(h: Array, s: Array, y: Array) -> Array:
    """Symmetric rank-one (Murtagh-Sargent) update, skipped when ill-conditioned."""
    r = y - h @ s
    rs = float(r @ s)
    if abs(rs) < 1e-8 * np.linalg.norm(r) * np.linalg.norm(s) or abs(rs) < 1e-14:
        return h
    return h + np.outer(r, r) / rs


def psb(h: Array, s: Array, y: Array) -> Array:
    """Powell-symmetric-Broyden update."""
    r = y - h @ s
    ss = float(s @ s)
    if ss < 1e-14:
        return h
    rs = float(r @ s)
    return h + (np.outer(r, s) + np.outer(s, r)) / ss - rs * np.outer(s, s) / ss**2


def bofill(h: Array, s: Array, y: Array) -> Array:
    """Bofill's update for saddle points: SR1 and PSB mixed by (r.s)^2/(|r|^2|s|^2).

    J. M. Bofill, J. Comput. Chem. 15, 1 (1994). Needs no positive curvature, so it
    keeps a transition state's negative eigenvalue.
    """
    r = y - h @ s
    ss = float(s @ s)
    rr = float(r @ r)
    if ss < 1e-14 or rr < 1e-28:
        return h
    rs = float(r @ s)
    phi = rs * rs / (rr * ss)
    out = (1.0 - phi) * psb(h, s, y)
    if phi > 1e-12:
        out += phi * sr1(h, s, y)
    else:
        out += phi * h
    return out


def ts_bfgs(h: Array, s: Array, y: Array) -> Array:
    """TS-BFGS: a BFGS-like update that needs no positive curvature.

    Bofill, Int. J. Quantum Chem. 94, 324 (2003); Anglada and Bofill, J. Comput.
    Chem. 19, 349 (1998). With j = y - H s and |H| the Hessian with its eigenvalues
    made positive, u = x / (x.s) for x = (s.y) y + (s.|H|s) |H|s, and
    H+ = H + u j^T + j u^T - (j.s) u u^T.
    """
    j = y - h @ s
    vals, vecs = np.linalg.eigh(0.5 * (h + h.T))
    abs_hs = vecs @ (np.abs(vals) * (vecs.T @ s))
    x = float(s @ y) * y + float(s @ abs_hs) * abs_hs
    xs = float(x @ s)
    if abs(xs) < 1e-14:
        return psb(h, s, y)
    u = x / xs
    return h + np.outer(u, j) + np.outer(j, u) - float(j @ s) * np.outer(u, u)


UPDATES = {"bfgs": bfgs, "sr1": sr1, "psb": psb, "bofill": bofill, "ts_bfgs": ts_bfgs}
