"""Geometry helpers in atomic units: masses, rigid-body motions, alignment, norms.

Positions are ``(n_atoms, 3)`` arrays in bohr; gradients in Hartree/bohr. Flattened
vectors run atom by atom (x0, y0, z0, x1, ...), as qcdata stores them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from qcref import periodic_table as pt

Array = NDArray[np.float64]


def masses(symbols: Sequence[str]) -> Array:
    """Atomic masses in amu (most common isotopes' standard atomic weights, PubChem)."""
    return np.array([float(getattr(pt, s.capitalize()).mass) for s in symbols])


def rigid_basis(positions: Array, weights: Array | None = None) -> Array:
    """Orthonormal columns spanning rigid translations and rotations.

    Args:
        positions: ``(n_atoms, 3)`` positions.
        weights: Per-atom weights (masses) for mass-weighted coordinates
            ``sqrt(m) * x``; None for plain Cartesians.

    Returns:
        ``(3 n_atoms, k)`` with k = 6, or 5 for a linear structure (3 for one atom).
    """
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    sw = np.sqrt(w)
    center = (w[:, None] * x).sum(axis=0) / w.sum()
    r = x - center
    vecs = np.zeros((6, x.size))
    for k in range(3):
        t = np.zeros_like(x)
        t[:, k] = sw
        vecs[k] = t.ravel()
        e = np.zeros(3)
        e[k] = 1.0
        vecs[3 + k] = (sw[:, None] * np.cross(e, r)).ravel()
    u, s, _ = np.linalg.svd(vecs.T, full_matrices=False)
    keep = np.nonzero(s > 1e-6 * np.max(s))[0]
    return np.ascontiguousarray(u[:, keep])


def internal_basis(positions: Array, weights: Array | None = None) -> Array:
    """Orthonormal columns spanning everything but rigid motion (the vibrations)."""
    rb = rigid_basis(positions, weights)
    q, _ = np.linalg.qr(rb, mode="complete")
    return np.ascontiguousarray(q[:, rb.shape[1] :])


def project_rigid(
    vector: Array, positions: Array, weights: Array | None = None
) -> Array:
    """Remove rigid translation and rotation from a flattened vector."""
    rb = rigid_basis(positions, weights)
    v = np.asarray(vector, dtype=float).ravel()
    return v - rb @ (rb.T @ v)


def align(mobile: Array, reference: Array, weights: Array | None = None) -> Array:
    """Return ``mobile`` rotated and translated onto ``reference`` (weighted Kabsch)."""
    a = np.asarray(mobile, dtype=float).reshape(-1, 3)
    b = np.asarray(reference, dtype=float).reshape(-1, 3)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    ca = (w[:, None] * a).sum(0) / w.sum()
    cb = (w[:, None] * b).sum(0) / w.sum()
    h = ((a - ca) * w[:, None]).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return (a - ca) @ rot + cb


def rmsd(
    a: Array, b: Array, weights: Array | None = None, aligned: bool = True
) -> float:
    """Root-mean-square deviation (bohr), after the best fit unless ``aligned=False``."""
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)
    if aligned:
        a = align(a, b, weights)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def max_atomic_norm(gradient: Array) -> float:
    """The largest per-atom norm of a flattened or ``(n, 3)`` gradient."""
    g = np.asarray(gradient, dtype=float).reshape(-1, 3)
    return float(np.sqrt((g * g).sum(axis=1)).max()) if g.size else 0.0


def rms(vector: Array) -> float:
    """Root mean square of the components."""
    v = np.asarray(vector, dtype=float).ravel()
    return float(np.sqrt((v * v).mean())) if v.size else 0.0
