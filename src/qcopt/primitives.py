"""Primitive internal coordinates, vectorized: values and Wilson B-matrix blocks.

Each kind of primitive is an integer array of atom indices, one row per coordinate.
``*_values`` return the coordinates (bohr, radians); ``*_derivatives`` return the
first derivatives as blocks of shape ``(m, k, 3)``: d q_i / d x_{atoms[i, a]}.
``scatter_b`` assembles blocks into a dense B matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .geometry import Array, cross

Index = NDArray[np.intp]


def _rows(x: Array, idx: Index, col: int) -> Array:
    return np.asarray(x, dtype=float).reshape(-1, 3)[idx[:, col]]


def stretch_values(x: Array, idx: Index) -> Array:
    """Bond lengths (bohr) of the pairs in ``idx`` ``(m, 2)``."""
    d = _rows(x, idx, 0) - _rows(x, idx, 1)
    return np.sqrt((d * d).sum(axis=1))


def stretch_derivatives(x: Array, idx: Index) -> Array:
    """Blocks ``(m, 2, 3)`` of d r / d x."""
    d = _rows(x, idx, 0) - _rows(x, idx, 1)
    u = d / np.sqrt((d * d).sum(axis=1))[:, None]
    return np.stack([u, -u], axis=1)


def bend_values(x: Array, idx: Index) -> Array:
    """Angles (radians) of the triples ``(m, 3)``, the middle atom at the vertex."""
    u = _rows(x, idx, 0) - _rows(x, idx, 1)
    v = _rows(x, idx, 2) - _rows(x, idx, 1)
    cos = (u * v).sum(1) / np.sqrt((u * u).sum(1) * (v * v).sum(1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def bend_derivatives(x: Array, idx: Index) -> Array:
    """Blocks ``(m, 3, 3)`` of d theta / d x (singular at 0 and 180 degrees)."""
    u = _rows(x, idx, 0) - _rows(x, idx, 1)
    v = _rows(x, idx, 2) - _rows(x, idx, 1)
    lu = np.sqrt((u * u).sum(1))
    lv = np.sqrt((v * v).sum(1))
    eu, ev = u / lu[:, None], v / lv[:, None]
    cos = np.clip((eu * ev).sum(1), -1.0, 1.0)
    sin = np.sqrt(np.maximum(1.0 - cos * cos, 1e-30))
    di = (cos[:, None] * eu - ev) / (lu * sin)[:, None]
    dk = (cos[:, None] * ev - eu) / (lv * sin)[:, None]
    return np.stack([di, -(di + dk), dk], axis=1)


def _torsion_vectors(x: Array, idx: Index) -> tuple[Array, ...]:
    f = _rows(x, idx, 0) - _rows(x, idx, 1)
    g = _rows(x, idx, 1) - _rows(x, idx, 2)
    h = _rows(x, idx, 3) - _rows(x, idx, 2)
    return f, g, h, cross(f, g), cross(h, g)


def torsion_values(x: Array, idx: Index) -> Array:
    """Dihedral angles (radians, in (-pi, pi]) of the quadruples ``(m, 4)``."""
    _, g, _, a, b = _torsion_vectors(x, idx)
    lg = np.sqrt((g * g).sum(1))
    return np.arctan2((cross(b, a) * g).sum(1) / lg, (a * b).sum(1))


def torsion_derivatives(x: Array, idx: Index) -> Array:
    """Blocks ``(m, 4, 3)`` of d phi / d x (Blondel and Karplus, 1996)."""
    f, g, h, a, b = _torsion_vectors(x, idx)
    a2 = (a * a).sum(1)
    b2 = (b * b).sum(1)
    lg = np.sqrt((g * g).sum(1))
    fg = (f * g).sum(1)
    hg = (h * g).sum(1)
    da = a / a2[:, None]
    db = b / b2[:, None]
    di = -lg[:, None] * da
    dl = lg[:, None] * db
    dj = lg[:, None] * da + (fg / lg)[:, None] * da - (hg / lg)[:, None] * db
    dk = (hg / lg)[:, None] * db - (fg / lg)[:, None] * da - lg[:, None] * db
    return np.stack([di, dj, dk, dl], axis=1)


def scatter_b(blocks: Array, idx: Index, n_atoms: int) -> Array:
    """Dense B matrix ``(m, 3 n_atoms)`` from derivative blocks."""
    m, k, _ = blocks.shape
    b = np.zeros((m, 3 * n_atoms))
    rows = np.repeat(np.arange(m), 3 * k)
    cols = (3 * idx[:, :, None] + np.arange(3)).reshape(m, -1).ravel()
    b[rows, cols] = blocks.reshape(m, -1).ravel()
    return b


def add_outer(hessian: Array, blocks: Array, idx: Index, weights: Array) -> None:
    """Add ``sum_i w_i b_i b_i^T`` into a dense Cartesian Hessian, in place.

    ``b_i`` are the B-matrix rows given as blocks; coordinates touching repeated
    atoms are handled by accumulation.
    """
    m, k, _ = blocks.shape
    if m == 0:
        return
    flat = blocks.reshape(m, 3 * k)
    cols = (3 * idx[:, :, None] + np.arange(3)).reshape(m, 3 * k)
    outer = weights[:, None, None] * flat[:, :, None] * flat[:, None, :]
    r = np.broadcast_to(cols[:, :, None], outer.shape)
    s = np.broadcast_to(cols[:, None, :], outer.shape)
    np.add.at(hessian, (r.ravel(), s.ravel()), outer.ravel())
