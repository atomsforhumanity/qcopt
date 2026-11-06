"""The coordinate systems an optimizer steps in: Cartesian or delocalized internal.

Both offer the same operations, so :func:`qcopt.algorithms.optimize` runs one loop:
gradients and Hessians into the coordinates, the eigen-decomposition the step is
computed from, a step back to Cartesians, and how a step is measured against the
trust radius.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import primitives as prim
from .geometry import Array, max_atomic_norm
from .internal import LINEAR_DEGREES, CoordinateError, DelocalizedInternals
from .space import CartesianSpace


class CartesianCoordinates:
    """Cartesian coordinates of the free atoms, rigid motion projected out.

    Args:
        n_atoms: Number of atoms.
        frozen: Atoms that do not move.
        atomic_trust: Measure steps by their largest atomic displacement (else norm).
    """

    name = "cartesian"
    residual = 0.0

    def __init__(
        self, n_atoms: int, frozen: Sequence[int] = (), atomic_trust: bool = True
    ) -> None:
        """Set up the projection."""
        self.space = CartesianSpace(n_atoms, frozen=frozen)
        self.atomic_trust = atomic_trust

    @property
    def n_locked(self) -> int:
        """Directions given a large curvature and never stepped along."""
        return self.space.n_locked

    def at(self, x: Array) -> None:
        """Move the projection to ``x``."""
        self.space.at(x)

    def gradient(self, x: Array, gradient: Array) -> Array:
        """The projected Cartesian gradient."""
        return self.space.project(np.asarray(gradient, dtype=float).ravel())

    def hessian(self, x: Array, hessian: Array, gradient: Array | None = None) -> Array:
        """The Cartesian Hessian (projection happens in :meth:`eigh`)."""
        return np.asarray(hessian, dtype=float)

    def eigh(self, h: Array) -> tuple[Array, Array]:
        """Eigen-decomposition with locked directions shifted up."""
        return self.space.eigh(h)

    def step(self, x: Array, d: Array) -> tuple[Array, Array, bool]:
        """New geometry; the step taken; always exact."""
        dx = self.space.project(d)
        return np.asarray(x).reshape(-1, 3) + dx.reshape(-1, 3), dx, True

    def size(self, d: Array) -> float:
        """Step size compared with the trust radius."""
        return max_atomic_norm(d) if self.atomic_trust else float(np.linalg.norm(d))

    def needs_rebuild(self, x: Array) -> bool:
        """Cartesians never need rebuilding."""
        return False


class InternalCoordinates:
    """Delocalized internal coordinates (see :mod:`qcopt.internal`).

    Args:
        symbols: Element symbols.
        x: Geometry where the coordinates are built.

    Raises:
        CoordinateError: The structure cannot be described (then use Cartesians).
    """

    n_locked = 0
    residual = 0.0

    def __init__(self, symbols: Sequence[str], x: Array, tric: bool = True) -> None:
        """Build the coordinates."""
        self.symbols = list(symbols)
        self.tric = tric
        self.name = "tric" if tric else "dlc"
        self.ic = DelocalizedInternals(symbols, x, tric=tric)

    def at(self, x: Array) -> None:
        """Nothing to move; the basis is fixed until rebuilt."""

    def gradient(self, x: Array, gradient: Array) -> Array:
        """Internal gradient; records how much of the Cartesian one it misses."""
        g = np.asarray(gradient, dtype=float).ravel()
        b = self.ic.b_matrix(x)
        gs = np.linalg.solve(b @ b.T, b @ g)
        rest = g - b.T @ gs
        xx = np.asarray(x, dtype=float).reshape(-1, 3)
        from .geometry import rigid_basis

        rb = rigid_basis(xx)
        rest -= rb @ (rb.T @ rest)
        self.residual = max_atomic_norm(rest)
        return gs

    def hessian(self, x: Array, hessian: Array, gradient: Array | None = None) -> Array:
        """Internal Hessian from a Cartesian one (with the gradient term if given)."""
        return self.ic.hessian(x, hessian, gradient)

    def eigh(self, h: Array) -> tuple[Array, Array]:
        """Plain eigen-decomposition."""
        vals, vecs = np.linalg.eigh(0.5 * (h + h.T))
        return vals, vecs

    def step(self, x: Array, d: Array) -> tuple[Array, Array, bool]:
        """Back-transform an internal step."""
        return self.ic.step(x, d)

    def size(self, d: Array) -> float:
        """Norm of the internal step."""
        return float(np.linalg.norm(d))

    def needs_rebuild(self, x: Array) -> bool:
        """Whether a bend has become linear, or a linear bend bent, since building."""
        p = self.ic.prims
        xx = np.asarray(x, dtype=float).reshape(-1, 3)
        if len(p.bends):
            if np.degrees(prim.bend_values(xx, p.bends)).max() >= LINEAR_DEGREES + 2.0:
                return True
        if len(p.linear):
            if np.degrees(prim.bend_values(xx, p.linear)).min() < LINEAR_DEGREES - 15.0:
                return True
        if p.fragments and p.max_rotation(xx) > 1.0:
            return True
        if len(p.torsions):
            a1 = np.degrees(prim.bend_values(xx, p.torsions[:, :3]))
            a2 = np.degrees(prim.bend_values(xx, p.torsions[:, 1:]))
            if max(a1.max(), a2.max()) >= LINEAR_DEGREES + 2.0:
                return True
        return False

    def rebuild(self, x: Array, h: Array) -> Array:
        """Rebuild at ``x``; returns the Hessian carried into the new coordinates."""
        hx = self.ic.to_cartesian_hessian(x, h)
        self.ic = DelocalizedInternals(self.symbols, x, tric=self.tric)
        return self.ic.hessian(x, hx)


def make_coordinates(
    kind: str,
    symbols: Sequence[str],
    x: Array,
    frozen: Sequence[int],
    atomic_trust: bool,
) -> CartesianCoordinates | InternalCoordinates:
    """Internal coordinates when asked for and possible, else Cartesians."""
    if kind in ("tric", "dlc") and not frozen and len(symbols) > 2:
        try:
            return InternalCoordinates(symbols, x, tric=kind == "tric")
        except CoordinateError:
            pass
    return CartesianCoordinates(len(symbols), frozen, atomic_trust)
