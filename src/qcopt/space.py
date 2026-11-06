"""The Cartesian space an optimizer moves in: free atoms, without rigid motion.

Steps are computed in the full 3N Cartesian space. Directions the optimizer must not
move along (rigid translation and rotation of an isolated system, and every
coordinate of a frozen atom) are removed from the gradient and given a large
positive curvature in the Hessian, so they are never stepped along and never chosen
as a transition-state mode. Optionally the space is mass-weighted (IRC).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import Array, rigid_basis

#: Curvature (Hartree/bohr^2) given to directions the optimizer must not move along.
LOCKED_CURVATURE = 1e3


class CartesianSpace:
    """Projection onto the free, internal motions of a structure.

    Args:
        n_atoms: Number of atoms.
        frozen: Atoms that do not move.
        project_rigid: Remove overall translation and rotation (isolated systems).
            Ignored when atoms are frozen: the frozen atoms then fix the frame.
        masses: Per-atom masses (amu) for a mass-weighted space, else None.
    """

    def __init__(
        self,
        n_atoms: int,
        *,
        frozen: Sequence[int] = (),
        project_rigid: bool = True,
        masses: Array | None = None,
    ) -> None:
        """Set up the space."""
        self.n_atoms = n_atoms
        self.frozen = np.asarray(sorted(set(int(i) for i in frozen)), dtype=int)
        self.project_rigid = project_rigid and self.frozen.size == 0 and n_atoms > 1
        self.masses = masses
        mask = np.ones(3 * n_atoms, dtype=bool)
        for a in self.frozen:
            mask[3 * a : 3 * a + 3] = False
        self.free = mask
        self._locked: Array = np.zeros((3 * n_atoms, 0))
        self._sqrt_m = (
            np.ones(3 * n_atoms) if masses is None else np.repeat(np.sqrt(masses), 3)
        )

    # --------------------------------------------------------- mass weighting

    def to_space(self, cartesian_step: Array) -> Array:
        """Cartesian displacement to (mass-weighted) space coordinates."""
        return np.asarray(cartesian_step, dtype=float).ravel() * self._sqrt_m

    def to_cartesian(self, step: Array) -> Array:
        """Space displacement back to a Cartesian displacement (bohr)."""
        return np.asarray(step, dtype=float).ravel() / self._sqrt_m

    def gradient_to_space(self, gradient: Array) -> Array:
        """Cartesian gradient to the space's gradient."""
        return np.asarray(gradient, dtype=float).ravel() / self._sqrt_m

    def hessian_to_space(self, hessian: Array) -> Array:
        """Cartesian Hessian to the space's Hessian."""
        w = 1.0 / self._sqrt_m
        return np.asarray(hessian, dtype=float) * w[:, None] * w[None, :]

    # --------------------------------------------------------------- projection

    def at(self, positions: Array) -> CartesianSpace:
        """Recompute the rigid-motion directions at ``positions``; returns self."""
        if self.project_rigid:
            self._locked = rigid_basis(positions, self.masses)
        return self

    def project(self, vector: Array) -> Array:
        """Remove locked directions (rigid motion, frozen atoms) from a vector."""
        v = np.asarray(vector, dtype=float).ravel().copy()
        v[~self.free] = 0.0
        if self._locked.shape[1]:
            v -= self._locked @ (self._locked.T @ v)
        return v

    def eigh(self, hessian: Array) -> tuple[Array, Array]:
        """Eigen-decomposition of the projected Hessian, locked directions shifted up."""
        h = np.asarray(hessian, dtype=float)
        if self._locked.shape[1]:
            p = np.eye(len(h)) - self._locked @ self._locked.T
            h = p @ h @ p + LOCKED_CURVATURE * (self._locked @ self._locked.T)
        if not self.free.all():
            h = h.copy()
            h[~self.free, :] = 0.0
            h[:, ~self.free] = 0.0
            h[~self.free, ~self.free] = LOCKED_CURVATURE
        h = 0.5 * (h + h.T)
        vals, vecs = np.linalg.eigh(h)
        return vals, vecs

    @property
    def n_locked(self) -> int:
        """Number of locked directions."""
        return int(self._locked.shape[1] + 3 * self.frozen.size)
