"""Geometric constraints: distances, angles and dihedrals held at target values.

Constraints are kept by projection, the same way in Cartesian and internal
coordinates: the constrained directions are removed from the gradient and the
Hessian (and given a large curvature so the step never moves along them), and each
step adds the smallest correction that brings the constraints to their targets
(linearized, at most ``max_correction`` per constraint per step). A constrained
optimization has converged when the gradient without the constraint forces meets the
convergence criteria and every constraint is within ``tolerance`` of its target.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import primitives as prim
from .geometry import Array
from .keywords import Constraints

_MAX_ANGLE_DEGREES = 175.0

_KINDS = {
    "distance": (2, prim.stretch_values, prim.stretch_derivatives),
    "angle": (3, prim.bend_values, prim.bend_derivatives),
    "dihedral": (4, prim.torsion_values, prim.torsion_derivatives),
}


class ConstraintSet:
    """The constraints of one optimization and their targets.

    Args:
        spec: The ``constraints`` keyword.
        positions: Starting geometry (bohr); targets left out keep its values.
        frozen: Atoms that do not move.

    Raises:
        ValueError: An entry names the wrong number of atoms or repeats one; the
            same constraint is given twice; a constraint involves only frozen atoms
            (it is held already); or an angle is constrained at or near 180 degrees,
            where its derivative vanishes (constrain distances instead).
    """

    def __init__(
        self, spec: Constraints, positions: Array, frozen: Sequence[int] = ()
    ) -> None:
        """Collect index arrays and targets (bohr, radians)."""
        x = np.asarray(positions, dtype=float).reshape(-1, 3)
        n_atoms = len(x)
        fixed = {int(a) for a in frozen}
        seen: set[tuple[int, ...]] = set()
        self.groups: list[tuple[str, np.ndarray, Array]] = []
        for kind, (k, values, _) in _KINDS.items():
            entries = getattr(spec, kind)
            if not entries:
                continue
            idx, targets = [], []
            for entry in entries:
                if len(entry) not in (k, k + 1):
                    raise ValueError(
                        f"a {kind} constraint names {k} atoms and optionally a value: {entry}"
                    )
                atoms = [int(a) for a in entry[:k]]
                if len(set(atoms)) != k or min(atoms) < 0 or max(atoms) >= n_atoms:
                    raise ValueError(f"bad atoms in {kind} constraint {entry}")
                key = min(tuple(atoms), tuple(reversed(atoms)))
                if key in seen:
                    raise ValueError(f"{kind} constraint {entry} is given twice")
                seen.add(key)
                if set(atoms) <= fixed:
                    raise ValueError(
                        f"{kind} constraint {entry} involves only frozen atoms"
                    )
                idx.append(atoms)
                targets.append(None if len(entry) == k else float(entry[k]))
            index = np.array(idx, dtype=np.intp)
            now = values(x, index)
            target = np.array(
                [
                    now[i]
                    if t is None
                    else (t if kind == "distance" else np.radians(t))
                    for i, t in enumerate(targets)
                ]
            )
            if kind == "angle" and np.degrees(target).max() > _MAX_ANGLE_DEGREES:
                raise ValueError(
                    f"angle constraints above {_MAX_ANGLE_DEGREES:g} degrees are not "
                    "supported (the angle's derivative vanishes at 180); constrain "
                    "distances instead"
                )
            self.groups.append((kind, index, target))
        self.size = sum(len(i) for _, i, _ in self.groups)
        self.n_atoms = n_atoms

    def violation(self, x: Array) -> Array:
        """``target - value`` of each constraint (dihedrals wrapped)."""
        xx = np.asarray(x, dtype=float).reshape(-1, 3)
        out = []
        for kind, index, target in self.groups:
            d = target - _KINDS[kind][1](xx, index)
            if kind == "dihedral":
                d = (d + np.pi) % (2 * np.pi) - np.pi
            out.append(d)
        return np.concatenate(out) if out else np.zeros(0)

    def jacobian(self, x: Array) -> Array:
        """``(n_constraints, 3 n_atoms)`` derivatives of the constrained values."""
        xx = np.asarray(x, dtype=float).reshape(-1, 3)
        rows = [
            prim.scatter_b(_KINDS[kind][2](xx, index), index, self.n_atoms)
            for kind, index, _ in self.groups
        ]
        return np.vstack(rows) if rows else np.zeros((0, 3 * self.n_atoms))


def project(jw: Array) -> tuple[Array, Array]:
    """Projectors onto the constrained directions and their complement.

    Args:
        jw: Constraint Jacobian in the working coordinates ``(k, n)``.

    Returns:
        ``(p_free, p_constrained)``, each ``(n, n)``.
    """
    gram = jw @ jw.T
    pc = jw.T @ np.linalg.solve(gram, jw)
    return np.eye(jw.shape[1]) - pc, pc


def correction(jw: Array, violation: Array, max_correction: float) -> Array:
    """Smallest step (working coordinates) that fixes the linearized violations."""
    v = np.asarray(violation, dtype=float)
    largest = float(np.abs(v).max()) if v.size else 0.0
    if largest > max_correction:
        v = v * (max_correction / largest)
    return jw.T @ np.linalg.solve(jw @ jw.T, v)


def free_gradient(jx: Array, gradient: Array) -> Array:
    """The Cartesian gradient without its constraint forces (least squares)."""
    g = np.asarray(gradient, dtype=float).ravel()
    if not len(jx):
        return g
    lam = np.linalg.lstsq(jx.T, g, rcond=None)[0]
    return g - jx.T @ lam
