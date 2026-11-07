"""Harmonic frequencies of a Cartesian Hessian, to tell minima from saddle points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qcref import constants as c

from .geometry import Array, internal_basis, masses

#: Modes below this (cm^-1, i.e. imaginary with |freq| above it) count as imaginary.
IMAGINARY_CUTOFF_CM = 10.0


@dataclass
class Frequencies:
    """Harmonic frequencies and modes.

    Attributes:
        wavenumbers: Frequencies in cm^-1, ascending; imaginary ones negative.
        modes: ``(n_modes, n_atoms, 3)`` Cartesian displacements of unit length, in
            the order of ``wavenumbers`` (qcdata's ``normal_modes_cartesian``).
    """

    wavenumbers: Array
    modes: Array

    def n_imaginary(self, cutoff_cm: float = IMAGINARY_CUTOFF_CM) -> int:
        """Number of modes below ``-cutoff_cm``."""
        return int((self.wavenumbers < -cutoff_cm).sum())


def frequencies(
    hessian: Array,
    symbols: Sequence[str],
    positions: Array,
    *,
    frozen: Sequence[int] = (),
    project: bool = True,
) -> Frequencies:
    """Diagonalize the mass-weighted Hessian.

    Args:
        hessian: ``(3N, 3N)`` Cartesian Hessian in Hartree/bohr^2.
        symbols: Element symbols.
        positions: ``(N, 3)`` positions in bohr.
        frozen: Atoms held fixed; their rows and columns are left out and nothing
            is projected (rigid motion of the free atoms is then real motion).
        project: Remove rigid translations and rotations (isolated molecules).

    Returns:
        Frequencies and unit-length Cartesian modes of the moving atoms (frozen
        atoms have zero displacement).
    """
    n = len(symbols)
    x = np.asarray(positions, dtype=float).reshape(n, 3)
    m = masses(symbols) / float(c.phys.ELECTRON_MASS_IN_U)  # in electron masses
    free = np.setdiff1d(np.arange(n), np.asarray(frozen, dtype=int))
    dof = (3 * free[:, None] + np.arange(3)).ravel()
    w = np.repeat(1.0 / np.sqrt(m[free]), 3)
    h = np.asarray(hessian, dtype=float).reshape(3 * n, 3 * n)[np.ix_(dof, dof)]
    hm = h * w[:, None] * w[None, :]
    hm = 0.5 * (hm + hm.T)
    # One atom is not projected: in a model potential its motion is real (and an
    # isolated atom's three frequencies are zero).
    if project and len(frozen) == 0 and n > 1:
        basis = internal_basis(x, m)
        vals, vecs = np.linalg.eigh(basis.T @ hm @ basis)
        vecs = basis @ vecs
    else:
        vals, vecs = np.linalg.eigh(hm)
    to_cm = float(c.phys.HARTREE_INVERSE_METER_RELATIONSHIP) / 100.0
    wavenumbers = np.sign(vals) * np.sqrt(np.abs(vals)) * to_cm
    cart = vecs * w[:, None]
    cart /= np.linalg.norm(cart, axis=0)
    modes = np.zeros((len(vals), n, 3))
    modes[:, free, :] = cart.T.reshape(len(vals), len(free), 3)
    return Frequencies(wavenumbers=wavenumbers, modes=modes)
