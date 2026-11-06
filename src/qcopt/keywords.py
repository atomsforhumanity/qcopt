"""qcopt's keywords, by calculation type, in atomic units (bohr, Hartree).

Unknown keywords are errors. The defaults are qcopt's own; the convergence default,
``gmax`` = 0.001 eV/angstrom on the largest atomic gradient, is the threshold at which
AFH's g-xTB transition states stop carrying spurious imaginary modes (mepd's KAIST
benchmark, 2026-10-19).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from qcref import constants as c

HessianSource = Literal["auto", "exact", "model", "identity", "finite_difference"]


class _Keywords(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Convergence(_Keywords):
    """When an optimization has converged; every criterion given must be met.

    Attributes:
        gmax: Largest per-atom gradient norm (Hartree/bohr).
        grms: Root-mean-square gradient component (Hartree/bohr), or None.
        dmax: Largest per-atom step (bohr), or None.
        energy: Largest energy change of the last step (Hartree), or None.
    """

    gmax: float = float(0.001 * c.EV_TO_HARTREE / c.ANGSTROM_TO_BOHR)
    grms: float | None = None
    dmax: float | None = None
    energy: float | None = None


class OptimizationKeywords(_Keywords):
    """Keywords of minimizations (``optimization``) and ``transition_state`` searches.

    Attributes:
        maxiter: Most steps (gradients at new geometries) before giving up.
        convergence: The convergence criteria.
        coordinates: ``tric``: delocalized internal coordinates with each fragment's
            translation and rotation (Wang and Song, 2016); ``dlc``: delocalized
            internals with fragments joined by extra bonds; ``cartesian``:
            Cartesians without rigid motion. Internals fall back to Cartesians when
            atoms are frozen or the structure cannot be described.
        hessian: The starting Hessian. ``auto``: the Hessian subprogram's when one is
            given, else Lindh's model for minima and finite differences of gradients
            for transition states. ``exact`` needs a Hessian subprogram, or uses
            finite differences of gradients.
        hessian_every: Recompute the exact Hessian every this many steps (0: never).
        recompute_hessian_on_loss: Transition states: recompute the exact Hessian when
            the updated one has lost its negative eigenvalue.
        min_steps_between_hessians: Steps between such recomputations, at least.
        update: The quasi-Newton update between exact Hessians; ``auto``: BFGS for
            minima, TS-BFGS for transition states.
        trust: Starting trust radius (bohr): the largest atomic displacement of a
            step, or with ``trust_type="norm"`` the norm of the whole step.
        trust_type: How a step is measured against the trust radius.
        trust_min: Smallest trust radius.
        trust_max: Largest trust radius.
        frozen_atoms: 0-based indices of atoms that do not move.
        mode: Transition states: a guide direction ``(n_atoms, 3)`` (such as a path's
            tangent at its highest image); the Hessian eigenvector overlapping it most
            is the one climbed. None: the lowest.
        follow: With a guide ``mode``, ``overlap`` keeps climbing the eigenvector most
            like the previous step's; ``lowest`` uses the guide only for the first
            step and then climbs the lowest.
        max_rejections: Minimizations: most consecutive steps undone because the
            energy rose, before the run continues anyway.
        frequencies: Compute the Hessian at the final structure and report its
            harmonic frequencies and number of imaginary modes.
        escape_saddles: Minimizations: check the converged structure's frequencies
            and, if it is a saddle point, step off it along the lowest mode and
            minimize again, up to this many times (0: no check).
        escape_step: Largest atomic displacement (bohr) of such a step.
        imaginary_cutoff: Frequencies below minus this (cm^-1) count as imaginary.
        fd_step: Step (bohr) of finite-difference Hessians.
    """

    maxiter: int = Field(default=500, ge=1)
    convergence: Convergence = Convergence()
    coordinates: Literal["tric", "dlc", "cartesian"] = "cartesian"
    hessian: HessianSource = "auto"
    hessian_every: int = Field(default=0, ge=0)
    recompute_hessian_on_loss: bool = True
    min_steps_between_hessians: int = Field(default=3, ge=1)
    update: Literal["bfgs", "bofill", "ts_bfgs", "sr1", "psb", "auto"] = "auto"
    trust: float = Field(default=0.2, gt=0)
    trust_type: Literal["atomic", "norm"] = "atomic"
    trust_min: float = Field(default=1e-3, gt=0)
    trust_max: float = Field(default=1.0, gt=0)
    frozen_atoms: list[int] = []
    mode: list[list[float]] | None = None
    follow: Literal["overlap", "lowest"] = "overlap"
    max_rejections: int = Field(default=3, ge=0)
    frequencies: bool = False
    escape_saddles: int = Field(default=0, ge=0)
    escape_step: float = Field(default=0.189, gt=0)
    imaginary_cutoff: float = 10.0
    fd_step: float = Field(default=0.005, gt=0)


class IRCKeywords(_Keywords):
    """Keywords of ``irc`` calculations.

    Attributes:
        direction: Which branches to follow from the transition state.
        step_size: Arc length of each step in mass-weighted coordinates
            (bohr amu^1/2).
        max_steps: Most steps per branch.
        method: ``lqa``: local quadratic approximation (Page and McIver, 1988) on
            the current Hessian.
        hessian: Hessian at the transition state; see OptimizationKeywords.
        hessian_every: Recompute the exact Hessian every this many steps (0: only
            at the transition state, then updated).
        update: Update between exact Hessians.
        gmax: A branch ends when its largest atomic gradient falls below this
            (Hartree/bohr) with positive curvature, or when the energy rises.
        frozen_atoms: Atoms that do not move.
        fd_step: Step (bohr) of finite-difference Hessians.
    """

    direction: Literal["both", "forward", "reverse"] = "both"
    step_size: float = Field(default=0.1, gt=0)
    max_steps: int = Field(default=300, ge=1)
    method: Literal["lqa"] = "lqa"
    hessian: HessianSource = "auto"
    hessian_every: int = Field(default=0, ge=0)
    update: Literal["bofill", "ts_bfgs", "bfgs", "sr1", "psb"] = "bofill"
    gmax: float = Field(default=5e-4, gt=0)
    frozen_atoms: list[int] = []
    fd_step: float = Field(default=0.005, gt=0)
