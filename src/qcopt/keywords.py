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


class Constraints(_Keywords):
    """Distances, angles and dihedrals held fixed during an optimization.

    Each entry lists 0-based atom indices and, optionally, the target value: a
    distance in bohr, an angle or dihedral in degrees. Without a value the starting
    value is kept.

    Attributes:
        distance: ``[i, j]`` or ``[i, j, bohr]``.
        angle: ``[i, j, k]`` or ``[i, j, k, degrees]``, vertex ``j``.
        dihedral: ``[i, j, k, l]`` or ``[i, j, k, l, degrees]``.
    """

    distance: list[list[float]] = []
    angle: list[list[float]] = []
    dihedral: list[list[float]] = []

    @property
    def size(self) -> int:
        """Number of constraints."""
        return len(self.distance) + len(self.angle) + len(self.dihedral)


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
        lost_curvature: The negative mode counts as lost once its curvature is above
            minus this (Hartree/bohr^2, or per internal unit).
        update: The quasi-Newton update between exact Hessians; ``auto``: BFGS for
            minima, TS-BFGS for transition states.
        trust: Starting trust radius (bohr): the largest atomic displacement of a
            step, or with ``trust_type="norm"`` the norm of the whole step.
        trust_type: How a step is measured against the trust radius.
        trust_min: Smallest trust radius.
        trust_max: Largest trust radius.
        frozen_atoms: 0-based indices of atoms that do not move.
        constraints: Distances, angles and dihedrals to hold (see Constraints).
        constraint_tolerance: Largest violation (bohr or radians) at convergence.
        max_correction: Largest change (bohr or radians) of a constrained value in
            one step while it is brought to its target.
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
    coordinates: Literal["tric", "dlc", "cartesian"] = "tric"
    hessian: HessianSource = "auto"
    hessian_every: int = Field(default=0, ge=0)
    recompute_hessian_on_loss: bool = True
    min_steps_between_hessians: int = Field(default=3, ge=1)
    lost_curvature: float = Field(default=1e-4, ge=0)
    update: Literal["bfgs", "bofill", "ts_bfgs", "sr1", "psb", "auto"] = "auto"
    trust: float = Field(default=0.2, gt=0)
    trust_type: Literal["atomic", "norm"] = "atomic"
    trust_min: float = Field(default=1e-3, gt=0)
    trust_max: float = Field(default=1.0, gt=0)
    frozen_atoms: list[int] = []
    constraints: Constraints | None = None
    constraint_tolerance: float = Field(default=1e-6, gt=0)
    max_correction: float = Field(default=0.1, gt=0)
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
            (bohr amu^1/2). A step that overshoots the valley floor (the energy
            rises, or the new gradient points along the step) is retried at half the
            length. Longer steps are cheaper: at g-xTB, 0.3 cost 35% fewer gradients
            than 0.1 on 100 Reaction-QM IRCs but ended one in another minimum, which
            0.1 did not; a corrector is needed before longer steps are safe.
        max_steps: Most steps per branch.
        method: ``lqa``: local quadratic approximation (Page and McIver, 1988) on
            the current Hessian; ``gs2``: Gonzalez and Schlegel's constrained step
            (1989), each point on the sphere around a pivot half a step down the
            gradient, found on the quadratic model (one gradient per step).
        hessian: Hessian at the transition state; see OptimizationKeywords.
        hessian_every: Recompute the exact Hessian every this many steps (0: only
            at the transition state, then updated).
        update: Update between exact Hessians.
        gmax: A branch ends when its largest atomic gradient falls below this
            (Hartree/bohr) with positive curvature. When the updated Hessian shows
            negative curvature there, the exact Hessian decides (at most twice per
            branch); without a Hessian program, a last step lowering the energy by
            less than ``flat_energy`` also ends it.
        flat_energy: Energy change (Hartree) below which a step counts as flat.
        frozen_atoms: Atoms that do not move.
        fd_step: Step (bohr) of finite-difference Hessians.
        minimize_ends: Minimize both ends after the IRC (both at once, starting
            from each branch's updated Hessian); the minimized structures become the
            path's first and last frames. An IRC stops short of the minimum, and
            the end can turn out to be another species once minimized.
        end_coordinates: Coordinates of those minimizations.
        end_convergence: Their convergence criteria.
        end_maxiter: Their most steps.
    """

    direction: Literal["both", "forward", "reverse"] = "both"
    step_size: float = Field(default=0.1, gt=0)
    max_steps: int = Field(default=300, ge=1)
    method: Literal["gs2", "lqa"] = "lqa"
    hessian: HessianSource = "auto"
    hessian_every: int = Field(default=0, ge=0)
    update: Literal["bofill", "ts_bfgs", "bfgs", "sr1", "psb"] = "bofill"
    gmax: float = Field(default=5e-4, gt=0)
    flat_energy: float = Field(default=1e-5, gt=0)
    frozen_atoms: list[int] = []
    fd_step: float = Field(default=0.005, gt=0)
    minimize_ends: bool = False
    end_coordinates: Literal["tric", "dlc", "cartesian"] = "tric"
    end_convergence: Convergence = Convergence()
    end_maxiter: int = Field(default=500, ge=1)
