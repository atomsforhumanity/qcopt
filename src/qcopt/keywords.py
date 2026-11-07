"""qcopt's keywords, by calculation type, in atomic units (bohr, Hartree).

Unknown keywords are errors. The defaults are qcopt's own; the convergence default,
``gmax`` = 0.001 eV/angstrom on the largest atomic gradient, is the threshold at which
AFH's g-xTB transition states stop carrying spurious imaginary modes (mepd's KAIST
benchmark, 2026-10-19).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
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
        maxiter: Most iterations before giving up: steps tried, including those whose
            back-transformation failed, and saddle escapes.
        convergence: The convergence criteria.
        coordinates: ``tric``: delocalized internal coordinates with each fragment's
            translation and rotation (Wang and Song, 2016); ``dlc``: delocalized
            internals with fragments joined by extra bonds; ``cartesian``:
            Cartesians without rigid motion. Internals fall back to Cartesians when
            atoms are frozen or the structure cannot be described.
        initial_hessian: The starting Hessian. ``auto``: the Hessian subprogram's
            when one is given, else Lindh's model for minima and finite differences of
            gradients for transition states. ``exact`` needs a Hessian subprogram, or
            uses finite differences of gradients. The Hessian subprogram may use a
            cheaper method than the gradients (such as GFN2-xTB Hessians for g-xTB or
            g-xTB Hessians for DFT); every frame still holds the gradient program's
            energy and gradient. On 100 Reaction-QM transition states at g-xTB, GFN2
            starting Hessians halved the total time and found a first-order saddle in
            all 100 (98 the same as from the exact start).
        hessian_every: Recompute the exact Hessian (the Hessian subprogram's, else
            finite differences of gradients) every this many steps (0: never). On the
            same benchmark at g-xTB, where a Hessian costs about 28 gradients, every
            schedule tried (1, 3, 5) cost more than the default; with a lower-level
            Hessian program it also replaces the updated Hessian, which loses the target
            method's curvature (GFN2 every step: 67 of 100 g-xTB saddles).
        recompute_hessian_on_loss: Transition states: recompute the Hessian when the
            updated one has lost its negative eigenvalue (with a Hessian subprogram,
            or when the search started from an exact or finite-difference one).
        min_steps_between_hessians: Steps between such recomputations, at least.
        lost_curvature: The negative mode counts as lost once its curvature is above
            minus this (Hartree/bohr^2, or per internal unit).
        update: The quasi-Newton update between exact Hessians; ``auto``: BFGS for
            minima, TS-BFGS for transition states.
        trust: Starting trust radius (kept between ``trust_min`` and ``trust_max``).
            In Cartesian coordinates, the largest atomic
            displacement of a step (bohr), or with ``trust_type="norm"`` the norm
            of the whole step. In internal coordinates (``tric``, ``dlc``), the norm
            of the step in those coordinates (bohr and radians, rotations scaled to
            lengths), whatever ``trust_type`` says.
        trust_type: How a Cartesian step is measured against the trust radius.
        trust_min: Smallest trust radius.
        trust_max: Largest trust radius.
        frozen_atoms: 0-based indices of atoms that do not move.
        constraints: Distances, angles and dihedrals to hold (see Constraints).
        constraint_tolerance: Largest violation (bohr or radians) at convergence.
        max_correction: Largest change (bohr or radians) of a constrained value in
            one step while it is brought to its target.
        mode: Transition states: a guide direction ``(n_atoms, 3)`` (such as a path's
            tangent at its highest image); the Hessian eigenvector overlapping it most
            is the one climbed. None: the lowest. Needs ``coordinates="cartesian"``.
        follow: With a guide ``mode``, ``overlap`` keeps climbing the eigenvector most
            like the previous step's; ``lowest`` uses the guide only for the first
            step and then climbs the lowest.
        max_rejections: Minimizations: most consecutive steps undone because the
            energy rose, before the run continues anyway.
        frequencies: Compute the Hessian at the final structure and report its
            harmonic frequencies and number of imaginary modes (from the Hessian
            subprogram, or finite differences of gradients without one). With the
            gradient method's own Hessians they decide: a minimization with an
            imaginary mode fails, and so does a transition-state search without
            exactly one. A cheaper method's are reported only.
        escape_saddles: Minimizations: check the converged structure's frequencies
            and, if it is a saddle point, step off it along the lowest mode and
            minimize again, up to this many times (0: no check). A structure still
            a saddle point after that fails (with the gradient method's own
            Hessians). Not with constraints.
        escape_step: Largest atomic displacement (bohr) of such a step.
        imaginary_cutoff: Frequencies below minus this (cm^-1) count as imaginary.
        fd_step: Step (bohr) of finite-difference Hessians.
    """

    maxiter: int = Field(default=500, ge=1)
    convergence: Convergence = Convergence()
    coordinates: Literal["tric", "dlc", "cartesian"] = "tric"
    initial_hessian: HessianSource = "auto"
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

    @model_validator(mode="after")
    def _trust_range(self) -> OptimizationKeywords:
        if self.trust_min > self.trust_max:
            raise ValueError("trust_min is larger than trust_max")
        return self

    @model_validator(mode="after")
    def _mode_in_cartesians(self) -> OptimizationKeywords:
        if self.mode is not None and self.coordinates != "cartesian":
            raise ValueError(
                "a guide mode is followed in Cartesian coordinates only: "
                "pass coordinates='cartesian' with mode"
            )
        return self


class IRCKeywords(_Keywords):
    """Keywords of ``irc`` calculations.

    Attributes:
        direction: Which branches to follow from the transition state.
        step_size: Arc length of each step in mass-weighted coordinates
            (bohr amu^1/2). A step that overshoots the valley floor (the energy
            rises, or the new gradient points along the step) is retried at half the
            length. On 100 Reaction-QM IRCs at g-xTB (axtb's default SCF accuracy),
            GS2 at 0.3 reached the same minima as Sella in all 100 with a median of 375
            gradients (ends included) against LQA at 0.1's 528; LQA at 0.3 and GS2 at
            0.5 each ended one in another minimum. With qcopt 0.2.0 at SCF accuracy
            0.01, GS2 at 0.3 matched Sella's minima in 94 (of the other 6, 4 ended in
            neighboring minima of the same species) with a median of 349.5 gradients,
            ends included; in 16 a branch reached ``max_steps``.
        max_steps: Most steps per branch. A branch that reaches it leaves the IRC
            unconverged (``success=False``, frames kept), unless ``minimize_ends``
            then converged its end.
        method: ``lqa``: local quadratic approximation (Page and McIver, 1988) on
            the current Hessian; ``gs2``: Gonzalez and Schlegel's constrained step
            (1989), each point on the sphere around a pivot half a step down the
            gradient, found on the quadratic model (one gradient per step).
        initial_hessian: Hessian at the transition state: ``auto``/``exact`` (the
            Hessian subprogram's, else finite differences) or ``finite_difference``.
            An IRC starts along the transition state's negative mode, so it needs
            one of these.
        hessian_every: Recompute the exact Hessian every this many steps (0: only
            at the transition state, then updated).
        update: Update between exact Hessians.
        gmax: A branch ends when its largest atomic gradient falls below this
            (Hartree/bohr) with positive curvature. When the updated Hessian shows
            negative curvature there, the exact Hessian decides (at most twice per
            branch); without a Hessian program, a last step lowering the energy by
            less than ``flat_energy`` also ends it. A branch can end on a flat
            shoulder of the path, where the gradient vanishes without a minimum;
            ``minimize_ends`` (or minimizing the ends afterwards) then finds it.
            From a very soft transition state the first step can already meet
            this test: from a g-xTB saddle with a 27.5i cm^-1 mode (Reaction-QM
            rxn_043), both branches ended after one step, about 20 kcal/mol above
            the minima their minimizations reached.
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
    step_size: float = Field(default=0.3, gt=0)
    max_steps: int = Field(default=300, ge=1)
    method: Literal["gs2", "lqa"] = "gs2"
    initial_hessian: HessianSource = "auto"
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

    @model_validator(mode="after")
    def _exact_start(self) -> IRCKeywords:
        if self.initial_hessian in ("model", "identity"):
            raise ValueError(
                f"an IRC needs the transition state's Hessian, not "
                f"initial_hessian={self.initial_hessian!r}"
            )
        return self
