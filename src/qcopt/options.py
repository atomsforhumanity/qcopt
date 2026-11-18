"""qcopt's settings: one options model per calculation type, made of setting groups.

A qcopt calculation is configured by the ``keywords`` of its ``ProgramInput``,
validated by the options model of its calculation type:

- ``optimization`` (a minimum): :class:`OptimizationOptions`
- ``transition_state``: :class:`TransitionStateOptions`
- ``irc``: :class:`IRCOptions`

The rules, shared with AFH's other programs (axtb, linus):

- Every setting is ``group.key`` or a top-level scalar, never deeper; a key never
  repeats its group's name. Tolerances end in ``_tol``; iteration limits are
  ``max_iter``.
- ``"auto"`` means qcopt decides; every result records what it decided in
  ``extras["resolved"]`` (the settings with every ``auto`` replaced, which reproduce
  the run, and the reason for each).
- Unknown keys are errors, with a suggestion; so are keys of another calculation
  type, and settings given that cannot affect the calculation (such as
  ``escape.step`` without ``escape.attempts``). Settings that act only if a condition
  arises during the run are allowed.
- Atomic units (bohr, Hartree) unless a key says otherwise: angles of constraints
  are in degrees, frequencies in cm^-1.

``qcopt.default_options(calctype)`` lists every setting with its default, and
``qcopt.options_schema(calctype)`` is the JSON Schema.
"""

from __future__ import annotations

import difflib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from qcref import constants as c

from .errors import InputError

CoordinateSystem = Literal["auto", "tric", "dlc", "cartesian"]
HessianUpdate = Literal["bfgs", "bofill", "ts_bfgs", "sr1", "psb"]


#: What a setting is about (qcdata's conventions, ``x-concern`` on every key):
#: ``model`` (what is computed: constraints), ``convergence`` (when it stops),
#: ``algorithm`` (how it gets there), ``resource`` (threads).
Concern = Literal[
    "model", "approximation", "convergence", "algorithm", "resource", "output"
]

#: Settings that mean the same in AFH's programs (qcdata's conventions, "Shared names").
SHARED = {
    "threads",
    "convergence.max_iter",
    "convergence.g_tol",
    "convergence.e_tol",
    "convergence.d_tol",
    "freq.imaginary_cutoff",
}


def _f(
    default: Any,
    *,
    unit: str | None = None,
    tier: str = "common",
    concern: Concern = "algorithm",
    **kw: Any,
) -> Any:
    """A field with its tier (common or expert), concern and unit in the JSON Schema."""
    extra: dict[str, Any] = {"x-tier": tier, "x-concern": concern}
    if unit is not None:
        extra["x-unit"] = unit
    return Field(default=default, json_schema_extra=extra, **kw)


def _c(default: Any, **kw: Any) -> Any:
    """A convergence criterion's field."""
    return _f(default, concern="convergence", **kw)


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


#: Convergence on the largest atomic gradient: 0.001 eV/angstrom, the threshold at
#: which AFH's g-xTB transition states stop carrying spurious imaginary modes.
G_TOL = float(0.001 * c.EV_TO_HARTREE / c.ANGSTROM_TO_BOHR)


# ------------------------------------------------------------------ groups


class ConvergenceSettings(_Settings):
    """When a minimization or transition-state search has converged.

    Every criterion given must be met; ``None`` leaves a criterion out.

    Attributes:
        max_iter: Most iterations before the run stops unconverged: steps tried
            (including those whose back-transformation failed) and saddle escapes.
        g_tol: Largest per-atom gradient norm (Hartree/bohr); the default is
            0.001 eV/angstrom.
        g_rms_tol: Root-mean-square gradient component (Hartree/bohr).
        d_tol: Largest per-atom displacement of the last step (bohr).
        e_tol: Energy change of the last step (Hartree).
    """

    max_iter: int = _c(500, ge=1)
    g_tol: float = _c(G_TOL, unit="Eh/bohr", gt=0)
    g_rms_tol: float | None = _c(None, unit="Eh/bohr", tier="expert", gt=0)
    d_tol: float | None = _c(None, unit="bohr", tier="expert", gt=0)
    e_tol: float | None = _c(None, unit="Eh", tier="expert", gt=0)


class IRCConvergenceSettings(_Settings):
    """When an IRC branch ends.

    A branch ends where its largest atomic gradient is below ``g_tol`` and the
    curvature is positive. Where the updated Hessian shows negative curvature, the
    exact Hessian decides (at most twice per branch); without one, a last step that
    changed the energy by less than ``e_tol`` also ends it.

    Attributes:
        max_iter: Most steps per branch; a branch that reaches it leaves the IRC
            unconverged unless ``ends.minimize`` then converges its end.
        g_tol: Largest per-atom gradient norm at a branch's end (Hartree/bohr).
        e_tol: Energy change (Hartree) below which a step counts as flat.
    """

    max_iter: int = _c(300, ge=1)
    g_tol: float = _c(5e-4, unit="Eh/bohr", gt=0)
    e_tol: float = _c(1e-5, unit="Eh", tier="expert", gt=0)


class TrustSettings(_Settings):
    """The trust region: how far one step may go.

    In Cartesian coordinates a step is measured by its largest atomic displacement
    (``measure="atomic"``) or its norm (``"norm"``); in internal coordinates (TRIC,
    DLC) always by its norm in those coordinates (bohr and radians, rotations scaled
    to lengths).

    Attributes:
        radius: Starting trust radius (bohr), kept between ``min`` and ``max``.
        min: Smallest trust radius (bohr).
        max: Largest trust radius (bohr).
        measure: How a Cartesian step is measured against the radius.
    """

    radius: float = _f(0.2, unit="bohr", gt=0)
    min: float = _f(1e-3, unit="bohr", tier="expert", gt=0)
    max: float = _f(1.0, unit="bohr", tier="expert", gt=0)
    measure: Literal["atomic", "norm"] = _f("atomic", tier="expert")

    @model_validator(mode="after")
    def _ordered(self) -> TrustSettings:
        if self.min > self.max:
            raise ValueError("trust.min is larger than trust.max")
        return self


class MinimizationTrustSettings(TrustSettings):
    """The trust region of a minimization (see :class:`TrustSettings`).

    Attributes:
        max_rejections: Most consecutive steps undone because the energy rose,
            before the run accepts the next one anyway.
    """

    max_rejections: int = _f(3, tier="expert", ge=0)


class HessianSettings(_Settings):
    """How the Hessian is started, updated and recomputed.

    The Hessian itself comes from the Hessian subprogram (``hessian_program=``),
    which may be a cheaper method than the gradients; without one, from finite
    differences of gradients. Frames always hold the gradient program's energies
    and gradients.

    Attributes:
        initial: The starting Hessian. ``auto``: the input's given Hessian
            (``ProgramInput.given``) when it has one, else the Hessian subprogram's
            when one is given, else Lindh's model Hessian for minima and finite
            differences for transition states. ``given``: the input's given Hessian.
            ``exact``: the Hessian subprogram's, or finite differences without one.
            ``model``: Lindh's model. ``identity``: 0.5 times the identity.
            ``finite_difference``: finite differences of gradients (only without a
            Hessian subprogram).
        update: The quasi-Newton update between exact Hessians; ``auto``: BFGS for
            minima, TS-BFGS for transition states.
        recompute_every: Recompute the exact Hessian every this many steps (0:
            never). With a lower-level Hessian program this replaces the updated
            Hessian and its target-method curvature.
        step: Displacement (bohr) of finite-difference Hessians.
    """

    initial: Literal[
        "auto", "given", "exact", "model", "identity", "finite_difference"
    ] = _f("auto")
    update: Literal["auto", "bfgs", "bofill", "ts_bfgs", "sr1", "psb"] = _f(
        "auto", tier="expert"
    )
    recompute_every: int = _f(0, ge=0)
    step: float = _f(0.005, unit="bohr", tier="expert", gt=0)


class TransitionStateHessianSettings(HessianSettings):
    """The Hessian of a transition-state search (see :class:`HessianSettings`).

    Attributes:
        recompute_on_loss: Recompute the exact Hessian when the updated one has
            lost its negative eigenvalue (with a Hessian subprogram, or when the
            search started from an exact or finite-difference Hessian).
        recompute_min_steps: Steps between such recomputations, at least.
        lost_curvature: The negative mode counts as lost once its curvature is
            above minus this (Hartree/bohr^2, or per internal unit).
    """

    recompute_on_loss: bool = _f(True)
    recompute_min_steps: int = _f(3, tier="expert", ge=1)
    lost_curvature: float = _f(1e-4, unit="Eh/bohr^2", tier="expert", ge=0)


class IRCHessianSettings(_Settings):
    """The Hessian of an IRC.

    An IRC starts along the transition state's negative mode, so it starts from an
    exact (or finite-difference) Hessian, then updates it.

    Attributes:
        initial: The transition state's Hessian. ``auto``: the input's given
            Hessian (``ProgramInput.given``) when it has one, else ``exact``.
            ``given``: the input's given Hessian. ``exact``: the Hessian
            subprogram's, else finite differences of gradients.
            ``finite_difference``: finite differences, without a Hessian
            subprogram.
        update: The update between exact Hessians.
        recompute_every: Recompute the exact Hessian every this many steps (0:
            only at the transition state).
        step: Displacement (bohr) of finite-difference Hessians.
    """

    initial: Literal["auto", "given", "exact", "finite_difference"] = _f("auto")
    update: HessianUpdate = _f("bofill", tier="expert")
    recompute_every: int = _f(0, ge=0)
    step: float = _f(0.005, unit="bohr", tier="expert", gt=0)


class ConstraintSettings(_Settings):
    """Atoms held fixed, and distances, angles and dihedrals held at targets.

    Each distance, angle or dihedral lists 0-based atom indices and optionally the
    target value; without one, the starting value is kept.

    Attributes:
        frozen: 0-based indices of atoms that do not move.
        distance: ``[i, j]`` or ``[i, j, bohr]``.
        angle: ``[i, j, k]`` or ``[i, j, k, degrees]``, vertex ``j``.
        dihedral: ``[i, j, k, l]`` or ``[i, j, k, l, degrees]``.
        tol: Largest violation (bohr or radians) at convergence.
        max_correction: Largest change (bohr or radians) of a constrained value in
            one step while it is brought to its target.
    """

    frozen: list[int] = _f([], concern="model")
    distance: list[list[float]] = _f([], concern="model")
    angle: list[list[float]] = _f([], concern="model")
    dihedral: list[list[float]] = _f([], concern="model")
    tol: float = _c(1e-6, tier="expert", gt=0)
    max_correction: float = _f(0.1, tier="expert", gt=0)

    @property
    def size(self) -> int:
        """Number of distance, angle and dihedral constraints."""
        return len(self.distance) + len(self.angle) + len(self.dihedral)


class IRCConstraintSettings(_Settings):
    """Atoms held fixed during an IRC.

    Attributes:
        frozen: 0-based indices of atoms that do not move.
    """

    frozen: list[int] = _f([], concern="model")


class ModeSettings(_Settings):
    """Which Hessian eigenvector a transition-state search climbs.

    Attributes:
        guide: A direction ``(n_atoms, 3)``, such as a path's tangent at its highest
            image: the eigenvector overlapping it most is climbed. None: the lowest.
            Needs ``coordinate_system="cartesian"``.
        follow: With a guide, ``overlap`` keeps climbing the eigenvector most like
            the previous step's; ``lowest`` uses the guide for the first step only.
    """

    guide: list[list[float]] | None = _f(None, unit="bohr")
    follow: Literal["overlap", "lowest"] = _f("overlap", tier="expert")


class FreqSettings(_Settings):
    """Harmonic frequencies at the final structure.

    Attributes:
        check: Compute the Hessian at the final structure and report its
            frequencies. With the gradient method's own Hessians they decide: a
            minimization with an imaginary mode fails, and so does a
            transition-state search without exactly one. A cheaper method's are
            reported only.
        imaginary_cutoff: Frequencies below minus this (cm^-1) count as imaginary.
    """

    check: bool = _f(False)
    imaginary_cutoff: float = _c(10.0, unit="cm^-1", tier="expert")


class EscapeSettings(_Settings):
    """Stepping off saddle points a minimization stops on.

    Attributes:
        attempts: Check the converged structure's frequencies and, at a saddle
            point, step off it along the lowest mode and minimize again, up to this
            many times (0: no check). Still a saddle point after that, the run
            fails (with the gradient method's own Hessians).
        step: Largest atomic displacement (bohr) of such a step.
    """

    attempts: int = _f(0, ge=0)
    step: float = _f(0.189, unit="bohr", tier="expert", gt=0)


class IRCSettings(_Settings):
    """The path: direction, step and integrator.

    Attributes:
        direction: Which branches to follow from the transition state.
        step: Arc length of each step in mass-weighted coordinates
            (bohr amu^1/2). A step that passes the valley floor is retried at half
            the length.
        method: ``gs2``: Gonzalez and Schlegel's constrained step (1989), one
            gradient per step; ``lqa``: the local quadratic approximation (Page and
            McIver, 1988) on the current Hessian.
    """

    direction: Literal["both", "forward", "reverse"] = _f("both")
    step: float = _f(0.3, unit="bohr amu^1/2", gt=0)
    method: Literal["gs2", "lqa"] = _f("gs2", tier="expert")


class EndSettings(_Settings):
    """Minimizing the ends of an IRC.

    An IRC stops short of the minimum (a branch ends once its gradient is below
    ``convergence.g_tol``), and the end can turn out to be another species once
    minimized. With ``minimize``, both ends are minimized together, each starting
    from its branch's updated Hessian, and the minimized structures become the
    path's first and last frames.

    Attributes:
        minimize: Minimize both ends after the IRC.
        coordinate_system: Coordinates of those minimizations.
        max_iter: Their most iterations.
        g_tol: Their largest per-atom gradient norm (Hartree/bohr).
        g_rms_tol: Their root-mean-square gradient (Hartree/bohr).
        d_tol: Their largest atomic displacement of the last step (bohr).
        e_tol: Their energy change of the last step (Hartree).
    """

    minimize: bool = _f(False)
    coordinate_system: CoordinateSystem = _f("auto", tier="expert")
    max_iter: int = _c(500, tier="expert", ge=1)
    g_tol: float = _c(G_TOL, unit="Eh/bohr", tier="expert", gt=0)
    g_rms_tol: float | None = _c(None, unit="Eh/bohr", tier="expert", gt=0)
    d_tol: float | None = _c(None, unit="bohr", tier="expert", gt=0)
    e_tol: float | None = _c(None, unit="Eh", tier="expert", gt=0)

    @property
    def convergence(self) -> ConvergenceSettings:
        """These settings as the end minimizations' convergence."""
        return ConvergenceSettings(
            max_iter=self.max_iter,
            g_tol=self.g_tol,
            g_rms_tol=self.g_rms_tol,
            d_tol=self.d_tol,
            e_tol=self.e_tol,
        )


# ------------------------------------------------------------ entry points


class _Options(_Settings):
    """Settings of one calculation type."""

    def changed(self) -> dict[str, Any]:
        """The settings given explicitly, as ``{"group.key": value}``."""
        out: dict[str, Any] = {}
        for name in self.model_fields_set:
            value = getattr(self, name)
            if isinstance(value, _Settings):
                for key in value.model_fields_set:
                    out[f"{name}.{key}"] = getattr(value, key)
            else:
                out[name] = value
        return out


class OptimizationOptions(_Options):
    """A minimization (``calctype="optimization"``): RS-RFO.

    Attributes:
        coordinate_system: ``tric``: delocalized internal coordinates with each
            fragment's translation and rotation (Wang and Song, 2016); ``dlc``:
            delocalized internals with fragments joined by extra bonds;
            ``cartesian``: Cartesians without rigid motion; ``auto``: TRIC, or
            Cartesians when atoms are frozen or there are fewer than three.
        convergence: When it has converged.
        trust: The trust region.
        hessian: The Hessian: start, update, recomputation.
        constraints: Frozen atoms; distances, angles and dihedrals held.
        freq: Frequencies at the final structure.
        escape: Stepping off saddle points.
        threads: CPU threads the calculation asks the runner for, shared by its
            gradient and Hessian calculations (qcopt's own steps take one); the
            runner grants at most its own. Default: what the children's plans ask.
    """

    coordinate_system: CoordinateSystem = _f("auto")
    convergence: ConvergenceSettings = ConvergenceSettings()
    trust: MinimizationTrustSettings = MinimizationTrustSettings()
    hessian: HessianSettings = HessianSettings()
    constraints: ConstraintSettings = ConstraintSettings()
    freq: FreqSettings = FreqSettings()
    escape: EscapeSettings = EscapeSettings()
    threads: int | None = _f(None, concern="resource", ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> OptimizationOptions:
        _check_internal(self.coordinate_system, self.constraints.frozen)
        if self.escape.attempts and self.constraints.size:
            raise ValueError(
                "escape.attempts steps off saddle points of unconstrained "
                "minimizations only; remove it or the distance, angle and dihedral "
                "constraints"
            )
        return self


class TransitionStateOptions(_Options):
    """A transition-state search (``calctype="transition_state"``): RS-P-RFO.

    Attributes:
        coordinate_system: As for :class:`OptimizationOptions`.
        convergence: When it has converged.
        trust: The trust region.
        hessian: The Hessian: start, update, recomputation.
        constraints: Frozen atoms; distances, angles and dihedrals held.
        mode: Which eigenvector is climbed.
        freq: Frequencies at the final structure.
        threads: CPU threads the calculation asks the runner for, shared by its
            gradient and Hessian calculations (qcopt's own steps take one); the
            runner grants at most its own. Default: what the children's plans ask.
    """

    coordinate_system: CoordinateSystem = _f("auto")
    convergence: ConvergenceSettings = ConvergenceSettings()
    trust: TrustSettings = TrustSettings()
    hessian: TransitionStateHessianSettings = TransitionStateHessianSettings()
    constraints: ConstraintSettings = ConstraintSettings()
    mode: ModeSettings = ModeSettings()
    freq: FreqSettings = FreqSettings()
    threads: int | None = _f(None, concern="resource", ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> TransitionStateOptions:
        _check_internal(self.coordinate_system, self.constraints.frozen)
        if self.mode.guide is not None and self.coordinate_system != "cartesian":
            raise ValueError(
                "mode.guide is followed in Cartesian coordinates only: pass "
                "coordinate_system='cartesian' with it"
            )
        return self


class IRCOptions(_Options):
    """An intrinsic reaction coordinate (``calctype="irc"``).

    Attributes:
        irc: Direction, step and integrator.
        convergence: When a branch ends.
        hessian: The Hessian: start, update, recomputation.
        constraints: Frozen atoms.
        ends: Minimizing the ends.
        threads: CPU threads the calculation asks the runner for, shared by its
            gradient and Hessian calculations (qcopt's own steps take one); the
            runner grants at most its own. Default: what the children's plans ask.
    """

    irc: IRCSettings = IRCSettings()
    convergence: IRCConvergenceSettings = IRCConvergenceSettings()
    hessian: IRCHessianSettings = IRCHessianSettings()
    constraints: IRCConstraintSettings = IRCConstraintSettings()
    ends: EndSettings = EndSettings()
    threads: int | None = _f(None, concern="resource", ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> IRCOptions:
        _check_internal(self.ends.coordinate_system, self.constraints.frozen, "ends.")
        return self


def _check_internal(system: str, frozen: list[int], prefix: str = "") -> None:
    if system in ("tric", "dlc") and frozen:
        raise ValueError(
            f"{prefix}coordinate_system={system!r} cannot hold atoms fixed: use "
            "'cartesian' (or 'auto', which does) with constraints.frozen"
        )


Options = OptimizationOptions | TransitionStateOptions | IRCOptions

OPTIONS: dict[str, type[_Options]] = {
    "optimization": OptimizationOptions,
    "transition_state": TransitionStateOptions,
    "irc": IRCOptions,
}


# ------------------------------------------------- parsing, with suggestions

#: qcopt 0.2's keywords and where they went, by calculation type ("*": all).
RENAMED: dict[str, str] = {
    "maxiter": "convergence.max_iter",
    "max_steps": "convergence.max_iter",
    "convergence.gmax": "convergence.g_tol",
    "convergence.grms": "convergence.g_rms_tol",
    "convergence.dmax": "convergence.d_tol",
    "convergence.energy": "convergence.e_tol",
    "gmax": "convergence.g_tol",
    "flat_energy": "convergence.e_tol",
    "coordinates": "coordinate_system",
    "initial_hessian": "hessian.initial",
    "hessian_every": "hessian.recompute_every",
    "recompute_hessian_on_loss": "hessian.recompute_on_loss",
    "min_steps_between_hessians": "hessian.recompute_min_steps",
    "lost_curvature": "hessian.lost_curvature",
    "update": "hessian.update",
    "fd_step": "hessian.step",
    "trust_type": "trust.measure",
    "trust_min": "trust.min",
    "trust_max": "trust.max",
    "max_rejections": "trust.max_rejections",
    "frozen_atoms": "constraints.frozen",
    "constraint_tolerance": "constraints.tol",
    "max_correction": "constraints.max_correction",
    "follow": "mode.follow",
    "frequencies": "freq.check",
    "imaginary_cutoff": "freq.imaginary_cutoff",
    "escape_saddles": "escape.attempts",
    "escape_step": "escape.step",
    "direction": "irc.direction",
    "step_size": "irc.step",
    "method": "irc.method",
    "minimize_ends": "ends.minimize",
    "end_coordinates": "ends.coordinate_system",
    "end_maxiter": "ends.max_iter",
    "end_convergence": "ends.g_tol (and ends.g_rms_tol, ends.d_tol, ends.e_tol)",
}

#: Old scalar keywords that are groups now.
_NOW_GROUPS = {
    "trust": "trust.radius",
    "mode": "mode.guide",
}


def _paths(model: type[_Options]) -> list[str]:
    out = []
    for name, field in model.model_fields.items():
        sub = field.annotation
        if isinstance(sub, type) and issubclass(sub, _Settings):
            out += [f"{name}.{k}" for k in sub.model_fields]
        else:
            out.append(name)
    return out


def _hint(path: str, calctype: str) -> str:
    if path in RENAMED:
        return f"; renamed {RENAMED[path]} in qcopt 0.3"
    others = sorted(
        ct
        for ct, m in OPTIONS.items()
        if ct != calctype and (path in _paths(m) or path in m.model_fields)
    )
    if others:
        return f"; only for {' and '.join(others)} calculations"
    valid = _paths(OPTIONS[calctype])
    valid += [p.split(".")[0] for p in valid]
    close = difflib.get_close_matches(path, sorted(set(valid)), n=1, cutoff=0.6)
    return f"; did you mean {close[0]!r}?" if close else ""


def _given(options: _Options, path: str) -> bool:
    """Whether the setting ``group.key`` (or a top-level one) was given explicitly."""
    group, _, key = path.partition(".")
    if not key:
        return group in options.model_fields_set
    value = getattr(options, group)
    return group in options.model_fields_set and key in value.model_fields_set


def without_effect(
    options: Options, settings: dict[str, Any] | None = None
) -> list[tuple[str, str]]:
    """Settings that cannot affect this calculation, with the reason.

    A setting that only acts if a condition arises during the run (a threshold for a
    case that may not occur) is not among them. With ``settings`` (the resolved
    tree), every such key present is listed; otherwise only those given explicitly.
    """
    rules: list[tuple[str, bool, str]] = []
    if isinstance(options, IRCOptions):
        if not options.ends.minimize:
            rules += [
                (f"ends.{k}", True, "acts only with ends.minimize=True")
                for k in EndSettings.model_fields
                if k != "minimize"
            ]
    else:
        constrained = options.constraints.size > 0
        rules += [
            (
                "constraints.tol",
                not constrained,
                "acts only with distance, angle or dihedral constraints",
            ),
            (
                "constraints.max_correction",
                not constrained,
                "acts only with distance, angle or dihedral constraints",
            ),
        ]
        escape = (
            isinstance(options, OptimizationOptions) and options.escape.attempts > 0
        )
        if isinstance(options, OptimizationOptions):
            rules.append(
                ("escape.step", not escape, "acts only with escape.attempts > 0")
            )
        rules.append(
            (
                "freq.imaginary_cutoff",
                not (options.freq.check or escape),
                "acts only with freq.check"
                + (
                    " or escape.attempts > 0"
                    if not isinstance(options, TransitionStateOptions)
                    else ""
                ),
            )
        )
        if isinstance(options, TransitionStateOptions):
            rules += [
                (
                    "mode.follow",
                    options.mode.guide is None,
                    "acts only with mode.guide",
                ),
                (
                    "hessian.recompute_min_steps",
                    not options.hessian.recompute_on_loss,
                    "acts only with hessian.recompute_on_loss=True",
                ),
            ]
    out = []
    for path, idle, why in rules:
        if not idle:
            continue
        if settings is None:
            present = _given(options, path)
        else:
            group, _, key = path.partition(".")
            present = key in settings.get(group, {})
        if present:
            out.append((path, why))
    return out


def parse_options(calctype: str, keywords: dict[str, Any] | None) -> Options:
    """Validate ``keywords`` as the options of ``calctype``.

    Raises:
        InputError: A key that does not exist (with a suggestion, where a qcopt 0.2
            keyword went, or which calculation type has it), an invalid value,
            settings that cannot be honoured together, or a setting given that
            cannot affect this calculation.
    """
    model = OPTIONS.get(calctype)
    if model is None:
        raise InputError(f"qcopt does not run {calctype!r} calculations")
    kw = {
        k: v.model_dump(exclude_unset=True) if isinstance(v, BaseModel) else v
        for k, v in (keywords or {}).items()
    }
    for old, new in _NOW_GROUPS.items():
        if old in kw and not isinstance(kw[old], dict | BaseModel):
            raise InputError(
                f"{old} is a group in qcopt 0.3: pass {old}={{'{new.split('.')[1]}': "
                f"...}} ({new})"
            )
    try:
        options: Options = model(**kw)  # type: ignore[assignment]
    except ValidationError as exc:
        lines, codes = [], set()
        for err in exc.errors():
            path = ".".join(str(p) for p in err["loc"])
            if err["type"] == "extra_forbidden":
                hint = _hint(path, calctype)
                codes.add("not_applicable" if "; only for " in hint else "unknown_key")
                lines.append(f"unknown setting {path!r}{hint}")
            elif err["loc"]:
                lines.append(f"{path}: {err['msg']}")
            else:
                lines.append(str(err["msg"]).removeprefix("Value error, "))
        code = next((c for c in ("unknown_key", "not_applicable") if c in codes), None)
        raise InputError(
            f"invalid qcopt {calctype} settings: " + "; ".join(lines), code
        ) from None
    idle = without_effect(options)
    if idle:
        raise InputError(
            f"invalid qcopt {calctype} settings: "
            + "; ".join(
                f"{path} cannot affect this calculation: it {why}" for path, why in idle
            ),
            "no_effect",
        )
    return options


def default_options(calctype: str) -> dict[str, Any]:
    """Every setting of ``calctype`` with its default, as nested dicts."""
    model = OPTIONS.get(calctype)
    if model is None:
        raise InputError(f"qcopt does not run {calctype!r} calculations")
    return model().model_dump()


def _annotate(prop: dict[str, Any], path: str) -> None:
    """``x-applies-to`` (the calculation types with this setting) and ``x-shared``."""
    prop["x-applies-to"] = {
        "calctypes": [
            ct
            for ct, model in OPTIONS.items()
            if path in _paths(model) or path in model.model_fields
        ]
    }
    if path in SHARED:
        prop["x-shared"] = True


def options_schema(calctype: str) -> dict[str, Any]:
    """The JSON Schema of ``calctype``'s settings.

    As qcdata's conventions ask, every setting carries ``x-tier`` (common, expert),
    ``x-concern`` (model, convergence, algorithm, resource), ``x-unit`` where it has
    one, ``x-applies-to`` (``{"calctypes": [...]}``) and ``x-shared`` for the settings
    AFH's programs share. A group carries the concern its settings share, else
    ``algorithm``.
    """
    model = OPTIONS.get(calctype)
    if model is None:
        raise InputError(f"qcopt does not run {calctype!r} calculations")
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    for name, prop in schema["properties"].items():
        _annotate(prop, name)
        ref = prop.get("$ref")
        if ref is None:
            continue
        group = defs[ref.rsplit("/", 1)[-1]]
        concerns = {sub["x-concern"] for sub in group["properties"].values()}
        prop["x-tier"] = "common"
        prop["x-concern"] = concerns.pop() if len(concerns) == 1 else "algorithm"
        for key, sub in group["properties"].items():
            _annotate(sub, f"{name}.{key}")
    return schema
