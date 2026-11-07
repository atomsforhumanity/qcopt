"""The optimizers as generators: minimization, transition-state search, IRC.

Each yields lists of :class:`~qcopt.driver.Ask` and returns a :class:`RunResult`.
Everything is in Cartesian coordinates (bohr, Hartree); see :mod:`qcopt.space` for
how rigid motion and frozen atoms are kept out of the steps.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qcref import periodic_table as pt

from . import hessians
from .constraints import ConstraintSet, correction, free_gradient, project
from .coordinates import CartesianCoordinates, InternalCoordinates, make_coordinates
from .driver import Algorithm, Answer, Ask
from .geometry import Array, masses, max_atomic_norm, rms
from .internal import CoordinateError
from .keywords import IRCKeywords, OptimizationKeywords
from .space import LOCKED_CURVATURE, CartesianSpace
from .steps import rfo_step, select_mode, update_trust
from .vibrations import Frequencies, frequencies


@dataclass
class RunResult:
    """What an algorithm produced.

    Attributes:
        frames: The answers kept as frames, in order.
        converged: Whether it met its convergence criteria.
        error: Why it stopped, when it failed.
        ts_index: For an IRC, the transition state's position in ``frames``.
        info: Counts and diagnostics, reported in the result's ``extras``.
        frequencies: Harmonic analysis of the final structure, if computed.
    """

    frames: list[Answer]
    converged: bool
    error: str | None = None
    ts_index: int | None = None
    info: dict[str, Any] = field(default_factory=dict)
    frequencies: Frequencies | None = None


class _Counter:
    """Counts the gradients and Hessians an algorithm received."""

    def __init__(self) -> None:
        self.gradients = 0
        self.hessians = 0

    def __call__(self, answers: list[Answer]) -> list[Answer]:
        for a in answers:  # the driver records what each answer cost
            self.gradients += a.n_gradients
            self.hessians += a.n_hessians
        return answers


@dataclass
class _Progress:
    """What an algorithm has done so far, for its result if it raises."""

    frames: Callable[[], list[Answer]] = list
    info: dict[str, Any] = field(default_factory=dict)
    ts_index: Callable[[], int | None] = lambda: None
    count: _Counter | None = None


def _guarded(body: Callable[[_Progress], Algorithm]) -> Algorithm:
    """Run an algorithm; an exception inside it becomes a failed result with its frames."""
    progress = _Progress()
    try:
        result = yield from body(progress)
        if progress.count is not None:  # early returns report their counts too
            result.info.setdefault("n_gradients", progress.count.gradients)
            result.info.setdefault("n_hessians", progress.count.hessians)
        return result
    except Exception as exc:  # noqa: BLE001 - reported as the result
        if progress.count is not None:
            progress.info.update(
                n_gradients=progress.count.gradients,
                n_hessians=progress.count.hessians,
            )
        return RunResult(
            frames=progress.frames(),
            converged=False,
            error="".join(traceback.format_exception(exc)),
            ts_index=progress.ts_index(),
            info=progress.info,
        )


def _hessian_source(
    kw: OptimizationKeywords | IRCKeywords, has_program: bool, ts: bool
) -> str:
    if kw.initial_hessian != "auto":
        return kw.initial_hessian
    if has_program:
        return "exact"
    return "finite_difference" if ts else "model"


def _converged(
    kw: OptimizationKeywords, gp: Array, step: Array | None, de: float | None
) -> bool:
    conv = kw.convergence
    gmax = max_atomic_norm(gp)
    if not np.isfinite(gmax) or gmax > conv.gmax:
        return False
    if conv.grms is not None and rms(gp) > conv.grms:
        return False
    if conv.dmax is not None and (step is None or max_atomic_norm(step) > conv.dmax):
        return False
    if conv.energy is not None and (de is None or abs(de) > conv.energy):
        return False
    return True


def optimize(
    positions: Array,
    symbols: Sequence[str],
    numbers: Sequence[int],
    kw: OptimizationKeywords,
    *,
    transition_state: bool,
    has_hessian_program: bool,
    start: tuple[Answer, Array] | None = None,
    separate_hessian: bool = False,
) -> Algorithm:
    """Minimize (RS-RFO) or find a first-order saddle point (RS-P-RFO).

    A transition-state search fails when it ends where the Hessian has no direction
    of negative curvature (checked with an exact Hessian when the updated one shows
    none), or several in an exact Hessian; with ``frequencies``, unless exactly one
    frequency is imaginary. A minimization fails when its frequencies show a saddle
    point, or when it ends higher than it started. Frequencies from a separate
    (lower-level) Hessian program are reported but decide nothing. An exception
    inside the algorithm ends it as a failure that keeps its frames.

    Args:
        positions: Starting geometry (bohr).
        symbols: Element symbols.
        numbers: Atomic numbers (for the model Hessian).
        kw: Keywords.
        transition_state: Climb one mode instead of minimizing.
        has_hessian_program: Whether exact Hessians come from a program (else
            finite differences of gradients).
        start: The calculation already done at ``positions`` and a Cartesian
            Hessian to start from (such as an IRC branch's updated one); then
            nothing is computed before the first step.
        separate_hessian: Hessians come from another calculation than the gradients
            (a cheaper method): they guide the steps but are not the method's own.
    """
    return (
        yield from _guarded(
            lambda progress: _optimize(
                positions,
                symbols,
                numbers,
                kw,
                transition_state=transition_state,
                has_hessian_program=has_hessian_program,
                start=start,
                separate_hessian=separate_hessian,
                progress=progress,
            )
        )
    )


def _optimize(
    positions: Array,
    symbols: Sequence[str],
    numbers: Sequence[int],
    kw: OptimizationKeywords,
    *,
    transition_state: bool,
    has_hessian_program: bool,
    start: tuple[Answer, Array] | None,
    separate_hessian: bool,
    progress: _Progress,
) -> Algorithm:
    count = progress.count = _Counter()
    n = len(symbols)
    x = np.asarray(positions, dtype=float).reshape(n, 3)
    cart = CartesianSpace(n, frozen=kw.frozen_atoms)
    coords = make_coordinates(
        kw.coordinates, symbols, x, kw.frozen_atoms, kw.trust_type == "atomic"
    )
    cons = (
        ConstraintSet(kw.constraints, x, kw.frozen_atoms)
        if kw.constraints is not None and kw.constraints.size
        else None
    )

    def _negative(hh: Array) -> Array:
        """The negative curvatures among the free directions, lowest first."""
        vals_ = _decompose(hh)[0]
        n_free_ = len(vals_) - coords.n_locked - (cons.size if cons else 0)
        free = vals_[:n_free_]
        return free[free < -kw.lost_curvature]

    def _n_negative(hh: Array) -> int:
        return len(_negative(hh))

    def _free(gradient: Any, xx: Array) -> Array:
        """Cartesian gradient without constraint forces or rigid motion."""
        gx = np.asarray(gradient, dtype=float).ravel()
        if cons is not None:
            gx = free_gradient(cons.jacobian(xx), gx)
        cart.at(xx)
        return cart.project(gx)

    def _satisfied(xx: Array) -> bool:
        if cons is None:
            return True
        return bool(np.abs(cons.violation(xx)).max() < kw.constraint_tolerance)

    def _decompose(hh: Array) -> tuple[Array, Array, Array, Array | None]:
        """Eigenpairs of the Hessian with constrained directions locked."""
        if cons is None:
            v, w = coords.eigh(hh)
            return v, w, g, None
        jw_ = coords.jacobian_to_space(x, cons.jacobian(x))
        pf, pc = project(jw_)
        v, w = coords.eigh(pf @ hh @ pf + LOCKED_CURVATURE * pc)
        return v, w, pf @ g, jw_

    def _rebuild(hh: Array) -> Array:
        """New internal coordinates at ``x``; Cartesians when they cannot be built."""
        nonlocal coords
        info["coordinate_rebuilds"] += 1
        assert isinstance(coords, InternalCoordinates)
        try:
            return coords.rebuild(x, hh)
        except CoordinateError:
            return _to_cartesian(hh, f"{info['coordinates']} -> cartesian")

    def _to_cartesian(hh: Array, name: str) -> Array:
        """Switch to Cartesian coordinates, keeping the Hessian."""
        nonlocal coords
        assert isinstance(coords, InternalCoordinates)
        hx = coords.ic.to_cartesian_hessian(x, hh)
        coords = CartesianCoordinates(n, kw.frozen_atoms, kw.trust_type == "atomic")
        coords.at(x)
        info["coordinates"] = name
        return hx + 1e-4 * np.eye(3 * n)

    frames: list[Answer] = []
    log: list[dict[str, Any]] = []
    info: dict[str, Any] = {
        "rejected_steps": 0,
        "hessian_recomputes": 0,
        "coordinate_rebuilds": 0,
        "backtransform_failures": 0,
        "coordinates": coords.name,
        "log": log,
    }
    progress.frames = lambda: list(frames)
    progress.info = info
    source = _hessian_source(kw, has_hessian_program, transition_state)
    exact_available = source in ("exact", "finite_difference")
    # Exact Hessians later on: the Hessian program's, or finite differences when the
    # run started from them; ``hessian_every`` asks for them in any case.
    can_recompute = exact_available or has_hessian_program
    if start is not None:
        a = start[0]
    else:
        (a,) = count((yield [Ask("hessian" if exact_available else "gradient", x)]))
    if not a.success:
        return RunResult(
            frames=[a], converged=False, error=f"first calculation failed: {a.error}"
        )
    frames.append(a)
    coords.at(x)
    g_raw = np.asarray(a.gradient, dtype=float).ravel()
    if start is not None:
        h = coords.hessian(x, np.asarray(start[1], dtype=float), g_raw)
    elif exact_available:
        h = coords.hessian(x, np.asarray(a.hessian, dtype=float), g_raw)
    elif source == "model":
        h = coords.hessian(x, hessians.lindh(numbers, x) + 1e-4 * np.eye(3 * n))
    else:
        h = coords.hessian(x, 0.5 * np.eye(3 * n))
    update_name = (
        kw.update
        if kw.update != "auto"
        else ("ts_bfgs" if transition_state else "bfgs")
    )
    update = hessians.UPDATES[update_name]

    mode_ref = None if kw.mode is None else np.asarray(kw.mode, dtype=float).ravel()
    trust = min(max(kw.trust, kw.trust_min), kw.trust_max)
    own = not separate_hessian  # exact Hessians are the gradient method's own
    rejections = 0
    failures = 0
    converged = False
    error: str | None = None
    last_step: Array | None = None
    last_de: float | None = None
    steps_since_hessian = 0
    h_exact = start is None and exact_available and own  # the method's own, at x
    g = coords.gradient(x, g_raw)
    freqs = None
    escapes = 0
    iterations = 0
    while True:
        cart.at(x)
        if _converged(kw, _free(a.gradient, x), last_step, last_de) and _satisfied(x):
            if transition_state:
                converged = True
                if kw.frequencies and own:
                    break  # the final frequencies decide (below)
                # An updated Hessian often shows a weak spurious second negative
                # direction (11 of 100 g-xTB searches, at most 0.12 of the first, all
                # positive in the exact Hessian), so only an exact Hessian fails a
                # search for having several. None, or a second one at least a quarter
                # as strong as the first, is checked with the exact Hessian.
                neg = _negative(h)
                n_neg = len(neg)
                doubtful = n_neg == 0 or (
                    n_neg > 1 and neg[1] < SECOND_NEGATIVE_RATIO * neg[0]
                )
                if doubtful and can_recompute and not h_exact:
                    (ha,) = count((yield [Ask("hessian", x)]))
                    if not ha.success:
                        error = (
                            f"the Hessian to check the saddle point failed: {ha.error}"
                        )
                        break
                    if ha.success:
                        h = coords.hessian(
                            x, np.asarray(ha.hessian), np.asarray(ha.gradient).ravel()
                        )
                        frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
                        info["hessian_recomputes"] += 1
                        steps_since_hessian = 0
                        h_exact = own
                        n_neg = _n_negative(h)
                if n_neg == 0 or (h_exact and n_neg > 1):
                    error = (
                        f"stopped at a stationary point with {n_neg} directions of "
                        "negative curvature, not a first-order saddle point"
                    )
                break
            if kw.escape_saddles == 0 or cons is not None:
                converged = True
                if h_exact and cons is None and _n_negative(h):
                    error = (
                        "stopped at a saddle point (negative curvature in the exact "
                        "Hessian); escape_saddles steps off saddle points"
                    )
                break
            # A minimization can stop on a saddle point, where the forces vanish too.
            (ha,) = count((yield [Ask("hessian", x)]))
            if not ha.success:
                converged = True
                error = f"the Hessian to check for a saddle point failed: {ha.error}"
                break
            frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
            freqs = frequencies(
                np.asarray(ha.hessian), symbols, x, frozen=kw.frozen_atoms
            )
            n_imag = freqs.n_imaginary(kw.imaginary_cutoff)
            if n_imag == 0 or escapes >= kw.escape_saddles:
                converged = True
                if n_imag and own:
                    error = (
                        f"still a saddle point ({n_imag} imaginary modes) after "
                        f"{escapes} escapes"
                    )
                break
            escapes += 1
            iterations += 1
            info["saddle_escapes"] = escapes
            mode_disp = np.asarray(freqs.modes[0], dtype=float)
            x = x + mode_disp / max_atomic_norm(mode_disp) * kw.escape_step
            (a,) = count((yield [Ask("gradient", x)]))
            if not a.success:
                return RunResult(
                    frames=frames + [a],
                    converged=False,
                    error=f"gradient failed: {a.error}",
                    info=info,
                )
            frames.append(a)
            coords.at(x)
            h = coords.hessian(x, np.asarray(ha.hessian))
            g = coords.gradient(x, np.asarray(a.gradient, dtype=float))
            trust, last_step, last_de, freqs = kw.trust, None, None, None
            continue
        if iterations >= kw.maxiter:
            break
        iterations += 1
        if coords.needs_rebuild(x) or coords.residual > max(
            0.2 * kw.convergence.gmax,
            1e-3 * max_atomic_norm(np.asarray(a.gradient)),
        ):
            h = _rebuild(h)
            g = coords.gradient(x, np.asarray(a.gradient, dtype=float))
        vals, vecs, g_eff, jw = _decompose(h)
        n_free = len(vals) - coords.n_locked - (cons.size if cons else 0)
        if (
            transition_state
            and can_recompute
            and kw.recompute_hessian_on_loss
            and steps_since_hessian >= kw.min_steps_between_hessians
            and vals[0] > -kw.lost_curvature
        ):
            (ha,) = count((yield [Ask("hessian", x)]))
            if ha.success:
                h = coords.hessian(
                    x, np.asarray(ha.hessian), np.asarray(ha.gradient).ravel()
                )
                frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
                info["hessian_recomputes"] += 1
                steps_since_hessian = 0
                h_exact = own
                vals, vecs, g_eff, jw = _decompose(h)
        mode = None
        if transition_state:
            if mode_ref is None or not isinstance(coords, CartesianCoordinates):
                mode = 0
            else:
                mode = select_mode(vecs, vals, cart.project(mode_ref), n_free)
                mode_ref = vecs[:, mode] if kw.follow == "overlap" else None
        step = rfo_step(
            vals,
            vecs,
            g_eff,
            trust,
            mode=mode,
            atomic=coords.name == "cartesian" and kw.trust_type == "atomic",
        )
        dw = step.dx
        if jw is not None and cons is not None:
            dw = dw + correction(jw, cons.violation(x), kw.max_correction)
        x_new, d, ok = coords.step(x, dw)
        if not ok:
            info["backtransform_failures"] += 1
            if isinstance(coords, InternalCoordinates):
                h = _rebuild(h)
                g = coords.gradient(x, np.asarray(a.gradient, dtype=float))
            trust = max(kw.trust_min, 0.5 * trust)
            if info["backtransform_failures"] > 20 and isinstance(
                coords, InternalCoordinates
            ):
                h = _to_cartesian(h, "cartesian (fallback)")
                g = coords.gradient(x, np.asarray(a.gradient, dtype=float))
            continue
        predicted = float(g @ d + 0.5 * d @ h @ d)
        (b,) = count((yield [Ask("gradient", x_new)]))
        if not b.success:
            failures += 1
            if failures > 3 or trust <= kw.trust_min:
                return RunResult(
                    frames=frames + [b],
                    converged=False,
                    error=f"gradient failed: {b.error}",
                    info=info,
                )
            trust = max(kw.trust_min, 0.25 * min(trust, coords.size(d)))
            continue
        failures = 0
        de = b.energy - a.energy
        noise = abs(de) < 1e-8 and abs(predicted) < 1e-8  # below SCF noise
        ratio = 1.0 if noise or abs(predicted) < 1e-14 else de / predicted
        size = coords.size(d)
        log.append(
            {
                "step": size,
                "trust": trust,
                "ratio": float(ratio),
                "lowest": float(vals[0]),
                "mode": None if mode is None else int(mode),
                "de": float(de),
                "gmax": max_atomic_norm(cart.project(np.asarray(a.gradient).ravel())),
            }
        )
        trust = update_trust(
            trust,
            ratio,
            size,
            minimum=kw.trust_min,
            maximum=kw.trust_max,
            ts=transition_state,
        )
        coords.at(x_new)
        g_new = coords.gradient(x_new, np.asarray(b.gradient, dtype=float))
        y = g_new - g
        if update_name == "bfgs" and kw.update == "auto" and vals[0] < -1e-10:
            h = hessians.bofill(h, d, y)  # BFGS cannot fix negative curvature
        else:
            h = update(h, d, y)
        reject = (
            not transition_state
            and de > 1e-8
            and ratio < 0.0
            and rejections < kw.max_rejections
            and size > kw.trust_min
        )
        if reject:
            rejections += 1
            info["rejected_steps"] += 1
            trust = max(kw.trust_min, min(trust, 0.5 * size))
            coords.at(x)
            continue
        rejections = 0
        last_step = (x_new - x).ravel()
        x, a, g = x_new, b, g_new
        frames.append(b)
        last_de = de
        steps_since_hessian += 1
        h_exact = False
        if kw.hessian_every and steps_since_hessian >= kw.hessian_every:
            (ha,) = count((yield [Ask("hessian", x)]))
            if ha.success:
                h = coords.hessian(
                    x, np.asarray(ha.hessian), np.asarray(ha.gradient).ravel()
                )
                frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
                info["hessian_recomputes"] += 1
                steps_since_hessian = 0
                h_exact = own
    if freqs is not None:
        info["n_imaginary"] = freqs.n_imaginary(kw.imaginary_cutoff)
    elif kw.frequencies:
        (ha,) = count((yield [Ask("hessian", x)]))
        if ha.success:
            frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
            freqs = frequencies(
                np.asarray(ha.hessian), symbols, x, frozen=kw.frozen_atoms
            )
            info["n_imaginary"] = n_imag = freqs.n_imaginary(kw.imaginary_cutoff)
            if converged and own and transition_state and n_imag != 1:
                error = error or (
                    f"stopped at a stationary point with {n_imag} "
                    "imaginary frequencies, not a first-order saddle point"
                )
            elif converged and own and not transition_state and n_imag and cons is None:
                error = error or (
                    f"stopped at a saddle point ({n_imag} imaginary frequencies); "
                    "escape_saddles steps off saddle points"
                )
        else:
            error = error or f"the Hessian for the frequencies failed: {ha.error}"
    cart.at(x)
    info.update(
        n_gradients=count.gradients,
        n_hessians=count.hessians,
        n_steps=len(frames) - 1,
        final_trust=trust,
        gmax=max_atomic_norm(_free(a.gradient, x)),
        hessian=source,
        update=update_name,
    )
    rose = float(frames[-1].energy) - float(frames[0].energy)
    if converged and not transition_state and cons is None and rose > 1e-6:
        # a minimization never ends above its start, unless the program changed
        # state (an SCF that landed on another solution, for example)
        error = error or f"ended {rose:.2e} Eh higher than it started"
    if freqs is not None and separate_hessian:
        info["frequencies_from"] = "hessian subprogram"
    if not converged:
        error = f"not converged in {kw.maxiter} steps"
    return RunResult(
        frames=frames, converged=converged, error=error, info=info, frequencies=freqs
    )


def _with_hessian(answer: Answer, hessian: Array) -> Answer:
    return Answer(
        positions=answer.positions,
        energy=answer.energy,
        gradient=answer.gradient,
        hessian=np.asarray(hessian, dtype=float),
        success=answer.success,
        error=answer.error,
        output=answer.output,
        program=answer.program,
        program_version=answer.program_version,
        n_gradients=answer.n_gradients,
        n_hessians=answer.n_hessians,
        hessian_answer=answer.hessian_answer,
    )


# A transition-state search whose updated Hessian ends with a second negative curvature
# at least this fraction of the first checks it with the exact Hessian.
SECOND_NEGATIVE_RATIO = 0.25

# -------------------------------------------------------------------------- IRC

# A branch whose steps keep passing the valley floor ("energy rose"), or whose steps
# have shrunk to nothing ("stalled", 3 in a row below 1e-3 of the step size), ends
# there when its largest atomic gradient is below this multiple of ``gmax`` (4 x 5e-4
# Hartree/bohr is about 0.1 eV/Angstrom, Sella's IRC criterion).
ENERGY_ROSE_GMAX = 4.0


def _gs2_step(vals: Array, gt: Array, ds: float) -> Array:
    """Gonzalez-Schlegel step of length ``ds`` (eigenbasis components), on the model.

    Gonzalez and Schlegel, J. Chem. Phys. 90, 2154 (1989): from the pivot half a step
    down the gradient, the next point lies on the sphere of radius ds/2 around the
    pivot where the gradient points at the pivot, i.e. the lowest point of the
    sphere. Found on the quadratic model (Hessian ``vals``, gradient ``gt`` at the
    current point) as a trust-region problem: p = -(H - lambda)^-1 g* with |p| = ds/2
    and lambda below the lowest eigenvalue, g* the model gradient at the pivot.
    """
    gnorm = float(np.linalg.norm(gt))
    if gnorm == 0.0:
        return np.zeros_like(gt)
    half = 0.5 * ds
    to_pivot = -half * gt / gnorm
    g_pivot = gt + vals * to_pivot
    hmin = float(np.min(vals))

    def size(mu: float) -> float:
        return float(np.linalg.norm(g_pivot / ((vals - hmin) + mu)))

    lo, hi = 1e-14, max(1.0, abs(hmin)) + float(np.linalg.norm(g_pivot)) / half + 1.0
    if size(lo) <= half:
        mu = lo  # hard case: the lowest mode carries no gradient at the pivot
    else:
        for _ in range(200):
            mu = float(np.sqrt(lo * hi))
            if size(mu) > half:
                lo = mu
            else:
                hi = mu
            if hi / lo < 1.0 + 1e-12:
                break
        mu = hi
    # In the hard case (the lowest mode carries no gradient at the pivot) p is
    # shorter than ds/2: the step is shortened rather than filled out sideways along
    # a mode the path does not follow.
    p = -g_pivot / ((vals - hmin) + mu)
    return to_pivot + p


def _lqa_step(vals: Array, gt: Array, ds: float) -> Array:
    """Local quadratic approximation step of arc length ``ds`` (eigenbasis components).

    Page and McIver, J. Chem. Phys. 88, 922 (1988): the steepest-descent path on the
    quadratic surface, x_i(t) = g_i (exp(-l_i t) - 1) / l_i, followed until its arc
    length is ``ds``.
    """
    active = np.abs(gt) > 0
    lam, g = vals[active], gt[active]

    def speed(t: Array) -> Array:
        e = np.exp(np.clip(-np.outer(t, lam), -150.0, 150.0))
        return np.sqrt(((g * e) ** 2).sum(axis=1))

    gnorm = float(np.linalg.norm(g))
    if gnorm == 0.0:
        return np.zeros_like(gt)
    # The speed varies on the scales 1/|lambda_i|, which can be far apart, so the arc
    # length is integrated on a geometric grid (fine near 0, where it changes fastest
    # when a curvature is large) up to a time long enough to cover ds.
    t_end = ds / gnorm
    t_min = 1e-6 * min(t_end, 1.0 / max(float(np.abs(lam).max()), 1e-300))
    previous = 0.0
    for _ in range(200):
        ts = np.concatenate([[0.0], np.geomspace(t_min, max(t_end, 2 * t_min), 4001)])
        v = speed(ts)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(ts))])
        if s[-1] >= ds:
            t = float(np.interp(ds, s, ts))
            break
        if s[-1] - previous <= 1e-12 * s[-1]:  # the path ends (at a minimum) before ds
            t = t_end
            break
        previous = float(s[-1])
        t_end *= 2.0
    else:
        t = t_end
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        small = np.abs(lam * t) < 1e-8
        comp = np.where(
            small,
            -g * t,
            g * np.expm1(np.clip(-lam * t, -150.0, 150.0)) / np.where(small, 1.0, lam),
        )
    out = np.zeros_like(gt)
    out[active] = comp
    return out


def lockstep(
    algorithms: dict[str, Algorithm],
) -> Generator[list[Ask], list[Answer], dict[str, Any]]:
    """Run several algorithms as one: each round's asks of all of them together.

    Returns each algorithm's result by name (an exception raised inside one is its
    result).
    """
    results: dict[str, Any] = {}
    pending: dict[str, list[Ask]] = {}
    for name, alg in algorithms.items():
        try:
            pending[name] = next(alg)
        except StopIteration as stop:
            results[name] = stop.value
        except Exception as exc:  # noqa: BLE001 - reported as that algorithm's result
            results[name] = exc
    while pending:
        order = list(pending)
        asks = [a for name in order for a in pending[name]]
        answers = yield asks
        nxt: dict[str, list[Ask]] = {}
        k = 0
        for name in order:
            mine = answers[k : k + len(pending[name])]
            k += len(pending[name])
            try:
                nxt[name] = algorithms[name].send(mine)
            except StopIteration as stop:
                results[name] = stop.value
            except Exception as exc:  # noqa: BLE001 - reported as that algorithm's result
                results[name] = exc
        pending = nxt
    return results


@dataclass
class _Branch:
    """One direction of an IRC."""

    name: str
    x: Array
    prev: Answer
    h: Array
    next: Array
    frames: list[Answer] = field(default_factory=list)
    done: bool = False
    reason: str | None = None
    since_hessian: int = 0
    steps: int = 0
    ds: float = 0.0
    need_hessian: bool = False
    end_checks: int = 0
    asked_hessian: bool = False
    skip_hessian: bool = False
    hessian_failures: int = 0
    last_de: float = float("inf")
    gmax: float = float("inf")
    sign: float = 1.0
    tiny_steps: int = 0


def irc(
    positions: Array,
    symbols: Sequence[str],
    kw: IRCKeywords,
    *,
    has_hessian_program: bool,
    separate_hessian: bool = False,
) -> Algorithm:
    """Follow the IRC down both sides of a transition state.

    Mass-weighted Cartesian coordinates with rigid motion projected out, so the path
    stays in the input's frame. The first step goes along the Hessian's negative
    mode; later steps are Gonzalez-Schlegel steps (or the local quadratic
    approximation) on the current Hessian (exact at the transition state, then
    updated). Both branches advance together. The IRC has converged when every branch
    reached its end, or, with ``minimize_ends``, when every end minimization
    converged; an exception inside it ends it as a failure that keeps its frames.
    """
    return (
        yield from _guarded(
            lambda progress: _irc(
                positions,
                symbols,
                kw,
                has_hessian_program=has_hessian_program,
                separate_hessian=separate_hessian,
                progress=progress,
            )
        )
    )


def _irc(
    positions: Array,
    symbols: Sequence[str],
    kw: IRCKeywords,
    *,
    has_hessian_program: bool,
    separate_hessian: bool,
    progress: _Progress,
) -> Algorithm:
    count = progress.count = _Counter()
    n = len(symbols)
    m = masses(symbols)
    m3 = np.repeat(m, 3)
    x0 = np.asarray(positions, dtype=float).reshape(n, 3)
    space = CartesianSpace(n, frozen=kw.frozen_atoms, masses=m)
    (ts,) = count((yield [Ask("hessian", x0)]))
    progress.frames = lambda: [ts]
    if not ts.success:
        return RunResult(
            frames=[ts],
            converged=False,
            error=f"Hessian at the transition state failed: {ts.error}",
            ts_index=0,
        )
    h0 = np.asarray(ts.hessian, dtype=float)
    space.at(x0)
    vals, vecs = space.eigh(space.hessian_to_space(h0))
    if vals[0] >= 0:
        return RunResult(
            frames=[ts],
            converged=False,
            error="the starting structure has no negative curvature",
            ts_index=0,
        )
    v = vecs[:, 0]
    v = v * np.sign(v[np.argmax(np.abs(v))])
    exact_available = _hessian_source(kw, has_hessian_program, True) in (
        "exact",
        "finite_difference",
    )
    update = hessians.UPDATES[kw.update]

    signs = {"forward": [1.0], "reverse": [-1.0], "both": [1.0, -1.0]}[kw.direction]
    branches = [
        _Branch(
            name="forward" if sign > 0 else "reverse",
            x=x0,
            prev=ts,
            h=h0.copy(),
            next=x0 + space.to_cartesian(sign * kw.step_size * v).reshape(n, 3),
            ds=kw.step_size,
            sign=sign,
        )
        for sign in signs
    ]

    def assemble() -> tuple[list[Answer], int]:
        """The path so far, reverse branch first, and the transition state's index."""
        by_name = {b.name: b for b in branches}
        rev = list(reversed(by_name["reverse"].frames)) if "reverse" in by_name else []
        fwd = by_name["forward"].frames if "forward" in by_name else []
        return rev + [ts] + fwd, len(rev)

    progress.frames = lambda: assemble()[0]
    progress.ts_index = lambda: assemble()[1]

    def next_point(b: _Branch) -> Array:
        """The next IRC point, ``b.ds`` along the path from the branch's last one."""
        space.at(b.x)
        g_mw = space.project(
            space.gradient_to_space(np.asarray(b.prev.gradient).ravel())
        )
        hv, hvec = space.eigh(space.hessian_to_space(b.h))
        n_free = len(hv) - space.n_locked
        gt = hvec.T @ g_mw
        gt[n_free:] = 0.0
        if kw.method == "gs2":
            sub = np.arange(n_free)  # locked directions carry no gradient and no step
            comp = np.zeros_like(gt)
            newton = -gt[sub] / hv[sub] if np.min(hv[:n_free]) > 0 else None
            if newton is not None and float(np.linalg.norm(newton)) <= b.ds:
                comp[sub] = newton  # the valley's minimum lies within this step
            else:
                comp[sub] = _gs2_step(hv[sub], gt[sub], b.ds)
            step_mw = hvec @ comp
        else:
            hv = np.where(hv >= LOCKED_CURVATURE * 0.5, 0.0, hv)
            step_mw = hvec @ _lqa_step(hv, gt, b.ds)
        return b.x + space.to_cartesian(space.project(step_mw)).reshape(n, 3)

    def end_test(b: _Branch, gradient: Array, de_last: float, exact: bool) -> bool:
        """Whether the branch ends (or waits for its exact Hessian) at its last point."""
        space.at(b.x)
        hv, _ = space.eigh(space.hessian_to_space(b.h))
        n_free = len(hv) - space.n_locked
        g_cart = np.asarray(gradient, dtype=float).ravel() * space.free
        b.gmax = max_atomic_norm(g_cart)
        if b.gmax < kw.gmax:
            if np.min(hv[:n_free]) > -1e-5:
                b.done, b.reason = True, "converged"
                return True
            if exact_available and b.end_checks < 2 and not exact:
                # Small gradient but the updated Hessian has negative curvature: a
                # flat shoulder of the path, or a minimum the updates blurred.
                # The exact Hessian here decides.
                b.end_checks += 1
                b.need_hessian = True
                b.next = b.x
                return True
            if (not exact_available or b.end_checks >= 2) and de_last < kw.flat_energy:
                b.done, b.reason = True, "converged (flat)"
                return True
        if b.steps >= kw.max_steps:
            b.done, b.reason = True, "max_steps"
            return True
        return False

    while True:
        active = [b for b in branches if not b.done]
        if not active:
            break
        asks = []
        for b in active:
            b.asked_hessian = b.need_hessian or bool(
                exact_available
                and kw.hessian_every
                and b.since_hessian + 1 >= kw.hessian_every
                and not b.skip_hessian
            )
            asks.append(Ask("hessian" if b.asked_hessian else "gradient", b.next))
        answers = count((yield asks))
        for b, ans in zip(active, answers, strict=True):
            if b.need_hessian:
                # The last point again, for its exact Hessian: no step was taken.
                b.need_hessian = False
                exact = ans.success and ans.hessian is not None
                if exact:
                    b.h = np.asarray(ans.hessian, dtype=float)
                    b.since_hessian = 0
                    b.frames[-1] = b.prev = _with_hessian(b.frames[-1], b.h)
                else:
                    b.end_checks = 2  # no exact Hessian here; the flat rule may end it
                    b.hessian_failures += 1
                    b.skip_hessian = True
                if not end_test(b, np.asarray(b.prev.gradient), b.last_de, exact):
                    b.next = next_point(b)
                continue
            if not ans.success:
                if b.asked_hessian and not b.skip_hessian:
                    # An optional Hessian failed: ask for the point's gradient alone,
                    # and continue the branch on updated Hessians.
                    b.hessian_failures += 1
                    b.skip_hessian = True
                    b.end_checks = 2
                    continue
                b.steps += 1
                b.done, b.reason = True, f"gradient failed: {ans.error}"
                continue
            b.steps += 1
            g_new = np.asarray(ans.gradient, dtype=float).ravel()
            dx = (np.asarray(ans.positions) - b.x).ravel()
            space.at(b.x)
            small = max_atomic_norm(g_new * space.free) < kw.gmax
            # Past the valley floor: the energy rose, or the new gradient points
            # along the step (a step can jump a shallow basin and still land lower)
            # while it is not yet small enough to end the branch.
            overshot = ans.energy > b.prev.energy + 1e-10 or (
                float(g_new @ dx) > 0.0 and not small
            )
            if overshot and not b.frames:
                # The first step went uphill or past a valley: shorter along the mode.
                b.ds *= 0.5
                if b.ds < kw.step_size / 16:
                    b.done, b.reason = True, "no descent from the transition state"
                else:
                    step = space.to_cartesian(b.sign * b.ds * v).reshape(n, 3)
                    b.next = x0 + step
                continue
            if overshot:
                # Overshot the valley floor: learn the curvature from the point
                # and retry from the last one with half the step, until it is tiny.
                if ans.hessian is None:
                    y = g_new - np.asarray(b.prev.gradient).ravel()
                    b.h = update(b.h, dx, y)
                b.ds *= 0.5
                if b.ds < kw.step_size / 16:
                    b.done, b.reason = True, "energy rose"
                else:
                    b.next = next_point(b)
                continue
            if ans.hessian is not None:
                b.h = np.asarray(ans.hessian, dtype=float)
                b.since_hessian = 0
            else:
                y = g_new - np.asarray(b.prev.gradient).ravel()
                b.h = update(b.h, dx, y)
                b.since_hessian += 1
            b.frames.append(ans)
            de_last = abs(float(ans.energy) - float(b.prev.energy))
            b.last_de = de_last
            step_mw = float(np.sqrt((m3 * dx * dx).sum()))
            b.tiny_steps = b.tiny_steps + 1 if step_mw < 1e-3 * kw.step_size else 0
            b.x, b.prev = np.asarray(ans.positions, dtype=float).reshape(n, 3), ans
            if end_test(b, g_new, de_last, ans.hessian is not None):
                continue
            if b.tiny_steps >= 3:  # the model sees no way down: the path ends here
                b.done, b.reason = True, "stalled"
                continue
            b.ds = min(kw.step_size, 2.0 * b.ds)
            b.next = next_point(b)

    end_info: dict[str, Any] = {}
    if kw.minimize_ends:
        numbers = [int(getattr(pt, sym.capitalize()).number) for sym in symbols]
        end_kw = OptimizationKeywords(
            coordinates=kw.end_coordinates,
            convergence=kw.end_convergence,
            maxiter=kw.end_maxiter,
            frozen_atoms=kw.frozen_atoms,
            initial_hessian="model",
        )
        subs = {
            b.name: optimize(
                b.x,
                symbols,
                numbers,
                end_kw,
                transition_state=False,
                has_hessian_program=has_hessian_program,
                start=(b.prev, b.h),
                separate_hessian=separate_hessian,
            )
            for b in branches
            if b.frames and not (b.reason or "").startswith("gradient failed")
        }
        ends = yield from lockstep(subs)  # type: ignore[arg-type]
        for name, res in ends.items():
            b = next(b for b in branches if b.name == name)
            if isinstance(res, BaseException) or not res.frames:
                end_info[name] = {"converged": False, "error": str(res)}
                continue
            end_info[name] = {
                "converged": res.converged and res.error is None,
                "error": res.error,
                "n_gradients": res.info.get("n_gradients", 0),
                "energy_drop": float(b.prev.energy - res.frames[-1].energy),
            }
            count.gradients += res.info.get("n_gradients", 0)
            count.hessians += res.info.get("n_hessians", 0)
            if not np.array_equal(
                np.asarray(res.frames[-1].positions),
                np.asarray(b.frames[-1].positions),
            ):
                b.frames.append(res.frames[-1])

    frames, ts_index = assemble()
    reasons = {b.name: b.reason for b in branches}
    failed = [r for r in reasons.values() if r and r.startswith("gradient failed")]
    unfinished = []
    for b in branches:
        # "energy rose": even a sixteenth of a step passed the valley floor, so the
        # path has reached the bottom along its direction; it ends there if its
        # gradient is close to the end criterion (within ENERGY_ROSE_GMAX of gmax).
        ended = (b.reason or "").startswith("converged") or (
            b.reason in ("energy rose", "stalled")
            and b.gmax < ENERGY_ROSE_GMAX * kw.gmax
        )
        end = end_info.get(b.name)
        if kw.minimize_ends and end is not None and not end["converged"]:
            unfinished.append(f"the {b.name} end minimization did not converge")
        elif b.frames and float(b.frames[-1].energy) >= float(ts.energy):
            unfinished.append(
                f"the {b.name} branch did not go downhill from the transition state"
            )
        elif not ended and not (kw.minimize_ends and end is not None):
            why = b.reason
            if b.reason in ("energy rose", "stalled"):
                why = f"{b.reason} with the gradient at {b.gmax:.1e} Hartree/bohr"
            unfinished.append(f"the {b.name} branch stopped: {why}")
    info = {
        "n_gradients": count.gradients,
        "n_hessians": count.hessians,
        "branch_end": reasons,
        "branch_steps": {b.name: len(b.frames) for b in branches},
        "end_minimization": end_info,
        "ts_imaginary_curvature": float(vals[0]),
    }
    problems = failed + [u for u in unfinished if not failed]
    return RunResult(
        frames=frames,
        converged=not problems,
        error="; ".join(problems) if problems else None,
        ts_index=ts_index,
        info=info,
    )
