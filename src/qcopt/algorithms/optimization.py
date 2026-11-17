"""Minimization (RS-RFO) and transition-state search (RS-P-RFO).

One loop serves both. Each iteration takes a rational-function step in the
eigenbasis of the current Hessian, inside the trust region, in the chosen
coordinate system (:mod:`qcopt.coordinates`), and updates the Hessian from the new
gradient. A minimization minimizes along every mode; a transition-state search
maximizes along one (the lowest, or the one following a guide direction).

Once the convergence criteria are met, the stationary point is verified:

- a transition state must have exactly one direction of negative curvature,
  checked with the method's own exact Hessian when the updated one is doubtful;
- a minimum must have none: with ``escape.attempts`` its frequencies are checked,
  and from a saddle point the run steps off along the lowest mode and continues;
- with ``freq.check``, the frequencies of the final structure decide.

Hessians from a cheaper method (a separate Hessian program) guide the steps, but
only the gradient method's own decide whether a stationary point is a minimum or a
saddle point.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from .. import hessians
from ..constraints import ConstraintSet, correction, free_gradient, project
from ..coordinates import CartesianCoordinates, InternalCoordinates, make_coordinates
from ..driver import Algorithm, Answer, Ask, Kind
from ..errors import ErrorKind
from ..geometry import Array, max_atomic_norm, rms
from ..internal import CoordinateError
from ..options import (
    ConvergenceSettings,
    OptimizationOptions,
    TransitionStateOptions,
)
from ..space import LOCKED_CURVATURE, CartesianSpace
from ..steps import rfo_step, select_mode, update_trust
from ..vibrations import frequencies
from .common import (
    Counter,
    Progress,
    RunResult,
    guarded,
    program_failed,
    resolve_initial_hessian,
    with_hessian,
)

#: A transition-state search whose updated Hessian ends with a second negative
#: curvature at least this fraction of the first checks it with the exact Hessian.
#: (An updated Hessian often shows a weak spurious second negative direction: 11 of
#: 100 g-xTB searches, at most 0.12 of the first, all positive in the exact Hessian.)
SECOND_NEGATIVE_RATIO = 0.25

#: Curvature (Hartree/bohr^2, or per internal unit) below minus which a minimum's
#: exact Hessian counts a direction as negative.
NEGATIVE_CURVATURE = 1e-4

#: Energy changes (Hartree) this small, with a prediction as small, are SCF noise:
#: the step counts as agreeing with the model.
NOISE_ENERGY = 1e-8

#: Back-transformation failures (internal coordinates) before the run continues in
#: Cartesian coordinates.
MAX_BACKTRANSFORM_FAILURES = 20

#: Failed gradient calculations in a row (each retried with a smaller step) before
#: the run stops.
MAX_GRADIENT_FAILURES = 3

#: A minimization that ends this much (Hartree) above its start fails: the program
#: changed state (an SCF that landed on another solution, for example).
ENERGY_ROSE = 1e-5


def optimize(
    positions: Array,
    symbols: Sequence[str],
    numbers: Sequence[int],
    options: OptimizationOptions | TransitionStateOptions,
    *,
    has_hessian_program: bool,
    start: tuple[Answer, Array] | None = None,
    given_hessian: Array | None = None,
    own_hessian: bool = True,
    lineage: Hashable | None = None,
) -> Algorithm:
    """Minimize (``OptimizationOptions``) or find a first-order saddle point.

    An exception inside the algorithm ends it as a failure that keeps its frames.

    Args:
        positions: Starting geometry (bohr).
        symbols: Element symbols.
        numbers: Atomic numbers (for the model Hessian).
        options: The settings; their type says which search.
        has_hessian_program: Whether exact Hessians come from a program (else
            finite differences of gradients).
        start: The calculation already done at ``positions`` and a Cartesian
            Hessian to start from (such as an IRC branch's updated one); then
            nothing is computed before the first step.
        given_hessian: A Hessian already known at ``positions`` (the input's
            ``ProgramInput.given``), the initial one with ``hessian.initial``
            ``auto`` or ``given``: a starting guess, like the model Hessian (it may
            be a cheaper method's).
        own_hessian: Hessians are the gradient method's own (the same program and
            method); otherwise (a cheaper method) they guide the steps but decide
            nothing about the stationary point.
        lineage: The lineage of every calculation it asks for (see
            :class:`~qcopt.driver.Ask`), such as the IRC branch an end minimization
            continues.
    """
    return (
        yield from guarded(
            lambda progress: _Optimization(
                positions,
                symbols,
                numbers,
                options,
                has_hessian_program=has_hessian_program,
                start=start,
                given_hessian=given_hessian,
                own_hessian=own_hessian,
                lineage=lineage,
                progress=progress,
            ).run()
        )
    )


def converged(
    criteria: ConvergenceSettings, gradient: Array, step: Array | None, de: float | None
) -> bool:
    """Whether a free gradient, the last step and its energy change meet ``criteria``."""
    gmax = max_atomic_norm(gradient)
    if not np.isfinite(gmax) or gmax > criteria.g_tol:
        return False
    if criteria.g_rms_tol is not None and rms(gradient) > criteria.g_rms_tol:
        return False
    if criteria.d_tol is not None and (
        step is None or max_atomic_norm(step) > criteria.d_tol
    ):
        return False
    if criteria.e_tol is not None and (de is None or abs(de) > criteria.e_tol):
        return False
    return True


class _Optimization:
    """The state of one minimization or transition-state search.

    ``run`` is the algorithm (a generator); its phases are the methods below, each
    yielding the calculations it needs.
    """

    def __init__(
        self,
        positions: Array,
        symbols: Sequence[str],
        numbers: Sequence[int],
        options: OptimizationOptions | TransitionStateOptions,
        *,
        has_hessian_program: bool,
        start: tuple[Answer, Array] | None,
        given_hessian: Array | None,
        own_hessian: bool,
        lineage: Hashable | None,
        progress: Progress,
    ) -> None:
        """Set up coordinates, constraints and bookkeeping; nothing is computed."""
        opts = self.opts = options
        self.ts = isinstance(options, TransitionStateOptions)
        self.symbols = list(symbols)
        self.numbers = list(numbers)
        self.start = start
        self.own = own_hessian  # exact Hessians are the gradient method's own
        self.lineage = lineage
        self.count = progress.count = Counter()
        n = self.n = len(symbols)
        frozen = opts.constraints.frozen
        self.x = np.asarray(positions, dtype=float).reshape(n, 3)
        self.cart = CartesianSpace(n, frozen=frozen)
        self.atomic_trust = opts.trust.measure == "atomic"
        self.coords: CartesianCoordinates | InternalCoordinates = make_coordinates(
            opts.coordinate_system, symbols, self.x, frozen, self.atomic_trust
        )
        self.cons = (
            ConstraintSet(opts.constraints, self.x, frozen)
            if opts.constraints.size
            else None
        )
        self.lost_curvature = (
            opts.hessian.lost_curvature
            if isinstance(opts, TransitionStateOptions)
            else NEGATIVE_CURVATURE
        )
        self.frames: list[Answer] = []
        self.log: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {
            "rejected_steps": 0,
            "hessian_recomputes": 0,
            "coordinate_rebuilds": 0,
            "backtransform_failures": 0,
            "coordinate_system": self.coords.name,
            "log": self.log,
        }
        progress.frames = lambda: list(self.frames)
        progress.info = self.info
        self.source = resolve_initial_hessian(
            opts.hessian.initial,
            has_hessian_program,
            self.ts,
            given_hessian is not None,
        )
        if self.source == "given" and given_hessian is None:
            raise ValueError("hessian.initial='given' needs a given Hessian")
        self.given_hessian = given_hessian if self.source == "given" else None
        self.exact_available = self.source in ("exact", "finite_difference")
        # Exact Hessians later on: the Hessian program's, or finite differences when
        # the run started from them; ``hessian.recompute_every`` asks in any case.
        self.can_recompute = self.exact_available or has_hessian_program
        self.update_name = (
            opts.hessian.update
            if opts.hessian.update != "auto"
            else ("ts_bfgs" if self.ts else "bfgs")
        )
        self.update = hessians.UPDATES[self.update_name]
        guide = opts.mode.guide if isinstance(opts, TransitionStateOptions) else None
        self.mode_ref = (
            None if guide is None else np.asarray(guide, dtype=float).ravel()
        )
        self.trust0 = min(max(opts.trust.radius, opts.trust.min), opts.trust.max)
        self.trust = self.trust0
        self.rejections = 0
        self.failures = 0
        self.converged = False
        self.error: str | None = None
        self.error_kind: ErrorKind | None = None
        self.reason: str | None = None
        self.cause: Answer | None = None
        self.last_step: Array | None = None
        self.last_de: float | None = None
        self.steps_since_hessian = 0
        self.h_exact = start is None and self.exact_available and self.own
        self.freqs: Any = None
        self.escapes = 0
        self.iterations = 0

    # ------------------------------------------------------------ the loop

    def run(self) -> Algorithm:
        """The algorithm: start, iterate until converged, verify, finish."""
        failed = yield from self._begin()
        if failed is not None:
            return failed
        while True:
            self.cart.at(self.x)
            if self._at_stationary_point():
                self.converged = True
                if self.ts:
                    yield from self._verify_saddle()
                    break
                if not self.opts.escape.attempts or self.cons is not None:  # type: ignore[union-attr]
                    self._verify_minimum()
                    break
                outcome = yield from self._escape_saddle()
                if isinstance(outcome, RunResult):
                    return outcome
                if outcome:
                    break
                continue
            if self.iterations >= self.opts.convergence.max_iter:
                break
            self.iterations += 1
            outcome = yield from self._iterate()
            if isinstance(outcome, RunResult):
                return outcome
        yield from self._final_frequencies()
        return self._result()

    def _begin(self) -> Algorithm:
        """The first calculation and the starting Hessian; a failed result or None."""
        x, n = self.x, self.n
        if self.start is not None:
            a = self.start[0]
        else:
            kind: Kind = "hessian" if self.exact_available else "gradient"
            first = Ask(kind, x, self.lineage)
            (a,) = self.count((yield [first]))
        if not a.success:
            return program_failed([a], a, "at the starting geometry")
        self.a = a
        self.frames.append(a)
        self.coords.at(x)
        g_raw = np.asarray(a.gradient, dtype=float).ravel()
        if self.start is not None:
            h0 = np.asarray(self.start[1], dtype=float)
            self.h = self.coords.hessian(x, h0, g_raw)
        elif self.given_hessian is not None:
            self.h = self.coords.hessian(x, np.asarray(self.given_hessian), g_raw)
        elif self.exact_available:
            self.h = self.coords.hessian(x, np.asarray(a.hessian, dtype=float), g_raw)
        elif self.source == "model":
            model = hessians.lindh(self.numbers, x) + 1e-4 * np.eye(3 * n)
            self.h = self.coords.hessian(x, model)
        else:
            self.h = self.coords.hessian(x, 0.5 * np.eye(3 * n))
        self.g = self.coords.gradient(x, g_raw)
        return None

    def _at_stationary_point(self) -> bool:
        """Whether the convergence criteria (and the constraints) are met."""
        free = self._free(self.a.gradient, self.x)
        if not converged(self.opts.convergence, free, self.last_step, self.last_de):
            return False
        if self.cons is None:
            return True
        tol = self.opts.constraints.tol
        return bool(np.abs(self.cons.violation(self.x)).max() < tol)

    def _iterate(self) -> Algorithm:
        """One step: decompose, step, compute, judge, update; a failed result or None."""
        opts, coords = self.opts, self.coords
        x, a = self.x, self.a
        g_scale = 1e-3 * max_atomic_norm(np.asarray(a.gradient))
        if coords.needs_rebuild(x) or coords.residual > max(
            0.2 * opts.convergence.g_tol, g_scale
        ):
            self.h = self._rebuild(self.h)
            self.g = self.coords.gradient(x, np.asarray(a.gradient, dtype=float))
        vals, vecs, g_eff, jw = self._decompose(self.h)
        n_free = len(vals) - self.coords.n_locked - self._n_constraints
        if self._lost_negative_mode(vals[0]):
            ha = yield from self._recompute_hessian()
            if ha.success:
                vals, vecs, g_eff, jw = self._decompose(self.h)
        mode = self._mode(vals, vecs, n_free)
        coords = self.coords
        step = rfo_step(
            vals,
            vecs,
            g_eff,
            self.trust,
            mode=mode,
            atomic=coords.name == "cartesian" and self.atomic_trust,
        )
        dw = step.dx
        if jw is not None and self.cons is not None:
            violation = self.cons.violation(x)
            dw = dw + correction(jw, violation, opts.constraints.max_correction)
        x_new, d, ok = coords.step(x, dw)
        if not ok:
            self._backtransform_failed()
            return None
        predicted = float(self.g @ d + 0.5 * d @ self.h @ d)
        (b,) = self.count((yield [Ask("gradient", x_new, self.lineage)]))
        if not b.success:
            self.failures += 1
            if self.failures > MAX_GRADIENT_FAILURES or self.trust <= opts.trust.min:
                where = f"at step {self.iterations}"
                return program_failed(self.frames + [b], b, where, self.info)
            self.trust = max(opts.trust.min, 0.25 * min(self.trust, coords.size(d)))
            return None
        self.failures = 0
        de = b.energy - a.energy
        noise = abs(de) < NOISE_ENERGY and abs(predicted) < NOISE_ENERGY
        ratio = 1.0 if noise or abs(predicted) < 1e-14 else de / predicted
        size = coords.size(d)
        self.log.append(
            {
                "step": size,
                "trust": self.trust,
                "ratio": float(ratio),
                "lowest": float(vals[0]),
                "mode": None if mode is None else int(mode),
                "de": float(de),
                "gmax": max_atomic_norm(
                    self.cart.project(np.asarray(a.gradient).ravel())
                ),
            }
        )
        self.trust = update_trust(
            self.trust,
            ratio,
            size,
            minimum=opts.trust.min,
            maximum=opts.trust.max,
            ts=self.ts,
        )
        coords.at(x_new)
        g_new = coords.gradient(x_new, np.asarray(b.gradient, dtype=float))
        self._update_hessian(d, g_new - self.g, vals[0])
        if self._reject(de, ratio, size):
            self.rejections += 1
            self.info["rejected_steps"] += 1
            self.trust = max(opts.trust.min, min(self.trust, 0.5 * size))
            coords.at(x)
            return None
        self.rejections = 0
        self.last_step = (x_new - x).ravel()
        self.x, self.a, self.g = x_new, b, g_new
        self.frames.append(b)
        self.last_de = de
        self.steps_since_hessian += 1
        self.h_exact = False
        every = opts.hessian.recompute_every
        if every and self.steps_since_hessian >= every:
            yield from self._recompute_hessian()
        return None

    # --------------------------------------------------- pieces of a step

    @property
    def _n_constraints(self) -> int:
        return self.cons.size if self.cons is not None else 0

    def _free(self, gradient: Any, x: Array) -> Array:
        """Cartesian gradient without constraint forces or rigid motion."""
        gx = np.asarray(gradient, dtype=float).ravel()
        if self.cons is not None:
            gx = free_gradient(self.cons.jacobian(x), gx)
        self.cart.at(x)
        return self.cart.project(gx)

    def _decompose(self, hh: Array) -> tuple[Array, Array, Array, Array | None]:
        """Eigenpairs of the Hessian with constrained directions locked.

        Returns the eigenvalues, eigenvectors, the gradient to step with, and the
        constraint Jacobian in the working coordinates (None without constraints).
        """
        coords, x = self.coords, self.x
        if self.cons is None:
            v, w = coords.eigh(hh)
            return v, w, self.g, None
        jw = coords.jacobian_to_space(x, self.cons.jacobian(x))
        pf, pc = project(jw)
        v, w = coords.eigh(pf @ hh @ pf + LOCKED_CURVATURE * pc)
        return v, w, pf @ self.g, jw

    def _negative(self, hh: Array) -> Array:
        """The negative curvatures among the free directions, lowest first."""
        vals = self._decompose(hh)[0]
        n_free = len(vals) - self.coords.n_locked - self._n_constraints
        free = vals[:n_free]
        return free[free < -self.lost_curvature]

    def _lost_negative_mode(self, lowest: float) -> bool:
        """Transition states: whether to recompute the Hessian, its mode lost."""
        opts = self.opts
        if not isinstance(opts, TransitionStateOptions):
            return False
        return bool(
            self.can_recompute
            and opts.hessian.recompute_on_loss
            and self.steps_since_hessian >= opts.hessian.recompute_min_steps
            and lowest > -opts.hessian.lost_curvature
        )

    def _mode(self, vals: Array, vecs: Array, n_free: int) -> int | None:
        """The eigenvector to climb (transition states), else None."""
        if not self.ts:
            return None
        if self.mode_ref is None or not isinstance(self.coords, CartesianCoordinates):
            return 0
        mode = select_mode(vecs, vals, self.cart.project(self.mode_ref), n_free)
        follow = self.opts.mode.follow  # type: ignore[union-attr]
        self.mode_ref = vecs[:, mode] if follow == "overlap" else None
        return mode

    def _update_hessian(self, d: Array, y: Array, lowest: float) -> None:
        """Quasi-Newton update; BFGS gives way to Bofill where curvature is negative."""
        if self.update_name == "bfgs" and self.opts.hessian.update == "auto":
            if lowest < -1e-10:  # BFGS cannot fix negative curvature
                self.h = hessians.bofill(self.h, d, y)
                return
        self.h = self.update(self.h, d, y)

    def _reject(self, de: float, ratio: float, size: float) -> bool:
        """Minimizations: undo a step that raised the energy against the model."""
        opts = self.opts
        if not isinstance(opts, OptimizationOptions):
            return False
        return bool(
            de > NOISE_ENERGY
            and ratio < 0.0
            and self.rejections < opts.trust.max_rejections
            and size > opts.trust.min
        )

    def _backtransform_failed(self) -> None:
        """An internal step that cannot be taken: rebuild, shrink, maybe Cartesians."""
        self.info["backtransform_failures"] += 1
        a = self.a
        if isinstance(self.coords, InternalCoordinates):
            self.h = self._rebuild(self.h)
            self.g = self.coords.gradient(self.x, np.asarray(a.gradient, dtype=float))
        self.trust = max(self.opts.trust.min, 0.5 * self.trust)
        failures = self.info["backtransform_failures"]
        if failures > MAX_BACKTRANSFORM_FAILURES and isinstance(
            self.coords, InternalCoordinates
        ):
            self.h = self._to_cartesian(self.h, "cartesian (fallback)")
            self.g = self.coords.gradient(self.x, np.asarray(a.gradient, dtype=float))

    def _rebuild(self, hh: Array) -> Array:
        """New internal coordinates at ``x``; Cartesians when they cannot be built."""
        self.info["coordinate_rebuilds"] += 1
        assert isinstance(self.coords, InternalCoordinates)
        try:
            return self.coords.rebuild(self.x, hh)
        except CoordinateError:
            name = f"{self.info['coordinate_system']} -> cartesian"
            return self._to_cartesian(hh, name)

    def _to_cartesian(self, hh: Array, name: str) -> Array:
        """Switch to Cartesian coordinates, keeping the Hessian."""
        assert isinstance(self.coords, InternalCoordinates)
        x, n = self.x, self.n
        hx = self.coords.ic.to_cartesian_hessian(x, hh)
        frozen = self.opts.constraints.frozen
        self.coords = CartesianCoordinates(n, frozen, self.atomic_trust)
        self.coords.at(x)
        self.info["coordinate_system"] = name
        return hx + 1e-4 * np.eye(3 * n)

    def _recompute_hessian(self) -> Algorithm:
        """The exact Hessian at ``x`` replaces the updated one; returns its answer."""
        x = self.x
        (ha,) = self.count((yield [Ask("hessian", x, self.lineage)]))
        if ha.success:
            hx = np.asarray(ha.hessian)
            self.h = self.coords.hessian(x, hx, np.asarray(ha.gradient).ravel())
            self.frames[-1] = with_hessian(self.frames[-1], hx)
            self.info["hessian_recomputes"] += 1
            self.steps_since_hessian = 0
            self.h_exact = self.own
        return ha

    # ------------------------------------------- verifying the stationary point

    def _fail(
        self,
        kind: ErrorKind,
        message: str,
        reason: str | None = None,
        cause: Answer | None = None,
    ) -> None:
        """Record why the run failed, unless an earlier reason stands."""
        if self.error is None:
            self.error, self.error_kind, self.reason = message, kind, reason
            self.cause = cause

    def _verify_saddle(self) -> Algorithm:
        """A transition state needs one direction of negative curvature, no more.

        With ``freq.check`` and the method's own Hessians the final frequencies
        decide (later). Otherwise an updated Hessian with none, or a strong second
        one, is checked with the exact Hessian; only an exact Hessian fails a search
        for having several.
        """
        if self.opts.freq.check and self.own:
            return
        neg = self._negative(self.h)
        n_neg = len(neg)
        doubtful = n_neg == 0 or (n_neg > 1 and neg[1] < SECOND_NEGATIVE_RATIO * neg[0])
        if doubtful and self.can_recompute and not self.h_exact and self.own:
            ha = yield from self._recompute_hessian()
            if not ha.success:
                self._fail(
                    "program",
                    f"the Hessian to check the saddle point failed: {ha.error}",
                    cause=ha,
                )
                return
            n_neg = len(self._negative(self.h))
        if n_neg == 0 or (self.h_exact and n_neg > 1):
            self._fail(
                "convergence",
                f"stopped at a stationary point with {n_neg} directions of "
                "negative curvature, not a first-order saddle point",
                "not_first_order_saddle",
            )

    def _verify_minimum(self) -> None:
        """A minimum whose exact Hessian (the method's own) has negative curvature."""
        if self.h_exact and self.cons is None and len(self._negative(self.h)):
            self._fail(
                "convergence",
                "stopped at a saddle point (negative curvature in the exact "
                "Hessian); escape.attempts steps off saddle points",
                "saddle_point",
            )

    def _escape_saddle(self) -> Algorithm:
        """Check the frequencies; at a saddle point, step off along the lowest mode.

        Returns True when the run is over, False to continue minimizing, or a failed
        result.
        """
        opts = self.opts
        assert isinstance(opts, OptimizationOptions)
        x = self.x
        (ha,) = self.count((yield [Ask("hessian", x, self.lineage)]))
        if not ha.success:
            self._fail(
                "program",
                f"the Hessian to check for a saddle point failed: {ha.error}",
                cause=ha,
            )
            return True
        hx = np.asarray(ha.hessian)
        self.frames[-1] = with_hessian(self.frames[-1], hx)
        frozen = opts.constraints.frozen
        self.freqs = frequencies(hx, self.symbols, x, frozen=frozen)
        n_imag = self.freqs.n_imaginary(opts.freq.imaginary_cutoff)
        if n_imag == 0 or self.escapes >= opts.escape.attempts:
            if n_imag and self.own:
                self._fail(
                    "convergence",
                    f"still a saddle point ({n_imag} imaginary modes) after "
                    f"{self.escapes} escapes",
                    "saddle_point",
                )
            return True
        self.converged = False
        self.escapes += 1
        self.iterations += 1
        self.info["saddle_escapes"] = self.escapes
        mode = np.asarray(self.freqs.modes[0], dtype=float)
        self.x = x = x + mode / max_atomic_norm(mode) * opts.escape.step
        (a,) = self.count((yield [Ask("gradient", x, self.lineage)]))
        if not a.success:
            where = "stepping off a saddle point"
            return program_failed(self.frames + [a], a, where, self.info)
        self.a = a
        self.frames.append(a)
        self.coords.at(x)
        self.h = self.coords.hessian(x, hx)
        self.g = self.coords.gradient(x, np.asarray(a.gradient, dtype=float))
        self.trust, self.last_step, self.last_de, self.freqs = (
            self.trust0,
            None,
            None,
            None,
        )
        return False

    def _final_frequencies(self) -> Algorithm:
        """With ``freq.check``, the final structure's frequencies (they may decide)."""
        opts = self.opts
        if self.freqs is not None:
            self.info["n_imaginary"] = self.freqs.n_imaginary(
                opts.freq.imaginary_cutoff
            )
            return
        if not opts.freq.check:
            return
        x = self.x
        (ha,) = self.count((yield [Ask("hessian", x, self.lineage)]))
        if not ha.success:
            self._fail(
                "program",
                f"the Hessian for the frequencies failed: {ha.error}",
                cause=ha,
            )
            return
        hx = np.asarray(ha.hessian)
        self.frames[-1] = with_hessian(self.frames[-1], hx)
        frozen = opts.constraints.frozen
        self.freqs = frequencies(hx, self.symbols, x, frozen=frozen)
        n_imag = self.freqs.n_imaginary(opts.freq.imaginary_cutoff)
        self.info["n_imaginary"] = n_imag
        if not (self.converged and self.own):
            return
        if self.ts and n_imag != 1:
            self._fail(
                "convergence",
                f"stopped at a stationary point with {n_imag} imaginary frequencies, "
                "not a first-order saddle point",
                "not_first_order_saddle",
            )
        elif not self.ts and n_imag and self.cons is None:
            self._fail(
                "convergence",
                f"stopped at a saddle point ({n_imag} imaginary frequencies); "
                "escape.attempts steps off saddle points",
                "saddle_point",
            )

    def _result(self) -> RunResult:
        """The finished run: counts, final checks, and why it failed if it did."""
        self.cart.at(self.x)
        self.info.update(
            n_gradients=self.count.gradients,
            n_hessians=self.count.hessians,
            n_steps=len(self.frames) - 1,
            final_trust=self.trust,
            gmax=max_atomic_norm(self._free(self.a.gradient, self.x)),
        )
        rose = float(self.frames[-1].energy) - float(self.frames[0].energy)
        if self.converged and not self.ts and self.cons is None and rose > ENERGY_ROSE:
            self._fail(
                "convergence",
                f"ended {rose:.2e} Eh higher than it started",
                "energy_rose",
            )
        if self.freqs is not None and not self.own:
            self.info["frequencies_from"] = "hessian subprogram"
        if not self.converged:
            self.error = f"not converged in {self.opts.convergence.max_iter} steps"
            self.error_kind, self.reason = "convergence", "max_iter"
        return RunResult(
            frames=self.frames,
            converged=self.converged,
            error=self.error,
            error_kind=self.error_kind,
            reason=self.reason,
            info=self.info,
            frequencies=self.freqs,
            cause=self.cause,
        )
