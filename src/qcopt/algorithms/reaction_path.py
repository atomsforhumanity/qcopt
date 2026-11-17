"""The intrinsic reaction coordinate: steepest descent in mass-weighted coordinates.

From the transition state, each branch first steps along the Hessian's negative
mode, then follows the path with Gonzalez-Schlegel steps (or the local quadratic
approximation) on the current Hessian: exact at the transition state, then updated.
Both branches advance together. A step past the valley floor (the energy rose, or
the new gradient points along the step) is retried at half the length.

A branch ends where its gradient is small and the curvature positive
(``convergence.g_tol``; the exact Hessian decides where the updated one disagrees),
after ``convergence.max_iter`` steps, or where it can go no further. With
``ends.minimize`` both ends are then minimized together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qcref import periodic_table as pt

from .. import hessians
from ..driver import Algorithm, Answer, Ask, Kind
from ..errors import ErrorKind
from ..geometry import Array, masses, max_atomic_norm
from ..options import (
    ConstraintSettings,
    HessianSettings,
    IRCOptions,
    OptimizationOptions,
)
from ..space import LOCKED_CURVATURE, CartesianSpace
from .common import (
    Counter,
    Progress,
    Report,
    RunResult,
    child_failed,
    guarded,
    lockstep,
    resolve_initial_hessian,
    with_hessian,
)
from .optimization import optimize

#: A branch whose steps keep passing the valley floor ("energy rose"), or whose steps
#: have shrunk to nothing ("stalled"), ends there when its largest atomic gradient is
#: below this multiple of ``convergence.g_tol`` (4 x 5e-4 Hartree/bohr is about
#: 0.1 eV/angstrom, Sella's IRC criterion).
ENERGY_ROSE_GMAX = 4.0

#: Mass-weighted curvature above minus which a branch's end counts as a minimum.
END_CURVATURE = 1e-5

#: Exact Hessians a branch may ask for to decide whether it has ended.
MAX_END_CHECKS = 2

#: The step is halved after an overshoot down to this fraction of ``irc.step``.
MIN_STEP_FRACTION = 1 / 16

#: Steps (mass-weighted) below this fraction of ``irc.step``, three in a row, mean
#: the model sees no way down: the branch has stalled.
TINY_STEP_FRACTION = 1e-3


def gs2_step(vals: Array, gt: Array, ds: float) -> Array:
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


def lqa_step(vals: Array, gt: Array, ds: float) -> Array:
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
        if s[-1] - previous <= 1e-6 * s[-1]:  # the path ends (at a minimum) before ds
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


def irc(
    positions: Array,
    symbols: Sequence[str],
    options: IRCOptions,
    *,
    has_hessian_program: bool,
    own_hessian: bool = True,
    given_hessian: Array | None = None,
    report: Report | None = None,
) -> Algorithm:
    """Follow the IRC down both sides of a transition state.

    Mass-weighted Cartesian coordinates with rigid motion projected out, so the path
    stays in the input's frame. The IRC has converged when every branch reached its
    end, or, with ``ends.minimize``, when every end minimization converged; an
    exception inside it ends it as a failure that keeps its frames.

    Args:
        positions: The transition state (bohr).
        symbols: Element symbols.
        options: The settings.
        has_hessian_program: Whether exact Hessians come from a program (else
            finite differences of gradients).
        own_hessian: Hessians are the gradient method's own (see
            :func:`~qcopt.algorithms.optimize`).
        given_hessian: A Hessian already known at the transition state (the input's
            ``ProgramInput.given``), used instead of computing one there (with
            ``hessian.initial`` ``auto`` or ``given``); only the gradient is computed.
        report: Called with each point's event: ``calctype`` ``irc``, ``line``
            (``forward`` or ``reverse``; none for the transition state), ``step``
            (the point's number on its line; 0 for the transition state),
            ``energy`` (Hartree) and ``g_max`` (the largest atomic gradient norm,
            Hartree/bohr). The end minimizations report as
            :func:`~qcopt.algorithms.optimize` does, with their branch as ``line``.
    """
    return (
        yield from guarded(
            lambda progress: _IRC(
                positions,
                symbols,
                options,
                has_hessian_program=has_hessian_program,
                own_hessian=own_hessian,
                given_hessian=given_hessian,
                report=report,
                progress=progress,
            ).run()
        )
    )


#: How an IRC branch ended (``record["branches"][name]["end"]``), with the words its
#: error message uses: ``converged`` (gradient and curvature criteria),
#: ``converged_flat`` (a flat end, checked with exact Hessians), ``max_iter``,
#: ``energy_rose`` (even a tiny step passed the valley floor), ``stalled`` (no way
#: down on the model), ``no_descent`` (no step down from the transition state) and
#: ``gradient_failed`` (a gradient calculation failed; the branch's ``error`` says
#: why).
BRANCH_ENDS = {
    "converged": "converged",
    "converged_flat": "converged (flat)",
    "max_iter": "max_iter",
    "energy_rose": "energy rose",
    "stalled": "stalled",
    "no_descent": "no descent from the transition state",
    "gradient_failed": "gradient failed",
}


@dataclass
class _Branch:
    """One direction of an IRC: where it is, its Hessian, and how it is doing."""

    name: str
    sign: float
    x: Array
    prev: Answer  # the calculation at ``x``
    h: Array  # the Cartesian Hessian at ``x`` (exact or updated)
    next: Array  # where the next calculation goes
    ds: float  # the current step length
    frames: list[Answer] = field(default_factory=list)
    done: bool = False
    reason: str | None = None  # how it ended: a key of BRANCH_ENDS
    error: str | None = None  # for gradient_failed: the calculation's error
    steps: int = 0
    since_hessian: int = 0
    need_hessian: bool = False  # the next ask is the last point again, for its Hessian
    asked_hessian: bool = False
    skip_hessian: bool = False  # optional Hessians failed here: updates only
    end_checks: int = 0
    hessian_failures: int = 0
    last_de: float = float("inf")
    gmax: float = float("inf")
    tiny_steps: int = 0
    failure: Answer | None = None  # the calculation that stopped it

    def finish(self, reason: str, error: str | None = None) -> None:
        """End the branch: ``reason`` is a key of :data:`BRANCH_ENDS`."""
        self.done, self.reason, self.error = True, reason, error

    @property
    def ended(self) -> bool:
        """Whether it reached the end of the path (rather than failing or stopping)."""
        return self.reason in ("converged", "converged_flat")

    @property
    def gradient_failed(self) -> bool:
        """Whether a gradient calculation stopped it."""
        return self.reason == "gradient_failed"

    def why(self) -> str:
        """How it ended, in words (``gradient failed: <error>``)."""
        text = BRANCH_ENDS.get(self.reason or "", str(self.reason))
        return f"{text}: {self.error}" if self.error is not None else text

    def end_record(self) -> dict[str, Any]:
        """Its entry in ``record["branches"]``.

        ``steps``: the gradient calculations it took, retried overshoots included (what
        ``convergence.max_iter`` counts); ``frames``: the points it kept.
        """
        entry: dict[str, Any] = {
            "end": self.reason,
            "steps": self.steps,
            "frames": len(self.frames),
        }
        if self.error is not None:
            entry["error"] = self.error
        return entry


class _IRC:
    """The state of one IRC; ``run`` is the algorithm."""

    def __init__(
        self,
        positions: Array,
        symbols: Sequence[str],
        options: IRCOptions,
        *,
        has_hessian_program: bool,
        own_hessian: bool,
        given_hessian: Array | None = None,
        progress: Progress,
        report: Report | None = None,
    ) -> None:
        """Set up the mass-weighted space; nothing is computed."""
        self.opts = options
        source = resolve_initial_hessian(
            options.hessian.initial,
            has_hessian_program,
            True,
            given_hessian is not None,
        )
        if source == "given" and given_hessian is None:
            raise ValueError("hessian.initial='given' needs a given Hessian")
        self.given_hessian = given_hessian if source == "given" else None
        self.symbols = list(symbols)
        self.has_hessian_program = has_hessian_program
        self.own = own_hessian
        self.progress = progress
        self.report = report
        self.count = progress.count = Counter()
        n = self.n = len(symbols)
        self.m3 = np.repeat(masses(symbols), 3)
        self.x0 = np.asarray(positions, dtype=float).reshape(n, 3)
        self.space = CartesianSpace(
            n, frozen=options.constraints.frozen, masses=masses(symbols)
        )
        self.update = hessians.UPDATES[options.hessian.update]
        self.branches: list[_Branch] = []

    def run(self) -> Algorithm:
        """The algorithm: the transition state, both branches, then the ends."""
        opts = self.opts
        given = self.given_hessian
        kind: Kind = "hessian" if given is None else "gradient"
        (ts,) = self.count((yield [Ask(kind, self.x0)]))
        self.ts = ts
        self.progress.frames = lambda: [ts]
        if not ts.success:
            return RunResult(
                frames=[ts],
                converged=False,
                error=child_failed(ts, "at the transition state"),
                error_kind="program",
                ts_index=0,
                cause=ts,
            )
        self._report(None, 0, ts)
        h0 = np.asarray(ts.hessian if given is None else given, dtype=float)
        self.space.at(self.x0)
        vals, vecs = self.space.eigh(self.space.hessian_to_space(h0))
        if vals[0] >= 0:
            return RunResult(
                frames=[ts],
                converged=False,
                error="the starting structure has no negative curvature",
                error_kind="input",
                ts_index=0,
            )
        self.ts_curvature = float(vals[0])
        v = vecs[:, 0]
        self.mode = v * np.sign(v[np.argmax(np.abs(v))])
        signs = {"forward": [1.0], "reverse": [-1.0], "both": [1.0, -1.0]}
        for sign in signs[opts.irc.direction]:
            self.branches.append(
                _Branch(
                    name="forward" if sign > 0 else "reverse",
                    sign=sign,
                    x=self.x0,
                    prev=ts,
                    h=h0.copy(),
                    next=self._along_mode(sign * opts.irc.step),
                    ds=opts.irc.step,
                )
            )
        self.progress.frames = lambda: self._assemble()[0]
        self.progress.ts_index = lambda: self._assemble()[1]
        while True:
            active = [b for b in self.branches if not b.done]
            if not active:
                break
            answers = self.count((yield [self._ask(b) for b in active]))
            for b, ans in zip(active, answers, strict=True):
                self._advance(b, ans)
        ends = (yield from self._minimize_ends()) if opts.ends.minimize else {}
        return self._result(ends)

    # ------------------------------------------------------------ one branch

    def _along_mode(self, length: float) -> Array:
        """The point ``length`` along the transition state's negative mode."""
        step = self.space.to_cartesian(length * self.mode)
        return self.x0 + step.reshape(self.n, 3)

    def _ask(self, b: _Branch) -> Ask:
        """The branch's next calculation: a gradient, or a Hessian when one is due."""
        every = self.opts.hessian.recompute_every
        b.asked_hessian = b.need_hessian or bool(
            every and b.since_hessian + 1 >= every and not b.skip_hessian
        )
        kind: Kind = "hessian" if b.asked_hessian else "gradient"
        return Ask(kind, b.next, lineage=b.name)

    def _advance(self, b: _Branch, ans: Answer) -> None:
        """Take a branch's answer: accept the step, retry it shorter, or end."""
        opts = self.opts
        if b.need_hessian:
            self._take_end_hessian(b, ans)
            return
        if not ans.success:
            if b.asked_hessian and not b.skip_hessian:
                # An optional Hessian failed: ask for the point's gradient alone,
                # and continue the branch on updated Hessians.
                b.hessian_failures += 1
                b.skip_hessian = True
                b.end_checks = MAX_END_CHECKS
                return
            b.steps += 1
            b.failure = ans
            b.finish("gradient_failed", str(ans.error))
            return
        b.steps += 1
        g_new = np.asarray(ans.gradient, dtype=float).ravel()
        dx = (np.asarray(ans.positions) - b.x).ravel()
        if self._overshot(b, ans, g_new, dx):
            if not b.frames:
                # The first step went past a valley: shorter along the mode.
                b.ds *= 0.5
                if b.ds < opts.irc.step * MIN_STEP_FRACTION:
                    b.finish("no_descent")
                else:
                    b.next = self._along_mode(b.sign * b.ds)
                return
            # Overshot the valley floor: learn the curvature from the point and retry
            # from the last one with half the step, until it is tiny.
            if ans.hessian is None:
                y = g_new - np.asarray(b.prev.gradient).ravel()
                b.h = self.update(b.h, dx, y)
            b.ds *= 0.5
            if b.ds < opts.irc.step * MIN_STEP_FRACTION:
                b.finish("energy_rose")
            else:
                b.next = self._next_point(b)
            return
        if ans.hessian is not None:
            b.h = np.asarray(ans.hessian, dtype=float)
            b.since_hessian = 0
        else:
            y = g_new - np.asarray(b.prev.gradient).ravel()
            b.h = self.update(b.h, dx, y)
            b.since_hessian += 1
        b.frames.append(ans)
        self._report(b.name, len(b.frames), ans)
        de_last = abs(float(ans.energy) - float(b.prev.energy))
        b.last_de = de_last
        step_mw = float(np.sqrt((self.m3 * dx * dx).sum()))
        tiny = step_mw < TINY_STEP_FRACTION * opts.irc.step
        b.tiny_steps = b.tiny_steps + 1 if tiny else 0
        b.x, b.prev = np.asarray(ans.positions, dtype=float).reshape(self.n, 3), ans
        if self._end_test(b, g_new, de_last, ans.hessian is not None):
            return
        if b.tiny_steps >= 3:  # the model sees no way down: the path ends here
            b.finish("stalled")
            return
        b.ds = min(opts.irc.step, 2.0 * b.ds)
        b.next = self._next_point(b)

    def _report(self, line: str | None, step: int, ans: Answer) -> None:
        """Report a point of the path (``report``), if anyone listens."""
        if self.report is None:
            return
        event: dict[str, Any] = {"calctype": "irc"}
        if line is not None:
            event["line"] = line
        g = np.asarray(ans.gradient, dtype=float).ravel() * self.space.free
        event.update(
            step=step, energy=float(ans.energy), g_max=float(max_atomic_norm(g))
        )
        self.report(event)

    def _take_end_hessian(self, b: _Branch, ans: Answer) -> None:
        """The last point again, for its exact Hessian: no step was taken."""
        b.need_hessian = False
        exact = ans.success and ans.hessian is not None
        if exact:
            b.h = np.asarray(ans.hessian, dtype=float)
            b.since_hessian = 0
            b.frames[-1] = b.prev = with_hessian(b.frames[-1], b.h)
        else:
            b.end_checks = MAX_END_CHECKS  # no exact Hessian: the flat rule may end it
            b.hessian_failures += 1
            b.skip_hessian = True
        if not self._end_test(b, np.asarray(b.prev.gradient), b.last_de, exact):
            b.next = self._next_point(b)

    def _overshot(self, b: _Branch, ans: Answer, g_new: Array, dx: Array) -> bool:
        """Whether a step went past the valley floor.

        Past the floor, the energy rose, or the new gradient points along the step (a
        step can jump a shallow basin and still land lower) while it is not yet
        small enough to end the branch. From the transition state a step across a
        valley is shortened; one that only ends higher is kept with the method's own
        Hessian (the structure is slightly off the saddle, and a shorter step would
        rise more), but shortened along a cheaper method's mode, which can point a
        little off the path.
        """
        self.space.at(b.x)
        small = max_atomic_norm(g_new * self.space.free) < self.opts.convergence.g_tol
        rose = ans.energy > b.prev.energy + 1e-10
        across = float(g_new @ dx) > 0.0 and not small
        if not b.frames:
            return across or (rose and not self.own)
        return rose or across

    def _next_point(self, b: _Branch) -> Array:
        """The next IRC point, ``b.ds`` along the path from the branch's last one."""
        space = self.space
        space.at(b.x)
        g_mw = space.project(
            space.gradient_to_space(np.asarray(b.prev.gradient).ravel())
        )
        hv, hvec = space.eigh(space.hessian_to_space(b.h))
        n_free = len(hv) - space.n_locked
        gt = hvec.T @ g_mw
        gt[n_free:] = 0.0
        if self.opts.irc.method == "gs2":
            sub = np.arange(n_free)  # locked directions carry no gradient and no step
            comp = np.zeros_like(gt)
            newton = -gt[sub] / hv[sub] if np.min(hv[:n_free]) > 0 else None
            if newton is not None and float(np.linalg.norm(newton)) <= b.ds:
                comp[sub] = newton  # the valley's minimum lies within this step
            else:
                comp[sub] = gs2_step(hv[sub], gt[sub], b.ds)
            step_mw = hvec @ comp
        else:
            hv = np.where(hv >= LOCKED_CURVATURE * 0.5, 0.0, hv)
            step_mw = hvec @ lqa_step(hv, gt, b.ds)
        return b.x + space.to_cartesian(space.project(step_mw)).reshape(self.n, 3)

    def _end_test(
        self, b: _Branch, gradient: Array, de_last: float, exact: bool
    ) -> bool:
        """Whether the branch ends (or waits for its exact Hessian) at its last point."""
        opts, space = self.opts, self.space
        space.at(b.x)
        hv, _ = space.eigh(space.hessian_to_space(b.h))
        n_free = len(hv) - space.n_locked
        g_cart = np.asarray(gradient, dtype=float).ravel() * space.free
        b.gmax = max_atomic_norm(g_cart)
        if b.gmax < opts.convergence.g_tol:
            if np.min(hv[:n_free]) > -END_CURVATURE:
                b.finish("converged")
                return True
            if b.end_checks < MAX_END_CHECKS and not exact:
                # Small gradient but the updated Hessian has negative curvature: a
                # flat shoulder of the path, or a minimum the updates blurred.
                # The exact Hessian here decides.
                b.end_checks += 1
                b.need_hessian = True
                b.next = b.x
                return True
            checked = b.end_checks >= MAX_END_CHECKS
            if checked and de_last < opts.convergence.e_tol:
                b.finish("converged_flat")
                return True
        if b.steps >= opts.convergence.max_iter:
            b.finish("max_iter")
            return True
        return False

    # ------------------------------------------------------------ the ends

    def _minimize_ends(self) -> Algorithm:
        """Minimize both ends together, each from its branch's updated Hessian."""
        opts = self.opts
        numbers = [int(getattr(pt, s.capitalize()).number) for s in self.symbols]
        end_options = OptimizationOptions(
            coordinate_system=opts.ends.coordinate_system,
            convergence=opts.ends.convergence,
            constraints=ConstraintSettings(frozen=opts.constraints.frozen),
            hessian=HessianSettings(initial="model"),
        )
        subs = {
            b.name: optimize(
                b.x,
                self.symbols,
                numbers,
                end_options,
                has_hessian_program=self.has_hessian_program,
                start=(b.prev, b.h),
                own_hessian=self.own,
                lineage=b.name,
                report=self.report,
            )
            for b in self.branches
            if b.frames and not b.gradient_failed
        }
        results = yield from lockstep(subs)  # type: ignore[arg-type]
        ends: dict[str, dict[str, Any]] = {}
        for name, res in results.items():
            b = next(b for b in self.branches if b.name == name)
            if isinstance(res, BaseException) or not res.frames:
                ends[name] = {"converged": False, "error": str(res)}
                continue
            ends[name] = {
                "converged": res.converged and res.error is None,
                "error": res.error,
                "n_gradients": res.info.get("n_gradients", 0),
                "energy_drop": float(b.prev.energy - res.frames[-1].energy),
            }
            self.count.gradients += res.info.get("n_gradients", 0)
            self.count.hessians += res.info.get("n_hessians", 0)
            if not np.array_equal(
                np.asarray(res.frames[-1].positions),
                np.asarray(b.frames[-1].positions),
            ):
                b.frames.append(res.frames[-1])
        return ends

    def _assemble(self) -> tuple[list[Answer], int]:
        """The path so far, reverse branch first, and the transition state's index."""
        by_name = {b.name: b for b in self.branches}
        rev = list(reversed(by_name["reverse"].frames)) if "reverse" in by_name else []
        fwd = by_name["forward"].frames if "forward" in by_name else []
        return rev + [self.ts] + fwd, len(rev)

    def _result(self, ends: dict[str, dict[str, Any]]) -> RunResult:
        """The path, whether every branch reached its end, and why not."""
        opts = self.opts
        frames, ts_index = self._assemble()
        failed = [b for b in self.branches if b.gradient_failed]
        unfinished: list[str] = []
        for b in self.branches:
            # "energy rose": even a sixteenth of a step passed the valley floor, so
            # the path has reached the bottom along its direction; it ends there if
            # its gradient is close to the end criterion.
            ended = b.ended or (
                b.reason in ("energy_rose", "stalled")
                and b.gmax < ENERGY_ROSE_GMAX * opts.convergence.g_tol
            )
            end = ends.get(b.name)
            if opts.ends.minimize and end is not None and not end["converged"]:
                why = end.get("error") or "did not converge"
                unfinished.append(f"the {b.name} end minimization failed: {why}")
            elif b.frames and float(b.frames[-1].energy) >= float(self.ts.energy):
                unfinished.append(
                    f"the {b.name} branch did not go downhill from the transition state"
                )
            elif not ended and not (opts.ends.minimize and end is not None):
                why = b.why()
                if b.reason in ("energy_rose", "stalled"):
                    why = f"{why} with the gradient at {b.gmax:.1e} Hartree/bohr"
                unfinished.append(f"the {b.name} branch stopped: {why}")
        info = {
            "n_gradients": self.count.gradients,
            "n_hessians": self.count.hessians,
            "branches": {b.name: b.end_record() for b in self.branches},
            "ends": ends,
            "ts_curvature": self.ts_curvature,
        }
        kind: ErrorKind | None = None
        if failed:
            problems = [
                child_failed(b.failure, f"on the {b.name} branch")
                for b in failed
                if b.failure is not None
            ]
            kind = "program"
        else:
            problems = unfinished
            kind = "convergence" if unfinished else None
        return RunResult(
            frames=frames,
            converged=not problems,
            error="; ".join(problems) if problems else None,
            error_kind=kind,
            reason="irc_incomplete" if kind == "convergence" else None,
            ts_index=ts_index,
            info=info,
            cause=failed[0].failure if failed else None,
        )
