"""The optimizers as generators: minimization, transition-state search, IRC.

Each yields lists of :class:`~qcopt.driver.Ask` and returns a :class:`RunResult`.
Everything is in Cartesian coordinates (bohr, Hartree); see :mod:`qcopt.space` for
how rigid motion and frozen atoms are kept out of the steps.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
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
        for a in answers:
            if a.hessian is not None:
                self.hessians += 1
                if a.n_gradients > 1:
                    self.gradients += a.n_gradients
            else:
                self.gradients += 1
        return answers


def _hessian_source(
    kw: OptimizationKeywords | IRCKeywords, has_program: bool, ts: bool
) -> str:
    if kw.hessian != "auto":
        return kw.hessian
    if has_program:
        return "exact"
    return "finite_difference" if ts else "model"


def _converged(
    kw: OptimizationKeywords, gp: Array, step: Array | None, de: float | None
) -> bool:
    conv = kw.convergence
    if max_atomic_norm(gp) > conv.gmax:
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
) -> Algorithm:
    """Minimize (RS-RFO) or find a first-order saddle point (RS-P-RFO).

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
    """
    count = _Counter()
    n = len(symbols)
    x = np.asarray(positions, dtype=float).reshape(n, 3)
    cart = CartesianSpace(n, frozen=kw.frozen_atoms)
    coords = make_coordinates(
        kw.coordinates, symbols, x, kw.frozen_atoms, kw.trust_type == "atomic"
    )
    cons = (
        ConstraintSet(kw.constraints, x)
        if kw.constraints is not None and kw.constraints.size
        else None
    )

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
            hx = coords.ic.to_cartesian_hessian(x, hh)
            coords = CartesianCoordinates(n, kw.frozen_atoms, kw.trust_type == "atomic")
            coords.at(x)
            info["coordinates"] = f"{info['coordinates']} -> cartesian"
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
    source = _hessian_source(kw, has_hessian_program, transition_state)
    exact_available = source in ("exact", "finite_difference")
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
    trust = kw.trust
    rejections = 0
    failures = 0
    converged = False
    last_step: Array | None = None
    last_de: float | None = None
    steps_since_hessian = 0
    g = coords.gradient(x, g_raw)
    freqs = None
    escapes = 0
    iterations = 0
    while iterations < kw.maxiter:
        iterations += 1
        cart.at(x)
        if _converged(kw, _free(a.gradient, x), last_step, last_de) and _satisfied(x):
            if transition_state or kw.escape_saddles == 0:
                converged = True
                break
            # A minimization can stop on a saddle point, where the forces vanish too.
            (ha,) = count((yield [Ask("hessian", x)]))
            if not ha.success:
                converged = True
                break
            frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
            freqs = frequencies(
                np.asarray(ha.hessian), symbols, x, frozen=kw.frozen_atoms
            )
            if (
                freqs.n_imaginary(kw.imaginary_cutoff) == 0
                or escapes >= kw.escape_saddles
            ):
                converged = True
                break
            escapes += 1
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
            and exact_available
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
            if info["backtransform_failures"] > 20:
                coords = CartesianCoordinates(
                    n, kw.frozen_atoms, kw.trust_type == "atomic"
                )
                coords.at(x)
                h = hessians.lindh(numbers, x) + 1e-4 * np.eye(3 * n)
                g = coords.gradient(x, np.asarray(a.gradient, dtype=float))
                info["coordinates"] = "cartesian (fallback)"
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
        if (
            exact_available
            and kw.hessian_every
            and steps_since_hessian >= kw.hessian_every
        ):
            (ha,) = count((yield [Ask("hessian", x)]))
            if ha.success:
                h = coords.hessian(
                    x, np.asarray(ha.hessian), np.asarray(ha.gradient).ravel()
                )
                frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
                info["hessian_recomputes"] += 1
                steps_since_hessian = 0
    if not converged:
        cart.at(x)
        converged = _converged(
            kw, _free(a.gradient, x), last_step, last_de
        ) and _satisfied(x)

    if freqs is not None:
        info["n_imaginary"] = freqs.n_imaginary(kw.imaginary_cutoff)
    elif kw.frequencies:
        (ha,) = count((yield [Ask("hessian", x)]))
        if ha.success:
            frames[-1] = _with_hessian(frames[-1], np.asarray(ha.hessian))
            freqs = frequencies(
                np.asarray(ha.hessian), symbols, x, frozen=kw.frozen_atoms
            )
            info["n_imaginary"] = freqs.n_imaginary(kw.imaginary_cutoff)
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
    error = None if converged else f"not converged in {kw.maxiter} steps"
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
    )


# -------------------------------------------------------------------------- IRC


def _lqa_step(vals: Array, gt: Array, ds: float) -> Array:
    """Local quadratic approximation step of arc length ``ds`` (eigenbasis components).

    Page and McIver, J. Chem. Phys. 88, 922 (1988): the steepest-descent path on the
    quadratic surface, x_i(t) = g_i (exp(-l_i t) - 1) / l_i, followed until its arc
    length is ``ds``.
    """
    active = np.abs(gt) > 0
    lam, g = vals[active], gt[active]

    def speed(t: Array) -> Array:
        e = np.exp(np.clip(-np.outer(t, lam), -100, 100))
        return np.sqrt(((g * e) ** 2).sum(axis=1))

    gnorm = float(np.linalg.norm(g))
    if gnorm == 0.0:
        return np.zeros_like(gt)
    t_end = 2.0 * ds / gnorm
    for _ in range(60):
        ts = np.linspace(0.0, t_end, 501)
        v = speed(ts)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(ts))])
        if s[-1] >= ds:
            t = float(np.interp(ds, s, ts))
            break
        t_end *= 2.0
    else:
        t = t_end
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.abs(lam * t) < 1e-8
        comp = np.where(
            small,
            -g * t,
            g * np.expm1(np.clip(-lam * t, -100, 100)) / np.where(small, 1.0, lam),
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


def irc(
    positions: Array,
    symbols: Sequence[str],
    kw: IRCKeywords,
    *,
    has_hessian_program: bool,
) -> Algorithm:
    """Follow the IRC down both sides of a transition state.

    Mass-weighted Cartesian coordinates with rigid motion projected out, so the path
    stays in the input's frame. The first step goes along the Hessian's negative
    mode; later steps follow the local quadratic approximation on the current
    Hessian (exact at the transition state, then updated). Both branches advance
    together.
    """
    count = _Counter()
    n = len(symbols)
    m = masses(symbols)
    x0 = np.asarray(positions, dtype=float).reshape(n, 3)
    space = CartesianSpace(n, frozen=kw.frozen_atoms, masses=m)
    (ts,) = count((yield [Ask("hessian", x0)]))
    if not ts.success:
        return RunResult(
            frames=[ts],
            converged=False,
            error=f"Hessian at the transition state failed: {ts.error}",
        )
    h0 = np.asarray(ts.hessian, dtype=float)
    space.at(x0)
    vals, vecs = space.eigh(space.hessian_to_space(h0))
    if vals[0] >= 0:
        return RunResult(
            frames=[ts],
            converged=False,
            error="the starting structure has no negative curvature",
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
        )
        for sign in signs
    ]

    def next_point(b: _Branch) -> Array:
        """The LQA step of length ``b.ds`` from the branch's last point."""
        space.at(b.x)
        g_mw = space.project(
            space.gradient_to_space(np.asarray(b.prev.gradient).ravel())
        )
        hv, hvec = space.eigh(space.hessian_to_space(b.h))
        n_free = len(hv) - space.n_locked
        gt = hvec.T @ g_mw
        gt[n_free:] = 0.0
        hv = np.where(hv >= LOCKED_CURVATURE * 0.5, 0.0, hv)
        step_mw = hvec @ _lqa_step(hv, gt, b.ds)
        return b.x + space.to_cartesian(space.project(step_mw)).reshape(n, 3)

    while True:
        active = [b for b in branches if not b.done]
        if not active:
            break
        asks = []
        for b in active:
            want_h = bool(
                exact_available
                and kw.hessian_every
                and b.since_hessian + 1 >= kw.hessian_every
            )
            asks.append(Ask("hessian" if want_h else "gradient", b.next))
        answers = count((yield asks))
        for b, ans in zip(active, answers, strict=True):
            b.steps += 1
            if not ans.success:
                b.done, b.reason = True, f"gradient failed: {ans.error}"
                continue
            dx = (np.asarray(ans.positions) - b.x).ravel()
            # Past the valley floor: the energy rose, or the new gradient points
            # along the step (a step can jump a shallow basin and still land lower).
            overshot = ans.energy > b.prev.energy + 1e-10 or (
                float(np.asarray(ans.gradient, dtype=float).ravel() @ dx) > 0.0
            )
            if overshot and b.frames:
                # Overshot the valley floor: learn the curvature from the point
                # and retry from the last one with half the step, until it is tiny.
                if ans.hessian is None:
                    y = (
                        np.asarray(ans.gradient).ravel()
                        - np.asarray(b.prev.gradient).ravel()
                    )
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
                y = (
                    np.asarray(ans.gradient).ravel()
                    - np.asarray(b.prev.gradient).ravel()
                )
                b.h = update(b.h, dx, y)
                b.since_hessian += 1
            b.frames.append(ans)
            b.x, b.prev = np.asarray(ans.positions, dtype=float).reshape(n, 3), ans
            space.at(b.x)
            hv, _ = space.eigh(space.hessian_to_space(b.h))
            n_free = len(hv) - space.n_locked
            g_cart = np.asarray(ans.gradient, dtype=float).ravel() * space.free
            if max_atomic_norm(g_cart) < kw.gmax and np.min(hv[:n_free]) > -1e-5:
                b.done, b.reason = True, "converged"
                continue
            if b.steps >= kw.max_steps:
                b.done, b.reason = True, "max_steps"
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
            hessian="model",
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
                "converged": res.converged,
                "n_gradients": res.info.get("n_gradients", 0),
                "energy_drop": float(b.prev.energy - res.frames[-1].energy),
            }
            count.gradients += res.info.get("n_gradients", 0)
            if res.frames[-1] is not b.frames[-1]:
                b.frames.append(res.frames[-1])

    by_name = {b.name: b for b in branches}
    rev_frames = (
        list(reversed(by_name["reverse"].frames)) if "reverse" in by_name else []
    )
    fwd_frames = by_name["forward"].frames if "forward" in by_name else []
    frames = rev_frames + [ts] + fwd_frames
    reasons = {b.name: b.reason for b in branches}
    failed = [r for r in reasons.values() if r and r.startswith("gradient failed")]
    info = {
        "n_gradients": count.gradients,
        "n_hessians": count.hessians,
        "branch_end": reasons,
        "branch_steps": {b.name: len(b.frames) for b in branches},
        "end_minimization": end_info,
        "ts_imaginary_curvature": float(vals[0]),
    }
    return RunResult(
        frames=frames,
        converged=not failed,
        error=failed[0] if failed else None,
        ts_index=len(rev_frames),
        info=info,
    )
