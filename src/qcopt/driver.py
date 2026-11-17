"""Runs many algorithms together, batching every gradient and Hessian they ask for.

An algorithm is a generator. It yields a list of :class:`Ask` (geometries where it
needs a gradient or a Hessian) and receives one :class:`Answer` per ask; when it
returns, its return value is its result. :func:`drive` advances every algorithm one
round at a time and hands all their asks to the evaluator in one call, so separate
optimizations, both branches of an IRC, and the displaced gradients of a
finite-difference Hessian are computed together (batched, or side by side).
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Generator, Hashable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import numpy as np

from .geometry import Array

Kind = Literal["gradient", "hessian"]


@dataclass
class Ask:
    """A calculation an algorithm needs.

    Attributes:
        kind: ``gradient`` (energy and gradient) or ``hessian`` (energy, gradient,
            and Hessian).
        positions: ``(n_atoms, 3)`` geometry in bohr.
        lineage: Which earlier calculations of the task this one continues (an IRC
            branch's name), so a program can start from the wavefunction at the
            previous geometry of the same line; None: the task's main line.
    """

    kind: Kind
    positions: Array
    lineage: Hashable | None = None


@dataclass
class Answer:
    """What was computed for an :class:`Ask`.

    Attributes:
        positions: The geometry (bohr).
        energy: Energy (Hartree).
        gradient: ``(n_atoms, 3)`` gradient (Hartree/bohr).
        hessian: ``(3N, 3N)`` Hessian (Hartree/bohr^2) for Hessian asks.
        success: Whether the calculation succeeded.
        error: Why it failed, in one line: the program's own message.
        error_kind: The failure's kind when the evaluator or the driver knows it and
            the output does not carry the program's own error (``numerical`` for a
            non-finite result); None: ``internal``.
        calctype: For a failure, the calculation that failed (``gradient`` or
            ``hessian``): a Hessian paired with a separate program's gradient fails
            in either.
        traceback: The failure's traceback or log, when there is one.
        output: The evaluator's record of the calculation (a qcdata
            ``ProgramOutput``, a session's ``Frame``, the program's raw result, or a
            ``qcdata.ErrorInfo``), used to build frames; its ``error`` is the
            program's own.
        program: The program that computed it.
        program_version: Its version.
        n_gradients: Gradient calculations this answer took, as the driver counts
            them for the algorithm (finite differences: all displaced gradients; a
            Hessian paired with a gradient: 1 when that gradient was computed for it,
            0 when it was reused; 0 when the evaluator itself failed).
        n_hessians: Hessian calculations it took (counted the same way).
        hessian_answer: For a Hessian from a separate program (a different program
            or method than the gradients), that program's own answer; the energy,
            gradient and output above are then the gradient program's.
    """

    positions: Array
    energy: float = float("nan")
    gradient: Array | None = None
    hessian: Array | None = None
    success: bool = True
    error: str | None = None
    error_kind: str | None = None
    calctype: str | None = None
    traceback: str | None = None
    output: Any = None
    program: str | None = None
    program_version: str | None = None
    n_gradients: int = 1
    n_hessians: int = 0
    hessian_answer: Answer | None = None


@dataclass
class Request:
    """An ask bound to the calculation it belongs to, as an evaluator receives it.

    Attributes:
        task: The task (calculation) index; requests of one task come in order and
            may share state, such as a wavefunction to start from.
        kind: ``gradient`` or ``hessian``.
        positions: Geometry in bohr.
        context: The task's :class:`TaskContext`.
        lineage: The ask's lineage (:class:`Ask`).
        commit: Whether the calculation becomes its lineage's history. The displaced
            gradients of a finite-difference Hessian do not: each starts from the
            centre's, as if it were the lineage's last.
    """

    task: int
    kind: Kind
    positions: Array
    context: TaskContext
    lineage: Hashable | None = None
    commit: bool = True


@dataclass
class TaskContext:
    """What an evaluator needs to know about a task's system and programs.

    Attributes:
        symbols: Element symbols.
        charge: Total charge.
        multiplicity: Spin multiplicity.
        gradient_spec: The qcdata ``ProgramSpec`` computing gradients.
        hessian_spec: The ``ProgramSpec`` computing Hessians, or None.
        fd_step: Finite-difference step (bohr) when no Hessian program is given.
        separate_hessian: The Hessian program differs from the gradient program (in
            program or method). Each Hessian ask is then answered with the gradient
            program's energy and gradient and the Hessian program's Hessian.
        extras: Anything else an evaluator may use.
    """

    symbols: list[str]
    charge: int = 0
    multiplicity: int = 1
    gradient_spec: Any = None
    hessian_spec: Any = None
    fd_step: float = 0.005
    separate_hessian: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    """Computes requests, in any order or all at once; returns answers in order."""

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Compute the requests; failures are answers with ``success=False``."""
        ...

    def supports_hessian(self, context: TaskContext) -> bool:
        """Whether Hessians can be computed directly (else finite differences).

        An evaluator may also define ``hessian_includes_gradient(context)``; when it
        returns False, each Hessian is paired with a gradient computed (or just
        computed) by the gradient program. It may define ``forget(task)``, called
        when a task's algorithm has finished, to release what it kept for the task.
        """
        ...


Algorithm = Generator[list[Ask], list[Answer], Any]


def _fd_points(x: Array, h: float) -> list[Array]:
    pts = []
    flat = np.asarray(x, dtype=float).ravel()
    for i in range(flat.size):
        for sign in (1.0, -1.0):
            y = flat.copy()
            y[i] += sign * h
            pts.append(y.reshape(-1, 3))
    return pts


def _fd_hessian(
    center: Answer | None, answers: list[Answer], x: Array, h: float
) -> Answer:
    n = x.size
    failed = [a for a in [*([center] if center else []), *answers] if not a.success]
    if failed:  # the first failed gradient, as the Hessian's failure
        return replace(
            failed[0],
            positions=x,
            n_gradients=len(answers) + (1 if center else 0),
            hessian_answer=None,
        )
    hess = np.zeros((n, n))
    for i in range(n):
        gp = np.asarray(answers[2 * i].gradient).ravel()
        gm = np.asarray(answers[2 * i + 1].gradient).ravel()
        hess[:, i] = (gp - gm) / (2.0 * h)
    hess = 0.5 * (hess + hess.T)
    base = center if center is not None else answers[0]
    return Answer(
        positions=x,
        energy=base.energy if center is not None else float("nan"),
        gradient=None if center is None else center.gradient,
        hessian=hess,
        output=None if center is None else center.output,
        program=base.program,
        program_version=base.program_version,
        n_gradients=len(answers) + (1 if center is not None else 0),
    )


def _finite(request: Request, answer: Answer, hessian_only: bool) -> Answer:
    """A successful answer that is incomplete or not finite becomes a failed one.

    ``hessian_only``: the Hessian part of a pair, whose energy and gradient come from
    the gradient program; every other answer needs a finite energy and gradient.
    """
    if not answer.success:
        return replace(answer, calctype=request.kind)
    needed: list[tuple[str, Any]] = []
    if not hessian_only:
        needed += [("energy", answer.energy), ("gradient", answer.gradient)]
    if request.kind == "hessian":
        needed.append(("Hessian", answer.hessian))
    for what, value in needed:
        if value is None:
            return replace(
                answer,
                success=False,
                error=f"no {what} computed",
                calctype=request.kind,
            )
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            return replace(
                answer,
                success=False,
                error=f"non-finite {what}",
                error_kind="numerical",
                calctype=request.kind,
            )
    return answer


def _hessian_brings_gradient(evaluator: Any, context: TaskContext) -> bool:
    """Whether the evaluator's Hessian answers carry the energy and gradient too."""
    check = getattr(evaluator, "hessian_includes_gradient", None)
    return True if check is None else bool(check(context))


def _remember(cache: dict[bytes, Answer], x: Array, answer: Answer) -> None:
    """Keep ``answer`` as the gradient at ``x``; only the latest few are kept."""
    cache[x.tobytes()] = answer
    while len(cache) > 4:
        cache.pop(next(iter(cache)))


def _combine(grad: Answer, hess: Answer, x: Array, fresh: bool) -> Answer:
    """The gradient program's answer with the separate Hessian program's Hessian."""
    for part in (grad, hess):
        if not part.success:  # the failed calculation, as the pair's failure
            return replace(
                part, positions=x, n_gradients=int(fresh), hessian_answer=hess
            )
    return Answer(
        positions=x,
        energy=grad.energy,
        gradient=grad.gradient,
        hessian=hess.hessian,
        output=grad.output,
        program=grad.program,
        program_version=grad.program_version,
        n_gradients=int(fresh),
        hessian_answer=hess,
    )


class _Plan:
    """How one ask is computed: the requests it adds to the round, and its reply.

    Subclasses: :class:`_Direct` (the evaluator computes it as asked),
    :class:`_Paired` (a Hessian from a separate program, with the gradient
    program's energy and gradient), :class:`_FiniteDifference` (a Hessian from
    displaced gradients), :class:`_Unsupported`.
    """

    def __init__(self, ask: Ask, x: Array) -> None:
        """Bind the ask and its geometry."""
        self.ask = ask
        self.x = x

    def reply(self, answers: list[Answer], recent: dict[bytes, Answer]) -> Answer:
        """The answer for the algorithm, from this round's answers."""
        raise NotImplementedError


class _Direct(_Plan):
    def __init__(self, ask: Ask, x: Array, index: int) -> None:
        super().__init__(ask, x)
        self.index = index

    def reply(self, answers: list[Answer], recent: dict[bytes, Answer]) -> Answer:
        reply = answers[self.index]
        if self.ask.kind == "gradient" and reply.success:
            _remember(recent, self.x, reply)
        hess = int(self.ask.kind == "hessian")
        return replace(reply, n_gradients=1 - hess, n_hessians=hess)


class _Paired(_Plan):
    """A separate program's Hessian with the gradient program's energy and gradient.

    The gradient is computed alongside, unless the task computed it at this
    geometry just before (``known``).
    """

    def __init__(
        self, ask: Ask, x: Array, gradient: int | Answer, hessian: int
    ) -> None:
        super().__init__(ask, x)
        self.gradient = gradient
        self.hessian = hessian

    def reply(self, answers: list[Answer], recent: dict[bytes, Answer]) -> Answer:
        fresh = isinstance(self.gradient, int)
        grad = (
            answers[self.gradient] if isinstance(self.gradient, int) else self.gradient
        )
        if fresh and grad.success:
            _remember(recent, self.x, grad)
        reply = _combine(grad, answers[self.hessian], self.x, fresh)
        return replace(reply, n_gradients=int(fresh), n_hessians=1)


class _FiniteDifference(_Plan):
    """A Hessian from central differences of gradients, with the center's gradient."""

    def __init__(self, ask: Ask, x: Array, start: int, stop: int, step: float) -> None:
        super().__init__(ask, x)
        self.start, self.stop, self.step = start, stop, step

    def reply(self, answers: list[Answer], recent: dict[bytes, Answer]) -> Answer:
        center = answers[self.start]
        reply = _fd_hessian(
            center, answers[self.start + 1 : self.stop], self.x, self.step
        )
        return replace(
            reply, n_gradients=self.stop - self.start, n_hessians=int(reply.success)
        )


class _Unsupported(_Plan):
    def reply(self, answers: list[Answer], recent: dict[bytes, Answer]) -> Answer:
        return Answer(
            positions=self.x,
            success=False,
            error=(
                "the evaluator cannot compute Hessians with the Hessian subprogram, "
                "which differs from the gradient one"
            ),
            error_kind="unsupported",
            calctype="hessian",
            n_gradients=0,
        )


@dataclass
class _Round:
    """One round's requests for the evaluator, and each task's plans."""

    requests: list[Request] = field(default_factory=list)
    plans: dict[int, list[_Plan]] = field(default_factory=dict)
    hessian_only: set[int] = field(default_factory=set)  # Hessian halves of pairs

    def add(self, request: Request) -> int:
        """Append a request; returns its index."""
        self.requests.append(request)
        return len(self.requests) - 1


class _Strategy:
    """For each task: how its Hessians are computed by this evaluator."""

    def __init__(self, evaluator: Evaluator, contexts: Sequence[TaskContext]) -> None:
        self.contexts = list(contexts)
        self.direct = [evaluator.supports_hessian(ctx) for ctx in contexts]
        # A Hessian is paired with the gradient program's energy and gradient when it
        # comes from another program, or when the evaluator's Hessians come without
        # them.
        self.paired = [
            ctx.separate_hessian or (d and not _hessian_brings_gradient(evaluator, ctx))
            for ctx, d in zip(contexts, self.direct, strict=True)
        ]

    def plan(
        self, rnd: _Round, task: int, ask: Ask, recent: dict[bytes, Answer]
    ) -> _Plan:
        """Add the requests for one ask to the round; returns its plan."""
        ctx = self.contexts[task]
        x = np.asarray(ask.positions, dtype=float).reshape(-1, 3)
        if ask.kind == "hessian" and self.paired[task]:
            if not self.direct[task]:
                return _Unsupported(ask, x)
            known = recent.get(x.tobytes())
            line = ask.lineage
            gradient: int | Answer = (
                rnd.add(Request(task, "gradient", x, ctx, line))
                if known is None
                else known
            )
            rnd.hessian_only.add(len(rnd.requests))
            hessian = rnd.add(Request(task, "hessian", x, ctx, line))
            return _Paired(ask, x, gradient, hessian)
        if ask.kind == "hessian" and not self.direct[task]:
            start = rnd.add(Request(task, "gradient", x, ctx, ask.lineage))
            for y in _fd_points(x, ctx.fd_step):
                rnd.add(Request(task, "gradient", y, ctx, ask.lineage, commit=False))
            return _FiniteDifference(ask, x, start, len(rnd.requests), ctx.fd_step)
        return _Direct(ask, x, rnd.add(Request(task, ask.kind, x, ctx, ask.lineage)))


def _evaluate(
    evaluator: Evaluator, rnd: _Round, timings: dict[str, float] | None
) -> tuple[list[Answer], bool]:
    """The round's answers, checked; True when the evaluator itself failed."""
    t0 = time.perf_counter()
    crashed = False
    requests = rnd.requests
    try:
        answers = list(evaluator(requests)) if requests else []
        if len(answers) != len(requests):
            raise RuntimeError(f"{len(answers)} answers for {len(requests)} requests")
    except Exception as exc:  # noqa: BLE001 - every request of the round failed
        crashed = True
        error = f"{type(exc).__name__}: {exc}"
        tb = "".join(traceback.format_exception(exc))
        # a ProgramError's own error is the cause; any other exception is a bug
        own = getattr(exc, "error", None)
        answers = [
            Answer(
                positions=r.positions,
                success=False,
                error=error,
                traceback=tb,
                output=own,
            )
            for r in requests
        ]
    if timings is not None:
        timings["subprograms"] = timings.get("subprograms", 0.0) + (
            time.perf_counter() - t0
        )
    answers = [
        _finite(r, a, k in rnd.hessian_only)
        for k, (r, a) in enumerate(zip(requests, answers, strict=True))
    ]
    return answers, crashed


def _start(alg: Algorithm) -> tuple[list[Ask] | None, Any]:
    """Prime an algorithm: its first asks, or (None, its result)."""
    try:
        return next(alg), None
    except StopIteration as stop:
        return None, stop.value
    except Exception as exc:  # noqa: BLE001 - reported as the task's result
        return None, exc


def _send(alg: Algorithm, replies: list[Answer]) -> tuple[list[Ask] | None, Any]:
    """Answer an algorithm's asks: its next asks, or (None, its result)."""
    try:
        return alg.send(replies), None
    except StopIteration as stop:
        return None, stop.value
    except Exception as exc:  # noqa: BLE001 - reported as the task's result
        return None, exc


def drive(
    algorithms: Sequence[Algorithm],
    contexts: Sequence[TaskContext],
    evaluator: Evaluator,
    *,
    timings: dict[str, float] | None = None,
) -> list[Any]:
    """Run algorithms to completion, computing each round's asks in one call.

    Args:
        algorithms: Started (not yet primed) generators.
        contexts: The task context of each algorithm.
        evaluator: Computes the requests.
        timings: If given, ``timings["subprograms"]`` accumulates the seconds spent in
            the evaluator.

    Returns:
        Each algorithm's return value, in order. An algorithm that raises has the
        exception as its result.
    """
    results: list[Any] = [None] * len(algorithms)
    pending: dict[int, list[Ask]] = {}
    for i, alg in enumerate(algorithms):
        asks, results[i] = _start(alg)
        if asks is not None:
            pending[i] = asks
    strategy = _Strategy(evaluator, contexts)
    # Each task's latest gradient answers by geometry, so a Hessian from a separate
    # program at a point just computed does not compute its gradient again.
    recent: list[dict[bytes, Answer]] = [{} for _ in algorithms]
    while pending:
        rnd = _Round()
        for i, asks in pending.items():
            rnd.plans[i] = [strategy.plan(rnd, i, ask, recent[i]) for ask in asks]
        answers, crashed = _evaluate(evaluator, rnd, timings)
        nxt: dict[int, list[Ask]] = {}
        for i, plans in rnd.plans.items():
            replies = [plan.reply(answers, recent[i]) for plan in plans]
            if crashed:  # nothing was computed
                replies = [replace(r, n_gradients=0, n_hessians=0) for r in replies]
            asks, results[i] = _send(algorithms[i], replies)
            if asks is not None:
                nxt[i] = asks
            else:
                _forget(evaluator, i)
        pending = nxt
    return results


def _forget(evaluator: Any, task: int) -> None:
    """Tell the evaluator a task is finished, if it wants to know."""
    forget = getattr(evaluator, "forget", None)
    if forget is not None:
        forget(task)
