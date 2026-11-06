"""Runs many algorithms together, batching every gradient and Hessian they ask for.

An algorithm is a generator. It yields a list of :class:`Ask` (geometries where it
needs a gradient or a Hessian) and receives one :class:`Answer` per ask; when it
returns, its return value is its result. :func:`drive` advances every algorithm one
round at a time and hands all their asks to the evaluator in one call, so separate
optimizations, both branches of an IRC, and the displaced gradients of a
finite-difference Hessian are computed together (batched, or side by side).
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
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
    """

    kind: Kind
    positions: Array


@dataclass
class Answer:
    """What was computed for an :class:`Ask`.

    Attributes:
        positions: The geometry (bohr).
        energy: Energy (Hartree).
        gradient: ``(n_atoms, 3)`` gradient (Hartree/bohr).
        hessian: ``(3N, 3N)`` Hessian (Hartree/bohr^2) for Hessian asks.
        success: Whether the calculation succeeded.
        error: Why it failed.
        output: The evaluator's record of the calculation (a qcdata
            ``ProgramOutput`` or the program's raw result), used to build frames.
        program: The program that computed it.
        program_version: Its version.
        n_gradients: Gradient calculations this answer took (finite differences;
            for a Hessian from a separate program, 1 when the gradient program's
            gradient was computed for it and 0 when it was reused).
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
    output: Any = None
    program: str | None = None
    program_version: str | None = None
    n_gradients: int = 1
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
    """

    task: int
    kind: Kind
    positions: Array
    context: TaskContext


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
        """Whether Hessians can be computed directly (else finite differences)."""
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
    if failed:
        return Answer(
            positions=x,
            success=False,
            error=f"finite-difference Hessian: {failed[0].error}",
            n_gradients=len(answers),
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


def _remember(cache: dict[bytes, Answer], x: Array, answer: Answer) -> None:
    """Keep ``answer`` as the gradient at ``x``; only the latest few are kept."""
    cache[x.tobytes()] = answer
    while len(cache) > 4:
        cache.pop(next(iter(cache)))


def _combine(grad: Answer, hess: Answer, x: Array, fresh: bool) -> Answer:
    """The gradient program's answer with the separate Hessian program's Hessian."""
    for part, what in ((grad, "gradient"), (hess, "Hessian")):
        if not part.success:
            return Answer(
                positions=x,
                success=False,
                error=f"{what} failed: {part.error}",
                output=part.output,
                program=part.program,
                program_version=part.program_version,
                n_gradients=int(fresh),
                hessian_answer=hess,
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


def drive(
    algorithms: Sequence[Algorithm],
    contexts: Sequence[TaskContext],
    evaluator: Evaluator,
) -> list[Any]:
    """Run algorithms to completion, computing each round's asks in one call.

    Args:
        algorithms: Started (not yet primed) generators.
        contexts: The task context of each algorithm.
        evaluator: Computes the requests.

    Returns:
        Each algorithm's return value, in order. An algorithm that raises has the
        exception as its result.
    """
    results: list[Any] = [None] * len(algorithms)
    pending: dict[int, list[Ask]] = {}
    for i, alg in enumerate(algorithms):
        try:
            pending[i] = next(alg)
        except StopIteration as stop:
            results[i] = stop.value
        except Exception as exc:  # noqa: BLE001 - reported as the task's result
            results[i] = exc
    direct = [evaluator.supports_hessian(ctx) for ctx in contexts]
    # Each task's latest gradient answers by geometry, so a Hessian from a separate
    # program at a point just computed does not compute its gradient again.
    recent: list[dict[bytes, Answer]] = [{} for _ in algorithms]
    while pending:
        requests: list[Request] = []
        plans: dict[int, list[tuple[str, int, int, Ask]]] = {}
        reused: dict[tuple[int, int], Answer] = {}
        for i, asks in pending.items():
            ctx = contexts[i]
            plan = []
            for ask in asks:
                x = np.asarray(ask.positions, dtype=float).reshape(-1, 3)
                start = len(requests)
                if ask.kind == "hessian" and ctx.separate_hessian:
                    if not direct[i]:
                        plan.append(("unsupported", start, start, ask))
                        continue
                    known = recent[i].get(x.tobytes())
                    if known is None:
                        requests.append(Request(i, "gradient", x, ctx))
                    else:
                        reused[(i, start)] = known
                    requests.append(Request(i, "hessian", x, ctx))
                    plan.append(("split", start, len(requests), ask))
                elif ask.kind == "hessian" and not direct[i]:
                    requests.append(Request(i, "gradient", x, ctx))
                    for y in _fd_points(x, ctx.fd_step):
                        requests.append(Request(i, "gradient", y, ctx))
                    plan.append(("fd", start, len(requests), ask))
                else:
                    requests.append(Request(i, ask.kind, x, ctx))
                    plan.append(("direct", start, len(requests), ask))
            plans[i] = plan
        answers = evaluator(requests) if requests else []
        nxt: dict[int, list[Ask]] = {}
        for i, plan in plans.items():
            replies = []
            for how, a, b, ask in plan:
                x = np.asarray(ask.positions, dtype=float).reshape(-1, 3)
                if how == "direct":
                    reply = answers[a]
                    if ask.kind == "gradient" and reply.success:
                        _remember(recent[i], x, reply)
                    replies.append(reply)
                elif how == "split":
                    fresh = b - a == 2
                    grad = answers[a] if fresh else reused[(i, a)]
                    if fresh and grad.success:
                        _remember(recent[i], x, grad)
                    replies.append(_combine(grad, answers[b - 1], x, fresh))
                elif how == "unsupported":
                    replies.append(
                        Answer(
                            positions=x,
                            success=False,
                            error=(
                                "the evaluator cannot compute Hessians with the "
                                "Hessian subprogram, which differs from the gradient one"
                            ),
                            n_gradients=0,
                        )
                    )
                else:
                    center = answers[a]
                    replies.append(
                        _fd_hessian(center, answers[a + 1 : b], x, contexts[i].fd_step)
                    )
            try:
                nxt[i] = algorithms[i].send(replies)
            except StopIteration as stop:
                results[i] = stop.value
            except Exception as exc:  # noqa: BLE001
                results[i] = exc
        pending = nxt
    return results
