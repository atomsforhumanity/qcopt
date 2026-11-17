"""What every algorithm shares: its result, cost counting, failure handling, nesting."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..driver import Algorithm, Answer, Ask
from ..errors import ErrorKind
from ..geometry import Array
from ..vibrations import Frequencies

#: Receives an algorithm's step events while it runs (small JSON objects; see
#: :func:`qcopt.calculation.run`).
Report = Callable[[dict[str, Any]], None]


@dataclass
class RunResult:
    """What an algorithm produced.

    Attributes:
        frames: The answers kept as frames, in order.
        converged: Whether it met its convergence criteria.
        error: Why it failed (None on success).
        error_kind: The kind of failure (:data:`qcopt.errors.ErrorKind`).
        reason: A short code for a ``convergence`` failure, such as ``max_iter``
            or ``not_first_order_saddle``.
        ts_index: For an IRC, the transition state's position in ``frames``.
        info: What happened (counts and diagnostics), the result's ``record``.
        frequencies: Harmonic analysis of the final structure, if computed.
        cause: For a ``program`` failure, the failed calculation's answer.
    """

    frames: list[Answer]
    converged: bool
    error: str | None = None
    error_kind: ErrorKind | None = None
    reason: str | None = None
    ts_index: int | None = None
    info: dict[str, Any] = field(default_factory=dict)
    frequencies: Frequencies | None = None
    cause: Answer | None = None


class Counter:
    """Counts the gradients and Hessians an algorithm received.

    The driver records what each answer cost (finite-difference Hessians count all
    their gradients; a gradient reused for a Hessian counts none).
    """

    def __init__(self) -> None:
        """Start at zero."""
        self.gradients = 0
        self.hessians = 0

    def __call__(self, answers: list[Answer]) -> list[Answer]:
        """Count ``answers`` and pass them on."""
        for a in answers:
            self.gradients += a.n_gradients
            self.hessians += a.n_hessians
        return answers


@dataclass
class Progress:
    """What an algorithm has done so far, for its result if it raises."""

    frames: Callable[[], list[Answer]] = list
    info: dict[str, Any] = field(default_factory=dict)
    ts_index: Callable[[], int | None] = lambda: None
    count: Counter | None = None


def guarded(body: Callable[[Progress], Algorithm]) -> Algorithm:
    """Run an algorithm; an exception inside it becomes a failed result with its frames.

    A linear-algebra or floating-point failure is a ``numerical`` error; anything else
    an ``internal`` one (a bug).
    """
    progress = Progress()
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
        numerical = isinstance(exc, np.linalg.LinAlgError | FloatingPointError)
        return RunResult(
            frames=progress.frames(),
            converged=False,
            error="".join(traceback.format_exception(exc)),
            error_kind="numerical" if numerical else "internal",
            ts_index=progress.ts_index(),
            info=progress.info,
        )


def child_failed(answer: Answer, where: str) -> str:
    """A failed gradient or Hessian calculation, as AFH's drivers report a child's.

    The program and calculation type first (``linus (gradient): ...``), then the
    program's own message, then where in the run it happened.
    """
    program = answer.program or "the evaluator"
    calctype = answer.calctype or "calculation"
    return f"{program} ({calctype}): {str(answer.error).rstrip('.')} ({where})"


def program_failed(
    frames: list[Answer],
    answer: Answer,
    where: str,
    info: dict[str, Any] | None = None,
) -> RunResult:
    """The result of a run stopped by a failed gradient or Hessian calculation."""
    return RunResult(
        frames=frames,
        converged=False,
        error=child_failed(answer, where),
        error_kind="program",
        info={} if info is None else info,
        cause=answer,
    )


def resolve_initial_hessian(
    initial: str, has_program: bool, ts: bool, given: bool = False
) -> str:
    """What ``hessian.initial="auto"`` means for this run, else ``initial``.

    ``given``: the input has a given Hessian (``ProgramInput.given``).
    """
    if initial != "auto":
        return initial
    if given:
        return "given"
    if has_program:
        return "exact"
    return "finite_difference" if ts else "model"


def with_hessian(answer: Answer, hessian: Array) -> Answer:
    """``answer`` holding ``hessian`` (an exact one computed at its geometry later)."""
    return replace(answer, hessian=np.asarray(hessian, dtype=float))


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
