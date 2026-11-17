"""Test helpers: qcopt inputs in a line, registered programs, the driver seam.

qcopt's interface is ``qcopt.compute(inp)``. Tests of what an algorithm does (steps,
rejections, Hessian schedules, failures of a child) often need a hand-made evaluator
instead: one that counts calls, breaks on the third, or returns noisy Hessians. They
run qcopt's own calculations on it through the seam between qcopt and its children
(:func:`run` with ``evaluator=``), the path ``compute`` takes with sessions.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
from qccompute.program import register_program, unregister_program
from qcdata import (
    ProgramError,
    ProgramInput,
    ProgramSpec,
    SinglePointData,
    Structure,
)

import qcopt
from qcopt.calculation import prepare
from qcopt.calculation import run as run_prepared
from qcopt.driver import Answer, Request, TaskContext
from qcopt.errors import InputError
from qcopt.program import program as qcopt_program


@contextmanager
def registered(*programs: Any) -> Iterator[None]:
    """Register programs for the block's duration."""
    for p in programs:
        register_program(p)
    try:
        yield
    finally:
        for p in programs:
            unregister_program(p.name)


class FunctionEvaluator:
    """Energies and gradients (optionally Hessians) from a Python function.

    A hand-made evaluator for the driver seam; ``qcopt.compute`` itself reaches
    programs only through sessions.

    Args:
        function: ``function(positions) -> (energy, gradient)``, as for
            ``qccompute.program.FunctionProgram``.
        hessian: ``hessian(positions) -> (3n, 3n)``, or None for finite differences
            of the gradient.
        name: Program name recorded as the frames' provenance.
    """

    def __init__(
        self,
        function: Callable[[Any], tuple[float, Any]],
        hessian: Callable[[Any], Any] | None = None,
        name: str = "function",
    ) -> None:
        """Wrap the functions."""
        self.function = function
        self.hessian = hessian
        self.name = name

    def supports_hessian(self, context: TaskContext) -> bool:
        """Whether a Hessian function was given."""
        return self.hessian is not None

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Evaluate each request."""
        out = []
        for r in requests:
            x = np.asarray(r.positions, dtype=float)
            try:
                e, g = self.function(x)
                h = self.hessian(x) if (r.kind == "hessian" and self.hessian) else None
                out.append(
                    Answer(
                        positions=x,
                        energy=float(e),
                        gradient=np.asarray(g, dtype=float).reshape(-1, 3),
                        hessian=None if h is None else np.asarray(h, dtype=float),
                        program=self.name,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported in the answer
                out.append(
                    Answer(
                        positions=x,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                        program=self.name,
                    )
                )
        return out


def qcopt_input(
    calctype: str,
    structure: Structure,
    program: ProgramSpec | dict[str, Any],
    hessian_program: ProgramSpec | dict[str, Any] | bool | None = None,
    given: SinglePointData | None = None,
    **keywords: Any,
) -> ProgramInput:
    """A qcopt input: gradients from ``program``, Hessians from ``hessian_program``.

    ``hessian_program=True`` computes Hessians with ``program`` itself.
    """
    gspec = dict(program.model_dump() if isinstance(program, ProgramSpec) else program)
    subprograms = [{**gspec, "calctype": "gradient"}]
    if hessian_program is True:
        subprograms.append({**gspec, "calctype": "hessian"})
    elif hessian_program:
        hspec = (
            hessian_program.model_dump()
            if isinstance(hessian_program, ProgramSpec)
            else hessian_program
        )
        subprograms.append({**hspec, "calctype": "hessian"})
    return ProgramInput(
        program="qcopt",
        calctype=calctype,
        structure=structure,
        keywords=keywords,
        subprograms=subprograms,
        given=given,
    )


def run(inputs: ProgramInput | Sequence[ProgramInput], evaluator: Any = None) -> Any:
    """Run qcopt inputs; a failed calculation is returned, not raised.

    Without an evaluator, through qcopt's program (``compute``, or ``compute_batch``
    for a list). With one, through the seam: the inputs are checked as ``compute``
    checks them, then run in lockstep on the evaluator. Either way an input qcopt
    cannot run raises ``ProgramError`` of kind ``input``.
    """
    single = isinstance(inputs, ProgramInput)
    items = [inputs] if isinstance(inputs, ProgramInput) else list(inputs)
    if evaluator is None:
        if single:
            try:
                return qcopt.compute(items[0])
            except ProgramError as exc:
                if exc.error.kind == "input":
                    raise
                return exc.output
        return qcopt_program.compute_batch(items)
    try:
        prepared = [prepare(inp) for inp in items]
    except InputError as exc:
        raise ProgramError.of("input", str(exc), code=exc.code) from None
    outs = run_prepared(prepared, evaluator)
    return outs[0] if single else outs


def minimize(
    structure: Structure, program: Any, *, evaluator: Any = None, **kw: Any
) -> Any:
    """An optimization to a minimum (:func:`qcopt_input` arguments as keywords)."""
    return run(qcopt_input("optimization", structure, program, **kw), evaluator)


def find_ts(
    structure: Structure, program: Any, *, evaluator: Any = None, **kw: Any
) -> Any:
    """A transition-state search (:func:`qcopt_input` arguments as keywords)."""
    return run(qcopt_input("transition_state", structure, program, **kw), evaluator)


def irc(structure: Structure, program: Any, *, evaluator: Any = None, **kw: Any) -> Any:
    """An IRC from a transition state (:func:`qcopt_input` arguments as keywords)."""
    return run(qcopt_input("irc", structure, program, **kw), evaluator)
