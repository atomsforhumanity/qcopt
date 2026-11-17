"""qcopt as an AFH program: ``qcopt.compute(inp)``.

qcopt is a program with subprograms (``qccompute.program``), registered as the
``qccompute.programs`` entry point ``qcopt``. Its Python interface is :func:`compute`;
many inputs at once go through a runner (``qccompute.compute`` with a list), which
calls :meth:`QcoptProgram.compute_batch` to run them in lockstep.

qcopt's settings are checked by :meth:`QcoptProgram.plan`. The gradients and Hessians
of every task come from sessions the runner opens (:mod:`qcopt.evaluators`): one per
task and program, warm-started from geometry to geometry, with each IRC branch its own
lineage. Called directly, without a runner, qcopt runs its children in this process.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qccompute.program import (
    Capabilities,
    Plan,
    Resources,
    RunContext,
    Session,
    find_program,
    plan_of,
)
from qcdata import (
    CalcType,
    Data,
    ErrorInfo,
    ProgramError,
    ProgramInput,
    ProgramOutput,
    Resolved,
)

from .calculation import CALCTYPES, PROGRAM, Prepared, _version, prepare, run
from .errors import InputError
from .evaluators import evaluator, same_spec
from .options import options_schema


def compute(
    inp: ProgramInput, ctx: RunContext | None = None
) -> ProgramOutput[ProgramInput, Data]:
    """Run one qcopt calculation: a minimum, a transition state, or an IRC.

    Args:
        inp: A ``ProgramInput`` with ``program="qcopt"``, ``calctype``
            ``optimization``, ``transition_state`` or ``irc``, the starting structure,
            qcopt's settings as ``keywords`` (docs/settings.md), a ``gradient``
            subprogram and optionally a ``hessian`` one, and optionally ``given``
            results at the structure, whose Hessian becomes the starting one.
        ctx: The runner's context; only a runner passes it. Without it the children
            run in this process.

    Returns:
        The output: ``OptimizationData`` (minima and transition states) or
        ``PathData`` (IRCs), with ``resolved`` settings and the run's ``record``.

    Raises:
        ProgramError: The calculation failed or did not converge: ``error.kind`` is
            ``input``, ``program`` (a gradient or Hessian failed; its error is the
            ``cause``), ``convergence`` or ``numerical``. Its ``output`` is the failed
            output, with the frames computed so far.
    """
    return program.compute(inp, ctx)


def _failure(out: ProgramOutput[ProgramInput, Data]) -> ProgramError:
    """The exception for a failed output, which it carries as ``output``."""
    assert out.error is not None  # every failed qcopt output has one
    return ProgramError(out.error, output=out)


class QcoptProgram:
    """qcopt under the AFH program protocol (``qccompute.program.Program``)."""

    name = PROGRAM

    def version(self) -> str:
        """Qcopt's version."""
        return _version() or "unknown"

    def capabilities(self) -> Capabilities:
        """Minima, transition states and IRCs; inputs of a batch advance together."""
        return Capabilities(
            calctypes=frozenset(CALCTYPES),
            sessions=False,
            batches=True,
            in_process=True,
            thread_safe=False,
        )

    def settings_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The JSON Schema of the settings (``keywords``) of ``calctype``."""
        return options_schema(CalcType(calctype).value)

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The record's shape: diagnostics, not stable across versions."""
        return {"type": "object"}

    def plan(self, inp: ProgramInput) -> Plan:
        """Check the input and resolve every ``auto``; resources from the children.

        Raises:
            ProgramError: Kind ``input``: an input qcopt cannot run. Kind ``program``:
                an installed child cannot run its settings (its error is the cause).
        """
        try:
            p = prepare(inp)
        except InputError as exc:
            raise ProgramError.of("input", str(exc), code=exc.code) from None
        return Plan(
            resolved=Resolved(settings=p.resolved, reasons=p.reasons),
            resources=_plan_children(inp),
        )

    def compute(
        self, inp: ProgramInput, ctx: RunContext | None = None
    ) -> ProgramOutput[ProgramInput, Data]:
        """Run one calculation (:func:`compute`).

        Raises:
            ProgramError: It failed; ``output`` is the failed output.
        """
        out = self.compute_batch([inp], ctx)[0]
        if not out.success:
            raise _failure(out)
        return out

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext | None = None
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Run several calculations in lockstep: one round's children in one call.

        Returns:
            One output per input, in order, failures in place (``success=False``,
            with ``error``): an input qcopt cannot run fails alone, with kind
            ``input``, and the others run.
        """
        if ctx is None or (ctx.open_session is None and ctx.compute is None):
            ctx = _direct(None if ctx is None else ctx.threads)
        outputs: list[ProgramOutput[ProgramInput, Data] | None] = []
        prepared: list[Prepared] = []
        for inp in inputs:
            try:
                prepared.append(prepare(inp))
            except InputError as exc:
                outputs.append(_input_failure(inp, exc))
            else:
                outputs.append(None)
        if not prepared:
            return [out for out in outputs if out is not None]
        ev = evaluator([p.context for p in prepared], ctx)
        try:
            done = iter(run(prepared, ev))
        finally:
            ev.close()
        return [out if out is not None else next(done) for out in outputs]


def _direct(threads: int | None) -> RunContext:
    """The context of a direct call: children open their sessions as qccompute does.

    Each child program's own sessions when it has them, otherwise a session over
    ``qccompute.compute``, so every program qccompute runs works as a child.
    """
    import qccompute

    def open_session(inp: ProgramInput, share: int | None, calctypes: Any) -> Session:
        return qccompute.session(inp, threads=share, calctypes=calctypes)

    return RunContext(threads=threads, open_session=open_session)


def _input_failure(
    inp: ProgramInput, exc: InputError
) -> ProgramOutput[ProgramInput, Data]:
    """The output of an input qcopt cannot run (nothing was computed)."""
    error = ErrorInfo(kind="input", message=str(exc), code=exc.code)
    return ProgramOutput.failure(inp, error)


def _plan_children(inp: ProgramInput) -> Resources:
    """Plan the installed children: their errors are the run's, their resources its own.

    A child's settings are checked here, before anything is computed (an unknown key, a
    setting that cannot take effect). Children run through qccompute's codecs are
    checked when they run. qcopt's own work is light: the resources are the largest
    child's.

    When the Hessians come from the gradient program's own calculation, one session
    serves both, and a setting is valid if it applies to either (a session's rule): only
    the Hessian, which computes the energy and gradient too, is planned.

    Raises:
        ProgramError: Kind ``program``, with the child's error as the cause.
    """
    structure = inp.require_structure()
    threads, memory, gpus = 1, None, 0
    hessians = [c for c in inp.subprograms if c.calctype == CalcType.hessian]
    for child in inp.subprograms:
        if child.calctype == CalcType.gradient and any(
            same_spec(h, child) for h in hessians
        ):
            continue
        program = find_program(child.program)
        if program is None:
            threads = max(threads, int(child.keywords.get("threads", 1)))
            continue
        try:
            res = plan_of(program, ProgramInput.from_spec(child, structure)).resources
        except ProgramError as exc:
            raise ProgramError(
                ErrorInfo(
                    kind="program",
                    message=f"{child.program} ({child.calctype}): {exc.error.message}",
                    program=child.program,
                    cause=exc.error,
                )
            ) from None
        threads = max(threads, res.threads)
        gpus = max(gpus, res.gpus)
        if res.memory_gib is not None:
            memory = max(memory or 0.0, res.memory_gib)
    return Resources(threads=threads, memory_gib=memory, gpus=gpus)


#: The object registered under the ``qccompute.programs`` entry point ``qcopt``.
program = QcoptProgram()
