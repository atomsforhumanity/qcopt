"""qcopt as an AFH program: ``qcopt.compute(inp)``.

qcopt is a program with subprograms (``qccompute.program``), registered as the
``qccompute.programs`` entry point ``qcopt``. Its Python interface is :func:`compute`;
many inputs at once go through a runner (``qccompute.compute`` with a list), which
calls :meth:`QcoptProgram.compute_batch` to run them in lockstep.

qcopt's settings are checked by :meth:`QcoptProgram.plan`. The gradients and Hessians
of every task come from sessions the runner opens (:mod:`qcopt.evaluators`): one per
task and program, warm-started from geometry to geometry, with each IRC branch its own
lineage. Called directly, qcopt hands itself to qccompute's default runner.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Sequence
from typing import Any

from qccompute.program import (
    Capabilities,
    Plan,
    Resources,
    RunContext,
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
    ProgramSpec,
    Resolved,
)
from qcref import constants as c
from qcref import periodic_table as pt

from .calculation import CALCTYPES, PROGRAM, Prepared, _version, prepare, run
from .errors import InputError
from .evaluators import evaluator, same_spec, unseen_axtb
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
        ctx: The runner's context; only a runner passes it. Without it qcopt hands
            itself to qccompute's default runner, which runs it on this machine.

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


#: qcopt's numerics epoch per calculation type. Bump a type's epoch whenever its golden
#: trajectories change (results for an unchanged input may change); never for
#: bit-identical changes.
NUMERICS_EPOCHS = {"optimization": 1, "transition_state": 1, "irc": 1}


@functools.cache
def _qcref_digest() -> str:
    """A short digest of the qcref data qcopt's numbers depend on.

    Atomic masses (mass-weighted IRCs, frequencies), covalent and van der Waals radii
    (internal-coordinate connectivity), and the constants of the frequency units.
    """
    elements = sorted((pt[key] for key in pt), key=lambda e: int(e.number))
    data = [
        [int(e.number)]
        + [
            None if v is None else repr(float(v))
            for v in (e.mass, e.cov_radius, e.vdw_radius)
        ]
        for e in elements
    ]
    data.append(
        [
            repr(float(c.phys.ELECTRON_MASS_IN_U)),
            repr(float(c.phys.HARTREE_INVERSE_METER_RELATIONSHIP)),
        ]
    )
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()[:12]


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
        """Minima, transition states and IRCs; inputs of a batch advance together.

        ``gpu``: its children may use GPUs (an ML potential), so the runner grants
        the GPUs its plan asks for (the children's) and passes them on in ``ctx``;
        a plan whose children use none gets none.
        """
        return Capabilities(
            calctypes=frozenset(CALCTYPES),
            sessions=False,
            batches=True,
            in_process=True,
            thread_safe=False,
            gpu=True,
        )

    def numerics(self, spec: ProgramInput | ProgramSpec) -> str:
        """What identifies qcopt's numbers for a calculation, for a store's reuse rule.

        The calculation type's epoch (:data:`NUMERICS_EPOCHS`) and a digest of the
        qcref data qcopt reads, such as ``"transition_state:1+qcref:1a2b3c4d5e6f"``. It
        changes whenever results for an unchanged input may change. The subprograms'
        numbers are their own programs' numerics, except axtb's where qcopt runs it in
        this process and no runner sees it (``"+axtb:0.1.1"``, :func:`unseen_axtb`).
        """
        calctype = CalcType(spec.calctype).value
        own = f"{calctype}:{NUMERICS_EPOCHS[calctype]}+qcref:{_qcref_digest()}"
        axtb = unseen_axtb(spec)
        return own if axtb is None else f"{own}+axtb:{axtb}"

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
            raise ProgramError.of(exc.kind, str(exc), code=exc.code) from None
        return Plan(
            resolved=Resolved(settings=p.resolved, reasons=p.reasons),
            resources=_plan_children(inp),
        )

    def compute(
        self, inp: ProgramInput, ctx: RunContext | None = None
    ) -> ProgramOutput[ProgramInput, Data]:
        """Run one calculation (:func:`compute`); called directly, on the default runner.

        Raises:
            ProgramError: It failed; ``output`` is the failed output.
        """
        if ctx is None:  # a direct call: the default runner runs it, with a ctx
            import qccompute

            return qccompute.compute(inp)
        out = self.compute_batch([inp], ctx)[0]
        if not out.success:
            raise _failure(out)
        return out

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext | None = None
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Run several calculations in lockstep: one round's children in one call.

        Called directly, the default runner runs them (and calls this with a ctx).

        Returns:
            One output per input, in order, failures in place (``success=False``,
            with ``error``): an input qcopt cannot run fails alone, with kind
            ``input``, and the others run.
        """
        if ctx is None:
            import qccompute

            return qccompute.compute(list(inputs), raise_exc=False)
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
            done = iter(run(prepared, ev, ctx.threads))
        finally:
            ev.close()
        return [out if out is not None else next(done) for out in outputs]


def _input_failure(
    inp: ProgramInput, exc: InputError
) -> ProgramOutput[ProgramInput, Data]:
    """The output of an input qcopt cannot run (nothing was computed)."""
    error = ErrorInfo(kind=exc.kind, message=str(exc), code=exc.code)
    return ProgramOutput.failure(inp, error)


def _plan_children(inp: ProgramInput) -> Resources:
    """Plan the installed children: their errors are the run's, their resources its own.

    A child's settings are checked here, before anything is computed (an unknown key, a
    setting that cannot take effect). Children run through qccompute's codecs are
    checked when they run. qcopt's own work is light: the resources are the largest
    child's (threads ``None``, the whole grant, if any child's are).

    When the Hessians come from the gradient program's own calculation, one session
    serves both, and a setting is valid if it applies to either (a session's rule): only
    the Hessian, which computes the energy and gradient too, is planned.

    Raises:
        ProgramError: Kind ``program``, with the child's error as the cause.
    """
    structure = inp.require_structure()
    threads: int | None = 1
    memory, gpus = None, 0
    hessians = [c for c in inp.subprograms if c.calctype == CalcType.hessian]
    for child in inp.subprograms:
        if child.calctype == CalcType.gradient and any(
            same_spec(h, child) for h in hessians
        ):
            continue
        program = find_program(child.program)
        if program is None:
            threads = _most(threads, int(child.keywords.get("threads", 1)))
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
        threads = _most(threads, res.threads)
        gpus = max(gpus, res.gpus)
        if res.memory_gib is not None:
            memory = max(memory or 0.0, res.memory_gib)
    return Resources(threads=threads, memory_gib=memory, gpus=gpus)


def _most(a: int | None, b: int | None) -> int | None:
    """The larger of two thread counts, where None (the whole grant) is the largest."""
    return None if a is None or b is None else max(a, b)


#: The object registered under the ``qccompute.programs`` entry point ``qcopt``.
program = QcoptProgram()
