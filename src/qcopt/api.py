"""Running qcopt: ``ProgramInput`` in, ``ProgramOutput`` out.

A qcopt calculation is a qcdata ``ProgramInput`` with ``program="qcopt"``, a
calculation type (``optimization``, ``transition_state`` or ``irc``), a structure,
qcopt's keywords (:mod:`qcopt.keywords`), and subprograms: one ``ProgramSpec`` with
``calctype="gradient"``, and optionally one with ``calctype="hessian"``. The result is
a ``ProgramOutput`` holding ``OptimizationData`` (minimizations and transition states)
or ``PathData`` (IRCs), every frame with the energy and gradient computed there.
"""

from __future__ import annotations

import socket
import time
import traceback
from collections.abc import Sequence
from dataclasses import replace
from importlib import metadata
from typing import Any, overload

import numpy as np
from qcdata import (
    CalcType,
    ExecutionInfo,
    Frame,
    OptimizationData,
    PathData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
)
from qcref import periodic_table as pt

from . import algorithms
from .driver import Answer, TaskContext, drive
from .evaluators import default_evaluator
from .keywords import IRCKeywords, OptimizationKeywords

PROGRAM = "qcopt"
CALCTYPES = (CalcType.optimization, CalcType.transition_state, CalcType.irc)


def _version() -> str | None:
    try:
        return metadata.version("qcopt")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return None


def _subprogram(inp: ProgramInput, calctype: CalcType) -> ProgramSpec | None:
    for child in inp.subprograms:
        if child.calctype == calctype:
            return child
    return None


def _separate(grad: ProgramSpec, hess: ProgramSpec | None) -> bool:
    """Whether Hessians come from another calculation than the gradients.

    Another program, method or keywords (such as a solvent or an SCF accuracy) is
    another energy surface: its energies and gradients never enter the frames.
    """
    return hess is not None and (
        hess.program != grad.program
        or hess.model != grad.model
        or dict(hess.keywords) != dict(grad.keywords)
    )


def _prepare(inp: ProgramInput) -> tuple[TaskContext, Any]:
    if inp.program != PROGRAM:
        raise ValueError(f"qcopt runs program='qcopt' inputs, not {inp.program!r}.")
    if inp.calctype not in CALCTYPES:
        raise ValueError(f"qcopt does not run {inp.calctype.value!r} calculations.")
    grad = _subprogram(inp, CalcType.gradient)
    if grad is None:
        raise ValueError("qcopt needs a subprogram with calctype='gradient'.")
    hess = _subprogram(inp, CalcType.hessian)
    extra = {c.calctype.value for c in inp.subprograms} - {"gradient", "hessian"}
    if extra:
        raise ValueError(
            f"qcopt subprograms are 'gradient' and 'hessian', not {sorted(extra)}."
        )
    kw: OptimizationKeywords | IRCKeywords
    kw = (
        IRCKeywords(**inp.keywords)
        if inp.calctype == CalcType.irc
        else OptimizationKeywords(**inp.keywords)
    )
    if hess is not None and kw.initial_hessian == "finite_difference":
        raise ValueError(
            "initial_hessian='finite_difference' takes the Hessian from gradients; "
            "with a Hessian subprogram, use 'exact' (or leave it 'auto')"
        )
    s = inp.require_structure()
    ctx = TaskContext(
        symbols=list(s.symbols),
        charge=s.charge,
        multiplicity=s.multiplicity,
        gradient_spec=grad,
        hessian_spec=hess,
        fd_step=kw.fd_step,
        separate_hessian=_separate(grad, hess),
        extras={"structure": s},
    )
    return ctx, kw


def _algorithm(inp: ProgramInput, ctx: TaskContext, kw: Any) -> algorithms.Algorithm:
    s = inp.require_structure()
    has_h = ctx.hessian_spec is not None
    if inp.calctype == CalcType.irc:
        return algorithms.irc(s.geometry, s.symbols, kw, has_hessian_program=has_h)
    numbers = [int(getattr(pt, sym.capitalize()).number) for sym in s.symbols]
    return algorithms.optimize(
        s.geometry,
        s.symbols,
        numbers,
        kw,
        transition_state=inp.calctype == CalcType.transition_state,
        has_hessian_program=has_h,
    )


def _frame(a: Answer, structure: Structure, keep_hessian: bool = True) -> Frame:
    """A qcdata frame for one answer.

    ``keep_hessian=False`` leaves out a Hessian computed by a separate (lower-level)
    program, which does not belong to this frame's method.
    """
    if not keep_hessian and a.hessian is not None:
        a = replace(a, hessian=None)
    out = a.output
    if isinstance(out, ProgramOutput):
        frame = Frame.from_output(out)
        if a.hessian is not None and frame.results.hessian is None:
            frame = frame.model_copy(
                update={
                    "results": frame.results.model_copy(update={"hessian": a.hessian})
                }
            )
        return frame
    results = SinglePointData(
        provenance=Provenance(
            program=a.program or "unknown", program_version=a.program_version
        ),
        energy=None if not a.success else a.energy,
        gradient=a.gradient,
        hessian=a.hessian,
    )
    return Frame(
        success=a.success,
        traceback=None if a.success else (a.error or "failed"),
        structure=structure.model_copy(
            update={"geometry": np.asarray(a.positions, dtype=float)}
        ),
        results=results,
    )


def _output(
    inp: ProgramInput, result: Any, wall: float
) -> ProgramOutput[ProgramInput, Any]:
    s = inp.require_structure()
    prov = Provenance(program=PROGRAM, program_version=_version())
    execution = ExecutionInfo(wall_time=wall, hostname=socket.gethostname())
    if isinstance(result, BaseException):
        tb = "".join(traceback.format_exception(result))
        data: Any = (
            PathData(provenance=prov)
            if inp.calctype == CalcType.irc
            else OptimizationData(provenance=prov)
        )
        return ProgramOutput(
            input_data=inp,
            success=False,
            traceback=tb,
            results=data,
            execution=execution,
        )
    gspec = _subprogram(inp, CalcType.gradient)
    keep = gspec is None or not _separate(gspec, _subprogram(inp, CalcType.hessian))
    frames = [_frame(a, s, keep) for a in result.frames]
    extras = dict(result.info)
    extras["converged"] = result.converged
    if result.frequencies is not None:
        f = result.frequencies
        if keep:
            frames[-1] = frames[-1].model_copy(
                update={
                    "results": frames[-1].results.model_copy(
                        update={
                            "freqs_wavenumber": [float(w) for w in f.wavenumbers],
                            "normal_modes_cartesian": f.modes,
                        }
                    )
                }
            )
        else:  # from the separate Hessian program: not the frame's own method
            hspec = _subprogram(inp, CalcType.hessian)
            assert hspec is not None
            extras["frequencies"] = {
                "program": hspec.program,
                "method": None if hspec.model is None else hspec.model.method,
                "wavenumbers": [float(w) for w in f.wavenumbers],
            }
    if inp.calctype == CalcType.irc:
        data = PathData(
            provenance=prov, frames=frames, ts_index=result.ts_index, extras=extras
        )
    else:
        data = OptimizationData(provenance=prov, frames=frames, extras=extras)
    return ProgramOutput(
        input_data=inp,
        success=result.error is None,
        traceback=result.error,
        results=data,
        execution=execution,
    )


@overload
def run(
    inputs: ProgramInput, *, evaluator: Any = None
) -> ProgramOutput[ProgramInput, Any]: ...
@overload
def run(
    inputs: Sequence[ProgramInput], *, evaluator: Any = None
) -> list[ProgramOutput[ProgramInput, Any]]: ...


def run(inputs: ProgramInput | Sequence[ProgramInput], *, evaluator: Any = None) -> Any:
    """Run one qcopt calculation, or several together.

    Several calculations advance in lockstep, and each round's gradients and
    Hessians, from all of them, go to the evaluator in one call: batched or side by
    side.

    Args:
        inputs: One ``ProgramInput`` or a list.
        evaluator: What computes gradients and Hessians (:mod:`qcopt.evaluators`);
            default: axtb in this process for axtb subprograms, else qccompute.

    Returns:
        One ``ProgramOutput`` or a list in input order. A calculation that fails or
        does not converge has ``success=False`` and keeps its frames.

    Raises:
        ValueError: An input qcopt cannot run (wrong program or calculation type,
            missing gradient subprogram, unknown or invalid keywords), checked for
            every input before anything is computed.
    """
    single = isinstance(inputs, ProgramInput)
    items: list[ProgramInput] = (
        [inputs] if isinstance(inputs, ProgramInput) else list(inputs)
    )
    prepared = [_prepare(inp) for inp in items]
    contexts = [ctx for ctx, _ in prepared]
    algs = [
        _algorithm(inp, ctx, kw) for inp, (ctx, kw) in zip(items, prepared, strict=True)
    ]
    ev = evaluator if evaluator is not None else default_evaluator(contexts)
    t0 = time.perf_counter()
    results = drive(algs, contexts, ev)
    wall = time.perf_counter() - t0
    outs = [_output(inp, res, wall) for inp, res in zip(items, results, strict=True)]
    return outs[0] if single else outs


def _input(
    calctype: CalcType,
    structure: Structure,
    gradient: ProgramSpec | dict[str, Any],
    hessian: ProgramSpec | dict[str, Any] | bool | None,
    keywords: dict[str, Any],
) -> ProgramInput:
    if not (hessian is None or isinstance(hessian, (bool, ProgramSpec, dict))):
        raise TypeError(
            "hessian= names the program computing Hessians (True: the gradient "
            "program; or a ProgramSpec or dict), not "
            f"{hessian!r}; the starting Hessian is the keyword initial_hessian="
        )
    gspec = (
        gradient
        if isinstance(gradient, ProgramSpec)
        else ProgramSpec(**{**gradient, "calctype": "gradient"})
    )
    subs = [gspec.model_copy(update={"calctype": CalcType.gradient})]
    if hessian is True:
        subs.append(gspec.model_copy(update={"calctype": CalcType.hessian}))
    elif isinstance(hessian, ProgramSpec):
        subs.append(hessian.model_copy(update={"calctype": CalcType.hessian}))
    elif isinstance(hessian, dict):
        subs.append(ProgramSpec(**{**hessian, "calctype": "hessian"}))
    return ProgramInput(
        program=PROGRAM,
        calctype=calctype,
        structure=structure,
        keywords=keywords,
        subprograms=subs,
    )


def minimize(
    structure: Structure,
    gradient: ProgramSpec | dict[str, Any],
    *,
    hessian: ProgramSpec | dict[str, Any] | bool | None = None,
    evaluator: Any = None,
    **keywords: Any,
) -> ProgramOutput[ProgramInput, OptimizationData]:
    """Minimize ``structure``; ``hessian=True`` uses the gradient program's Hessians.

    ``hessian`` is the program computing Hessians: True for the gradient program,
    or a ``ProgramSpec`` or dict for another one (such as a cheaper method; the
    frames keep the gradient program's energies and gradients). Other keyword
    arguments are qcopt's keywords (:class:`qcopt.keywords.OptimizationKeywords`), for example
    ``initial_hessian``.
    """
    inp = _input(CalcType.optimization, structure, gradient, hessian, keywords)
    return run(inp, evaluator=evaluator)


def find_ts(
    structure: Structure,
    gradient: ProgramSpec | dict[str, Any],
    *,
    hessian: ProgramSpec | dict[str, Any] | bool | None = None,
    evaluator: Any = None,
    **keywords: Any,
) -> ProgramOutput[ProgramInput, OptimizationData]:
    """Optimize a transition-state guess to a first-order saddle point.

    ``hessian`` is the program computing Hessians: True for the gradient program,
    or a ``ProgramSpec`` or dict for another one (such as a cheaper method; the
    frames keep the gradient program's energies and gradients). Other keyword
    arguments are qcopt's keywords (:class:`qcopt.keywords.OptimizationKeywords`), for example
    ``initial_hessian``.
    """
    inp = _input(CalcType.transition_state, structure, gradient, hessian, keywords)
    return run(inp, evaluator=evaluator)


def irc(
    structure: Structure,
    gradient: ProgramSpec | dict[str, Any],
    *,
    hessian: ProgramSpec | dict[str, Any] | bool | None = None,
    evaluator: Any = None,
    **keywords: Any,
) -> ProgramOutput[ProgramInput, PathData]:
    """Follow the IRC down both sides of a transition state.

    ``hessian`` is the program computing Hessians: True for the gradient program,
    or a ``ProgramSpec`` or dict for another one (such as a cheaper method; the
    frames keep the gradient program's energies and gradients). Other keyword
    arguments are qcopt's keywords (:class:`qcopt.keywords.IRCKeywords`), for example
    ``initial_hessian``.
    """
    inp = _input(CalcType.irc, structure, gradient, hessian, keywords)
    return run(inp, evaluator=evaluator)
