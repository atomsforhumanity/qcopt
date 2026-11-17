"""One qcopt calculation: a ``ProgramInput`` checked, run, and its ``ProgramOutput``.

A qcopt calculation is a qcdata ``ProgramInput`` with ``program="qcopt"``, a
calculation type (``optimization``, ``transition_state`` or ``irc``), a structure,
qcopt's settings as ``keywords`` (:mod:`qcopt.options`), subprograms (one
``ProgramSpec`` with ``calctype="gradient"``, and optionally one with
``calctype="hessian"``), and optionally ``given`` results at the structure, whose
Hessian is the starting one. :func:`prepare` checks an input before anything is
computed; :func:`run` runs prepared calculations in lockstep on an evaluator
(:mod:`qcopt.evaluators`). :mod:`qcopt.program` is the entry point.

The result is a ``ProgramOutput`` holding ``OptimizationData`` (minima and transition
states) or ``PathData`` (IRCs), every frame with the energy and gradient computed
there, and qcdata's standard slots:

- ``resolved``: ``settings``, the settings with every ``auto`` replaced (a valid
  ``keywords`` that reproduces the run), and ``reasons``, why each was chosen;
- ``record``: what happened (counts, rebuilds, rejected steps, IRC branch ends, the
  step log);
- ``error``, on failure: ``kind`` (:mod:`qcopt.errors`), ``message``, ``code`` (such
  as ``max_iter``), and for a failed gradient or Hessian calculation its ``program``
  and its own error (``cause``);
- ``execution.timings``: ``subprograms`` (seconds in the gradient and Hessian
  programs), ``optimizer`` (qcopt's own), ``total``. Calculations run together as a
  list advance in lockstep, so each one's timings are the list's;
- ``execution.threads``: the threads this calculation ran with: the most its
  children recorded, else its share of the threads granted to the list.
"""

from __future__ import annotations

import socket
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Any

import numpy as np
from qcdata import (
    Accuracy,
    CalcType,
    ErrorInfo,
    ExecutionInfo,
    Frame,
    OptimizationData,
    PathData,
    ProgramError,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    Resolved,
    SinglePointData,
    Structure,
    VibrationalAnalysis,
)
from qcref import periodic_table as pt

from . import algorithms
from .algorithms.common import RunResult, resolve_initial_hessian
from .constraints import ConstraintSet
from .driver import Answer, Evaluator, TaskContext, drive
from .errors import InputError, UnsupportedError
from .evaluators import surface_keywords
from .geometry import Array
from .options import (
    IRCOptions,
    Options,
    TransitionStateOptions,
    parse_options,
    without_effect,
)

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

    Another program, method or keywords that define the surface (such as a solvent
    or an SCF accuracy, but not threads or how the Hessian is computed) is another
    energy surface: its energies and gradients never enter the frames.
    """
    return hess is not None and (
        hess.program != grad.program
        or hess.model != grad.model
        or surface_keywords(hess) != surface_keywords(grad)
    )


def _same_method(grad: ProgramSpec, hess: ProgramSpec | None) -> bool:
    """Whether Hessians are the gradient method's own (the same program and method)."""
    return hess is None or (hess.program == grad.program and hess.model == grad.model)


def _given_hessian(inp: ProgramInput) -> tuple[Array, str] | None:
    """The Hessian of the input's given results, and where it came from.

    qcopt uses one field of ``ProgramInput.given``: the Hessian, as its starting one
    (an optimization's initial Hessian, an IRC's at the transition state). It is a
    starting guess, so any program's will do (a GFN2 Hessian for a DFT search); it
    never becomes a frame's own Hessian, and every energy and gradient is computed.
    The other fields are left alone.

    Raises:
        InputError: ``not_applicable``: the given results have no Hessian.
    """
    given = inp.given
    if given is None:
        return None
    if given.hessian is None:
        raise InputError(
            "qcopt uses a given Hessian (its starting Hessian), and this input's "
            "given results have none",
            "not_applicable",
        )
    prov = given.provenance
    source = " ".join(str(v) for v in (prov.program, prov.program_version) if v)
    return np.asarray(given.hessian, dtype=float), source


@dataclass
class Prepared:
    """One input, checked: its settings, what ``auto`` became, its task context."""

    inp: ProgramInput
    options: Options
    context: TaskContext
    own_hessian: bool
    given_hessian: Array | None
    resolved: dict[str, Any]
    reasons: dict[str, str]


def _resolve(
    options: Options, n_atoms: int, has_hessian_program: bool, given: str | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """The settings with every ``auto`` replaced, and why each became what it did.

    Settings that cannot affect the calculation are left out, so the tree, given back
    as ``keywords``, is valid and reproduces the run. ``given``: where the input's
    given Hessian came from, when it has one.
    """
    settings = options.model_dump()
    reasons: dict[str, str] = {}
    for path, _ in without_effect(options, settings):
        group, _, key = path.partition(".")
        del settings[group][key]
    if has_hessian_program:
        del settings["hessian"]["step"]

    def system(path: str, value: str, frozen: list[int]) -> str:
        if value != "auto":
            return value
        if frozen:
            reasons[path] = "atoms are frozen: Cartesian coordinates"
            return "cartesian"
        if n_atoms < 3:
            reasons[path] = "fewer than three atoms: Cartesian coordinates"
            return "cartesian"
        reasons[path] = "TRIC (internal coordinates with fragment rigid motion)"
        return "tric"

    if isinstance(options, IRCOptions):
        if given is not None:
            settings["hessian"]["initial"] = "given"
            reasons["hessian.initial"] = f"the given Hessian ({given})"
        elif options.hessian.initial == "auto":
            initial = resolve_initial_hessian("auto", has_hessian_program, True)
            settings["hessian"]["initial"] = initial
            reasons["hessian.initial"] = (
                "the Hessian subprogram's at the transition state"
                if has_hessian_program
                else "no Hessian subprogram: finite differences of gradients"
            )
        if options.ends.minimize:
            settings["ends"]["coordinate_system"] = system(
                "ends.coordinate_system",
                options.ends.coordinate_system,
                options.constraints.frozen,
            )
        return settings, reasons
    ts = isinstance(options, TransitionStateOptions)
    settings["coordinate_system"] = system(
        "coordinate_system", options.coordinate_system, options.constraints.frozen
    )
    if given is not None:
        settings["hessian"]["initial"] = "given"
        reasons["hessian.initial"] = f"the given Hessian ({given})"
    elif options.hessian.initial == "auto":
        initial = resolve_initial_hessian("auto", has_hessian_program, ts)
        settings["hessian"]["initial"] = initial
        reasons["hessian.initial"] = {
            "exact": "a Hessian subprogram is given",
            "model": "minimum without a Hessian subprogram: Lindh's model Hessian",
            "finite_difference": "transition state without a Hessian subprogram: "
            "finite differences of gradients",
        }[initial]
    if options.hessian.update == "auto":
        settings["hessian"]["update"] = "ts_bfgs" if ts else "bfgs"
        reasons["hessian.update"] = (
            "TS-BFGS for transition states"
            if ts
            else "BFGS for minima (Bofill while the Hessian has negative curvature)"
        )
    return settings, reasons


def _without_accuracy(keywords: Any) -> dict[str, Any]:
    """The keywords without the shared ``accuracy``, once checked: qcopt's settings.

    ``accuracy`` (qcdata's ``Accuracy``: energy, gradient and Hessian targets) is a
    keyword every AFH program reads the same way. qcopt cannot honour targets yet: how
    tightly it converges is its ``convergence`` settings, and how accurate the
    gradients are is the gradient program's own settings. A target is refused as
    ``unsupported``; a malformed one as any program refuses it.

    Raises:
        InputError: A malformed ``accuracy`` (its code: ``unknown_key``,
            ``no_effect``, or none).
        UnsupportedError: An ``accuracy`` that sets a target.
    """
    settings = dict(keywords)
    try:
        target = Accuracy.from_keywords(settings)
    except ProgramError as exc:
        raise InputError(exc.error.message, exc.error.code) from None
    settings.pop("accuracy", None)
    if target is not None:
        raise UnsupportedError(
            "qcopt does not take accuracy targets yet: its convergence settings "
            "(convergence) set how tightly it converges, and the gradient program's "
            "own keywords how accurate the gradients are"
        )
    return settings


def prepare(inp: ProgramInput) -> Prepared:
    """Check one input before anything is computed.

    Raises:
        InputError: Not a qcopt input, a calculation type qcopt does not run, a
            missing or extra subprogram, or invalid settings.
        UnsupportedError: An ``accuracy`` target (:func:`_without_accuracy`).
    """
    if inp.program != PROGRAM:
        raise InputError(f"qcopt runs program='qcopt' inputs, not {inp.program!r}")
    if inp.calctype not in CALCTYPES:
        raise InputError(f"qcopt does not run {inp.calctype.value!r} calculations")
    grad = _subprogram(inp, CalcType.gradient)
    if grad is None:
        raise InputError("qcopt needs a subprogram with calctype='gradient'")
    hess = _subprogram(inp, CalcType.hessian)
    extra = {c.calctype.value for c in inp.subprograms} - {"gradient", "hessian"}
    if extra:
        raise InputError(
            f"qcopt subprograms are 'gradient' and 'hessian', not {sorted(extra)}"
        )
    options = parse_options(inp.calctype.value, _without_accuracy(inp.keywords))
    if hess is not None and options.hessian.initial == "finite_difference":
        raise InputError(
            "hessian.initial='finite_difference' takes the Hessian from gradients; "
            "with a Hessian subprogram, use 'exact' (or leave it 'auto')"
        )
    if hess is not None and "step" in options.hessian.model_fields_set:
        raise InputError(
            "hessian.step cannot affect this calculation: it is the displacement of "
            "finite-difference Hessians, and Hessians come from the Hessian subprogram",
            "no_effect",
        )
    given = _given_hessian(inp)
    initial = options.hessian.initial
    if initial == "given" and given is None:
        raise InputError(
            "hessian.initial='given' takes the Hessian from ProgramInput.given, and "
            "this input has none"
        )
    if given is not None and initial not in ("auto", "given"):
        raise InputError(
            f"qcopt uses only a given Hessian, and hessian.initial={initial!r} leaves "
            "it unused (leave it 'auto', or set 'given')",
            "not_applicable",
        )
    s = inp.require_structure()
    if not isinstance(options, IRCOptions) and options.constraints.size:
        try:  # checked here, before anything is computed
            ConstraintSet(options.constraints, s.geometry, options.constraints.frozen)
        except ValueError as exc:
            raise InputError(str(exc)) from None
    ctx = TaskContext(
        symbols=list(s.symbols),
        charge=s.charge,
        multiplicity=s.multiplicity,
        gradient_spec=grad,
        hessian_spec=hess,
        fd_step=options.hessian.step,
        separate_hessian=_separate(grad, hess),
        extras={"structure": s},
    )
    resolved, reasons = _resolve(
        options, len(s.symbols), hess is not None, None if given is None else given[1]
    )
    return Prepared(
        inp=inp,
        options=options,
        context=ctx,
        own_hessian=_same_method(grad, hess),
        given_hessian=None if given is None else given[0],
        resolved=resolved,
        reasons=reasons,
    )


def _algorithm(p: Prepared) -> algorithms.common.Algorithm:
    s = p.inp.require_structure()
    has_h = p.context.hessian_spec is not None
    if isinstance(p.options, IRCOptions):
        return algorithms.irc(
            s.geometry,
            s.symbols,
            p.options,
            has_hessian_program=has_h,
            own_hessian=p.own_hessian,
            given_hessian=p.given_hessian,
        )
    numbers = [int(getattr(pt, sym.capitalize()).number) for sym in s.symbols]
    return algorithms.optimize(
        s.geometry,
        s.symbols,
        numbers,
        p.options,
        has_hessian_program=has_h,
        own_hessian=p.own_hessian,
        given_hessian=p.given_hessian,
    )


def _frame(a: Answer, structure: Structure, keep_hessian: bool = True) -> Frame:
    """A qcdata frame for one answer.

    ``keep_hessian=False`` leaves out a Hessian computed by a separate (lower-level)
    program, which does not belong to this frame's method.
    """
    if not keep_hessian and a.hessian is not None:
        a = replace(a, hessian=None)
    out = a.output
    if isinstance(out, Frame):  # from a program session
        if a.hessian is None and out.results.hessian is not None:
            out = out.model_copy(
                update={"results": out.results.model_copy(update={"hessian": None})}
            )
        elif a.hessian is not None and out.results.hessian is None:
            out = out.model_copy(
                update={
                    "results": out.results.model_copy(update={"hessian": a.hessian})
                }
            )
        return out
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
        traceback=None if a.success else (a.traceback or a.error or "failed"),
        structure=structure.model_copy(
            update={"geometry": np.asarray(a.positions, dtype=float)}
        ),
        results=results,
    )


def _cause(answer: Answer | None) -> tuple[str | None, ErrorInfo | None]:
    """The program of a failed child calculation, and its error.

    The program's own error when its output carries one; otherwise one made from the
    answer (an exception in the evaluator is ``internal``, as qccompute treats a
    program's bug), so callers can always follow ``cause``.
    """
    if answer is None:
        return None, None
    out = answer.output
    child = out if isinstance(out, ErrorInfo) else getattr(out, "error", None)
    if not isinstance(child, ErrorInfo):
        child = ErrorInfo(
            kind=answer.error_kind or "internal", message=answer.error or "failed"
        )
    return answer.program, child


def _output(
    p: Prepared,
    result: RunResult | BaseException,
    timings: dict[str, float],
    threads: int | None = None,
) -> ProgramOutput[ProgramInput, Any]:
    inp = p.inp
    s = inp.require_structure()
    prov = Provenance(program=PROGRAM, program_version=_version())
    execution = ExecutionInfo(
        wall_time=timings["total"],
        hostname=socket.gethostname(),
        timings=timings,
        threads=threads,
    )
    irc = isinstance(p.options, IRCOptions)
    common: dict[str, Any] = {
        "input_data": inp,
        "execution": execution,
        "resolved": Resolved(settings=p.resolved, reasons=p.reasons),
    }
    if isinstance(result, BaseException):  # the driver reports a crash this way
        message = "".join(traceback.format_exception(result))
        data: Any = (
            PathData(provenance=prov) if irc else OptimizationData(provenance=prov)
        )
        return ProgramOutput(
            success=False,
            traceback=message,
            error=ErrorInfo(kind="internal", message=message),
            results=data,
            **common,
        )
    keep = not p.context.separate_hessian
    frames = [_frame(a, s, keep) for a in result.frames]
    used = [f.execution.threads for f in frames if f.execution.threads]
    if used:  # what its children ran with, not the whole list's grant
        common["execution"] = execution.model_copy(update={"threads": max(used)})
    record = dict(result.info)
    record["converged"] = result.converged
    extras: dict[str, Any] = {}
    if result.frequencies is not None:
        f = result.frequencies
        if keep:
            analysis = VibrationalAnalysis(
                masses=f.masses,
                projection="translations_rotations" if f.projected else "none",
                frequencies=f.wavenumbers,
                normal_modes_cartesian=f.modes,
            )
            results = frames[-1].results.model_copy(
                update={"vibrational_analysis": analysis}
            )
            frames[-1] = frames[-1].model_copy(update={"results": results})
        else:  # from a separate Hessian calculation: not the frame's own
            hspec = p.context.hessian_spec
            extras["frequencies"] = {
                "program": hspec.program,
                "method": None if hspec.model is None else hspec.model.method,
                "wavenumbers": [float(w) for w in f.wavenumbers],
            }
    if irc:
        data = PathData(
            provenance=prov, frames=frames, ts_index=result.ts_index, extras=extras
        )
    else:
        data = OptimizationData(provenance=prov, frames=frames, extras=extras)
    error = None
    if result.error is not None:
        program, cause = _cause(result.cause)
        error = ErrorInfo(
            kind=result.error_kind or "internal",
            message=result.error,
            code=result.reason,
            program=program,
            cause=cause,
        )
    # the message, then the failed child's own traceback when there is one
    tb = result.error
    if tb is not None and result.cause is not None and result.cause.traceback:
        tb = f"{tb}\n\n{result.cause.traceback}"
    return ProgramOutput(
        success=result.error is None,
        traceback=tb,
        error=error,
        record=record,
        results=data,
        **common,
    )


def run(
    prepared: Sequence[Prepared], evaluator: Evaluator, threads: int | None = None
) -> list[ProgramOutput[ProgramInput, Any]]:
    """Run checked calculations in lockstep on one evaluator.

    Each round's gradients and Hessians, from all of them, go to the evaluator in one
    call, to be batched or run side by side.

    Args:
        prepared: The calculations, checked by :func:`prepare`.
        evaluator: What computes their gradients and Hessians.
        threads: The threads granted, shared by their subprograms. Each output's
            ``execution.threads`` is what its own children recorded, else its share.

    Returns:
        One output per calculation, in order. One that fails or does not converge has
        ``success=False``, keeps its frames, and says why in its typed ``error``.
    """
    contexts = [p.context for p in prepared]
    algs = [_algorithm(p) for p in prepared]
    timings: dict[str, float] = {"subprograms": 0.0}
    t0 = time.perf_counter()
    results = drive(algs, contexts, evaluator, timings=timings)
    timings["total"] = time.perf_counter() - t0
    timings["optimizer"] = timings["total"] - timings["subprograms"]
    share = None if threads is None else max(1, threads // max(1, len(prepared)))
    return [
        _output(p, res, timings, share)
        for p, res in zip(prepared, results, strict=True)
    ]
