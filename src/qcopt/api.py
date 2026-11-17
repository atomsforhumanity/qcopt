"""Running qcopt: ``ProgramInput`` in, ``ProgramOutput`` out.

A qcopt calculation is a qcdata ``ProgramInput`` with ``program="qcopt"``, a
calculation type (``optimization``, ``transition_state`` or ``irc``), a structure,
qcopt's settings as ``keywords`` (:mod:`qcopt.options`), and subprograms: one
``ProgramSpec`` with ``calctype="gradient"``, and optionally one with
``calctype="hessian"``. The result is a ``ProgramOutput`` holding
``OptimizationData`` (minima and transition states) or ``PathData`` (IRCs), every
frame with the energy and gradient computed there, and qcdata's standard slots:

- ``resolved``: ``settings``, the settings with every ``auto`` replaced (a valid
  ``keywords`` that reproduces the run), and ``reasons``, why each was chosen;
- ``record``: what happened (counts, rebuilds, rejected steps, IRC branch ends, the
  step log);
- ``error``, on failure: ``kind`` (:mod:`qcopt.errors`), ``message``, ``code`` (such
  as ``max_iter``), and for a failed gradient or Hessian calculation its ``program``
  and its own error (``cause``);
- ``execution.timings``: ``evaluator`` (seconds in the gradient and Hessian
  programs), ``optimizer`` (qcopt's own), ``total``.
"""

from __future__ import annotations

import socket
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Any, overload

import numpy as np
from pydantic import BaseModel
from qcdata import (
    CalcType,
    ErrorInfo,
    ExecutionInfo,
    Frame,
    OptimizationData,
    PathData,
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
from .driver import Answer, TaskContext, drive
from .errors import InputError
from .evaluators import default_evaluator, surface_keywords
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


def _same_spec(a: ProgramSpec, b: ProgramSpec) -> bool:
    """Whether two specs differ only in their calculation type."""
    return a.model_copy(update={"calctype": b.calctype}) == b


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
class _Prepared:
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


def _prepare(inp: ProgramInput) -> _Prepared:
    """Check one input before anything is computed.

    Raises:
        InputError: Not a qcopt input, a calculation type qcopt does not run, a
            missing or extra subprogram, or invalid settings.
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
    options = parse_options(inp.calctype.value, dict(inp.keywords))
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
    return _Prepared(
        inp=inp,
        options=options,
        context=ctx,
        own_hessian=_same_method(grad, hess),
        given_hessian=None if given is None else given[0],
        resolved=resolved,
        reasons=reasons,
    )


def _algorithm(p: _Prepared) -> algorithms.common.Algorithm:
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
    p: _Prepared, result: RunResult | BaseException, timings: dict[str, float]
) -> ProgramOutput[ProgramInput, Any]:
    inp = p.inp
    s = inp.require_structure()
    prov = Provenance(program=PROGRAM, program_version=_version())
    execution = ExecutionInfo(
        wall_time=timings["total"], hostname=socket.gethostname(), timings=timings
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
            default (:func:`~qcopt.evaluators.default_evaluator`): axtb in this
            process, a program's own sessions, or qccompute.

    Returns:
        One ``ProgramOutput`` or a list in input order. A calculation that fails or
        does not converge has ``success=False``, keeps its frames, and says why in
        its typed ``error`` (and ``traceback``).

    Raises:
        InputError: An input qcopt cannot run (wrong program or calculation type,
            missing gradient subprogram, unknown or invalid settings), checked for
            every input before anything is computed.
    """
    single = isinstance(inputs, ProgramInput)
    items: list[ProgramInput] = (
        [inputs] if isinstance(inputs, ProgramInput) else list(inputs)
    )
    prepared = [_prepare(inp) for inp in items]
    contexts = [p.context for p in prepared]
    algs = [_algorithm(p) for p in prepared]
    ev = evaluator if evaluator is not None else default_evaluator(contexts)
    timings: dict[str, float] = {"evaluator": 0.0}
    t0 = time.perf_counter()
    try:
        results = drive(algs, contexts, ev, timings=timings)
    finally:
        if evaluator is None and hasattr(ev, "close"):
            ev.close()  # the sessions it opened
    timings["total"] = time.perf_counter() - t0
    timings["optimizer"] = timings["total"] - timings["evaluator"]
    outs = [_output(p, res, timings) for p, res in zip(prepared, results, strict=True)]
    return outs[0] if single else outs


def _keywords(options: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments as qcdata keywords (setting groups given as models allowed)."""
    return {
        k: v.model_dump(exclude_unset=True) if isinstance(v, BaseModel) else v
        for k, v in options.items()
    }


def _input(
    calctype: CalcType,
    structure: Structure,
    program: ProgramSpec | dict[str, Any],
    hessian_program: ProgramSpec | dict[str, Any] | bool | None,
    options: dict[str, Any],
    given: SinglePointData | None = None,
) -> ProgramInput:
    if not (
        hessian_program is None
        or isinstance(hessian_program, bool | ProgramSpec | dict)
    ):
        raise InputError(
            "hessian_program= names the program computing Hessians (True: the "
            f"gradient program; or a ProgramSpec or dict), not {hessian_program!r}"
        )
    gspec = (
        program
        if isinstance(program, ProgramSpec)
        else ProgramSpec(**{**program, "calctype": "gradient"})
    )
    subs = [gspec.model_copy(update={"calctype": CalcType.gradient})]
    if hessian_program is True:
        subs.append(gspec.model_copy(update={"calctype": CalcType.hessian}))
    elif isinstance(hessian_program, ProgramSpec):
        subs.append(hessian_program.model_copy(update={"calctype": CalcType.hessian}))
    elif isinstance(hessian_program, dict):
        subs.append(ProgramSpec(**{**hessian_program, "calctype": "hessian"}))
    return ProgramInput(
        program=PROGRAM,
        calctype=calctype,
        structure=structure,
        keywords=_keywords(options),
        subprograms=subs,
        given=given,
    )


def minimize(
    structure: Structure,
    program: ProgramSpec | dict[str, Any],
    *,
    hessian_program: ProgramSpec | dict[str, Any] | bool | None = None,
    given: SinglePointData | None = None,
    evaluator: Any = None,
    **options: Any,
) -> ProgramOutput[ProgramInput, OptimizationData]:
    """Minimize ``structure`` on ``program``'s energy surface.

    Args:
        structure: The starting structure.
        program: The program computing energies and gradients, as a
            ``ProgramSpec`` or a dict (``program``, ``model``, ``keywords``).
        hessian_program: The program computing Hessians: True for ``program``
            itself, or another one (such as a cheaper method; frames keep
            ``program``'s energies and gradients). None: no Hessians (model Hessian,
            finite differences where an exact one is needed).
        given: Results already known at ``structure`` (``ProgramInput.given``, such
            as a frame's ``results``): their Hessian is the initial one, from any
            program (a cheaper method's will do); their other fields are not used.
        evaluator: What computes the gradients and Hessians (default: axtb in this
            process, a program's own sessions, or qccompute; see :func:`run`).
        **options: The settings (:class:`qcopt.options.OptimizationOptions`), such as
            ``convergence={"g_tol": 1e-5}`` or ``hessian={"recompute_every": 5}``.
    """
    inp = _input(
        CalcType.optimization, structure, program, hessian_program, options, given
    )
    return run(inp, evaluator=evaluator)


def find_ts(
    structure: Structure,
    program: ProgramSpec | dict[str, Any],
    *,
    hessian_program: ProgramSpec | dict[str, Any] | bool | None = None,
    given: SinglePointData | None = None,
    evaluator: Any = None,
    **options: Any,
) -> ProgramOutput[ProgramInput, OptimizationData]:
    """Optimize a transition-state guess to a first-order saddle point.

    Arguments as for :func:`minimize`; the settings are
    :class:`qcopt.options.TransitionStateOptions`. With ``freq={"check": True}``
    the last frame holds the transition state's Hessian, which :func:`irc` reuses.
    """
    inp = _input(
        CalcType.transition_state, structure, program, hessian_program, options, given
    )
    return run(inp, evaluator=evaluator)


def irc(
    start: Structure | ProgramOutput[ProgramInput, OptimizationData],
    program: ProgramSpec | dict[str, Any],
    *,
    hessian_program: ProgramSpec | dict[str, Any] | bool | None = None,
    given: SinglePointData | None = None,
    evaluator: Any = None,
    **options: Any,
) -> ProgramOutput[ProgramInput, PathData]:
    """Follow the IRC down both sides of a transition state.

    Args:
        start: The transition state, or the output of :func:`find_ts`. From an
            output, the IRC starts at its final structure and reuses its last frame's
            Hessian (``freq.check``) when this IRC would compute the same one (the
            same Hessian program, method and settings): only the gradient is then
            computed at the transition state (``hessian.initial`` is ``given``).
        program: The program computing energies and gradients (see
            :func:`minimize`).
        hessian_program: The program computing Hessians (see :func:`minimize`).
        given: Results already known at the transition state
            (``ProgramInput.given``), with a structure as ``start``: their Hessian is
            the transition state's.
        evaluator: What computes the gradients and Hessians (see :func:`run`).
        **options: The settings (:class:`qcopt.options.IRCOptions`).
    """
    if isinstance(start, ProgramOutput):
        if given is not None:
            raise InputError(
                "irc() takes given= with a structure; from a search's output it "
                "reuses the output's last frame"
            )
        structure = _searched(start)
        inp = _input(CalcType.irc, structure, program, hessian_program, options)
        reused = _reusable(start, inp)
        if reused is not None:
            inp = ProgramInput(**{**dict(inp), "given": reused})
    else:
        inp = _input(CalcType.irc, start, program, hessian_program, options, given)
    return run(inp, evaluator=evaluator)


def _searched(out: ProgramOutput[Any, Any]) -> Structure:
    """The transition state a qcopt search found (its final structure)."""
    inp = out.input_data
    if not (
        isinstance(inp, ProgramInput)
        and inp.program == PROGRAM
        and inp.calctype == CalcType.transition_state
    ):
        raise InputError(
            "irc() starts from a structure or the output of a qcopt "
            "transition-state search"
        )
    if not out.success:
        message = out.error.message if out.error is not None else "it failed"
        raise InputError(
            f"the transition-state search did not succeed ({message}); to follow the "
            "IRC from its last structure anyway, pass out.results.final_structure"
        )
    return out.results.final_structure


def _reusable(
    out: ProgramOutput[ProgramInput, OptimizationData], inp: ProgramInput
) -> SinglePointData | None:
    """A search's transition-state Hessian, when this IRC would compute the same one.

    The same program, method and settings computing Hessians (the Hessian
    subprogram, else the gradient program by finite differences): the IRC is then
    the one it would have been, without computing the Hessian again.
    """
    last = out.results.frames[-1].results
    if last.hessian is None:
        return None
    search = out.input_data
    grad, ts_grad = (
        _subprogram(inp, CalcType.gradient),
        _subprogram(search, CalcType.gradient),
    )
    assert grad is not None and ts_grad is not None  # checked by qcopt's inputs
    source = _subprogram(inp, CalcType.hessian) or grad
    ts_source = _subprogram(search, CalcType.hessian) or ts_grad
    if not _same_spec(ts_source, source):
        return None
    return SinglePointData(provenance=last.provenance, hessian=last.hessian)
