"""Evaluators: what computes the gradients and Hessians the optimizers ask for.

- :class:`AxtbEvaluator` calls axtb in this process, one ``axtb.Calculator`` per
  calculation, so each step starts from the previous wavefunction; calculations run
  side by side in threads (axtb releases the GIL).
- :class:`ComputeEvaluator` hands every request to qccompute (or any qccodec
  ``Compute``), one list per round, so any program with a codec works and children
  are batched or run side by side by the runner.
- :class:`FunctionEvaluator` calls a Python function, for model surfaces and tests.
- :func:`default_evaluator` picks axtb in-process where it can, else qccompute.
"""

from __future__ import annotations

import os
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from typing import Any

import numpy as np

from .driver import Answer, Request, TaskContext
from .geometry import Array


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _map_tasks(
    requests: Sequence[Request],
    run_one: Callable[[Request], Answer],
    workers: int,
) -> list[Answer]:
    """Run each task's requests in order, different tasks side by side."""
    groups: dict[int, list[int]] = defaultdict(list)
    for k, r in enumerate(requests):
        groups[r.task].append(k)
    answers: list[Answer | None] = [None] * len(requests)

    def run_group(indices: list[int]) -> None:
        for k in indices:
            answers[k] = run_one(requests[k])

    if workers <= 1 or len(groups) == 1:
        for indices in groups.values():
            run_group(indices)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
            list(pool.map(run_group, groups.values()))
    return [a for a in answers if a is not None]


class FunctionEvaluator:
    """Energies and gradients (optionally Hessians) from a Python function.

    Args:
        function: ``function(positions) -> (energy, gradient)``; positions ``(n, 3)``
            in bohr, energy in Hartree, gradient ``(n, 3)`` in Hartree/bohr.
        hessian: ``hessian(positions) -> (3n, 3n)`` array, or None for finite
            differences of the gradient.
        name: Program name recorded as the frames' provenance.
    """

    def __init__(
        self,
        function: Callable[[Array], tuple[float, Array]],
        hessian: Callable[[Array], Array] | None = None,
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
                        program=self.name,
                    )
                )
        return out


# axtb.Calculator keywords a ProgramSpec may set; everything else is rejected.
_AXTB_CALCULATOR = frozenset(
    {
        "accuracy",
        "etemp",
        "max_iter",
        "threads",
        "conventions",
        "mixer",
        "alpb",
        "gbsa",
        "solvation_state",
        "solvation_correction",
        "correction_conventions",
    }
)
_AXTB_HESSIAN = frozenset({"hessian_method", "hessian_step", "hessian_accuracy"})

# Keywords that change how a calculation runs or how its Hessian is computed, but not
# its energy surface, by program; ``threads`` for every program.
_NOT_SURFACE: dict[str, frozenset[str]] = {
    "axtb": _AXTB_HESSIAN | {"threads", "max_iter", "mixer"}
}


def surface_keywords(spec: Any) -> dict[str, Any]:
    """A spec's keywords that define its energy surface."""
    skip = _NOT_SURFACE.get(getattr(spec, "program", ""), frozenset({"threads"}))
    return {k: v for k, v in dict(spec.keywords).items() if k not in skip}


class AxtbEvaluator:
    """axtb in this process: one Calculator per calculation, reused step to step.

    Args:
        workers: Calculations run side by side (threads); default: CPU cores
            divided by the threads each calculation uses.
    """

    def __init__(self, workers: int | None = None) -> None:
        """Import axtb and set up the calculator cache."""
        import axtb

        self._axtb = axtb
        self.workers = workers
        self.version = _version("axtb")
        # (task, kind) -> (the calculator's setup, the calculator); a new setup (another
        # run's task with the same index) replaces it
        self._calcs: dict[tuple[int, str], tuple[tuple[Any, ...], Any]] = {}

    @staticmethod
    def handles(context: TaskContext) -> bool:
        """Whether this evaluator can compute the task (axtb gradients)."""
        spec = context.gradient_spec
        if spec is None or getattr(spec, "program", None) != "axtb":
            return False
        hs = context.hessian_spec
        return hs is None or getattr(hs, "program", None) == "axtb"

    def supports_hessian(self, context: TaskContext) -> bool:
        """Axtb Hessians are analytic; used when a Hessian subprogram is given."""
        return context.hessian_spec is not None

    def _settings(self, spec: Any) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if getattr(spec, "program", None) != "axtb":
            raise ValueError(
                f"the axtb evaluator computes axtb specs, not {spec.program!r}"
            )
        if spec.model is None or not spec.model.method:
            raise ValueError("axtb needs model.method, such as gfn2 or gxtb-v2.0.1")
        method = spec.model.method
        kw = dict(spec.keywords)
        unknown = set(kw) - _AXTB_CALCULATOR - _AXTB_HESSIAN
        if unknown:
            raise ValueError(
                f"axtb keywords not supported by qcopt's axtb evaluator: {sorted(unknown)}"
            )
        calc_kw = {k: v for k, v in kw.items() if k in _AXTB_CALCULATOR}
        hess_kw = {
            k.removeprefix("hessian_"): v for k, v in kw.items() if k in _AXTB_HESSIAN
        }
        return method, calc_kw, hess_kw

    def _calculator(self, r: Request, kind: str) -> tuple[Any, dict[str, Any]]:
        ctx = r.context
        spec = ctx.hessian_spec if kind == "hessian" else ctx.gradient_spec
        method, calc_kw, hess_kw = self._settings(spec)
        same = (
            kind == "hessian"
            and ctx.hessian_spec.model == ctx.gradient_spec.model
            and {
                k: v
                for k, v in ctx.hessian_spec.keywords.items()
                if k in _AXTB_CALCULATOR
            }
            == {
                k: v
                for k, v in ctx.gradient_spec.keywords.items()
                if k in _AXTB_CALCULATOR
            }
        )
        key = (r.task, "gradient" if (kind == "gradient" or same) else "hessian")
        from qcref import periodic_table as pt

        numbers = [int(getattr(pt, s.capitalize()).number) for s in ctx.symbols]
        setup = (method, numbers, ctx.charge, ctx.multiplicity, calc_kw)
        cached = self._calcs.get(key)
        calc = cached[1] if cached is not None and cached[0] == setup else None
        if calc is None:
            calc = self._axtb.Calculator(
                method,
                numbers,
                np.asarray(r.positions, dtype=float),
                charge=float(ctx.charge),
                uhf=int(ctx.multiplicity) - 1,
                **calc_kw,
            )
            self._calcs[key] = (setup, calc)
        return calc, hess_kw

    def _run(self, r: Request) -> Answer:
        x = np.asarray(r.positions, dtype=float)
        try:
            calc, hess_kw = self._calculator(r, r.kind)
            if r.kind == "hessian":
                res = calc.hessian(x, vibrations=False, **hess_kw)
                hess = np.asarray(res["hessian"], dtype=float)
            else:
                res = calc.singlepoint(x, gradient=True)
                hess = None
            return Answer(
                positions=x,
                energy=float(res["energy"]),
                gradient=np.asarray(res["gradient"], dtype=float).reshape(-1, 3),
                hessian=hess,
                program="axtb",
                program_version=self.version,
                output={"iterations": res.get("iterations")},
            )
        except Exception as exc:  # noqa: BLE001 - reported in the answer
            return Answer(
                positions=x,
                success=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                program="axtb",
                program_version=self.version,
            )

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Compute the requests; one task's in order, tasks side by side."""
        workers = self.workers
        if workers is None:
            threads = max(
                [1]
                + [
                    int(r.context.gradient_spec.keywords.get("threads", 1))
                    for r in requests
                ]
            )
            workers = max(
                1,
                (
                    len(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity")
                    else os.cpu_count() or 1
                )
                // threads,
            )
        return _map_tasks(requests, self._run, workers)

    def forget(self, task: int) -> None:
        """Drop a finished task's calculators."""
        for key in [k for k in self._calcs if k[0] == task]:
            del self._calcs[key]


class ComputeEvaluator:
    """Children computed by qccompute, or any qccodec ``Compute``, one list per round.

    Args:
        compute: ``compute(inputs: list[ProgramInput]) -> list[ProgramOutput]``;
            default ``qccompute.compute``.
    """

    def __init__(self, compute: Callable[..., Any] | None = None) -> None:
        """Bind the compute callable."""
        if compute is None:
            from qccompute import compute as qccompute_compute

            def compute(inputs: Any) -> Any:
                return qccompute_compute(inputs, raise_exc=False)

        self.compute = compute

    def supports_hessian(self, context: TaskContext) -> bool:
        """Hessians come from the Hessian subprogram when one is given."""
        return context.hessian_spec is not None

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Compute all requests in one call."""
        from qcdata import ProgramInput, Structure

        inputs = []
        for r in requests:
            ctx = r.context
            spec = ctx.hessian_spec if r.kind == "hessian" else ctx.gradient_spec
            structure = ctx.extras.get("structure")
            if structure is None:
                structure = Structure(
                    symbols=ctx.symbols,
                    geometry=r.positions,
                    charge=ctx.charge,
                    multiplicity=ctx.multiplicity,
                )
            else:
                structure = structure.model_copy(
                    update={"geometry": np.asarray(r.positions, dtype=float)}
                )
            inputs.append(ProgramInput.from_spec(spec, structure))
        outputs = self.compute(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        return [
            _compute_answer(r, out) for r, out in zip(requests, outputs, strict=True)
        ]

    def hessian_includes_gradient(self, context: TaskContext) -> bool:
        """No: a codec's Hessian output need not carry the gradient (xtb's doesn't)."""
        return False


def _compute_answer(r: Request, out: Any) -> Answer:
    """One request's answer from its ``ProgramOutput``; unreadable ones fail."""
    x = np.asarray(r.positions, dtype=float)
    res = getattr(out, "results", None)
    prov = getattr(res, "provenance", None)
    program = getattr(prov, "program", None)
    version = getattr(prov, "program_version", None)
    if not getattr(out, "success", False):
        error = getattr(out, "traceback", None) or "the calculation failed"
        return Answer(
            positions=x,
            success=False,
            error=error,
            output=out,
            program=program,
            program_version=version,
        )
    try:
        energy = getattr(res, "energy", None)
        gradient = getattr(res, "gradient", None)
        hessian = getattr(res, "hessian", None)
        if r.kind == "gradient" and (energy is None or gradient is None):
            raise ValueError("the output has no energy or gradient")
        if r.kind == "hessian" and hessian is None:
            raise ValueError("the output has no Hessian")
        return Answer(
            positions=x,
            energy=float("nan") if energy is None else float(energy),
            gradient=None
            if gradient is None
            else np.asarray(gradient, dtype=float).reshape(-1, 3),
            hessian=None if hessian is None else np.asarray(hessian, dtype=float),
            output=out,
            program=program,
            program_version=version,
        )
    except Exception as exc:  # noqa: BLE001 - reported in the answer
        return Answer(
            positions=x,
            success=False,
            error=f"reading the {program or 'program'} output: {exc}",
            output=out,
            program=program,
            program_version=version,
        )


class RoutingEvaluator:
    """Sends each task to the evaluator chosen for it; one call per evaluator."""

    def __init__(
        self,
        evaluators: Sequence[Any],
        choice: Sequence[int],
        contexts: Sequence[TaskContext] | None = None,
    ) -> None:
        """Bind evaluators, the index of the one for each task, and the tasks."""
        self.evaluators = list(evaluators)
        self.choice = list(choice)
        self._by_context = (
            {}
            if contexts is None
            else {
                id(ctx): self.evaluators[e]
                for ctx, e in zip(contexts, choice, strict=True)
            }
        )

    def _for(self, context: TaskContext) -> Any:
        return self._by_context.get(id(context))

    def supports_hessian(self, context: TaskContext) -> bool:
        """Delegate to the task's evaluator."""
        ev = self._for(context)
        if ev is None:
            return context.hessian_spec is not None
        return bool(ev.supports_hessian(context))

    def hessian_includes_gradient(self, context: TaskContext) -> bool:
        """Delegate to the task's evaluator; for an unknown task, False if any says so."""
        ev = self._for(context)
        candidates = self.evaluators if ev is None else [ev]
        for e in candidates:
            check = getattr(e, "hessian_includes_gradient", None)
            if check is not None and not check(context):
                return False
        return True

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Split by evaluator, compute, and merge back in order."""
        groups: dict[int, list[int]] = defaultdict(list)
        for k, r in enumerate(requests):
            groups[self.choice[r.task]].append(k)
        answers: list[Answer | None] = [None] * len(requests)
        for e, idx in groups.items():
            got = self.evaluators[e]([requests[k] for k in idx])
            for k, a in zip(idx, got, strict=True):
                answers[k] = a
        return [a for a in answers if a is not None]


def default_evaluator(contexts: Sequence[TaskContext]) -> Any:
    """Axtb in-process for axtb tasks when axtb is installed, qccompute otherwise."""
    try:
        import axtb  # noqa: F401

        have_axtb = True
    except ImportError:
        have_axtb = False
    use_axtb = [have_axtb and AxtbEvaluator.handles(ctx) for ctx in contexts]
    if all(use_axtb):
        return AxtbEvaluator()
    if not any(use_axtb):
        return ComputeEvaluator()
    return RoutingEvaluator(
        [AxtbEvaluator(), ComputeEvaluator()],
        [0 if u else 1 for u in use_axtb],
        contexts,
    )
