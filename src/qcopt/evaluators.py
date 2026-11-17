"""Evaluators: what computes the gradients and Hessians the optimizers ask for.

qcopt has no route of its own to programs. Every gradient and Hessian comes from a
session the runner opens (``RunContext.session``): the program's own sessions, which
warm-start from step to step, or a session over its ``compute``
(:class:`SessionEvaluator`).

Until axtb has sessions of its own, axtb tasks run on :class:`AxtbEvaluator`: axtb in
this process, one warm-started calculator per task. :func:`evaluator` sends them there
and every other task to the sessions (:class:`RoutingEvaluator`).
"""

from __future__ import annotations

import os
import re
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from typing import Any

import numpy as np
from qccompute.program import RunContext, Session, capabilities_of, find_program
from qcdata import (
    CalcType,
    ErrorInfo,
    Frame,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
)
from qcdata import Request as SessionRequest

from .driver import Answer, Request, TaskContext


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


_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)(: |$)")


def summary(error: Any, traceback_text: str | None) -> str:
    """A failed calculation's message in one line.

    The first line of the program's own error (``ErrorInfo.message``: AFH's programs
    and qccompute attach one to every failure). For a program that reports only a
    traceback, its last exception line (``ValueError: ...``, not the notes some
    libraries print after it), else its last line.
    """
    message = getattr(error, "message", None)
    if message and message.strip():
        return message.strip().splitlines()[0].strip()
    text = traceback_text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if _EXCEPTION_LINE.match(line):
            return line
    return lines[-1] if lines else "the calculation failed"


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
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
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

    def forget(self, task: int) -> None:
        """Pass on that a task has finished."""
        forget = getattr(self.evaluators[self.choice[task]], "forget", None)
        if forget is not None:
            forget(task)

    def close(self) -> None:
        """Close the evaluators that hold something open (program sessions)."""
        for ev in self.evaluators:
            close = getattr(ev, "close", None)
            if close is not None:
                close()

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


def same_spec(a: ProgramSpec, b: ProgramSpec) -> bool:
    """Whether two specs differ only in their calculation type."""
    return a.model_copy(update={"calctype": b.calctype}) == b


def axtb_in_process(context: TaskContext) -> bool:
    """Whether a task runs on qcopt's own axtb evaluator.

    Only until axtb is a program with sessions: a session over axtb's ``compute`` would
    make every gradient a separate calculation (30x slower on small molecules), where
    qcopt's in-process evaluator keeps one warm-started calculator per task.
    """
    if not AxtbEvaluator.handles(context):
        return False
    program = find_program("axtb")
    if program is not None and capabilities_of(program).sessions:
        return False
    try:
        import axtb  # noqa: F401
    except ImportError:
        return False
    return True


def evaluator(
    contexts: Sequence[TaskContext], ctx: RunContext
) -> SessionEvaluator | RoutingEvaluator:
    """What computes each task: sessions from ``ctx``, axtb in process until it has its own.

    Args:
        contexts: The tasks.
        ctx: The runner's context: the grant and the way to open child sessions.

    Returns:
        The evaluator; ``close()`` closes the sessions it opened.
    """

    def open_session(
        inp: ProgramInput, threads: int | None, calctypes: set[str]
    ) -> Session:
        return ctx.session(inp, threads=threads, calctypes=calctypes)

    sessions = SessionEvaluator(open_session, ctx.threads)
    in_process = [axtb_in_process(c) for c in contexts]
    if not any(in_process):
        return sessions
    return RoutingEvaluator(
        [AxtbEvaluator(workers=1), sessions],
        [0 if a else 1 for a in in_process],
        contexts,
    )


#: Opens a session: (input, threads, calctypes) -> Session, as ``RunContext.open_session``.
OpenSession = Callable[[ProgramInput, "int | None", "set[str]"], Session]


class SessionEvaluator:
    """Gradients and Hessians from program sessions: one per task and program.

    Each task's requests of a round go to its sessions in one ``compute_many`` call
    (so the 6N displaced gradients of a finite-difference Hessian are one batch);
    tasks run side by side in threads (the runner makes that safe), as many at a time
    as the granted threads allow. A request continues its lineage's history (the
    task, and the IRC branch); the displaced gradients of a finite-difference Hessian
    start from the centre's and do not become history. A task's sessions close when it
    finishes.

    Args:
        open_session: Opens a session from the child's input (its spec at the task's
            starting structure), its threads and the calctypes it serves:
            ``RunContext.open_session``, or ``qccompute.session`` with those keywords.
        threads: The threads to share among the tasks' sessions (None: the
            programs' own settings).
    """

    def __init__(self, open_session: OpenSession, threads: int | None = None) -> None:
        """Bind the session opener; no session is opened yet."""
        self.open_session = open_session
        self.threads = threads
        self._sessions: dict[tuple[int, str], Session] = {}
        self._tasks: set[int] = set()

    def supports_hessian(self, context: TaskContext) -> bool:
        """Hessians come from the Hessian subprogram's session when one is given."""
        return context.hessian_spec is not None

    def hessian_includes_gradient(self, context: TaskContext) -> bool:
        """Yes: a session's Hessian comes with the energy and gradient."""
        return True

    def _share(self, n_tasks: int) -> int | None:
        if self.threads is None:
            return None
        return max(1, self.threads // max(1, n_tasks))

    @staticmethod
    def _role(r: Request) -> tuple[str, ProgramSpec]:
        """The session a request goes to: its role, and the program spec it opens."""
        ctx = r.context
        if r.kind == "hessian" and ctx.hessian_spec is not None:
            if not same_spec(ctx.hessian_spec, ctx.gradient_spec):
                return "hessian", ctx.hessian_spec
        return "gradient", ctx.gradient_spec

    def _session(self, r: Request, n_tasks: int) -> Session:
        ctx = r.context
        role, spec = self._role(r)
        key = (r.task, role)
        if key not in self._sessions:
            structure = ctx.extras.get("structure") or Structure(
                symbols=ctx.symbols,
                geometry=r.positions,
                charge=ctx.charge,
                multiplicity=ctx.multiplicity,
            )
            # the calctypes this session serves: the gradient program's also the
            # Hessians when they come from the same calculation
            calctypes = {role}
            if role == "gradient" and ctx.hessian_spec is not None:
                if same_spec(ctx.hessian_spec, ctx.gradient_spec):
                    calctypes.add("hessian")
            self._sessions[key] = self.open_session(
                ProgramInput.from_spec(spec, structure),
                self._share(n_tasks),
                calctypes,
            )
        return self._sessions[key]

    def _workers(self, tasks: Sequence[list[int]], requests: Sequence[Request]) -> int:
        """How many tasks run at once: all, or as many as the granted threads allow.

        A task needs its share of the grant, or the threads its programs' settings ask
        for (``threads``; a program run through ``qccompute.compute`` takes those),
        whichever is more.
        """
        if self.threads is None:
            return len(tasks)
        share = self._share(len(tasks)) or 1
        need = max(
            max([share] + [_threads(self._role(requests[k])[1]) for k in ks])
            for ks in tasks
        )
        return max(1, min(len(tasks), self.threads // need))

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Compute a round: each task's requests in order, tasks side by side."""
        by_task: dict[int, list[int]] = defaultdict(list)
        for k, r in enumerate(requests):
            by_task[r.task].append(k)
        n_tasks = len(by_task)
        answers: list[Answer | None] = [None] * len(requests)

        def run_task(indices: list[int]) -> None:
            by_session: dict[int, tuple[Session, list[int]]] = {}
            for k in indices:
                try:
                    session = self._session(requests[k], n_tasks)
                except Exception as exc:  # noqa: BLE001 - this request's answer
                    # the program refused its settings or could not start: this
                    # task's request fails with the program's error, not the round
                    answers[k] = _not_opened(
                        requests[k], self._role(requests[k])[1], exc
                    )
                    continue
                by_session.setdefault(id(session), (session, []))[1].append(k)
            for session, ks in by_session.values():
                frames = _compute_many(session, [requests[k] for k in ks])
                for k, frame in zip(ks, frames, strict=True):
                    answers[k] = _answer(requests[k], frame, session)

        workers = self._workers(list(by_task.values()), requests)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run_task, by_task.values()))
        else:
            for indices in by_task.values():
                run_task(indices)
        return [a for a in answers if a is not None]

    def forget(self, task: int) -> None:
        """Close a finished task's sessions."""
        for key in [k for k in self._sessions if k[0] == task]:
            self._sessions.pop(key).close()

    def close(self) -> None:
        """Close every session."""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()


def _threads(spec: ProgramSpec) -> int:
    """The threads a spec's settings ask for (1 if they do not say)."""
    threads = spec.keywords.get("threads")
    return threads if isinstance(threads, int) and threads > 0 else 1


def _compute_many(session: Session, requests: Sequence[Request]) -> list[Frame]:
    """One session's requests as session requests, one frame each, failures in place.

    A session runs every request, then raises ``ProgramErrors`` holding every frame in
    order when some failed. One that raises anything else fails every request.
    """
    batch = [
        SessionRequest(
            positions=np.asarray(r.positions, dtype=float).reshape(-1, 3),
            calctype=CalcType(r.kind),
            lineage=(r.task, r.lineage),
            commit=r.commit,
        )
        for r in requests
    ]
    try:
        return session.compute_many(batch)
    except ProgramErrors as group:
        return list(group.results)
    except ProgramError as exc:
        return [_failed(r, exc.error) for r in requests]
    except Exception as exc:  # noqa: BLE001 - reported in the answers
        error = ErrorInfo(kind="internal", message=f"{type(exc).__name__}: {exc}")
        return [_failed(r, error) for r in requests]


def _not_opened(r: Request, spec: ProgramSpec, exc: Exception) -> Answer:
    """A request whose session could not be opened, failed with the program's error."""
    error = (
        exc.error
        if isinstance(exc, ProgramError)
        else ErrorInfo(kind="internal", message=f"{type(exc).__name__}: {exc}")
    )
    return Answer(
        positions=np.asarray(r.positions, dtype=float),
        success=False,
        error=summary(error, None),
        output=_failed(r, error, spec.program),
        program=spec.program,
    )


def _failed(r: Request, error: ErrorInfo, program: str = "unknown") -> Frame:
    ctx = r.context
    structure = Structure(
        symbols=ctx.symbols,
        geometry=np.asarray(r.positions, dtype=float),
        charge=ctx.charge,
        multiplicity=ctx.multiplicity,
    )
    return Frame(
        success=False,
        traceback=error.message,
        error=error,
        structure=structure,
        results=SinglePointData(provenance=Provenance(program=program)),
    )


def _answer(r: Request, frame: Frame, session: Session) -> Answer:
    """A frame from a session as the driver's answer (the frame becomes its output)."""
    x = np.asarray(r.positions, dtype=float)
    res = frame.results
    if not frame.success:
        return Answer(
            positions=x,
            success=False,
            error=summary(frame.error, frame.traceback),
            traceback=frame.traceback,
            output=frame,
            program=session.program,
            program_version=session.program_version,
        )
    return Answer(
        positions=x,
        energy=float("nan") if res.energy is None else float(res.energy),
        gradient=None
        if res.gradient is None
        else np.asarray(res.gradient, dtype=float).reshape(-1, 3),
        hessian=None if res.hessian is None else np.asarray(res.hessian, dtype=float),
        output=frame,
        program=session.program,
        program_version=session.program_version,
    )
