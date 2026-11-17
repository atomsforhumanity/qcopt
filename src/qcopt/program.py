"""qcopt as an AFH program (``qccompute.program``), and the evaluator over program sessions.

Registered as the ``qccompute.programs`` entry point ``qcopt``, so qccompute runs
``ProgramInput(program="qcopt", ...)`` natively, in its own process: qcopt's settings
are checked by :meth:`QcoptProgram.plan`, and the gradients and Hessians of every task
come from sessions of the child programs (:class:`SessionEvaluator`): one per task and
program, warm-started from geometry to geometry, with each IRC branch its own lineage.
A program without sessions of its own gets a session over ``qccompute.compute`` from
the runner, so every child program works.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from qccompute.program import (
    Capabilities,
    Plan,
    Resources,
    RunContext,
    Session,
    find_program,
    local_context,
    plan_of,
)
from qcdata import (
    CalcType,
    Data,
    ErrorInfo,
    Frame,
    ProgramError,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Resolved,
    Structure,
)
from qcdata import Request as SessionRequest

from .api import CALCTYPES, PROGRAM, _prepare, _same_spec, _version, run
from .driver import Answer, Request, TaskContext
from .errors import InputError
from .evaluators import AxtbEvaluator, RoutingEvaluator, axtb_in_process, summary
from .options import options_schema

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
            if not _same_spec(ctx.hessian_spec, ctx.gradient_spec):
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
                if _same_spec(ctx.hessian_spec, ctx.gradient_spec):
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
    """One session's requests as session requests; a session that raises fails all."""
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
    from qcdata import Provenance, SinglePointData

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


class QcoptProgram:
    """qcopt under the AFH program protocol (``qccompute.program.Program``)."""

    name = PROGRAM

    def version(self) -> str:
        """Qcopt's version."""
        return _version() or "unknown"

    def capabilities(self) -> Capabilities:
        """Minima, transition states and IRCs; a driver; inputs advance together."""
        return Capabilities(
            calctypes=frozenset(CALCTYPES),
            sessions=False,
            batches=True,
            in_process=True,
            thread_safe=False,
            driver=True,
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
            p = _prepare(inp)
        except InputError as exc:
            raise ProgramError.of("input", str(exc), code=exc.code) from None
        return Plan(
            resolved=Resolved(settings=p.resolved, reasons=p.reasons),
            resources=_plan_children(inp),
        )

    def compute(
        self, inp: ProgramInput, ctx: RunContext
    ) -> ProgramOutput[ProgramInput, Data]:
        """Run one calculation; its children through sessions of ``ctx``."""
        return self.compute_batch([inp], ctx)[0]

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Run several calculations in lockstep; one output per input, in order."""
        if ctx.open_session is None and ctx.compute is None:
            # a direct call, without a runner: children in this process
            ctx = local_context(threads=ctx.threads)

        def open_session(
            inp: ProgramInput, threads: int | None, calctypes: set[str]
        ) -> Session:
            return ctx.session(inp, threads=threads, calctypes=calctypes)

        outputs: list[ProgramOutput[ProgramInput, Data] | None] = []
        valid: list[ProgramInput] = []
        in_process: list[bool] = []
        for inp in inputs:  # an input qcopt cannot run fails alone, with kind input
            try:
                prepared = _prepare(inp)
            except InputError as exc:
                outputs.append(_input_failure(inp, exc))
            else:
                outputs.append(None)
                valid.append(inp)
                in_process.append(axtb_in_process(prepared.context))
        sessions = SessionEvaluator(open_session, ctx.threads)
        evaluator: Any = sessions
        if any(in_process):
            evaluator = RoutingEvaluator(
                [AxtbEvaluator(workers=1), sessions],
                [0 if a else 1 for a in in_process],
            )
        try:
            done = iter(run(valid, evaluator=evaluator) if valid else [])
        finally:
            sessions.close()
        return [out if out is not None else next(done) for out in outputs]


def _input_failure(
    inp: ProgramInput, exc: InputError
) -> ProgramOutput[ProgramInput, Data]:
    """The output of an input qcopt cannot run (nothing was computed)."""
    from qcdata import OptimizationData, PathData, Provenance

    prov = Provenance(program=PROGRAM, program_version=_version())
    data: Any = (
        PathData(provenance=prov)
        if inp.calctype == CalcType.irc
        else OptimizationData(provenance=prov)
    )
    return ProgramOutput(
        input_data=inp,
        success=False,
        traceback=str(exc),
        error=ErrorInfo(kind="input", message=str(exc), code=exc.code),
        results=data,
    )


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
            _same_spec(h, child) for h in hessians
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
