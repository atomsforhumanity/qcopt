"""Evaluators: what computes the gradients and Hessians the optimizers ask for.

qcopt has no route of its own to programs. Every gradient and Hessian comes from a
session the runner opens (``RunContext.session``): the program's own sessions, which
warm-start from step to step, or a session over its ``compute``
(:class:`SessionEvaluator`).

A program without sessions that evaluates many inputs at once (an ML potential) gets each round's requests of every task
as one list through ``ctx.compute`` (:class:`ComputeEvaluator`), so the runner can batch
them. :func:`route` chooses per task, :func:`evaluator` builds the evaluator.
"""

from __future__ import annotations

import re
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


#: The concerns (qcdata's conventions: ``x-concern`` on every settings key) of the keys
#: that define a calculation's energy surface. Convergence, algorithm, resource and
#: output keys do not: a Hessian computed tighter or another way is still the same
#: method's.
SURFACE_CONCERNS = frozenset({"model", "approximation"})


def surface_keywords(spec: Any) -> dict[str, Any]:
    """A spec's keywords that define its energy surface.

    The program's settings schema says (the keys whose ``x-concern`` is in
    :data:`SURFACE_CONCERNS`; a key it does not know or gives no concern counts). Without
    a schema (a program not found, or one that publishes none), every key but
    ``threads``.
    """
    keywords = dict(spec.keywords)
    schema = _settings_schema(spec)
    if schema is None:
        return {k: v for k, v in keywords.items() if k != "threads"}
    return _surface(keywords, schema, schema)


def _settings_schema(spec: Any) -> dict[str, Any] | None:
    program = find_program(spec.program)
    read = getattr(program, "settings_schema", None)
    if read is None:
        return None
    try:
        schema = read(spec.calctype)
    except Exception:  # noqa: BLE001 - no schema to read: count every key
        return None
    return schema if isinstance(schema, dict) else None


def _node(node: Any, root: dict[str, Any]) -> dict[str, Any]:
    """A schema node with a local ``$ref`` (or a nullable ``anyOf`` of one) resolved."""
    if not isinstance(node, dict):
        return {}
    for option in node.get("anyOf", ()):
        if isinstance(option, dict) and option.get("type") != "null":
            node = {**option, **{k: v for k, v in node.items() if k != "anyOf"}}
            break
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}
        node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    return node


def _surface(
    values: dict[str, Any], schema: dict[str, Any], root: dict[str, Any]
) -> dict[str, Any]:
    props = _node(schema, root).get("properties") or {}
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key not in props:  # unknown to the schema: it may define the surface
            out[key] = value
            continue
        node = _node(props[key], root)
        concern = node.get("x-concern")
        if concern is not None:
            if concern in SURFACE_CONCERNS:
                out[key] = value
        elif isinstance(value, dict) and "properties" in node:
            inner = _surface(value, node, root)
            if inner:
                out[key] = inner
        else:
            out[key] = value
    return out


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


def route(context: TaskContext) -> str:
    """How a task's gradients are computed: ``compute`` or ``sessions``.

    - ``compute``: a gradient program without sessions that evaluates many inputs at
      once (an ML potential on a GPU): every task's requests of a round in one list
      (:class:`ComputeEvaluator`), which the runner hands it as one batch;
    - ``sessions``: everything else, the program's own sessions (warm-started) or a
      session over its ``compute`` (:class:`SessionEvaluator`).
    """
    program = find_program(context.gradient_spec.program)
    if program is not None:
        caps = capabilities_of(program)
        if caps.batches and not caps.sessions:
            return "compute"
    return "sessions"


def evaluator(
    contexts: Sequence[TaskContext], ctx: RunContext
) -> SessionEvaluator | RoutingEvaluator:
    """What computes each task (:func:`route`), all through ``ctx``.

    Args:
        contexts: The tasks.
        ctx: The runner's context: the grant and the way to run subprograms.

    Returns:
        The evaluator; ``close()`` closes the sessions it opened.
    """

    def open_session(
        inp: ProgramInput, threads: int | None, calctypes: set[str]
    ) -> Session:
        return ctx.session(inp, threads=threads, calctypes=calctypes)

    sessions = SessionEvaluator(open_session, ctx.threads)
    routes = [route(c) for c in contexts]
    if set(routes) == {"sessions"}:
        return sessions
    makers: dict[str, Callable[[], Any]] = {
        "compute": lambda: ComputeEvaluator(ctx),
        "sessions": lambda: sessions,
    }
    used = [k for k in makers if k in routes]
    return RoutingEvaluator(
        [makers[k]() for k in used], [used.index(k) for k in routes], contexts
    )


class ComputeEvaluator:
    """Every task's requests of a round as one list through ``ctx.compute``.

    For programs without sessions that evaluate many inputs at once: the runner hands
    the list to the program as one batch, so the gradients of many minimizations are
    one GPU pass per round.

    Args:
        ctx: The runner's context.
    """

    def __init__(self, ctx: RunContext) -> None:
        """Bind the context."""
        self.ctx = ctx

    def supports_hessian(self, context: TaskContext) -> bool:
        """Hessians come from the Hessian subprogram when one is given."""
        return context.hessian_spec is not None

    def hessian_includes_gradient(self, context: TaskContext) -> bool:
        """No: a program's Hessian output need not carry the gradient."""
        return False

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Compute all requests in one list; failures in place."""
        inputs = []
        for r in requests:
            ctx = r.context
            spec = ctx.hessian_spec if r.kind == "hessian" else ctx.gradient_spec
            assert spec is not None  # Hessians are asked for only with a Hessian spec
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
        outputs = self.ctx.compute(inputs, raise_exc=False)
        return [
            _compute_answer(r, out) for r, out in zip(requests, outputs, strict=True)
        ]


def _compute_answer(r: Request, out: Any) -> Answer:
    """One request's answer from its ``ProgramOutput``; unreadable ones fail."""
    x = np.asarray(r.positions, dtype=float)
    res = getattr(out, "results", None)
    prov = getattr(res, "provenance", None)
    program = getattr(prov, "program", None)
    version = getattr(prov, "program_version", None)
    if not getattr(out, "success", False):
        tb = getattr(out, "traceback", None)
        return Answer(
            positions=x,
            success=False,
            error=summary(getattr(out, "error", None), tb),
            traceback=tb,
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


#: Opens a session: (input, threads, calctypes) -> Session, as ``RunContext.session`` does.
OpenSession = Callable[[ProgramInput, "int | None", "set[str]"], Session]


class SessionEvaluator:
    """Gradients and Hessians from program sessions: one per task and program.

    Each task's requests of a round go to its sessions in one ``compute_many`` call
    (so the 6N displaced gradients of a finite-difference Hessian are one batch);
    tasks run side by side in threads (the runner makes that safe): as many at a time
    as qcopt's thread count allows, each task's sessions opened with the count divided
    by the tasks running (at least 1), as qccompute's rule for a parent's children
    asks. A request continues its lineage's history (the
    task, and the IRC branch); the displaced gradients of a finite-difference Hessian
    start from the centre's and do not become history. A task's sessions close when it
    finishes.

    Args:
        open_session: Opens a session from the child's input (its spec at the task's
            starting structure), its threads and the calctypes it serves:
            ``RunContext.session`` with those keywords.
        threads: qcopt's thread count, shared by the tasks running side by side
            (None: not granted; each session takes its program's own).
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

    def _split(
        self, tasks: Sequence[list[int]], requests: Sequence[Request]
    ) -> tuple[int, int | None]:
        """How many tasks run at once, and each one's sessions' threads.

        As many as the count allows when each takes at least the threads its programs'
        settings ask for (``threads``; 1 if they do not say), and the count divided
        among those, at least 1.
        """
        if self.threads is None:
            return len(tasks), None
        need = max(_threads(self._role(requests[k])[1]) for ks in tasks for k in ks)
        workers = max(1, min(len(tasks), self.threads // need))
        return workers, max(1, self.threads // workers)

    @staticmethod
    def _role(r: Request) -> tuple[str, ProgramSpec]:
        """The session a request goes to: its role, and the program spec it opens."""
        ctx = r.context
        if r.kind == "hessian" and ctx.hessian_spec is not None:
            if not same_spec(ctx.hessian_spec, ctx.gradient_spec):
                return "hessian", ctx.hessian_spec
        return "gradient", ctx.gradient_spec

    def _session(self, r: Request, threads: int | None) -> Session:
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
                threads,
                calctypes,
            )
        return self._sessions[key]

    def __call__(self, requests: Sequence[Request]) -> list[Answer]:
        """Compute a round: each task's requests in order, tasks side by side."""
        by_task: dict[int, list[int]] = defaultdict(list)
        for k, r in enumerate(requests):
            by_task[r.task].append(k)
        workers, threads = self._split(list(by_task.values()), requests)
        answers: list[Answer | None] = [None] * len(requests)

        def run_task(indices: list[int]) -> None:
            by_session: dict[int, tuple[Session, list[int]]] = {}
            for k in indices:
                try:
                    session = self._session(requests[k], threads)
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
