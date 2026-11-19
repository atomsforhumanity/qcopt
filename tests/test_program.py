"""qcopt as an AFH program: run natively by qccompute, children through sessions."""

import numpy as np
import pytest
import qccompute
import qccompute.program as program_protocol
from qccompute.program import FunctionProgram
from qccompute.testing import toy
from qcdata import (
    CalcType,
    Frame,
    ProgramError,
    ProgramInput,
    ProgramSpec,
    SinglePointData,
    Structure,
)

import qcopt
from qcopt.calculation import prepare
from qcopt.evaluators import SessionEvaluator, evaluator
from qcopt.program import program

from .helpers import irc, minimize, run
from .surfaces import (
    MB_MINIMA,
    MB_SADDLES,
    muller_brown,
    muller_brown_hessian,
)

SIGMA = toy.DEFAULT_SIGMA
R_MIN = 2 ** (1 / 6) * SIGMA  # the Lennard-Jones pair minimum
TOY = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})


@pytest.fixture(autouse=True)
def toy_registered():
    program_protocol.register_program(toy.ToyProgram())
    yield
    program_protocol.unregister_program("toy")


def _trimer(scale=1.1):
    r = scale * R_MIN
    x = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [0.4 * r, 0.8 * r, 0.1]])
    return Structure(symbols=["Ar"] * 3, geometry=x)


def _input(calctype="optimization", structure=None, children=(TOY,), **keywords):
    return ProgramInput(
        program="qcopt",
        calctype=calctype,
        structure=structure or _trimer(),
        keywords=keywords,
        subprograms=list(children),
    )


def _distances(x):
    x = np.asarray(x).reshape(-1, 3)
    return sorted(
        float(np.linalg.norm(x[i] - x[j]))
        for i in range(len(x))
        for j in range(i + 1, len(x))
    )


def test_qccompute_runs_qcopt_natively_on_a_session_program():
    out = qccompute.compute(_input())
    assert out.success, out.traceback
    assert np.allclose(
        _distances(out.results.final_structure.geometry), R_MIN, atol=2e-3
    )
    # qcopt's own slots, straight through (no codec in between)
    assert out.resolved.settings["hessian"]["initial"] == "model"
    assert out.record["converged"] is True
    assert out.execution.timings["subprograms"] > 0
    # the frames are the session's: its program and its per-step record
    frames = out.results.frames
    assert all(f.results.provenance.program == "toy" for f in frames)
    assert frames[0].record["warm_start"] is False
    assert all(f.record["warm_start"] for f in frames[1:])


def test_plan_resolves_settings_and_states_no_threads():
    # qccompute 0.23: a calculation runs on its threads, else its caller's count
    spec = TOY.model_copy(update={"keywords": {"threads": 3}})
    plan = program.plan(_input(children=[spec]))
    assert plan.resolved.settings["coordinate_system"] == "tric"
    assert "threads" not in plan.resources.model_dump(exclude_unset=True)
    with pytest.raises(ProgramError) as info:
        program.plan(_input(nonsense=1))
    assert info.value.error.kind == "input"


class _ScalingProgram:
    """A program whose plan names its memory, as DFT's does (and no threads)."""

    def plan(self, inp):
        from qcdata import Resolved

        return program_protocol.Plan(
            resolved=Resolved(settings={}),
            resources=program_protocol.Resources(memory_gib=3.0),
        )


def test_qcopt_s_plan_takes_its_children_s_memory_not_their_threads(monkeypatch):
    scaling = _ScalingProgram()
    monkeypatch.setattr(
        "qcopt.program.find_program",
        lambda name: scaling if name == "dft" else program_protocol.find_program(name),
    )
    dft = ProgramSpec(program="dft", calctype="hessian", model={"method": "b3lyp"})
    res = program.plan(_input(children=[TOY, dft])).resources
    assert "threads" not in res.model_dump(exclude_unset=True)
    assert res.memory_gib == 3.0


class _FailingSession:
    program, program_version = "broken", "1"

    def compute_many(self, requests):
        from qccompute.program import failed_frame
        from qcdata import ErrorInfo

        error = ProgramError(
            ErrorInfo(kind="convergence", message="SCF did not converge")
        )
        s = Structure(symbols=["Ar"] * 3, geometry=np.zeros((3, 3)))
        return [failed_frame(s, error, self.program) for _ in requests]

    def close(self):
        pass


def test_a_failing_child_is_a_program_error_with_its_own_error_attached():
    ev = SessionEvaluator(lambda inp, t, c: _FailingSession())
    out = run(_input(), evaluator=ev)
    assert not out.success
    assert out.error.kind == "program"
    assert out.error.program == "broken"
    assert out.error.cause is not None and out.error.cause.kind == "convergence"
    # AFH's rule for a child's failure: the child and calctype first
    assert out.error.message == (
        "broken (gradient): SCF did not converge (at the starting geometry)"
    )


def test_the_python_api_runs_a_program_with_sessions_through_them():
    out = qcopt.compute(_input())
    assert out.success, out.traceback
    frames = out.results.frames
    assert frames[0].record["warm_start"] is False
    assert all(f.record["warm_start"] for f in frames[1:])


def test_every_task_runs_on_sessions_the_runner_opens():
    other = ProgramSpec(
        program="no-such-program", calctype="gradient", model={"method": "x"}
    )
    ctx = program_protocol.RunContext()
    contexts = [prepare(_input()).context, prepare(_input(children=[other])).context]
    assert isinstance(evaluator(contexts, ctx), SessionEvaluator)
    out = program.compute(_input(), ctx)
    assert out.results.frames[-1].results.provenance.program == "toy"


def test_several_inputs_advance_together():
    inputs = [_input(structure=_trimer(s)) for s in (1.05, 1.1, 1.2)]
    outs = program.compute_batch(inputs, program_protocol.RunContext())
    assert all(o.success for o in outs)
    # through the runner: a list of qcopt inputs is one batch
    assert all(o.success for o in qccompute.compute(inputs))


def test_a_child_that_refuses_its_settings_fails_only_its_own_calculation():
    bad = TOY.model_copy(update={"keywords": {"nonsense": 1}})
    outs = program.compute_batch(
        [_input(children=[bad]), _input()], program_protocol.RunContext()
    )
    assert not outs[0].success
    assert (outs[0].error.kind, outs[0].error.program) == ("program", "toy")
    assert outs[0].error.message.startswith("toy (gradient): ")
    assert (outs[0].error.cause.kind, outs[0].error.cause.code) == (
        "input",
        "unknown_key",
    )
    assert outs[1].success, outs[1].traceback


def test_plan_checks_the_settings_of_its_children():
    assert program.plan(_input()).resolved.settings
    bad = TOY.model_copy(update={"keywords": {"nonsense": 1}})
    with pytest.raises(ProgramError) as info:
        program.plan(_input(children=[bad]))
    error = info.value.error
    assert (error.kind, error.program) == ("program", "toy")
    assert (error.cause.kind, error.cause.code) == ("input", "unknown_key")


class _HessianSettingsProgram:
    """A program whose ``freq`` settings apply to Hessians only (as linus's do)."""

    def plan(self, inp):
        from qcdata import Resolved

        if "freq" in inp.keywords and str(inp.calctype) != "hessian":
            raise ProgramError.of(
                "input", "freq applies to Hessians", code="not_applicable"
            )
        return program_protocol.Plan(
            resolved=Resolved(settings={}), resources=program_protocol.Resources()
        )


def test_plan_takes_one_spec_for_gradients_and_hessians_as_one_session(monkeypatch):
    monkeypatch.setattr(
        "qcopt.program.find_program", lambda name: _HessianSettingsProgram()
    )
    spec = TOY.model_copy(update={"keywords": {"freq": {"analysis": "off"}}})
    hessian = spec.model_copy(update={"calctype": CalcType.hessian})
    program.plan(_input(children=[spec, hessian]))  # one session: valid for Hessians
    with pytest.raises(ProgramError) as info:  # gradients alone
        program.plan(_input(children=[spec]))
    assert info.value.error.cause.code == "not_applicable"


def test_the_program_meets_the_afh_contract():
    from qccompute.testing import check_program

    check_program(program, [_input()])


def test_a_direct_call_runs_the_children_in_this_process():
    # without a runner, qcopt finds its children among the registered and installed
    # programs
    out = qcopt.compute(_input())
    assert out.results.frames[-1].results.provenance.program == "toy"
    assert program.compute(_input(), program_protocol.RunContext()).success
    program_protocol.unregister_program("toy")
    with pytest.raises(ProgramError) as info:
        qcopt.compute(_input())
    error = info.value.error
    assert (error.kind, error.program) == ("program", "toy")


def test_a_failing_hessian_program_is_the_one_named():
    # gradients from toy, Hessians from a program whose sessions fail: the message,
    # program and cause are the Hessian program's
    def open_session(inp, t, c):
        if inp.program == "broken":
            return _FailingSession()
        return program_protocol.RunContext().session(inp, threads=t, calctypes=c)

    broken = ProgramSpec(program="broken", calctype="hessian", model={"method": "x"})
    out = run(_input(children=[TOY, broken]), evaluator=SessionEvaluator(open_session))
    assert not out.success
    assert out.error.message == (
        "broken (hessian): SCF did not converge (at the starting geometry)"
    )
    assert (out.error.program, out.error.cause.kind) == ("broken", "convergence")


def test_a_failing_toy_session_fails_the_run_with_its_error():
    spec = TOY.model_copy(update={"keywords": {"fail": "convergence"}})
    out = qccompute.compute(_input(children=[spec]), raise_exc=False)
    assert not out.success
    assert (out.error.kind, out.error.program) == ("program", "toy")
    assert out.error.cause.kind == "convergence"


class _MBSession:
    """A Müller-Brown session that records each request's lineage."""

    program, program_version = "muller-brown", "1"

    def __init__(self, structure, log, commits=None):
        self.structure, self.log, self.commits = structure, log, commits

    def compute_many(self, requests):
        out = []
        for r in requests:
            self.log.append((str(r.calctype), r.lineage))
            if self.commits is not None:
                self.commits.append((r.lineage, r.commit))
            x = np.asarray(r.positions).reshape(-1, 3)
            e, g = muller_brown(x)
            h = muller_brown_hessian(x) if str(r.calctype) == "hessian" else None
            out.append(
                Frame(
                    success=True,
                    structure=self.structure.model_copy(update={"geometry": x}),
                    results=SinglePointData(
                        provenance={"program": self.program},
                        energy=e,
                        gradient=g,
                        hessian=h,
                    ),
                )
            )
        return out

    def close(self):
        self.log.append(("closed", None))


def test_irc_branches_and_their_end_minimizations_keep_their_own_lineage():
    log = []
    ev = SessionEvaluator(lambda inp, t, c: _MBSession(inp.structure, log))
    out = irc(
        Structure(symbols=["H"], geometry=[[*MB_SADDLES[0], 0.0]]),
        {"program": "muller-brown"},
        hessian_program=True,
        evaluator=ev,
        ends={"minimize": True},
    )
    assert out.success, out.traceback
    lineages = {lin for kind, lin in log if kind != "closed"}
    assert lineages == {(0, None), (0, "forward"), (0, "reverse")}
    assert log[-1] == ("closed", None)  # the task's session closed when it finished
    ends = sorted(
        (
            out.results.frames[0].structure.geometry[0, 0],
            out.results.frames[-1].structure.geometry[0, 0],
        )
    )
    assert np.isclose(ends[0], MB_MINIMA[0][0], atol=1e-3)


def test_a_calculation_s_threads_setting_is_what_it_asks_the_runner_for():
    # qcdata's shared resource key: the runner grants it, at most its own
    two = TOY.model_copy(update={"keywords": {"threads": 2}})
    with qccompute.Runner(threads=2) as runner:
        one = runner.compute(_input(children=[two], threads=1))
        many = runner.compute(_input(children=[two], threads=4))
    assert one.execution.threads == 1
    assert not one.warnings
    assert [(w.code, w.path) for w in many.warnings] == [("resource_capped", "threads")]
    assert one.resolved.settings["threads"] == 1


def test_without_threads_a_calculation_runs_on_its_caller_s_count():
    two = TOY.model_copy(update={"keywords": {"threads": 2}})
    with qccompute.Runner(threads=2) as runner:
        assert runner.compute(_input(children=[two])).execution.threads == 2


def test_each_calculation_of_a_list_reports_the_threads_it_ran_with():
    # ExecutionInfo.threads is what the calculation ran with: four calculations on
    # 8 threads ran with 2 each, not with the list's 8
    two = TOY.model_copy(update={"keywords": {"threads": 2}})
    inputs = [
        _input(structure=_trimer(1.1 + 0.01 * k), children=[two]) for k in range(4)
    ]
    outs = qccompute.Runner(threads=8).compute(inputs)
    assert [o.execution.threads for o in outs] == [2, 2, 2, 2]


class _ThreadedMBSession(_MBSession):
    """A Müller-Brown session whose frames record that they ran on 3 threads."""

    def compute_many(self, requests):
        return [
            f.model_copy(
                update={"execution": f.execution.model_copy(update={"threads": 3})}
            )
            for f in super().compute_many(requests)
        ]


def test_the_threads_children_recorded_are_the_calculations():
    ev = SessionEvaluator(lambda inp, t, c: _ThreadedMBSession(inp.structure, []))
    out = minimize(
        Structure(symbols=["H"], geometry=[[-0.5, 1.4, 0.0]]),
        {"program": "muller-brown"},
        evaluator=ev,
    )
    assert out.success, out.traceback
    assert out.execution.threads == 3


def test_displaced_gradients_start_from_the_centre_and_are_not_kept():
    # no Hessian program: the transition state's Hessian from finite differences
    log, commits = [], []
    ev = SessionEvaluator(lambda inp, t, c: _MBSession(inp.structure, log, commits))
    out = irc(
        Structure(symbols=["H"], geometry=[[*MB_SADDLES[0], 0.0]]),
        {"program": "muller-brown"},
        evaluator=ev,
        convergence={"max_iter": 2},
    )
    assert out.results.frames
    first = commits[:7]  # the centre, then 6 displacements, in one lineage
    assert [c for _, c in first] == [True] + [False] * 6
    assert {lin for lin, _ in first} == {(0, None)}


def test_tasks_run_side_by_side_on_the_count_split_among_them():
    from qcopt.driver import Request

    def split(threads, children, n=4):
        ctx = prepare(_input(children=children)).context
        requests = [Request(t, "gradient", np.zeros((3, 3)), ctx) for t in range(n)]
        return SessionEvaluator(None, threads)._split([[k] for k in range(n)], requests)

    two = TOY.model_copy(update={"keywords": {"threads": 2}})
    # (tasks at once, each one's sessions' threads)
    assert split(None, [TOY]) == (4, None)  # not granted: the programs' own
    assert split(8, [TOY]) == (4, 2)  # 8 // 4
    assert split(8, [TOY], n=16) == (8, 1)
    assert split(3, [TOY]) == (3, 1)
    assert split(8, [two]) == (4, 2)  # each child asks for 2
    assert split(4, [two]) == (2, 2)
    assert split(1, [two]) == (1, 1)


class _GroupSession(_MBSession):
    """Runs every request, then raises one group when some failed, as sessions do."""

    def compute_many(self, requests):
        from qccompute.program import failed_frame
        from qcdata import ErrorInfo, ProgramErrors

        frames = super().compute_many(requests)
        if len(requests) > 1:  # the displaced gradients: the third fails
            error = ProgramError(ErrorInfo(kind="convergence", message="no SCF"))
            frames[2] = failed_frame(frames[2].structure, error, self.program)
            raise ProgramErrors("1 of the requests failed", [error], frames)
        return frames


def test_a_session_s_group_of_failures_keeps_every_frame():
    log = []
    ev = SessionEvaluator(lambda inp, t, c: _GroupSession(inp.structure, log))
    out = irc(
        Structure(symbols=["H"], geometry=[[*MB_SADDLES[0], 0.0]]),
        {"program": "muller-brown"},
        evaluator=ev,
    )
    assert not out.success
    # the failed displacement's own error, not the group's
    assert (out.error.program, out.error.cause.kind) == ("muller-brown", "convergence")
    assert out.error.cause.message == "no SCF"


def test_numerics_name_the_calculation_type_and_qcref_s_data():
    from qcopt.program import NUMERICS_EPOCHS

    ts = program.numerics(_input("transition_state"))
    assert ts == program.numerics(_input("transition_state"))  # stable
    assert ts.startswith(f"transition_state:{NUMERICS_EPOCHS['transition_state']}+")
    assert program.numerics(TOY.model_copy(update={"calctype": "irc"})).startswith(
        "irc:"
    )
    assert ts.split("+qcref:")[1] == program.numerics(_input()).split("+qcref:")[1]


def test_numerics_are_qcopt_s_own_and_the_runner_names_each_child():
    pytest.importorskip("axtb")
    axtb = ProgramSpec(program="axtb", calctype="gradient", model={"method": "gfn2"})
    inp = _input(children=[axtb], convergence={"max_iter": 1})
    assert program.numerics(inp) == program.numerics(_input())
    out = qccompute.compute(inp, raise_exc=False)
    assert set(out.execution.identities) == {"qcopt", "axtb"}
    assert "+axtb" not in out.execution.identities["qcopt"]


class _BatchedSurface(FunctionProgram):
    """A program without sessions that computes batches, as an ML potential does."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []

    def capabilities(self):
        return super().capabilities().model_copy(update={"batches": True})

    def compute_batch(self, inputs, ctx=None):
        self.batches.append(len(inputs))
        outs = []
        for inp in inputs:
            try:
                outs.append(self.compute(inp, ctx))
            except ProgramError as e:
                outs.append(e.output)
        return outs


def test_a_round_of_a_batch_program_is_one_batch_across_tasks():
    from .helpers import qcopt_input, registered

    surface = _BatchedSurface("mb-batched", muller_brown)
    starts = [(-0.5, 1.4), (0.6, 0.1), (-0.1, 0.5)]
    inputs = [
        qcopt_input(
            "optimization",
            Structure(symbols=["H"], geometry=[[x, y, 0.0]]),
            {"program": "mb-batched"},
        )
        for x, y in starts
    ]
    with registered(surface):
        together = qccompute.compute(inputs)
        assert max(surface.batches) == 3  # every task's gradient of a round at once
        alone = [qcopt.compute(i) for i in inputs]
    for a, b in zip(together, alone, strict=True):
        assert [f.results.energy for f in a.results.frames] == [
            f.results.energy for f in b.results.frames
        ]


def test_the_package_has_its_version() -> None:
    from importlib import metadata

    assert qcopt.__version__ == metadata.version("qcopt") == program.version()
