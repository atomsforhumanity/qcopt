"""What a running calculation tells its caller: step events and its own wall time."""

import numpy as np
import pytest
import qccompute
from qccompute.program import FunctionProgram
from qcdata import SinglePointData, Structure

from .helpers import qcopt_input, registered
from .surfaces import MB_SADDLES, muller_brown, muller_brown_hessian

MB = {"program": "muller-brown"}


@pytest.fixture(autouse=True)
def mb_registered():
    with registered(
        FunctionProgram("muller-brown", muller_brown, muller_brown_hessian)
    ):
        yield


def _h(x, y):
    return Structure(symbols=["H"], geometry=[[x, y, 0.0]])


def _steps(events, **match):
    return [e for e in events if all(e.get(k) == v for k, v in match.items())]


def test_each_frame_of_each_calculation_is_one_step_event():
    events = []
    inputs = [
        qcopt_input("optimization", _h(-0.3, 1.2), MB),
        qcopt_input("optimization", _h(0.9, -0.2), MB),
    ]
    outs = qccompute.compute(inputs, progress=events.append)
    assert all(o.success for o in outs)
    assert {e["path"][0] for e in events} == {"qcopt"}
    for task, out in enumerate(outs):
        mine = _steps(events, task=task)
        frames = out.results.frames
        assert [e["step"] for e in mine] == list(range(len(frames)))
        assert [e["energy"] for e in mine] == [f.results.energy for f in frames]
        assert {e["event"] for e in mine} == {"step"}
        assert {e["calctype"] for e in mine} == {"optimization"}
        assert all("line" not in e for e in mine)
        assert mine[-1]["g_max"] < out.resolved.settings["convergence"]["g_tol"]


def test_each_calculation_of_a_list_reports_its_own_wall_time():
    short = qcopt_input("optimization", _h(0.62, 0.03), MB)  # next to a minimum
    long = qcopt_input("optimization", _h(-0.3, 1.2), MB)
    a, b = qccompute.compute([short, long])
    assert len(a.results.frames) < len(b.results.frames)
    batch = a.execution.timings["batch"]
    assert b.execution.timings["batch"] == batch
    assert 0 < a.execution.wall_time < b.execution.wall_time <= batch


def test_an_irc_reports_its_points_by_branch_and_then_its_end_minimizations():
    events = []
    inp = qcopt_input(
        "irc", _h(*MB_SADDLES[0]), MB, hessian_program=True, ends={"minimize": True}
    )
    out = qccompute.compute(inp, progress=events.append)
    assert out.success, out.traceback
    frames, ts = out.results.frames, out.results.ts_index
    (start,) = _steps(events, calctype="irc", step=0)
    assert "line" not in start
    assert start["energy"] == frames[ts].results.energy
    for line in ("forward", "reverse"):
        points = [e for e in _steps(events, calctype="irc", line=line)]
        assert [e["step"] for e in points] == list(range(1, len(points) + 1))
        minimization = _steps(events, calctype="optimization", line=line)
        assert minimization[0]["step"] == 0  # from the branch's last point
        assert minimization[0]["energy"] == points[-1]["energy"]
    # the path's frames are its points, then each branch's minimized end where the
    # minimization moved it
    ends = {frames[0].results.energy, frames[-1].results.energy}
    last = {
        line: _steps(events, calctype="optimization", line=line)[-1]
        for line in ("forward", "reverse")
    }
    assert ends == {e["energy"] for e in last.values()}
    moved = sum(e["step"] > 0 for e in last.values())
    assert len(_steps(events, calctype="irc")) == len(frames) - moved


def test_listening_changes_no_number():
    inp = qcopt_input("transition_state", _h(-0.7, 0.6), MB, hessian_program=True)
    quiet = qccompute.compute(inp)
    heard = qccompute.compute(inp, progress=lambda e: None)
    xq = [f.structure.geometry for f in quiet.results.frames]
    xh = [f.structure.geometry for f in heard.results.frames]
    assert len(xq) == len(xh)
    assert all(np.array_equal(p, q) for p, q in zip(xq, xh, strict=True))


def test_the_record_names_the_frames_where_hessians_were_computed():
    inp = qcopt_input(
        "transition_state",
        _h(-0.7, 0.6),
        MB,
        hessian_program=True,
        freq={"check": True},
    )
    out = qccompute.compute(inp)
    assert out.success, out.traceback
    frames = out.results.frames
    computed = [k for k, f in enumerate(frames) if f.results.hessian is not None]
    assert out.record["hessian_frames"] == computed
    assert computed[0] == 0 and computed[-1] == len(frames) - 1
    assert len(computed) <= out.record["n_hessians"]


def test_a_given_hessian_is_not_one_of_the_computed():
    s = _h(*MB_SADDLES[0])
    x = s.geometry
    energy, gradient = muller_brown(x)
    given = SinglePointData(
        provenance={"program": "muller-brown"},
        energy=energy,
        gradient=gradient,
        hessian=muller_brown_hessian(x),
    )
    inp = qcopt_input("irc", s, MB, hessian_program=True, given=given)
    out = qccompute.compute(inp)
    assert out.success, out.traceback
    assert out.resolved.settings["hessian"]["initial"] == "given"
    assert out.results.ts_index not in out.record["hessian_frames"]
    assert out.results.frames[out.results.ts_index].results.hessian is None


def test_each_irc_branch_ends_with_a_code_the_record_schema_names():
    from qcopt.algorithms.reaction_path import BRANCH_ENDS
    from qcopt.program import program

    inp = qcopt_input(
        "irc", _h(*MB_SADDLES[0]), MB, hessian_program=True, convergence={"max_iter": 3}
    )
    out = qccompute.compute(inp, raise_exc=False)
    assert (out.error.kind, out.error.code) == ("convergence", "irc_incomplete")
    ends = {name: b["end"] for name, b in out.record["branches"].items()}
    assert ends == {"forward": "max_iter", "reverse": "max_iter"}
    for b in out.record["branches"].values():  # max_iter counts steps, not frames
        assert b["steps"] >= 3 and b["frames"] <= b["steps"]
    assert "the forward branch stopped: max_iter" in out.error.message
    schema = program.record_schema("irc")["properties"]["branches"]
    assert schema["additionalProperties"]["properties"]["end"]["enum"] == list(
        BRANCH_ENDS
    )
    # with its ends minimized, the same IRC is a success
    inp = inp.model_copy(
        update={"keywords": {**inp.keywords, "ends": {"minimize": True}}}
    )
    out = qccompute.compute(inp)
    assert out.success
    assert {b["end"] for b in out.record["branches"].values()} == {"max_iter"}
