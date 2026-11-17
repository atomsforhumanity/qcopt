"""A run its caller stops, or that runs out of time, ends at its next step.

It keeps what it reached and fails with kind ``stopped`` (or ``timeout``), as the
runner's contract for programs that work in steps asks (``ctx.stop_requested()``,
``ctx.timeout``).
"""

import threading

import numpy as np
import pytest
import qccompute
from qccompute.program import FunctionProgram
from qcdata import ProgramError, Structure

from qcopt.calculation import prepare
from qcopt.calculation import run as run_prepared

from .helpers import FunctionEvaluator, qcopt_input, registered, run
from .surfaces import MB_SADDLES, muller_brown, muller_brown_hessian

MB = {"program": "muller-brown"}


def _h(x, y):
    return Structure(symbols=["H"], geometry=[[x, y, 0.0]])


class _After:
    """A halt that lets ``rounds`` rounds of gradients run, then ends the run."""

    def __init__(self, rounds=None, kind="stopped", message="stopped on request"):
        self.rounds, self.kind, self.message, self.seen = rounds, kind, message, 0

    def __call__(self):
        self.seen += 1
        if self.rounds is not None and self.seen > self.rounds:
            return self.kind, self.message
        return None


def _run(inp, halt=None):
    ev = FunctionEvaluator(muller_brown, muller_brown_hessian, name="muller-brown")
    return run_prepared([prepare(inp)], ev, halt=halt)[0]


def _computed(out):
    return [f for f in out.results.frames if f.success]


def test_a_stopped_minimization_keeps_the_frames_it_reached():
    inp = qcopt_input("optimization", _h(-0.3, 1.2), MB)
    full = _run(inp)
    assert full.success and len(full.results.frames) > 6
    out = _run(inp, _After(4))
    assert not out.success
    assert out.error.kind == "stopped"
    assert out.error.message.startswith("stopped on request")
    assert out.error.program is None and out.error.cause is None  # qcopt's own stop
    assert out.record["n_gradients"] == 4
    computed = _computed(out)
    assert 0 < len(computed) <= 4
    for a, b in zip(computed, full.results.frames, strict=False):
        np.testing.assert_array_equal(a.structure.geometry, b.structure.geometry)


def test_a_run_out_of_time_fails_with_kind_timeout():
    inp = qcopt_input("transition_state", _h(-0.7, 0.6), MB, hessian_program=True)
    out = _run(inp, _After(3, "timeout", "out of time"))
    assert out.error.kind == "timeout"
    assert out.error.message.startswith("out of time")


def test_a_stopped_irc_ends_each_branch_as_stopped():
    inp = qcopt_input("irc", _h(*MB_SADDLES[0]), MB, hessian_program=True)
    out = _run(inp, _After(5))
    assert out.error.kind == "stopped"
    assert {b["end"] for b in out.record["branches"].values()} == {"stopped"}
    assert all(f.success for f in out.results.frames)  # the path's computed points


def test_a_stop_during_the_end_minimizations_is_a_stop():
    inp = qcopt_input(
        "irc", _h(*MB_SADDLES[0]), MB, hessian_program=True, ends={"minimize": True}
    )
    count = _After()
    full = _run(inp, count)
    assert full.success
    out = _run(inp, _After(count.seen - 1))  # the last round: an end minimization's
    assert {b["end"] for b in out.record["branches"].values()} <= {
        "converged",
        "converged_flat",
    }
    ends = out.record["ends"].values()
    assert any("stopped on request" in (e.get("error") or "") for e in ends)
    assert out.error.kind == "stopped"
    assert out.error.code is None


def test_runner_stop_reaches_a_running_calculation():
    calls = [0]
    reached, go = threading.Event(), threading.Event()

    def mb(pos):
        calls[0] += 1
        if calls[0] == 5:
            reached.set()
            go.wait(10)
        return muller_brown(pos)

    inp = qcopt_input("optimization", _h(-0.3, 1.2), {"program": "mb-slow"})
    with (
        registered(FunctionProgram("mb-slow", mb)),
        qccompute.Runner(threads=1) as runner,
    ):
        future = runner.submit(inp)
        assert reached.wait(10)
        runner.stop(future)
        go.set()
        with pytest.raises(ProgramError) as exc:
            future.result(timeout=30)
    out = exc.value.output
    assert out.error.kind == "stopped"
    assert calls[0] == 5  # nothing computed after the stop
    assert out.record["n_gradients"] == 5


def test_a_child_that_stopped_stops_the_run():
    calls = [0]

    def mb(pos):
        calls[0] += 1
        if calls[0] == 4:
            raise ProgramError.of("stopped", "stopped on request at SCF cycle 3")
        return muller_brown(pos)

    with registered(FunctionProgram("mb-stops", mb)):
        out = run(qcopt_input("optimization", _h(-0.3, 1.2), {"program": "mb-stops"}))
    assert out.error.kind == "stopped"
    assert out.error.program == "mb-stops"
    assert out.error.cause.kind == "stopped"
    assert calls[0] == 4  # no retry of a stopped calculation
