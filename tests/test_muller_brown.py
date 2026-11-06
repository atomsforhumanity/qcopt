import numpy as np
import pytest
from qcdata import (
    CalcType,
    OptimizationData,
    PathData,
    ProgramInput,
    ProgramSpec,
    Structure,
)

import qcopt

from .surfaces import MB_MINIMA, MB_SADDLES, muller_brown, muller_brown_hessian

SPEC = {"program": "muller-brown"}


def _structure(x, y):
    return Structure(symbols=["H"], geometry=[[x, y, 0.0]])


def _evaluator(hessian=True):
    return qcopt.FunctionEvaluator(
        muller_brown, muller_brown_hessian if hessian else None, name="muller-brown"
    )


def _xy(structure):
    return structure.geometry[0, :2]


@pytest.mark.parametrize(
    "start, expected", [((-0.4, 1.2), 0), ((0.3, 0.2), 1), ((0.0, 0.6), 2)]
)
@pytest.mark.parametrize("hessian", [None, True])
def test_minimize_finds_each_minimum(start, expected, hessian):
    out = qcopt.minimize(
        _structure(*start), SPEC, hessian=hessian, evaluator=_evaluator(), trust=0.1
    )
    assert out.success
    assert isinstance(out.results, OptimizationData)
    assert np.allclose(_xy(out.results.final_structure), MB_MINIMA[expected], atol=1e-4)
    assert out.results.provenance.program == "qcopt"
    assert out.results.frames[0].results.provenance.program == "muller-brown"
    assert out.results.extras["converged"]


@pytest.mark.parametrize("start, expected", [((-0.7, 0.6), 0), ((0.25, 0.3), 1)])
@pytest.mark.parametrize("hessian", [True, None])  # program Hessian, finite differences
def test_find_ts_finds_each_saddle(start, expected, hessian):
    out = qcopt.find_ts(
        _structure(*start),
        SPEC,
        hessian=hessian,
        evaluator=_evaluator(hessian is True),
        trust=0.1,
    )
    assert out.success, out.traceback
    assert np.allclose(
        _xy(out.results.final_structure), MB_SADDLES[expected], atol=1e-4
    )


@pytest.mark.parametrize("saddle, ends", [(0, (0, 2)), (1, (1, 2))])
@pytest.mark.parametrize("every", [0, 1])
def test_irc_connects_the_minima(saddle, ends, every):
    out = qcopt.irc(
        _structure(*MB_SADDLES[saddle]),
        SPEC,
        hessian=True,
        evaluator=_evaluator(),
        step_size=0.05,
        gmax=1e-4,
        hessian_every=every,
    )
    assert out.success, out.traceback
    path = out.results
    assert isinstance(path, PathData)
    assert np.allclose(_xy(path.transition_state.structure), MB_SADDLES[saddle])
    found = sorted(
        int(np.argmin([np.linalg.norm(_xy(f.structure) - m) for m in MB_MINIMA]))
        for f in (path.frames[0], path.frames[-1])
    )
    assert found == sorted(ends)
    e = path.energies
    assert np.all(np.diff(e[: path.ts_index + 1]) > 0)
    assert np.all(np.diff(e[path.ts_index :]) < 0)


def test_run_many_inputs_together_batches_requests():
    calls = []

    class Recording(qcopt.FunctionEvaluator):
        def __call__(self, requests):
            calls.append(len(requests))
            return super().__call__(requests)

    ev = Recording(muller_brown, muller_brown_hessian)
    grad = ProgramSpec(program="mb", calctype=CalcType.gradient)
    inputs = [
        ProgramInput(
            program="qcopt",
            calctype="optimization",
            structure=_structure(*s),
            subprograms=[grad],
            keywords={"trust": 0.1},
        )
        for s in [(-0.4, 1.2), (0.3, 0.2), (0.0, 0.6)]
    ]
    outs = qcopt.run(inputs, evaluator=ev)
    assert [o.success for o in outs] == [True, True, True]
    assert calls[0] == 3  # the first round asks all three at once
    for out, m in zip(outs, MB_MINIMA, strict=True):
        assert np.allclose(_xy(out.results.final_structure), m, atol=1e-4)


def test_failed_gradient_is_reported_with_frames():
    def broken(pos):
        if pos[0, 0] > 0.0:
            raise RuntimeError("SCF did not converge")
        return muller_brown(pos)

    out = qcopt.minimize(
        _structure(-0.05, 0.47),
        SPEC,
        evaluator=qcopt.FunctionEvaluator(broken),
        trust=0.5,
        maxiter=5,
    )
    # either it converged without crossing x=0, or the failure is reported with frames kept
    assert out.success or (out.traceback and out.results.frames)

    def always(pos):
        raise RuntimeError("no")

    out = qcopt.minimize(
        _structure(0.0, 0.0), SPEC, evaluator=qcopt.FunctionEvaluator(always)
    )
    assert not out.success
    assert "no" in out.traceback


def test_unknown_keywords_are_errors():
    with pytest.raises(ValueError):
        qcopt.minimize(_structure(0, 0), SPEC, evaluator=_evaluator(), not_a_keyword=1)


def test_not_converged_keeps_frames_and_fails():
    out = qcopt.minimize(_structure(-0.4, 1.2), SPEC, evaluator=_evaluator(), maxiter=2)
    assert not out.success
    assert len(out.results.frames) >= 2
    assert "not converged" in out.traceback


def test_output_round_trips_through_json():
    out = qcopt.find_ts(
        _structure(-0.7, 0.6),
        SPEC,
        hessian=True,
        evaluator=_evaluator(),
        frequencies=True,
    )
    again = type(out).model_validate_json(out.model_dump_json())
    assert again.results.frames[-1].results.freqs_wavenumber
    assert again.results.extras["n_imaginary"] == 1


@pytest.mark.parametrize("step", [0.15, 0.3])
def test_irc_with_long_steps_still_reaches_the_minima(step):
    out = qcopt.irc(
        _structure(*MB_SADDLES[0]),
        SPEC,
        hessian=True,
        evaluator=_evaluator(),
        step_size=step,
        gmax=1e-4,
    )
    assert out.success, out.traceback
    path = out.results
    found = sorted(
        int(np.argmin([np.linalg.norm(_xy(f.structure) - m) for m in MB_MINIMA]))
        for f in (path.frames[0], path.frames[-1])
    )
    assert found == [0, 2]


def test_irc_minimize_ends_lands_on_the_minima():
    out = qcopt.irc(
        _structure(*MB_SADDLES[0]),
        SPEC,
        hessian=True,
        evaluator=_evaluator(),
        step_size=0.1,
        minimize_ends=True,
    )
    assert out.success, out.traceback
    path = out.results
    ends = sorted(
        (_xy(path.frames[0].structure), _xy(path.frames[-1].structure)),
        key=lambda p: p[0],
    )
    assert np.allclose(ends[0], MB_MINIMA[0], atol=1e-4)
    assert np.allclose(ends[1], MB_MINIMA[2], atol=1e-4)
    info = path.extras["end_minimization"]
    assert info["forward"]["converged"] and info["reverse"]["converged"]
    assert np.allclose(_xy(path.transition_state.structure), MB_SADDLES[0])
