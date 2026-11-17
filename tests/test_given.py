"""Starting from results already known: ``ProgramInput.given`` and ``irc(search)``."""

import numpy as np
import pytest
from qcdata import SinglePointData, Structure

import qcopt

from .surfaces import MB_MINIMA, muller_brown, muller_brown_hessian

SPEC = {"program": "muller-brown"}


def _structure(x, y):
    return Structure(symbols=["H"], geometry=[[x, y, 0.0]])


def _evaluator():
    return qcopt.FunctionEvaluator(
        muller_brown, muller_brown_hessian, name="muller-brown"
    )


def _known(x, y, point=True, program="muller-brown", **more):
    xyz = np.array([[x, y, 0.0]])
    e, g = muller_brown(xyz)
    return SinglePointData(
        provenance={"program": program},
        energy=e if point else None,
        gradient=g if point else None,
        hessian=muller_brown_hessian(xyz),
        **more,
    )


def _irc(start, **kw):
    return qcopt.irc(
        start,
        SPEC,
        hessian_program=True,
        evaluator=_evaluator(),
        irc={"step": 0.05},
        **kw,
    )


@pytest.fixture(scope="module")
def search():
    out = qcopt.find_ts(
        _structure(-0.7, 0.6),
        SPEC,
        hessian_program=True,
        evaluator=_evaluator(),
        trust={"radius": 0.1},
        freq={"check": True},
    )
    assert out.success, out.traceback
    assert out.results.frames[-1].results.hessian is not None
    return out


def test_an_irc_from_a_search_does_not_compute_its_hessian_again(search):
    fresh = _irc(search.results.final_structure)
    reused = _irc(search)
    assert reused.success, reused.traceback
    assert fresh.resolved.settings["hessian"]["initial"] == "exact"
    assert reused.resolved.settings["hessian"]["initial"] == "given"
    assert "muller-brown" in reused.resolved.reasons["hessian.initial"]
    # one gradient at the transition state instead of a Hessian
    assert reused.record["n_hessians"] == fresh.record["n_hessians"] - 1
    assert reused.record["n_gradients"] == fresh.record["n_gradients"] + 1
    # the same transition-state Hessian, so the same path
    for a, b in zip(fresh.results.frames, reused.results.frames, strict=True):
        assert np.array_equal(a.structure.geometry, b.structure.geometry)
        assert a.results.energy == b.results.energy


def test_an_irc_reuses_only_the_hessian_it_would_compute(search):
    other = {"program": "muller-brown", "model": {"method": "another"}}
    out = qcopt.irc(
        search, other, hessian_program=True, evaluator=_evaluator(), irc={"step": 0.05}
    )
    assert out.input_data.given is None
    assert out.resolved.settings["hessian"]["initial"] == "exact"
    # another gradient program, the same Hessian program as the search's
    out = qcopt.irc(
        search, other, hessian_program=SPEC, evaluator=_evaluator(), irc={"step": 0.05}
    )
    assert out.input_data.given.hessian is not None


def test_an_irc_from_a_search_without_frequencies_computes_the_hessian():
    out = qcopt.find_ts(
        _structure(-0.7, 0.6), SPEC, hessian_program=True, evaluator=_evaluator()
    )
    irc = _irc(out)
    assert irc.input_data.given is None
    assert irc.resolved.settings["hessian"]["initial"] == "exact"


def test_an_irc_does_not_start_from_a_failed_search():
    out = qcopt.find_ts(
        _structure(-0.7, 0.6),
        SPEC,
        hessian_program=True,
        evaluator=_evaluator(),
        convergence={"max_iter": 1},
    )
    assert not out.success
    with pytest.raises(qcopt.InputError, match="did not succeed"):
        _irc(out)


def test_a_given_frame_s_results_start_a_run_as_they_stand():
    # every field a program's frame carries; qcopt uses the Hessian, from any program
    x, y = -0.4, 1.2
    known = _known(x, y, program="cheap", scf_iterations=7, calcinfo_natoms=1)
    out = qcopt.minimize(_structure(x, y), SPEC, given=known, evaluator=_evaluator())
    assert out.success, out.traceback
    assert out.resolved.settings["hessian"]["initial"] == "given"
    assert out.resolved.reasons["hessian.initial"] == "the given Hessian (cheap)"
    assert np.allclose(
        out.results.final_structure.geometry[0, :2], MB_MINIMA[0], atol=1e-4
    )
    # a starting guess: no Hessian is computed, and no frame holds the given one
    assert out.record["n_hessians"] == 0
    assert out.results.frames[0].results.hessian is None


@pytest.mark.parametrize(
    "given, keywords, match, code",
    [
        (
            _known(0.0, 0.5).model_copy(update={"hessian": None}),
            {},
            "have none",
            "not_applicable",
        ),
        (
            _known(0.0, 0.5),
            {"hessian": {"initial": "model"}},
            "leaves it unused",
            "not_applicable",
        ),
        (None, {"hessian": {"initial": "given"}}, "has none", None),
    ],
)
def test_given_results_qcopt_cannot_use_are_refused(given, keywords, match, code):
    with pytest.raises(qcopt.InputError, match=match) as info:
        qcopt.minimize(
            _structure(0.0, 0.5), SPEC, given=given, evaluator=_evaluator(), **keywords
        )
    assert info.value.code == code
