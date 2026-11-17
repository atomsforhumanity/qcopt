"""Starting from results already known: ``ProgramInput.given``."""

import numpy as np
import pytest
from qcdata import ProgramError, SinglePointData, Structure

import qcopt

from .helpers import FunctionProgram, qcopt_input, registered
from .surfaces import MB_MINIMA, muller_brown, muller_brown_hessian

SPEC = {"program": "muller-brown"}


@pytest.fixture(autouse=True)
def muller_brown_program():
    with registered(
        FunctionProgram("muller-brown", muller_brown, muller_brown_hessian)
    ):
        yield


def _structure(x, y):
    return Structure(symbols=["H"], geometry=[[x, y, 0.0]])


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


def _irc(structure, given=None):
    return qcopt.compute(
        qcopt_input(
            "irc",
            structure,
            SPEC,
            hessian_program=True,
            given=given,
            irc={"step": 0.05},
        )
    )


def test_an_irc_starts_from_its_search_s_transition_state_hessian():
    search = qcopt.compute(
        qcopt_input(
            "transition_state",
            _structure(-0.7, 0.6),
            SPEC,
            hessian_program=True,
            trust={"radius": 0.1},
            freq={"check": True},
        )
    )
    last = search.results.frames[-1]
    assert last.results.hessian is not None  # freq.check
    fresh = _irc(last.structure)
    reused = _irc(last.structure, given=last.results)
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


def test_a_given_frame_s_results_start_a_run_as_they_stand():
    # every field a program's frame carries; qcopt uses the Hessian, from any program
    x, y = -0.4, 1.2
    known = _known(x, y, program="cheap", scf_iterations=7, calcinfo_natoms=1)
    out = qcopt.compute(
        qcopt_input("optimization", _structure(x, y), SPEC, given=known)
    )
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
    with pytest.raises(ProgramError, match=match) as info:
        qcopt.compute(
            qcopt_input(
                "optimization", _structure(0.0, 0.5), SPEC, given=given, **keywords
            )
        )
    assert (info.value.error.kind, info.value.error.code) == ("input", code)
