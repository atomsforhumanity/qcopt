"""qcopt run by qccompute, with axtb children."""

import numpy as np
import pytest
import qccompute
from qcdata import (
    OptimizationData,
    PathData,
    ProgramError,
    ProgramInput,
    ProgramSpec,
    Structure,
)

import qcopt

pytest.importorskip("axtb")

ANG = 1.8897261259077824
GUESS = Structure(
    symbols=["C", "N", "H"],
    geometry=np.array([[0, 0, 0], [1.18, 0, 0], [0.6, 1.1, 0]]) * ANG,
)
SPEC = {"program": "axtb", "model": {"method": "gfn2"}}


def _input(calctype, structure=GUESS, hessians=True, **keywords):
    subprograms = [ProgramSpec(calctype="gradient", **SPEC)]
    if hessians:
        subprograms.append(ProgramSpec(calctype="hessian", **SPEC))
    return ProgramInput(
        program="qcopt",
        calctype=calctype,
        structure=structure,
        keywords=keywords,
        subprograms=subprograms,
    )


def test_transition_state_and_irc_through_qccompute():
    ts = qccompute.compute(_input("transition_state", freq={"check": True}))
    assert ts.success
    assert isinstance(ts.results, OptimizationData)
    assert ts.results.provenance.program == "qcopt"
    assert ts.record["n_imaginary"] == 1
    assert ts.resolved.settings["freq"]["check"] is True
    irc = qccompute.compute(
        _input("irc", ts.results.final_structure, irc={"step": 0.3})
    )
    assert irc.success
    assert isinstance(irc.results, PathData)
    assert irc.results.ts_index is not None


def test_bad_keywords_fail_in_qccompute():
    out = qccompute.compute(_input("optimization", nonsense=1), raise_exc=False)
    assert not out.success
    assert "nonsense" in out.traceback


@pytest.mark.xfail(
    raises=ProgramError,
    strict=True,
    reason="axtb 0.1.0's codec builds a pre-0.24 qcdata Wavefunction",
)
def test_children_without_sessions_run_through_qccompute(monkeypatch):
    # axtb as qccompute runs it (its codec, one calculation per gradient), not on
    # qcopt's in-process axtb evaluator: the path of every program without sessions.
    # Gradients only: axtb 0.1.0's codec Hessians do not validate under qcdata 0.24
    # (they carry the removed freqs_wavenumber fields).
    monkeypatch.setattr("qcopt.evaluators.axtb_in_process", lambda context: False)
    out = qcopt.compute(_input("optimization", hessians=False))
    assert out.success
    assert out.results.frames[-1].results.provenance.program == "axtb"
