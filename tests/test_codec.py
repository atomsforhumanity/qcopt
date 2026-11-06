import numpy as np
import pytest
from qcdata import OptimizationData, PathData, ProgramInput, ProgramSpec, Structure

import qcopt

qccompute = pytest.importorskip("qccompute")
pytest.importorskip("axtb")

ANG = 1.8897261259077824
GUESS = Structure(
    symbols=["C", "N", "H"],
    geometry=np.array([[0, 0, 0], [1.18, 0, 0], [0.6, 1.1, 0]]) * ANG,
)
SPEC = {"program": "axtb", "model": {"method": "gfn2"}}


def _input(calctype, structure=GUESS, **keywords):
    return ProgramInput(
        program="qcopt",
        calctype=calctype,
        structure=structure,
        keywords=keywords,
        subprograms=[
            ProgramSpec(calctype="gradient", **SPEC),
            ProgramSpec(calctype="hessian", **SPEC),
        ],
    )


def test_transition_state_and_irc_through_qccompute():
    ts = qccompute.compute(_input("transition_state", frequencies=True))
    assert ts.success
    assert isinstance(ts.results, OptimizationData)
    assert ts.results.provenance.program == "qcopt"
    assert ts.results.extras["n_imaginary"] == 1
    irc = qccompute.compute(_input("irc", ts.results.final_structure, step_size=0.3))
    assert irc.success
    assert isinstance(irc.results, PathData)
    assert irc.results.ts_index is not None


def test_bad_keywords_fail_in_qccompute():
    out = qccompute.compute(_input("optimization", nonsense=1), raise_exc=False)
    assert not out.success
    assert "nonsense" in out.traceback


def test_children_through_qccompute():
    out = qcopt.run(_input("optimization"), evaluator=qcopt.ComputeEvaluator())
    assert out.success
    assert out.results.frames[-1].results.provenance.program == "axtb"
