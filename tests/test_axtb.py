import numpy as np
import pytest
from qcdata import Structure

import qcopt
from qcopt.geometry import rmsd

axtb = pytest.importorskip("axtb")
pytestmark = pytest.mark.axtb

ANG = 1.8897261259077824
GFN2 = {"program": "axtb", "model": {"method": "gfn2"}}

WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=np.array([[0, 0, 0], [0.0, 0.9, 0.3], [0.0, -0.8, 0.4]]) * ANG,
)
# HCN <-> HNC: a bent guess between the two
HCN_TS_GUESS = Structure(
    symbols=["C", "N", "H"],
    geometry=np.array([[0, 0, 0], [1.18, 0, 0], [0.6, 1.1, 0]]) * ANG,
)


def test_water_minimum_in_process():
    out = qcopt.minimize(WATER, GFN2, frequencies=True, hessian=True)
    assert out.success, out.traceback
    assert out.results.extras["n_imaginary"] == 0
    f = out.results.frames[-1]
    assert f.results.provenance.program == "axtb"
    assert np.abs(f.results.gradient).max() < 2e-5


def test_hcn_isomerization_ts_and_irc():
    ts = qcopt.find_ts(HCN_TS_GUESS, GFN2, hessian=True, frequencies=True)
    assert ts.success, ts.traceback
    assert ts.results.extras["n_imaginary"] == 1
    path = qcopt.irc(ts.results.final_structure, GFN2, hessian=True, step_size=0.15)
    assert path.success, path.traceback
    ends = [
        path.results.frames[0].structure.geometry,
        path.results.frames[-1].structure.geometry,
    ]

    def h_on(g):  # which heavy atom the H sits nearer to
        return "C" if np.linalg.norm(g[2] - g[0]) < np.linalg.norm(g[2] - g[1]) else "N"

    assert sorted(h_on(g) for g in ends) == ["C", "N"]
    # the IRC stays in the input's frame (no drift)
    assert (
        rmsd(
            path.results.transition_state.structure.geometry,
            ts.results.final_structure.geometry,
            aligned=False,
        )
        < 1e-10
    )


def test_lockstep_minimizations_match_single_runs():
    from qcdata import CalcType, ProgramInput, ProgramSpec

    grad = ProgramSpec(
        program="axtb", calctype=CalcType.gradient, model={"method": "gfn2"}
    )
    shifted = WATER.model_copy(update={"geometry": WATER.geometry * 1.05})
    inputs = [
        ProgramInput(
            program="qcopt", calctype="optimization", structure=s, subprograms=[grad]
        )
        for s in (WATER, shifted)
    ]
    together = qcopt.run(inputs)
    alone = [qcopt.run(i) for i in inputs]
    for a, b in zip(together, alone, strict=True):
        assert a.success and b.success
        assert np.allclose(
            a.results.final_structure.geometry,
            b.results.final_structure.geometry,
            atol=1e-10,
        )


GXTB = {
    "program": "axtb",
    "model": {"method": "gxtb-v2.0.1"},
    "keywords": {"accuracy": 0.01},  # SCF noise well below the convergence threshold
}


def test_gxtb_ts_from_gfn2_hessians():
    """g-xTB gradients with GFN2 Hessians: frames hold g-xTB energies throughout."""
    ts = qcopt.find_ts(HCN_TS_GUESS, GXTB, hessian=GFN2, hessian_every=2)
    assert ts.success, ts.traceback
    ex = ts.results.extras
    assert ex["n_hessians"] >= 2
    check = axtb.Calculator(
        "gxtb-v2.0.1", [6, 7, 1], ts.results.frames[0].structure.geometry, accuracy=0.01
    )
    for frame in (ts.results.frames[0], ts.results.frames[-1]):
        e = check.singlepoint(frame.structure.geometry)["energy"]
        assert frame.results.energy == pytest.approx(e, abs=1e-7)
        assert frame.results.hessian is None
    # A g-xTB Hessian at the result: still a first-order saddle.
    h = check.hessian(ts.results.final_structure.geometry, vibrations=False)["hessian"]
    f = qcopt.frequencies(
        np.asarray(h), ["C", "N", "H"], ts.results.final_structure.geometry
    )
    assert f.n_imaginary(10.0) == 1
