import numpy as np
import pytest
from qcdata import Structure

import qcopt
from qcopt import primitives as prim

axtb = pytest.importorskip("axtb")
pytestmark = pytest.mark.axtb

ANG = 1.8897261259077824
GFN2 = {"program": "axtb", "model": {"method": "gfn2"}}
# butane, staggered-ish
BUTANE = Structure(
    symbols=["C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H"],
    geometry=np.array(
        [
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [2.08, 1.43, 0.0],
            [3.61, 1.43, 0.1],
            [-0.39, -1.02, 0.0],
            [-0.39, 0.51, 0.89],
            [-0.39, 0.51, -0.89],
            [1.9, -0.52, 0.89],
            [1.9, -0.52, -0.89],
            [1.7, 1.95, 0.89],
            [1.7, 1.95, -0.89],
            [4.0, 2.45, 0.1],
            [4.0, 0.92, 0.99],
            [4.0, 0.92, -0.79],
        ]
    )
    * ANG,
)


@pytest.mark.parametrize("coordinates", ["cartesian", "tric"])
def test_dihedral_and_distance_targets(coordinates):
    out = qcopt.minimize(
        BUTANE,
        GFN2,
        hessian=True,
        coordinates=coordinates,
        constraints={"dihedral": [[0, 1, 2, 3, 60.0]], "distance": [[0, 1, 2.95]]},
    )
    assert out.success, out.traceback
    x = out.results.final_structure.geometry
    phi = np.degrees(prim.torsion_values(x, np.array([[0, 1, 2, 3]]))[0])
    r = prim.stretch_values(x, np.array([[0, 1]]))[0]
    assert abs(abs(phi) - 60.0) < 1e-4
    assert abs(r - 2.95) < 1e-5
    # unconstrained gradient is not zero: the constraints hold forces
    assert np.abs(out.results.frames[-1].results.gradient).max() > 1e-4


def test_keep_starting_angle():
    x0 = BUTANE.geometry
    a0 = prim.bend_values(x0, np.array([[0, 1, 2]]))[0]
    out = qcopt.minimize(
        BUTANE,
        GFN2,
        hessian=True,
        coordinates="tric",
        constraints={"angle": [[0, 1, 2]]},
    )
    assert out.success, out.traceback
    a = prim.bend_values(out.results.final_structure.geometry, np.array([[0, 1, 2]]))[0]
    assert abs(a - a0) < 1e-6


def test_bad_constraint_is_an_error():
    out = qcopt.minimize(BUTANE, GFN2, constraints={"distance": [[0, 0]]})
    assert not out.success
    assert "bad atoms" in out.traceback
