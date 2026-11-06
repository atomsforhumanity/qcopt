import numpy as np
import pytest
from qcdata import Structure

import qcopt
from qcopt.geometry import rigid_basis
from qcopt.internal import CoordinateError, DelocalizedInternals, build_primitives

ANG = 1.8897261259077824
RNG = np.random.default_rng(3)

MOLECULES = {
    "water": (["O", "H", "H"], [[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]]),
    "hcn": (["C", "N", "H"], [[0, 0, 0], [1.16, 0, 0], [-1.07, 0, 0]]),
    "propyne": (
        ["C", "C", "C", "H", "H", "H", "H"],
        [
            [0, 0, 0],
            [1.20, 0, 0],
            [2.66, 0, 0],
            [-1.06, 0, 0],
            [3.05, 1.0, 0],
            [3.05, -0.5, 0.87],
            [3.05, -0.5, -0.87],
        ],
    ),
    "ketene": (
        ["C", "C", "O", "H", "H"],
        [[0, 0, 0], [1.31, 0, 0], [2.47, 0, 0], [-0.55, 0.94, 0], [-0.55, -0.94, 0]],
    ),
    "water dimer": (
        ["O", "H", "H", "O", "H", "H"],
        [
            [0, 0, 0],
            [0.96, 0, 0],
            [-0.24, 0.93, 0],
            [2.9, 0, 0],
            [3.2, 0.5, 0.75],
            [3.2, 0.5, -0.75],
        ],
    ),
    "hcn + water": (
        ["C", "N", "H", "O", "H", "H"],
        [
            [0, 0, 0],
            [1.16, 0, 0],
            [-1.07, 0, 0],
            [0.5, 3.0, 0],
            [1.3, 3.4, 0],
            [-0.2, 3.6, 0],
        ],
    ),
    "argon + methane": (
        ["Ar", "C", "H", "H", "H", "H"],
        [
            [0, 0, 0],
            [3.6, 0, 0],
            [4.0, 1.0, 0],
            [4.0, -0.5, 0.9],
            [4.0, -0.5, -0.9],
            [2.5, 0, 0],
        ],
    ),
}


def _geometry(name):
    sym, x = MOLECULES[name]
    return sym, np.array(x, float) * ANG + RNG.normal(0, 0.02, (len(sym), 3))


def _fd_b(ic, x, h=1e-6):
    cols = []
    for j in range(x.size):
        y = x.ravel().copy()
        y[j] += h
        plus = ic.values(y.reshape(-1, 3))
        y[j] -= 2 * h
        cols.append((plus - ic.values(y.reshape(-1, 3))) / (2 * h))
    return np.array(cols).T


@pytest.mark.parametrize("name", list(MOLECULES))
@pytest.mark.parametrize("tric", [True, False])
def test_b_matrix_and_back_transformation(name, tric):
    sym, x = _geometry(name)
    ic = DelocalizedInternals(sym, x, tric=tric)
    n = len(sym)
    # a nearly linear molecule (HCN with noise) still has 6 rigid motions
    expected = 3 * n if tric else 3 * n - rigid_basis(x).shape[1]
    assert ic.n == expected
    if tric:  # DLC's B has rigid motion projected out, so it is not d(values)/dx
        assert np.allclose(ic.b_matrix(x), _fd_b(ic, x), atol=1e-7)
    ds = RNG.normal(0, 0.05, ic.n)
    x_new, taken, ok = ic.step(x, ds)
    assert ok
    assert np.allclose(taken, ds, atol=1e-7)
    assert np.allclose(ic.values(x_new) - ic.values(x), ds, atol=1e-7)


def test_rotation_derivatives_after_a_finite_rotation():
    sym, x = _geometry("water dimer")
    ic = DelocalizedInternals(sym, x, tric=True)
    t = 0.7
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    y = x.copy()
    c = y[:3].mean(0)
    y[:3] = (y[:3] - c) @ rot.T + c
    assert np.allclose(ic.b_matrix(y), _fd_b(ic, y), atol=1e-7)
    assert ic.prims.max_rotation(y) == pytest.approx(t, abs=1e-6)


def test_internal_gradient_reproduces_the_cartesian_one():
    sym, x = _geometry("water dimer")
    ic = DelocalizedInternals(sym, x, tric=True)
    g = RNG.normal(size=x.size)
    gs = ic.gradient(x, g)
    assert np.allclose(ic.b_matrix(x).T @ gs, g, atol=1e-10)  # TRIC spans all 3N


def test_ketene_gets_an_out_of_plane_coordinate():
    sym, x = _geometry("ketene")
    p = build_primitives(sym, x, tric=True)
    assert any(t[1] == 0 for t in p.torsions.tolist())  # improper at the CH2 carbon
    DelocalizedInternals(sym, x, tric=True)


def test_incomplete_primitives_raise():
    sym, x = _geometry("water")
    p = build_primitives(sym, x)
    p_bad = type(p)(
        bonds=p.bonds,
        bends=p.bends[:0],
        linear=p.linear,
        linear_axes=p.linear_axes,
        torsions=p.torsions,
        n_atoms=3,
    )
    with pytest.raises(CoordinateError):
        DelocalizedInternals(sym, x, primitives=p_bad)


def test_hessian_transform_includes_the_gradient_term():
    """H_s from a quadratic surface in Cartesians equals finite differences of g_s."""
    sym, x = _geometry("water")
    ic = DelocalizedInternals(sym, x, tric=False)
    a = RNG.normal(size=(9, 9))
    hx = a @ a.T
    x0 = x.ravel().copy()

    def grad(y):
        return hx @ (y.ravel() - x0) + 0.3  # constant part: gradient nonzero at x

    hs = ic.hessian(x, hx, grad(x))
    h = 1e-4
    cols = []
    for i in range(ic.n):
        d = np.zeros(ic.n)
        d[i] = h
        xp, _, _ = ic.step(x, d)
        xm, _, _ = ic.step(x, -d)
        cols.append((ic.gradient(xp, grad(xp)) - ic.gradient(xm, grad(xm))) / (2 * h))
    assert np.allclose(hs, np.array(cols).T, atol=1e-4 * np.abs(hs).max())


axtb = pytest.importorskip("axtb")
GFN2 = {"program": "axtb", "model": {"method": "gfn2"}}


@pytest.mark.axtb
@pytest.mark.parametrize("coordinates", ["tric", "dlc"])
def test_minimize_a_complex_in_internals(coordinates):
    sym, x = _geometry("water dimer")
    out = qcopt.minimize(
        Structure(symbols=sym, geometry=x),
        GFN2,
        hessian=True,
        coordinates=coordinates,
        frequencies=True,
    )
    assert out.success, out.traceback
    assert out.results.extras["coordinates"] == coordinates
    assert out.results.extras["n_imaginary"] == 0


@pytest.mark.axtb
def test_saddle_escape():
    # Planar NH3 with a model Hessian: the gradient never leaves the plane, so the
    # minimization stops on the planar saddle unless the frequencies are checked.
    sym = ["N", "H", "H", "H"]
    x = (
        np.array([[0, 0, 0], [1.01, 0, 0], [-0.505, 0.875, 0], [-0.505, -0.875, 0]])
        * ANG
    )
    flat = qcopt.minimize(
        Structure(symbols=sym, geometry=x),
        GFN2,
        coordinates="cartesian",
        frequencies=True,
    )
    assert flat.results.extras["n_imaginary"] == 1
    out = qcopt.minimize(
        Structure(symbols=sym, geometry=x),
        GFN2,
        coordinates="cartesian",
        escape_saddles=2,
        frequencies=True,
    )
    assert out.success
    assert out.results.extras["n_imaginary"] == 0
    assert out.results.extras["saddle_escapes"] == 1
    assert out.results.final_energy < flat.results.final_energy
