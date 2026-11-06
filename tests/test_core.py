import numpy as np
import pytest

from qcopt import hessians, primitives
from qcopt.geometry import align, internal_basis, max_atomic_norm, rigid_basis, rmsd
from qcopt.steps import rfo_step
from qcopt.vibrations import frequencies

RNG = np.random.default_rng(7)


def _fd(fn, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    out = []
    for i in range(x.size):
        d = np.zeros(x.size)
        d[i] = h
        out.append(
            (
                fn((x.ravel() + d).reshape(x.shape))
                - fn((x.ravel() - d).reshape(x.shape))
            )
            / (2 * h)
        )
    return np.array(out).T  # (m, 3n)


@pytest.mark.parametrize(
    "values, derivs, k",
    [
        (primitives.stretch_values, primitives.stretch_derivatives, 2),
        (primitives.bend_values, primitives.bend_derivatives, 3),
        (primitives.torsion_values, primitives.torsion_derivatives, 4),
    ],
)
def test_primitive_derivatives_match_finite_differences(values, derivs, k):
    x = RNG.normal(size=(6, 3)) * 1.5
    idx = np.array([RNG.permutation(6)[:k] for _ in range(5)], dtype=np.intp)
    b = primitives.scatter_b(derivs(x, idx), idx, 6)
    fd = _fd(lambda y: values(y, idx), x)
    assert np.allclose(b, fd, atol=1e-6)


def test_torsion_sign_and_range():
    x = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1.5], [0, 1.0, 1.5]])
    phi = primitives.torsion_values(x, np.array([[0, 1, 2, 3]]))[0]
    assert np.isclose(abs(phi), np.pi / 2)


def test_rigid_basis_spans_rotations_and_translations():
    x = RNG.normal(size=(5, 3))
    rb = rigid_basis(x)
    assert rb.shape == (15, 6)
    assert np.allclose(rb.T @ rb, np.eye(6), atol=1e-12)
    ib = internal_basis(x)
    assert ib.shape == (15, 9)
    assert np.allclose(rb.T @ ib, 0, atol=1e-12)
    linear = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 2.5]])
    assert rigid_basis(linear).shape[1] == 5


def test_align_and_rmsd():
    x = RNG.normal(size=(6, 3))
    t = np.radians(40)
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    y = x @ rot.T + np.array([1.0, -2.0, 0.5])
    assert np.allclose(align(y, x), x, atol=1e-10)
    assert rmsd(y, x) < 1e-10


@pytest.mark.parametrize("name", ["bfgs", "sr1", "psb", "bofill", "ts_bfgs"])
def test_updates_satisfy_the_secant_condition(name):
    n = 6
    a = RNG.normal(size=(n, n))
    true = a @ a.T - 2.0 * np.eye(n) if name != "bfgs" else a @ a.T + np.eye(n)
    h = np.eye(n)
    s = RNG.normal(size=n)
    y = true @ s
    new = hessians.UPDATES[name](h, s, y)
    assert np.allclose(new, new.T)
    assert np.allclose(new @ s, y, atol=1e-8)


def test_lindh_is_psd_and_rigid_invariant():
    numbers = [8, 1, 1, 6, 1, 1, 1]
    x = np.array(
        [
            [0, 0, 0],
            [1.8, 0, 0],
            [-0.5, 1.7, 0],
            [-1.5, -2.2, 0.3],
            [-1.2, -3.0, 2.1],
            [-3.4, -1.6, 0.0],
            [-1.0, -3.6, -1.3],
        ]
    )
    h = hessians.lindh(numbers, x)
    assert np.allclose(h, h.T)
    vals = np.linalg.eigvalsh(h)
    assert vals.min() > -1e-10
    rb = rigid_basis(x)
    assert np.abs(h @ rb).max() < 1e-8


@pytest.mark.parametrize("atomic", [False, True])
def test_rfo_step_respects_trust_and_descends(atomic):
    n = 9
    a = RNG.normal(size=(n, n))
    h = a @ a.T + 0.1 * np.eye(n)
    g = RNG.normal(size=n)
    vals, vecs = np.linalg.eigh(h)
    step = rfo_step(vals, vecs, g, 0.05, atomic=atomic)
    size = max_atomic_norm(step.dx) if atomic else np.linalg.norm(step.dx)
    assert size <= 0.05 * (1 + 1e-6)
    assert np.isclose(size, 0.05, rtol=1e-3)
    assert g @ step.dx < 0
    assert step.predicted < 0
    big = rfo_step(vals, vecs, 1e-6 * g, 10.0)
    assert not big.restricted


def test_prfo_climbs_the_chosen_mode():
    vals = np.array([-0.5, 0.3, 0.8, 1.2])
    vecs = np.eye(4)
    g = np.array([0.1, 0.2, -0.1, 0.05])
    step = rfo_step(vals, vecs, g, 1.0, mode=0)
    assert step.dx[0] * g[0] > 0  # uphill along the negative mode
    assert np.all(step.dx[1:] * g[1:] < 0)  # downhill along the others


def test_frequencies_of_a_harmonic_diatomic():
    from qcref import constants as c

    from qcopt.geometry import masses

    x = np.array([[0, 0, 0], [0, 0, 1.4]])
    k = 0.37  # Hartree/bohr^2 along the bond
    u = np.array([0, 0, -1, 0, 0, 1.0]) / np.sqrt(2)
    h = 2 * k * np.outer(u, u)
    f = frequencies(h, ["H", "H"], x)
    m = masses(["H"])[0] / float(c.phys.ELECTRON_MASS_IN_U)
    expected = (
        np.sqrt(k / (m / 2)) * float(c.phys.HARTREE_INVERSE_METER_RELATIONSHIP) / 100
    )
    assert len(f.wavenumbers) == 1
    assert np.isclose(f.wavenumbers[0], expected, rtol=1e-8)
    assert f.n_imaginary() == 0
    assert np.isclose(np.linalg.norm(f.modes[0]), 1.0)
