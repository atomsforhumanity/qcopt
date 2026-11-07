"""Failures, checks and keyword combinations: each test failed on qcopt 0.1.0."""

from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

import qcopt
import qcopt.algorithms as algorithms
from qcopt.algorithms import lockstep
from qcopt.coordinates import InternalCoordinates, make_coordinates
from qcopt.driver import Answer
from qcopt.geometry import rigid_basis

from .surfaces import (
    MB_MINIMA,
    MB_SADDLES,
    muller_brown,
    muller_brown_hessian,
    shoulder,
    shoulder_hessian,
    triatomic,
)

MB = {"program": "muller-brown"}


def _atom(x, y=0.0, z=0.0):
    return Structure(symbols=["H"], geometry=[[x, y, z]])


def _mb(point):
    return _atom(*point)


def _ev(hessian=True):
    return qcopt.FunctionEvaluator(
        muller_brown, muller_brown_hessian if hessian else None
    )


# ----------------------------------------------------------- success and checks


def test_ts_search_started_at_a_minimum_fails():
    for frequencies in (False, True):
        out = qcopt.find_ts(
            _mb(MB_MINIMA[1]),
            MB,
            hessian=True,
            evaluator=_ev(),
            frequencies=frequencies,
        )
        assert not out.success
        assert "not a first-order saddle point" in out.traceback


def test_ts_search_still_succeeds_at_a_saddle():
    out = qcopt.find_ts(_mb((-0.7, 0.6)), MB, hessian=True, evaluator=_ev(), trust=0.1)
    assert out.success, out.traceback


def test_nan_gradient_fails_and_keeps_frames():
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        e, g = muller_brown(x)
        return (e, g * np.nan) if calls["n"] > 3 else (e, g)

    out = qcopt.minimize(_atom(0.3, 0.2), MB, evaluator=qcopt.FunctionEvaluator(fn))
    assert not out.success
    assert "non-finite gradient" in out.traceback
    assert len(out.results.frames) >= 3


def test_an_exception_inside_the_algorithm_keeps_the_frames(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("frequencies broke")

    monkeypatch.setattr(algorithms, "frequencies", broken)
    out = qcopt.minimize(
        _atom(0.3, 0.2), MB, hessian=True, evaluator=_ev(), frequencies=True
    )
    assert not out.success
    assert "frequencies broke" in out.traceback
    assert len(out.results.frames) > 1


def test_a_failing_evaluator_fails_the_tasks_and_keeps_their_frames():
    class Breaks(qcopt.FunctionEvaluator):
        calls = 0

        def __call__(self, requests):
            Breaks.calls += 1
            if Breaks.calls >= 4:  # down for good: retries fail too
                raise RuntimeError("the program crashed")
            return super().__call__(requests)

    out = qcopt.minimize(_atom(0.3, 0.2), MB, evaluator=Breaks(muller_brown))
    assert not out.success
    assert "evaluator failed: RuntimeError: the program crashed" in out.traceback
    assert len(out.results.frames) == 3  # the three computed before the crash


def test_a_failed_hessian_for_the_frequencies_fails_the_run():
    def no_hessian(x):
        raise RuntimeError("no Hessian today")

    ev = qcopt.FunctionEvaluator(muller_brown, no_hessian)
    out = qcopt.minimize(
        _atom(0.3, 0.2),
        MB,
        hessian=True,
        initial_hessian="model",
        evaluator=ev,
        frequencies=True,
    )
    assert not out.success
    assert "the Hessian for the frequencies failed" in out.traceback


def test_saddle_escapes_that_run_out_fail():
    out = qcopt.minimize(
        _mb(MB_SADDLES[1]),
        MB,
        hessian=True,
        evaluator=_ev(),
        escape_saddles=1,
        escape_step=1e-7,
    )
    assert not out.success
    assert "still a saddle point" in out.traceback


def test_saddle_escape_is_skipped_with_constraints():
    x = np.array([[2.0, 0, 0], [0, 0, 0], [-1.7, 1.0, 0.0]])
    out = qcopt.minimize(
        Structure(symbols=["O", "O", "O"], geometry=x),
        {"program": "f"},
        hessian=True,
        evaluator=qcopt.FunctionEvaluator(
            triatomic, lambda p: _fd_hessian(triatomic, p)
        ),
        escape_saddles=2,
        constraints={"distance": [[0, 1, 2.3]]},
    )
    assert out.success, out.traceback
    assert "saddle_escapes" not in out.results.extras
    assert out.results.extras["n_hessians"] == 1  # the starting one only


def _fd_hessian(fn, p, h=1e-5):
    x = np.asarray(p, dtype=float).ravel()
    cols = []
    for i in range(x.size):
        y = x.copy()
        y[i] += h
        gp = fn(y.reshape(-1, 3))[1].ravel()
        y[i] -= 2 * h
        gm = fn(y.reshape(-1, 3))[1].ravel()
        cols.append((gp - gm) / (2 * h))
    m = np.array(cols).T
    return 0.5 * (m + m.T)


# ----------------------------------------------------------------- keywords


def test_guide_mode_needs_cartesian_coordinates():
    mode = [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="Cartesian coordinates only"):
        qcopt.find_ts(_mb((-0.7, 0.6)), MB, hessian=True, evaluator=_ev(), mode=mode)
    out = qcopt.find_ts(
        _mb((-0.7, 0.6)),
        MB,
        hessian=True,
        evaluator=_ev(),
        mode=mode,
        coordinates="cartesian",
        trust=0.1,
    )
    assert out.success


def test_irc_needs_the_transition_state_hessian():
    for start in ("model", "identity"):
        with pytest.raises(ValueError, match="needs the transition state's Hessian"):
            qcopt.irc(
                _mb(MB_SADDLES[0]),
                MB,
                hessian=True,
                evaluator=_ev(),
                initial_hessian=start,
            )


def test_finite_differences_with_a_hessian_program_are_rejected():
    with pytest.raises(ValueError, match="finite_difference"):
        qcopt.find_ts(
            _mb((-0.7, 0.6)),
            MB,
            hessian=True,
            evaluator=_ev(),
            initial_hessian="finite_difference",
        )


def test_hessian_every_works_from_a_model_start():
    out = qcopt.minimize(
        _atom(0.3, 0.2),
        MB,
        hessian=True,
        evaluator=_ev(),
        initial_hessian="model",
        hessian_every=1,
    )
    assert out.success
    assert out.results.extras["n_hessians"] > 0


# ------------------------------------------------------------- separate Hessians


class _Solvated:
    """Müller-Brown gradients in 'solvent'; gas-phase ones 100 Eh higher."""

    def supports_hessian(self, context):
        return True

    def __call__(self, requests):
        out = []
        for r in requests:
            x = np.asarray(r.positions, dtype=float)
            spec = (
                r.context.hessian_spec
                if r.kind == "hessian"
                else r.context.gradient_spec
            )
            shift = 0.0 if spec.keywords.get("solvent") else 100.0
            e, g = muller_brown(x)
            h = muller_brown_hessian(x) if r.kind == "hessian" else None
            out.append(
                Answer(
                    positions=x,
                    energy=e + shift,
                    gradient=g,
                    hessian=h,
                    program="muller-brown",
                )
            )
        return out


def test_a_hessian_program_with_other_keywords_is_separate():
    solvated = {"program": "muller-brown", "keywords": {"solvent": "water"}}
    gas = {"program": "muller-brown"}
    out = qcopt.find_ts(
        _mb((-0.7, 0.6)),
        solvated,
        hessian=gas,
        evaluator=_Solvated(),
        trust=0.1,
        frequencies=True,
    )
    assert out.success, out.traceback
    for frame in out.results.frames:
        e, _ = muller_brown(frame.structure.geometry)
        assert frame.results.energy == pytest.approx(e)  # never the gas-phase one
        assert frame.results.hessian is None
    # frequencies from the other calculation are labelled, not put on the frame
    assert not out.results.frames[-1].results.freqs_wavenumber
    assert out.results.extras["frequencies"]["program"] == "muller-brown"


def test_compute_evaluator_pairs_hessians_without_gradients():
    calls = {"gradient": 0, "hessian": 0}

    def compute(inputs):
        outs = []
        for inp in inputs:
            x = np.asarray(inp.structure.geometry, dtype=float)
            e, g = muller_brown(x)
            kind = inp.calctype.value
            calls[kind] += 1
            results = SimpleNamespace(
                energy=e,
                gradient=g if kind == "gradient" else None,  # as xtb's codec
                hessian=muller_brown_hessian(x) if kind == "hessian" else None,
                provenance=SimpleNamespace(
                    program="muller-brown", program_version=None
                ),
            )
            outs.append(SimpleNamespace(success=True, results=results, traceback=None))
        return outs

    ev = qcopt.ComputeEvaluator(compute)
    out = qcopt.find_ts(_mb((-0.7, 0.6)), MB, hessian=True, evaluator=ev, trust=0.1)
    assert out.success, out.traceback
    assert calls["hessian"] >= 1


# ----------------------------------------------------------------------- IRC


def _irc_frames_x(out):
    return [float(f.structure.geometry[0, 0]) for f in out.results.frames]


def test_irc_that_runs_out_of_steps_fails():
    out = qcopt.irc(_mb(MB_SADDLES[0]), MB, hessian=True, evaluator=_ev(), max_steps=2)
    assert not out.success
    assert "max_steps" in out.traceback
    assert len(out.results.frames) == 5  # kept


def test_irc_with_unconverged_end_minimizations_fails():
    out = qcopt.irc(
        _mb(MB_SADDLES[0]),
        MB,
        hessian=True,
        evaluator=_ev(),
        minimize_ends=True,
        end_maxiter=1,
        end_coordinates="cartesian",
        end_convergence={"gmax": 1e-12},
    )
    assert not out.success
    assert "end minimization did not converge" in out.traceback


class _NoisyHessians(qcopt.FunctionEvaluator):
    """The shoulder surface; Hessian answers come back 1e-9 Eh higher (SCF noise)."""

    def __call__(self, requests):
        answers = super().__call__(requests)
        for r, a in zip(requests, answers, strict=True):
            if r.kind == "hessian" and np.abs(r.positions).sum() > 0:
                a.energy += 1e-9
        return answers


def test_irc_end_check_does_not_overwrite_frames():
    ev = _NoisyHessians(shoulder, shoulder_hessian)
    out = qcopt.irc(
        _atom(0.0),
        {"program": "f"},
        hessian=True,
        evaluator=ev,
        direction="forward",
        step_size=0.26,
        update="bfgs",
    )
    x = _irc_frames_x(out)
    gaps = np.diff(x)
    assert np.all(gaps > 0)
    assert gaps.max() < 0.27  # one step (ds / sqrt(m_H)) at most: none replaced


def test_irc_continues_when_an_optional_hessian_fails():
    class OnlyFirstHessian(qcopt.FunctionEvaluator):
        hessians = 0

        def __call__(self, requests):
            out = []
            for r in requests:
                if r.kind == "hessian":
                    OnlyFirstHessian.hessians += 1
                    if OnlyFirstHessian.hessians > 1:
                        x = np.asarray(r.positions, dtype=float)
                        out.append(Answer(positions=x, success=False, error="no"))
                        continue
                out.extend(super().__call__([r]))
            return out

    ev = OnlyFirstHessian(shoulder, shoulder_hessian)
    out = qcopt.irc(
        _atom(0.0),
        {"program": "f"},
        hessian=True,
        evaluator=ev,
        direction="forward",
        step_size=0.26,
        update="bfgs",
        hessian_every=3,
    )
    assert out.success, out.traceback
    assert _irc_frames_x(out)[-1] > 1.5  # past the shoulder, toward the minimum at 2


def test_gs2_irc_does_not_step_sideways():
    out = qcopt.irc(_mb(MB_SADDLES[0]), MB, hessian=True, evaluator=_ev())
    assert out.success
    z = [abs(float(f.structure.geometry[0, 2])) for f in out.results.frames]
    assert max(z) < 1e-8


# ---------------------------------------------------------------- bookkeeping


def test_failed_finite_difference_hessians_count_their_gradients():
    counter = algorithms._Counter()
    counter([Answer(positions=np.zeros((1, 3)), success=False, n_gradients=7)])
    assert counter.gradients == 7
    assert counter.hessians == 0


def test_lockstep_reports_an_exception_while_starting():
    def broken():
        raise RuntimeError("before the first ask")
        yield []  # pragma: no cover

    gen = lockstep({"a": broken()})
    with pytest.raises(StopIteration) as stop:
        next(gen)
    assert isinstance(stop.value.value["a"], RuntimeError)


# --------------------------------------------------------------- coordinates


def _hcn_like(off_axis):
    return np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [-2.0, off_axis, 0.0]])


def test_a_nearly_linear_molecule_has_five_rigid_motions():
    for off in (0.0, 1e-5, 1e-3):
        assert rigid_basis(_hcn_like(off)).shape[1] == 5
    assert rigid_basis(_hcn_like(0.5)).shape[1] == 6


def test_dlc_survives_a_linear_molecule_bending():
    x = _hcn_like(1e-7)  # linear to the coordinates when built, then it bends
    out = qcopt.minimize(
        Structure(symbols=["O", "C", "O"], geometry=x),
        {"program": "f"},
        evaluator=qcopt.FunctionEvaluator(triatomic),
        coordinates="dlc",
    )
    assert out.success, out.traceback


def _h2o2(dihedral):
    """H-O-O-H with the given dihedral (degrees), bohr."""
    t = np.radians(dihedral)
    o1, o2 = np.array([0.0, 0.0, 0.0]), np.array([2.8, 0.0, 0.0])
    h1 = o1 + 1.8 * np.array([-0.3, 0.95, 0.0])
    h2 = o2 + 1.8 * np.array([0.3, 0.95 * np.cos(t), 0.95 * np.sin(t)])
    return np.array([h1, o1, o2, h2])


def test_coordinates_rebuild_before_a_torsion_wraps():
    coords = make_coordinates("tric", ["H", "O", "O", "H"], _h2o2(120.0), [], True)
    assert isinstance(coords, InternalCoordinates)
    assert not coords.needs_rebuild(_h2o2(150.0))
    assert coords.needs_rebuild(_h2o2(-140.0))  # 100 degrees on, across 180


def test_bad_constraints_are_reported():
    x = np.array([[2.0, 0, 0], [0, 0, 0], [-1.7, 1.0, 0.0]])
    s = Structure(symbols=["O", "O", "O"], geometry=x)
    ev = qcopt.FunctionEvaluator(triatomic)
    cases = [
        ({"constraints": {"distance": [[0, 1], [1, 0]]}}, "given twice"),
        (
            {"constraints": {"distance": [[0, 1]]}, "frozen_atoms": [0, 1]},
            "only frozen",
        ),
        ({"constraints": {"angle": [[0, 1, 2, 180.0]]}}, "constrain distances instead"),
    ]
    for keywords, message in cases:
        out = qcopt.minimize(s, {"program": "f"}, evaluator=ev, **keywords)
        assert not out.success
        assert message in out.traceback


def test_cartesian_fallback_keeps_the_hessian(monkeypatch):
    original = InternalCoordinates.step
    calls = {"n": 0}

    def failing(self, x, d):
        calls["n"] += 1
        new, taken, ok = original(self, x, d)
        return new, taken, ok and calls["n"] > 21

    monkeypatch.setattr(InternalCoordinates, "step", failing)
    x = np.array([[2.05, 0, 0], [0, 0, 0], [-1.8, 1.0, 0.0]])

    def surface(p):  # a saddle along the 0-1 stretch
        e, g = triatomic(p)
        r = np.linalg.norm(p[0] - p[1])
        u = (p[0] - p[1]) / r
        e = e - 0.6 * (r - 2.0) ** 2
        g = g.copy()
        g[0] -= 1.2 * (r - 2.0) * u
        g[1] += 1.2 * (r - 2.0) * u
        return e, g

    ev = qcopt.FunctionEvaluator(surface, lambda p: _fd_hessian(surface, p))
    out = qcopt.find_ts(
        Structure(symbols=["O", "O", "O"], geometry=x),
        {"program": "f"},
        hessian=True,
        evaluator=ev,
        recompute_hessian_on_loss=False,
    )
    ex = out.results.extras
    assert "cartesian" in ex["coordinates"]
    assert all(s["lowest"] < 0 for s in ex["log"])  # the negative mode survived


@pytest.mark.skipif(find_spec("axtb") is None, reason="needs axtb")
def test_a_reused_axtb_evaluator_uses_each_runs_method():
    water = Structure(
        symbols=["O", "H", "H"],
        geometry=[[0.0, 0.0, 0.0], [1.81, 0.0, 0.0], [-0.45, 1.75, 0.0]],
    )
    ev = qcopt.AxtbEvaluator()
    gfn2 = {"program": "axtb", "model": {"method": "gfn2"}}
    gxtb = {"program": "axtb", "model": {"method": "gxtb-v2.0.1"}}
    first = qcopt.minimize(water, gfn2, evaluator=ev, maxiter=1)
    again = qcopt.minimize(water, gxtb, evaluator=ev, maxiter=1)
    fresh = qcopt.minimize(water, gxtb, evaluator=qcopt.AxtbEvaluator(), maxiter=1)
    e = [o.results.frames[0].results.energy for o in (first, again, fresh)]
    assert e[1] == pytest.approx(e[2]) and e[1] != pytest.approx(e[0])
    xtb = qcopt.minimize(
        water, {"program": "xtb", "model": {"method": "gfn2"}}, evaluator=ev, maxiter=1
    )
    assert not xtb.success
    assert "computes axtb specs" in xtb.traceback
