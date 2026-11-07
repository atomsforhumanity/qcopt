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
    assert "end minimization failed: not converged" in out.traceback


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
    for keywords, message in cases:  # rejected before anything is computed
        with pytest.raises(ValueError, match=message):
            qcopt.minimize(s, {"program": "f"}, evaluator=ev, **keywords)


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


# ------------------------------------------------- round 2 (review of the fixes)


def test_ts_check_also_runs_on_the_last_iteration():
    x, y = MB_MINIMA[1]
    out = qcopt.find_ts(
        _atom(x + 0.01, y), MB, hessian=True, evaluator=_ev(), maxiter=2
    )
    assert not out.success


def _two_shoulders(c=0.02, k=0.05):
    from numpy.polynomial import polynomial as P

    g = c * P.polyfromroots([0, 1, 1, 2, 2, 3])  # TS at 0, shoulders 1 and 2, min 3
    e, h = P.polyint(g), P.polyder(g)

    def fn(p):
        x, yy, zz = (float(v) for v in p[0])
        grad = np.array([[P.polyval(x, g), 2 * k * yy, 2 * k * zz]])
        return P.polyval(x, e) + k * (yy * yy + zz * zz), grad

    def hess(p):
        return np.diag([P.polyval(float(p[0, 0]), h), 2 * k, 2 * k])

    return fn, hess


def test_irc_end_recheck_keeps_going_down_a_shoulder():
    # At step 0.26 the second exact end check comes on the shoulder at 2 while the
    # path still goes down (other steps end on the shoulder at 1, where the gradient
    # vanishes and the curvature is zero: an IRC cannot tell that from a minimum).
    fn, hess = _two_shoulders()
    for ds in (0.26,):
        out = qcopt.irc(
            _atom(0.0),
            {"program": "f"},
            hessian=True,
            evaluator=qcopt.FunctionEvaluator(fn, hess),
            direction="forward",
            step_size=ds,
            update="bfgs",
        )
        assert out.success, out.traceback
        assert _irc_frames_x(out)[-1] > 2.8  # the minimum at 3, not the shoulder at 2


def test_a_nan_energy_with_a_hessian_fails():
    class NaNAtHessians(qcopt.FunctionEvaluator):
        def __call__(self, requests):
            out = super().__call__(requests)
            for r, a in zip(requests, out, strict=True):
                if r.kind == "hessian":
                    a.energy = float("nan")
            return out

    ev = NaNAtHessians(muller_brown, muller_brown_hessian)
    out = qcopt.find_ts(_mb((-0.8, 0.6)), MB, hessian=True, evaluator=ev)
    assert not out.success
    assert "non-finite energy" in out.traceback


def _oco(angle):
    t = np.radians(angle / 2)
    return np.array(
        [
            [-2.2 * np.sin(t), 2.2 * np.cos(t), 0],
            [0, 0, 0],
            [2.2 * np.sin(t), 2.2 * np.cos(t), 0],
        ]
    )


def test_bent_molecules_keep_three_rotations():
    for angle, rigid in ((180.0, 5), (179.9, 5), (179.0, 6), (176.0, 6), (120.0, 6)):
        assert rigid_basis(_oco(angle)).shape[1] == rigid, angle
    for angle in (176.0, 174.5, 173.0):  # DLC builds with ordinary bends here
        coords = make_coordinates("dlc", ["O", "C", "O"], _oco(angle), [], True)
        assert coords.name == "dlc", angle


def test_hessian_keywords_do_not_make_a_separate_program():
    from qcdata import ProgramSpec

    from qcopt.api import _separate

    def spec(calctype, **kw):
        return ProgramSpec(
            program="axtb", calctype=calctype, model={"method": "gfn2"}, keywords=kw
        )

    grad = spec("gradient", accuracy=0.01)
    assert not _separate(
        grad, spec("hessian", accuracy=0.01, hessian_accuracy=1e-4, threads=4)
    )
    assert _separate(grad, spec("hessian", accuracy=0.01, alpb="water"))


def test_routing_without_contexts_pairs_hessians_conservatively():
    from qcopt.driver import TaskContext
    from qcopt.evaluators import RoutingEvaluator

    route = RoutingEvaluator(
        [qcopt.FunctionEvaluator(muller_brown), qcopt.ComputeEvaluator(lambda i: [])],
        [0],
    )
    ctx = TaskContext(symbols=["H"])
    assert route.hessian_includes_gradient(ctx) is False


def test_counts_survive_an_exception(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("frequencies broke")

    monkeypatch.setattr(algorithms, "frequencies", broken)
    out = qcopt.minimize(
        _atom(0.3, 0.2), MB, hessian=True, evaluator=_ev(), frequencies=True
    )
    assert out.results.extras["n_gradients"] > 0
    assert out.results.extras["n_hessians"] >= 1


# --------------------------------------- round 2 (end-to-end scenario matrix)


def _saddle_eigen():
    h = muller_brown_hessian(np.array([[*MB_SADDLES[0], 0.0]]))[:2, :2]
    return np.linalg.eigh(h)


class _LowLevel(qcopt.FunctionEvaluator):
    """Müller-Brown gradients; Hessians from another 'program' given as a function."""

    def __init__(self, low_hessian):
        super().__init__(muller_brown, low_hessian)

    def supports_hessian(self, context):
        return True


def test_lower_level_frequencies_do_not_decide_a_ts_search():
    def low(p):  # Müller-Brown's, with an extra negative mode along z
        h = muller_brown_hessian(p)
        h[2, 2] = -1e-4
        return h

    out = qcopt.find_ts(
        _mb((-0.7, 0.6)),
        MB,
        hessian={"program": "low"},
        evaluator=_LowLevel(low),
        trust=0.1,
        frequencies=True,
    )
    assert out.success, out.traceback
    assert out.results.extras["n_imaginary"] == 2  # reported, labelled
    assert out.results.extras["frequencies"]["program"] == "low"


def test_irc_that_cannot_descend_fails():
    vals, vecs = _saddle_eigen()

    def swapped(p):  # the negative mode where the path goes up
        h = np.zeros((3, 3))
        h[:2, :2] = vals[0] * np.outer(vecs[:, 1], vecs[:, 1]) + vals[1] * np.outer(
            vecs[:, 0], vecs[:, 0]
        )
        h[2, 2] = 1.0
        return h

    out = qcopt.irc(
        _mb(MB_SADDLES[0]), MB, hessian={"program": "low"}, evaluator=_LowLevel(swapped)
    )
    assert not out.success
    ts = out.results.frames[out.results.ts_index].results.energy
    assert all(f.results.energy <= ts + 1e-12 for f in out.results.frames)


def test_minimization_that_ends_higher_than_it_started_fails():
    calls = {"n": 0}

    def changes_state(x):  # the program lands on another solution after 3 calls
        calls["n"] += 1
        e, g = muller_brown(x)
        return (e + (0.01 if calls["n"] > 3 else 0.0)), g

    out = qcopt.minimize(
        _atom(-0.5, 1.4), MB, evaluator=qcopt.FunctionEvaluator(changes_state)
    )
    assert not out.success
    assert "higher than it started" in out.traceback


def test_trust_starts_within_its_bounds():
    out = qcopt.minimize(
        _atom(0.3, 0.2),
        MB,
        evaluator=_ev(False),
        coordinates="cartesian",
        trust=0.5,
        trust_max=0.05,
    )
    assert out.results.extras["log"][0]["trust"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="trust_min"):
        qcopt.minimize(
            _atom(0.3, 0.2), MB, evaluator=_ev(False), trust_min=0.5, trust_max=0.1
        )


def test_counts_are_the_calculations_run():
    class Breaks(qcopt.FunctionEvaluator):
        computed = 0
        calls = 0

        def __call__(self, requests):
            Breaks.calls += 1
            if Breaks.calls >= 4:
                raise RuntimeError("down")
            Breaks.computed += len(requests)
            return super().__call__(requests)

    out = qcopt.minimize(_atom(0.3, 0.2), MB, evaluator=Breaks(muller_brown))
    assert not out.success
    assert out.results.extras["n_gradients"] == Breaks.computed == 3


def test_minimized_ends_are_not_duplicated():
    # the branch ends after an exact end check; the end minimization takes no step
    out = qcopt.irc(
        _atom(0.0),
        {"program": "f"},
        hessian=True,
        evaluator=qcopt.FunctionEvaluator(shoulder, shoulder_hessian),
        direction="forward",  # the surface has no minimum on the other side
        update="bfgs",
        minimize_ends=True,
        end_convergence={"gmax": 1e-2},
    )
    assert out.success, out.traceback
    x = [f.structure.geometry.tobytes() for f in out.results.frames]
    assert all(a != b for a, b in zip(x, x[1:], strict=False))


def test_coincident_atoms_fall_back_to_cartesians():
    x = np.array([[0.0, 0, 0], [0.0, 0, 0], [1.8, 0.2, 0.0]])
    coords = make_coordinates("tric", ["O", "H", "H"], x, [], True)
    assert coords.name == "cartesian"


def test_an_irc_that_cannot_start_points_at_its_frame():
    out = qcopt.irc(_mb(MB_MINIMA[1]), MB, hessian=True, evaluator=_ev())
    assert not out.success
    assert out.results.ts_index == 0


# ------------------------------------------------ round 2 (numerical cross-checks)

_ANG = 1.8897261246257702


def _c4h2(bend, at=0):
    """H-C-C-C-C-H along z, bent by ``bend`` degrees at atom ``at + 1``."""
    x = [np.zeros(3)]
    for length in (1.06, 1.21, 1.37, 1.21, 1.06):
        x.append(x[-1] + np.array([0.0, 0.0, length * _ANG]))
    x = np.array(x)
    t = np.radians(bend)
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    pivot = x[at + 1].copy()
    x[at + 2 :] = (x[at + 2 :] - pivot) @ rot.T + pivot
    return x


def test_tric_back_transformation_reports_a_flipped_rotation():
    x = _c4h2(10.0)
    coords = make_coordinates("tric", ["H", "C", "C", "C", "C", "H"], x, [], True)
    rng = np.random.default_rng(1)
    for _ in range(300):
        d = rng.normal(size=coords.ic.n)
        d -= coords._rigid @ (coords._rigid.T @ d)
        d *= 0.2 / np.linalg.norm(d)
        _, taken, ok = coords.step(x, d)
        assert not ok or np.abs(taken - d).max() < 1e-2


def test_coordinates_need_no_rebuild_right_after_building():
    symbols = ["H", "C", "C", "C", "C", "H"]
    for at in range(4):
        for bend in (5.0, 10.0, 20.0):
            x = _c4h2(bend, at)
            for kind in ("tric", "dlc"):
                coords = make_coordinates(kind, symbols, x, [], True)
                if isinstance(coords, InternalCoordinates):
                    assert not coords.needs_rebuild(x), (kind, at, bend)


def test_prfo_climbs_with_a_tiny_gradient_along_the_mode():
    from qcopt.steps import rfo_step

    for g_mode in (1e-12, 1e-10, 1e-8):
        step = rfo_step(
            np.array([0.1, 0.5, 0.7]),
            np.eye(3),
            np.array([g_mode, 1e-3, 1e-3]),
            0.2,
            mode=0,
        )
        assert abs(step.dx[0]) == pytest.approx(0.2, rel=1e-3)


def test_lqa_steps_have_the_requested_arc_length():
    from qcopt.algorithms import _lqa_step

    step = _lqa_step(np.array([-0.05, 0.3]), np.array([1e-5, 1e-5]), 0.3)
    assert np.linalg.norm(step) == pytest.approx(0.3, rel=1e-3)
    end = _lqa_step(np.array([0.01, 0.5]), np.array([2e-3, 2e-3]), 0.3)
    assert np.linalg.norm(end) == pytest.approx(0.2, rel=1e-3)  # the path's end


def test_bofill_meets_the_secant_condition_on_tiny_steps():
    from qcopt.hessians import bofill

    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    true = a + a.T
    b = rng.normal(size=(6, 6))
    h = true + 1e-3 * (b + b.T)  # a nearly right Hessian: r.s is tiny
    for size in (1e-4, 1e-6):
        s = rng.normal(size=6)
        s *= size / np.linalg.norm(s)
        y = true @ s
        assert np.linalg.norm(bofill(h, s, y) @ s - y) < 1e-6 * np.linalg.norm(y)


def test_degenerate_torsions_are_not_built():
    x = np.array(
        [  # HC4H, slightly bent; its torsion 0-1-4-5 has a 177 degree angle
            [0.1184293896, 0.0518202649, -0.0691388191],
            [-0.0497054628, -0.0678762621, 1.9527073886],
            [0.0046758639, -0.4437923246, 4.2800432079],
            [-0.02491774, -0.8025865723, 6.773194157],
            [0.0226992628, -1.2659818831, 8.8400672525],
            [-0.0218614617, -1.564193342, 11.0505842694],
        ]
    )
    for kind in ("tric", "dlc"):
        coords = make_coordinates(kind, ["H", "C", "C", "C", "C", "H"], x, [], True)
        if isinstance(coords, InternalCoordinates):  # else Cartesians: also fine
            assert not coords.needs_rebuild(x), kind


# ------------------------------------------------ round 3 (matrix rerun findings)


def test_a_failed_finite_difference_hessian_is_no_hessian():
    calls = {"n": 0}

    def fails_once(x):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("one displaced gradient failed")
        return muller_brown(x)

    out = qcopt.find_ts(
        _mb((-0.7, 0.6)), MB, evaluator=qcopt.FunctionEvaluator(fails_once)
    )
    assert not out.success
    assert out.results.extras["n_hessians"] == 0
    assert out.results.extras["n_gradients"] == calls["n"]


def test_minimization_at_a_saddle_with_its_exact_hessian_fails():
    out = qcopt.minimize(_mb(MB_SADDLES[1]), MB, hessian=True, evaluator=_ev())
    assert not out.success
    assert "saddle point" in out.traceback


def test_ts_search_checks_a_strong_second_negative_mode():
    def fn(p):  # a second-order saddle at the origin; y is symmetric, so never sampled
        x, y, z = (float(v) for v in p[0])
        return -0.1 * x * x - 0.05 * y * y + 0.2 * z * z, np.array(
            [[-0.2 * x, -0.1 * y, 0.4 * z]]
        )

    def hess(p):
        return np.diag([-0.2, -0.1, 0.4])

    out = qcopt.find_ts(
        _atom(0.05),
        {"program": "f"},
        hessian=True,
        evaluator=qcopt.FunctionEvaluator(fn, hess),
    )
    assert not out.success
    assert "2 directions of negative curvature" in out.traceback


# --------------------------------------------- round 4 (review of the round-2 fixes)


def _soft_top():
    """A barrier whose top is soft (curvature -1e-4) and steep beyond |x| ~ 0.5."""
    a, c, e, k = 1e-4, 0.1, 0.025, 0.05

    def fn(p):
        x, y, z = (float(v) for v in p[0])
        en = -0.5 * a * x**2 - c / 8 * x**8 + e / 10 * x**10 + 0.5 * k * (y * y + z * z)
        return en, np.array([[-a * x - c * x**7 + e * x**9, k * y, k * z]])

    def hess(p):
        x = float(p[0, 0])
        return np.diag([-a - 7 * c * x**6 + 9 * e * x**8, k, k])

    return fn, hess


def test_irc_from_a_transition_state_slightly_off_the_saddle():
    fn, hess = _soft_top()
    ev = qcopt.FunctionEvaluator(fn, hess)
    ts = qcopt.find_ts(_atom(0.18), {"program": "f"}, hessian=True, evaluator=ev)
    assert ts.success  # converged as it is: the gradient is below gmax
    out = qcopt.irc(_atom(0.18), {"program": "f"}, hessian=True, evaluator=ev)
    assert out.success, out.traceback
    x = _irc_frames_x(out)
    assert x[0] == pytest.approx(-2.0, abs=0.05) and x[-1] == pytest.approx(
        2.0, abs=0.05
    )


def test_trust_after_a_saddle_escape_stays_within_trust_max():
    out = qcopt.minimize(
        _mb(MB_SADDLES[1]),
        MB,
        hessian=True,
        evaluator=_ev(),
        escape_saddles=1,
        coordinates="cartesian",
        trust=5.0,
        trust_max=0.05,
    )
    assert out.results.extras.get("saddle_escapes") == 1
    assert max(s["trust"] for s in out.results.extras["log"]) <= 0.05


def test_same_method_hessians_with_other_settings_still_decide():
    other = {"program": "muller-brown", "keywords": {"memory": 1}}  # not the surface
    out = qcopt.minimize(
        _mb(MB_SADDLES[1]), MB, hessian=other, evaluator=_ev(), frequencies=True
    )
    assert not out.success
    assert "saddle point" in out.traceback


def test_lqa_finds_the_end_of_a_short_path_quickly(monkeypatch):
    from qcopt.algorithms import _lqa_step

    calls = {"n": 0}
    real = np.geomspace

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(algorithms.np, "geomspace", counting)
    vals = np.linspace(0.01, 1.0, 30)
    step = _lqa_step(vals, 1e-3 * np.ones(30), 1.0)  # the path is shorter than 1.0
    assert np.allclose(step, -1e-3 / vals, rtol=1e-6)  # its end: the Newton step
    assert calls["n"] < 30
