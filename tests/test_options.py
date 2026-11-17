"""Settings: groups, suggestions, what auto became, error kinds in the results."""

import numpy as np
import pytest
from qcdata import ProgramInput, Structure

import qcopt
from qcopt.options import parse_options

from .surfaces import MB_MINIMA, muller_brown, muller_brown_hessian

MB = {"program": "muller-brown"}


def _ev(hessian=True):
    return qcopt.FunctionEvaluator(
        muller_brown, muller_brown_hessian if hessian else None, name="muller-brown"
    )


def _atom(x, y):
    return Structure(symbols=["H"], geometry=[[x, y, 0.0]])


WATER = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],
)


@pytest.mark.parametrize("calctype", ["optimization", "transition_state", "irc"])
def test_defaults_and_schema_cover_every_setting(calctype):
    defaults = qcopt.default_options(calctype)
    schema = qcopt.options_schema(calctype)
    assert set(defaults) == set(schema["properties"])
    assert defaults["convergence"]["max_iter"] >= 1
    # every group's settings carry a tier for documentation
    for name in schema["$defs"]:
        for key, prop in schema["$defs"][name]["properties"].items():
            assert prop.get("tier") in ("common", "expert"), (name, key)


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"convergenc": {}}, "did you mean 'convergence'"),
        ({"maxiter": 10}, "renamed convergence.max_iter"),
        ({"convergence": {"gmax": 1e-4}}, "renamed convergence.g_tol"),
        ({"trust": 0.3}, "trust is a group"),
        ({"coordinate_system": "tric", "constraints": {"frozen": [0]}}, "cannot hold"),
        ({"trust": {"min": 0.5, "max": 0.1}}, "trust.min is larger"),
    ],
)
def test_bad_settings_say_what_to_do(keywords, message):
    with pytest.raises(qcopt.InputError, match=message):
        parse_options("optimization", keywords)


@pytest.mark.parametrize(
    ("calctype", "keywords", "message"),
    [
        ("optimization", {"mode": {"follow": "lowest"}}, "only for transition_state"),
        ("optimization", {"irc": {"step": 0.1}}, "only for irc"),
        (
            "optimization",
            {"hessian": {"recompute_on_loss": False}},
            "only for transition_state",
        ),
        ("irc", {"escape": {"attempts": 1}}, "only for optimization"),
        ("irc", {"convergence": {"g_rms_tol": 1e-5}}, "only for optimization"),
    ],
)
def test_settings_of_another_calculation_type_are_errors(calctype, keywords, message):
    with pytest.raises(qcopt.InputError, match=message):
        parse_options(calctype, keywords)


@pytest.mark.parametrize(
    ("calctype", "keywords", "path"),
    [
        ("irc", {"ends": {"g_tol": 1e-3}}, "ends.g_tol"),
        ("optimization", {"escape": {"step": 0.1}}, "escape.step"),
        ("optimization", {"freq": {"imaginary_cutoff": 20.0}}, "freq.imaginary_cutoff"),
        ("optimization", {"constraints": {"tol": 1e-8}}, "constraints.tol"),
        ("transition_state", {"mode": {"follow": "lowest"}}, "mode.follow"),
        (
            "transition_state",
            {"hessian": {"recompute_on_loss": False, "recompute_min_steps": 5}},
            "hessian.recompute_min_steps",
        ),
    ],
)
def test_settings_that_cannot_affect_the_calculation_are_errors(
    calctype, keywords, path
):
    with pytest.raises(qcopt.InputError, match=f"{path} cannot affect") as info:
        parse_options(calctype, keywords)
    assert info.value.code == "no_effect"


def test_settings_acting_only_if_a_condition_arises_are_allowed():
    parse_options("optimization", {"escape": {"attempts": 2, "step": 0.1}})
    parse_options("optimization", {"trust": {"max_rejections": 5}})
    parse_options("transition_state", {"hessian": {"lost_curvature": 1e-3}})
    parse_options("optimization", {"freq": {"check": True, "imaginary_cutoff": 20.0}})


def test_the_finite_difference_step_needs_finite_differences():
    with pytest.raises(qcopt.InputError, match="hessian.step cannot affect"):
        qcopt.find_ts(
            _atom(-0.8, 0.6),
            MB,
            hessian_program=True,
            evaluator=_ev(),
            hessian={"step": 0.01},
        )
    out = qcopt.find_ts(
        _atom(-0.8, 0.6), MB, evaluator=_ev(False), hessian={"step": 0.01}
    )
    assert out.success, out.traceback


def test_input_errors_are_value_errors_with_a_kind():
    with pytest.raises(ValueError) as info:
        qcopt.minimize(_atom(0.3, 0.2), MB, evaluator=_ev(), nonsense=1)
    assert info.value.kind == "input"


def test_groups_can_be_given_as_models():
    out = qcopt.minimize(
        _atom(0.3, 0.2),
        MB,
        evaluator=_ev(),
        convergence=qcopt.ConvergenceSettings(g_tol=1e-6),
        trust=qcopt.MinimizationTrustSettings(radius=0.1),
    )
    assert out.success, out.traceback
    assert out.input_data.keywords["convergence"] == {"g_tol": 1e-6}
    settings = out.resolved.settings
    assert settings["convergence"]["g_tol"] == 1e-6
    assert settings["trust"]["radius"] == 0.1


def test_auto_is_resolved_with_reasons():
    out = qcopt.find_ts(_atom(-0.8, 0.6), MB, evaluator=_ev(False))
    resolved = out.resolved
    assert resolved.settings["coordinate_system"] == "cartesian"  # one atom
    assert resolved.settings["hessian"]["initial"] == "finite_difference"
    assert resolved.settings["hessian"]["update"] == "ts_bfgs"
    assert set(resolved.reasons) == {
        "coordinate_system",
        "hessian.initial",
        "hessian.update",
    }
    frozen = qcopt.minimize(
        WATER,
        MB,
        evaluator=qcopt.FunctionEvaluator(_harmonic),
        constraints={"frozen": [0]},
    )
    assert frozen.resolved.settings["coordinate_system"] == ("cartesian")
    assert "frozen" in frozen.resolved.reasons["coordinate_system"]


def _harmonic(x):
    d = np.asarray(x) - WATER.geometry
    return float((d * d).sum()), 2 * d


@pytest.mark.parametrize("calctype", ["optimization", "transition_state", "irc"])
def test_resolved_settings_reproduce_the_run(calctype):
    start = {
        "optimization": (-0.5, 1.4),
        "transition_state": (-0.8, 0.6),
        "irc": (-0.822002, 0.624313),
    }[calctype]
    first = qcopt.run(
        ProgramInput(
            program="qcopt",
            calctype=calctype,
            structure=_atom(*start),
            keywords={"ends": {"minimize": True}} if calctype == "irc" else {},
            subprograms=[
                {**MB, "calctype": "gradient"},
                {**MB, "calctype": "hessian"},
            ],
        ),
        evaluator=_ev(),
    )
    settings = first.resolved.settings
    again = qcopt.run(
        first.input_data.model_copy(update={"keywords": settings}), evaluator=_ev()
    )
    assert again.resolved.settings == settings
    assert len(again.results.frames) == len(first.results.frames)
    for a, b in zip(first.results.frames, again.results.frames, strict=True):
        assert np.array_equal(a.structure.geometry, b.structure.geometry)


def test_failures_report_their_kind_and_code():
    unconverged = qcopt.minimize(
        _atom(-0.4, 1.2), MB, evaluator=_ev(), convergence={"max_iter": 2}
    )
    assert (unconverged.error.kind, unconverged.error.code) == (
        "convergence",
        "max_iter",
    )

    def broken(x):
        raise RuntimeError("the program crashed")

    failed = qcopt.minimize(
        _atom(-0.4, 1.2), MB, evaluator=qcopt.FunctionEvaluator(broken)
    )
    assert failed.error.kind == "program"
    assert failed.error.program == "function"

    at_minimum = qcopt.find_ts(
        _atom(*MB_MINIMA[0]), MB, hessian_program=True, evaluator=_ev()
    )
    assert not at_minimum.success
    assert (at_minimum.error.kind, at_minimum.error.code) == (
        "convergence",
        "not_first_order_saddle",
    )
    ok = qcopt.minimize(_atom(-0.4, 1.2), MB, evaluator=_ev())
    assert ok.success and ok.error is None


def test_record_says_what_happened_and_how_long_it_took():
    out = qcopt.minimize(_atom(-0.4, 1.2), MB, evaluator=_ev())
    record = out.record
    assert record["converged"] is True
    assert record["n_gradients"] >= len(out.results.frames)  # rejected steps too
    assert record["coordinate_system"] == "cartesian"
    timings = out.execution.timings
    assert timings["total"] >= timings["evaluator"] >= 0.0
    assert timings["optimizer"] == pytest.approx(
        timings["total"] - timings["evaluator"]
    )
    path = qcopt.irc(
        _atom(-0.822002, 0.624313),
        MB,
        hessian_program=True,
        evaluator=_ev(),
        ends={"minimize": True},
    )
    branches = path.record["branches"]
    assert set(branches) == {"forward", "reverse"}
    assert all(b["end"].startswith("converged") for b in branches.values())
    assert set(path.record["ends"]) == {"forward", "reverse"}


def test_the_settings_reference_is_current():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "settings_doc.py"
    done = subprocess.run([sys.executable, str(script), "--check"], check=False)
    assert done.returncode == 0, (
        "run scripts/settings_doc.py to update docs/settings.md"
    )


@pytest.mark.parametrize(
    "keywords, code",
    [
        ({"nonsense": 1}, "unknown_key"),
        ({"irc": {"step": 0.1}}, "not_applicable"),  # an IRC setting
    ],
)
def test_unknown_settings_carry_afh_codes(keywords, code):
    with pytest.raises(qcopt.InputError) as info:
        parse_options("optimization", keywords)
    assert info.value.code == code
