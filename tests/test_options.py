"""Settings: groups, suggestions, what auto became, error kinds in the results."""

import numpy as np
import pytest
from qccompute.program import FunctionProgram
from qcdata import ProgramError, ProgramInput, Structure

import qcopt
from qcopt.errors import InputError
from qcopt.options import default_options, options_schema, parse_options
from qcopt.program import program as qcopt_program

from .helpers import (
    FunctionEvaluator,
    find_ts,
    irc,
    minimize,
    qcopt_input,
    registered,
    run,
)
from .surfaces import MB_MINIMA, muller_brown, muller_brown_hessian

MB = {"program": "muller-brown"}


def _ev(hessian=True):
    return FunctionEvaluator(
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
    defaults = default_options(calctype)
    schema = options_schema(calctype)
    assert set(defaults) == set(schema["properties"])
    assert defaults["convergence"]["max_iter"] >= 1
    # qcdata's conventions: every setting carries x-tier, x-concern, x-applies-to
    concerns = {
        "model",
        "approximation",
        "convergence",
        "algorithm",
        "resource",
        "output",
    }
    props = [(name, prop) for name, prop in schema["properties"].items()]
    for name in schema["$defs"]:
        props += list(schema["$defs"][name]["properties"].items())
    for key, prop in props:
        assert prop.get("x-tier") in ("common", "expert"), key
        assert prop.get("x-concern") in concerns, key
        assert calctype in prop["x-applies-to"]["calctypes"], key
    top = schema["properties"]
    assert top["threads"]["x-concern"] == "resource" and top["threads"]["x-shared"]
    assert top["convergence"]["x-concern"] == "convergence"


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
    with pytest.raises(InputError, match=message):
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
    with pytest.raises(InputError, match=message):
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
    with pytest.raises(InputError, match=f"{path} cannot affect") as info:
        parse_options(calctype, keywords)
    assert info.value.code == "no_effect"


def test_settings_acting_only_if_a_condition_arises_are_allowed():
    parse_options("optimization", {"escape": {"attempts": 2, "step": 0.1}})
    parse_options("optimization", {"trust": {"max_rejections": 5}})
    parse_options("transition_state", {"hessian": {"lost_curvature": 1e-3}})
    parse_options("optimization", {"freq": {"check": True, "imaginary_cutoff": 20.0}})


def test_the_finite_difference_step_needs_finite_differences():
    with pytest.raises(ProgramError, match="hessian.step cannot affect"):
        find_ts(
            _atom(-0.8, 0.6),
            MB,
            hessian_program=True,
            evaluator=_ev(),
            hessian={"step": 0.01},
        )
    out = find_ts(_atom(-0.8, 0.6), MB, evaluator=_ev(False), hessian={"step": 0.01})
    assert out.success, out.traceback


def test_input_errors_raise_with_kind_input_and_the_failed_output():
    with registered(FunctionProgram("muller-brown", muller_brown)):
        with pytest.raises(ProgramError, match="unknown setting 'nonsense'") as info:
            qcopt.compute(qcopt_input("optimization", _atom(0.3, 0.2), MB, nonsense=1))
    assert (info.value.error.kind, info.value.error.code) == ("input", "unknown_key")
    out = info.value.output
    assert not out.success and out.results.frames == []


@pytest.mark.parametrize(
    "accuracy, kind, code",
    [
        ({"gradient": 1e-5}, "unsupported", None),  # a target qcopt cannot honour yet
        (0.01, "input", None),  # a bare number, as every program refuses it
        ({"grad": 1e-5}, "input", "unknown_key"),
    ],
)
def test_the_shared_accuracy_keyword_is_read_as_every_program_reads_it(
    accuracy, kind, code
):
    inp = qcopt_input("optimization", _atom(0.3, 0.2), MB, accuracy=accuracy)
    with registered(FunctionProgram("muller-brown", muller_brown)):
        with pytest.raises(ProgramError, match="accuracy") as info:
            qcopt.compute(inp)
        (failed,) = qcopt_program.compute_batch([inp])
    assert (info.value.error.kind, info.value.error.code) == (kind, code)
    assert (failed.error.kind, failed.error.code) == (kind, code)


def test_no_accuracy_target_is_no_accuracy():
    out = minimize(_atom(0.3, 0.2), MB, evaluator=_ev(False), accuracy=None)
    assert out.success, out.traceback


def test_auto_is_resolved_with_reasons():
    out = find_ts(_atom(-0.8, 0.6), MB, evaluator=_ev(False))
    resolved = out.resolved
    assert resolved.settings["coordinate_system"] == "cartesian"  # one atom
    assert resolved.settings["hessian"]["initial"] == "finite_difference"
    assert resolved.settings["hessian"]["update"] == "ts_bfgs"
    assert set(resolved.reasons) == {
        "coordinate_system",
        "hessian.initial",
        "hessian.update",
    }
    frozen = minimize(
        WATER,
        MB,
        evaluator=FunctionEvaluator(_harmonic),
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
    first = run(
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
    again = run(
        first.input_data.model_copy(update={"keywords": settings}), evaluator=_ev()
    )
    assert again.resolved.settings == settings
    assert len(again.results.frames) == len(first.results.frames)
    for a, b in zip(first.results.frames, again.results.frames, strict=True):
        assert np.array_equal(a.structure.geometry, b.structure.geometry)


def test_failures_raise_with_their_kind_code_and_output():
    def broken(x):
        raise RuntimeError("the program crashed")

    surfaces = (
        FunctionProgram("muller-brown", muller_brown, muller_brown_hessian),
        FunctionProgram("broken", broken),
    )
    with registered(*surfaces):
        with pytest.raises(ProgramError) as info:
            qcopt.compute(
                qcopt_input(
                    "optimization", _atom(-0.4, 1.2), MB, convergence={"max_iter": 2}
                )
            )
        error, out = info.value.error, info.value.output
        assert (error.kind, error.code) == ("convergence", "max_iter")
        assert out.results.frames  # the frames so far, to restart from
        assert out.resolved.settings["convergence"]["max_iter"] == 2

        with pytest.raises(ProgramError) as info:
            qcopt.compute(
                qcopt_input("optimization", _atom(-0.4, 1.2), {"program": "broken"})
            )
        error = info.value.error
        assert (error.kind, error.program) == ("program", "broken")
        assert "the program crashed" in error.cause.message

        with pytest.raises(ProgramError) as info:
            qcopt.compute(
                qcopt_input(
                    "transition_state", _atom(*MB_MINIMA[0]), MB, hessian_program=True
                )
            )
        error = info.value.error
        assert (error.kind, error.code) == ("convergence", "not_first_order_saddle")

        ok = qcopt.compute(qcopt_input("optimization", _atom(-0.4, 1.2), MB))
    assert ok.success and ok.error is None


def test_record_says_what_happened_and_how_long_it_took():
    out = minimize(_atom(-0.4, 1.2), MB, evaluator=_ev())
    record = out.record
    assert record["converged"] is True
    assert record["n_gradients"] >= len(out.results.frames)  # rejected steps too
    assert record["coordinate_system"] == "cartesian"
    timings = out.execution.timings
    assert timings["total"] >= timings["subprograms"] >= 0.0
    assert timings["optimizer"] == pytest.approx(
        timings["total"] - timings["subprograms"]
    )
    path = irc(
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
    with pytest.raises(InputError) as info:
        parse_options("optimization", keywords)
    assert info.value.code == code
