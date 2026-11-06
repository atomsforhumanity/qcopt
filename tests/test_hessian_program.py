"""The Hessian subprogram: how it is named, and Hessians from another method."""

import numpy as np
import pytest
from qcdata import Structure

import qcopt
from qcopt.driver import Answer

from .surfaces import MB_SADDLES, muller_brown, muller_brown_hessian

SPEC = {"program": "muller-brown"}
LOW = {"program": "low-level"}  # a cheaper method computing the Hessians
OFFSET = 100.0  # the low-level method's energies differ from the target's


def _structure(x, y):
    return Structure(symbols=["H"], geometry=[[x, y, 0.0]])


class TwoLevels:
    """Gradients from Müller-Brown; Hessians from a scaled, shifted copy of it.

    Hessian requests are answered the way a separate program answers them: with its
    own energy and gradient, which must not end up in the frames.
    """

    def __init__(self):
        self.asked = {"gradient": [], "hessian": []}

    def supports_hessian(self, context):
        return context.hessian_spec is not None

    def __call__(self, requests):
        out = []
        for r in requests:
            x = np.asarray(r.positions, dtype=float)
            self.asked[r.kind].append(x.copy())
            e, g = muller_brown(x)
            if r.kind == "hessian":
                assert r.context.hessian_spec.program == "low-level"
                out.append(
                    Answer(
                        positions=x,
                        energy=e + OFFSET,
                        gradient=1.2 * np.asarray(g),
                        hessian=1.2 * muller_brown_hessian(x),
                        program="low-level",
                    )
                )
            else:
                assert r.context.gradient_spec.program == "muller-brown"
                out.append(
                    Answer(positions=x, energy=e, gradient=g, program="muller-brown")
                )
        return out


def test_hessian_argument_is_the_program_not_the_starting_hessian():
    with pytest.raises(TypeError, match="initial_hessian"):
        qcopt.minimize(_structure(0.3, 0.2), SPEC, hessian="model")


def test_initial_hessian_keyword_reaches_the_helpers():
    ev = qcopt.FunctionEvaluator(muller_brown, muller_brown_hessian)
    out = qcopt.minimize(
        _structure(0.3, 0.2),
        SPEC,
        hessian=True,
        initial_hessian="model",
        evaluator=ev,
        trust=0.1,
    )
    assert out.success
    assert out.results.extras["hessian"] == "model"
    assert out.results.extras["n_hessians"] == 0


@pytest.mark.parametrize("every", [0, 1])
def test_lower_level_hessians_keep_the_target_energies(every):
    ev = TwoLevels()
    out = qcopt.find_ts(
        _structure(-0.7, 0.6),
        SPEC,
        hessian=LOW,
        evaluator=ev,
        hessian_every=every,
        trust=0.1,
    )
    assert out.success
    assert np.allclose(
        out.results.final_structure.geometry[0, :2], MB_SADDLES[0], atol=1e-4
    )
    for frame in out.results.frames:
        e, _ = muller_brown(frame.structure.geometry)
        assert frame.results.energy == pytest.approx(e)  # never the low-level one
        assert frame.results.provenance.program == "muller-brown"
        assert frame.results.hessian is None  # a low-level Hessian is not the frame's
    ex = out.results.extras
    assert ex["n_hessians"] == len(ev.asked["hessian"]) >= 1 + every
    assert ex["n_gradients"] == len(ev.asked["gradient"])
    # A Hessian at a point whose gradient is known does not compute it again.
    seen = {x.tobytes() for x in ev.asked["gradient"]}
    assert len(seen) == len(ev.asked["gradient"])


def test_lower_level_hessians_need_an_evaluator_that_computes_them():
    ev = qcopt.FunctionEvaluator(muller_brown)  # gradients only
    out = qcopt.find_ts(_structure(-0.7, 0.6), SPEC, hessian=LOW, evaluator=ev)
    assert not out.success
    assert "cannot compute Hessians" in out.traceback
