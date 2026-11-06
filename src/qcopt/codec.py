"""qccodec codec: qcopt as a program qccompute runs, ``ProgramInput(program="qcopt")``.

Installed as the ``qccodec.codecs`` entry point ``qcopt``, so with qcopt installed::

    from qccompute import compute
    out = compute(ProgramInput(program="qcopt", calctype="transition_state", ...))

runs exactly what ``qcopt.run`` does. Children (the gradient and Hessian subprograms) go
to axtb in the calculation's own process when they are axtb's and axtb is installed;
otherwise to the runner's compute callback, one list per round, so the runner batches
them or runs them side by side like any other calculations.
"""

from __future__ import annotations

from typing import Any

from qccodec import Codec, PythonInput
from qccodec.exceptions import CalculationError
from qccodec.native import NativeOutput, PythonOutput
from qcdata import Data, ProgramInput, ProgramOutput

from .api import CALCTYPES, PROGRAM, _prepare, run
from .evaluators import AxtbEvaluator, ComputeEvaluator


def _threads(inp: ProgramInput) -> int:
    return max(
        [int(child.keywords.get("threads", 1)) for child in inp.subprograms] + [1]
    )


def encode(inp: ProgramInput) -> PythonInput:
    """Check the input the way ``qcopt.run`` would, then hand it to :func:`invoke`."""
    _prepare(inp)  # fails here, with the reason, on bad keywords or subprograms
    extra: dict[str, Any] = {}
    if "compute_callback" in PythonInput.__dataclass_fields__:  # qccodec > 0.21
        extra["compute_callback"] = "compute"
    return PythonInput(
        function=invoke,
        kwargs={"inp": inp.model_copy(deep=True)},
        child_callback="compute_child",
        threads=_threads(inp),
        **extra,
    )


def invoke(
    inp: ProgramInput,
    compute: Any = None,
    compute_child: Any = None,
) -> Any:
    """Run qcopt; children in-process (axtb) or through the runner's callback."""
    ctx, _ = _prepare(inp)
    evaluator: Any
    try:
        use_axtb = AxtbEvaluator.handles(ctx)
        if use_axtb:
            import axtb  # noqa: F401
    except ImportError:
        use_axtb = False
    if use_axtb:
        evaluator = AxtbEvaluator(workers=1)
    else:
        if compute is None:
            if compute_child is None:
                raise RuntimeError(
                    "qcopt computes child calculations, but its runner provided no "
                    "compute callback."
                )
            one = compute_child

            def compute(inputs: Any) -> list[ProgramOutput[Any, Any]]:
                return [one(i) for i in inputs]

        evaluator = ComputeEvaluator(compute)
    return run(inp, evaluator=evaluator)


def decode(inp: ProgramInput, out: NativeOutput) -> Data:
    """Qcopt's results; a failed or unconverged run raises with them attached."""
    assert isinstance(out, PythonOutput)
    result = out.value
    if result is None:
        raise CalculationError(f"qcopt did not return a result: {out.error}")
    if not result.success:
        raise CalculationError(result.traceback or "qcopt failed", data=result.results)
    return result.results


codec = Codec(
    PROGRAM,
    calctypes=set(CALCTYPES),
    encode=encode,
    decode=decode,
)
