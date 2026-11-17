"""Many minimizations with an ML potential: one at a time vs. as one list.

Every optimization is a ``qcopt`` ``ProgramInput`` with an ML-potential gradient
subprogram. One at a time (``qcopt.compute(inp)``), each round's gradient is a call of
its own; as one list (``qccompute.compute(inputs)``), qcopt runs them in lockstep and
hands each round's gradients from every optimization to the runner as one batch, which
the ML potential evaluates together (on a GPU, if one is granted).

usage: batch_demo.py OUT.json --data DIR --n 64 [--model aimnet2] [--modes list sequential]

DIR holds one folder per reaction with start.xyz and end.xyz (Angstrom); the first N of the
sorted starts, then ends, are minimized. ``subprograms_s`` in the output is the time the
optimizations spent waiting for gradients (summed over them); the rest of ``wall`` is
qcopt's own work and the runner's.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from qcref import constants as c  # noqa: E402


def structures(data: str, n: int):
    from qcdata import Structure

    files = sorted(glob.glob(f"{data}/*/start.xyz")) + sorted(
        glob.glob(f"{data}/*/end.xyz")
    )
    out = []
    for f in files[:n]:
        lines = open(f).read().splitlines()
        k = int(lines[0])
        sym = [line.split()[0] for line in lines[2 : 2 + k]]
        xyz = np.array(
            [[float(v) for v in line.split()[1:4]] for line in lines[2 : 2 + k]]
        )
        out.append(Structure(symbols=sym, geometry=xyz * c.ANGSTROM_TO_BOHR))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--data", required=True, help="folder of <reaction>/{start,end}.xyz")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--model", default="aimnet2")
    p.add_argument("--modes", nargs="+", default=["list", "sequential"])
    p.add_argument("--maxiter", type=int, default=500)
    a = p.parse_args()

    import qccompute
    from qcdata import CalcType, ProgramError, ProgramInput, ProgramSpec

    import qcopt

    grad = ProgramSpec(
        program=a.model, calctype=CalcType.gradient, model={"method": a.model}
    )
    inputs = [
        ProgramInput(
            program="qcopt",
            calctype="optimization",
            structure=s,
            subprograms=[grad],
            keywords={"convergence": {"max_iter": a.maxiter}},
        )
        for s in structures(a.data, a.n)
    ]
    # start the model once before timing
    qccompute.compute(ProgramInput.from_spec(grad, inputs[0].structure))
    record = {"n": len(inputs), "model": a.model, "modes": {}}
    for mode in a.modes:
        t0 = time.perf_counter()
        if mode == "list":
            outs = qccompute.compute(inputs, raise_exc=False)
        else:
            outs = []
            for inp in inputs:
                try:
                    outs.append(qcopt.compute(inp))
                except ProgramError as exc:
                    outs.append(exc.output)
        wall = time.perf_counter() - t0
        grads = [o.record.get("n_gradients", 0) for o in outs]
        waiting = sum(o.execution.timings.get("subprograms", 0.0) for o in outs)
        record["modes"][mode] = {
            "wall": wall,
            "converged": sum(bool(o.success) for o in outs),
            "gradients": int(sum(grads)),
            "subprograms_s": waiting,
            "energies": [
                float(o.results.final_energy) if o.results.frames else None
                for o in outs
            ],
        }
        print(
            mode,
            f"wall {wall:.1f} s, {sum(grads)} gradients, "
            f"{record['modes'][mode]['converged']}/{len(outs)} converged",
            flush=True,
        )
    if {"list", "sequential"} <= set(record["modes"]):
        e1 = np.array(record["modes"]["list"]["energies"], dtype=float)
        e2 = np.array(record["modes"]["sequential"]["energies"], dtype=float)
        record["max_energy_difference"] = float(np.nanmax(np.abs(e1 - e2)))
        print(
            "largest energy difference between modes:", record["max_energy_difference"]
        )
    json.dump(record, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
