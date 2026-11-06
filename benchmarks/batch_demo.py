"""Many minimizations with an ML potential: one at a time vs. in lockstep.

Every optimization runs through qccompute (``qcopt.ComputeEvaluator``). One at a time,
each gradient is its own call; in lockstep (``qcopt.run(list)``) each round's
gradients from all optimizations go to qccompute in one call, which the AIMNet2 codec
evaluates in batches.

usage: batch_demo.py OUT.json --n 64 [--model aimnet2] [--modes lockstep sequential]
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

DATA = "/home/cbh/dev/afh/research/qcopt-bench/data/reaction_qm/benchmark"


def structures(n: int):
    from qcdata import Structure

    files = sorted(glob.glob(f"{DATA}/*/start.xyz")) + sorted(
        glob.glob(f"{DATA}/*/end.xyz")
    )
    out = []
    for f in files[:n]:
        lines = open(f).read().splitlines()
        k = int(lines[0])
        sym = [line.split()[0] for line in lines[2 : 2 + k]]
        xyz = np.array(
            [[float(v) for v in line.split()[1:4]] for line in lines[2 : 2 + k]]
        )
        out.append(Structure(symbols=sym, geometry=xyz * 1.8897261259077824))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--model", default="aimnet2")
    p.add_argument("--modes", nargs="+", default=["lockstep", "sequential"])
    p.add_argument("--maxiter", type=int, default=500)
    a = p.parse_args()

    import qccompute
    from qcdata import CalcType, ProgramInput, ProgramSpec

    import qcopt

    calls = []

    def compute(inputs):
        calls.append(len(inputs))
        return qccompute.compute(inputs, raise_exc=False)

    grad = ProgramSpec(
        program="aimnet2", calctype=CalcType.gradient, model={"method": a.model}
    )
    inputs = [
        ProgramInput(
            program="qcopt",
            calctype="optimization",
            structure=s,
            subprograms=[grad],
            keywords={"maxiter": a.maxiter},
        )
        for s in structures(a.n)
    ]
    # load the model once before timing
    qccompute.compute(ProgramInput.from_spec(grad, inputs[0].structure))
    record = {"n": len(inputs), "model": a.model, "modes": {}}
    for mode in a.modes:
        calls.clear()
        ev = qcopt.ComputeEvaluator(compute)
        t0 = time.perf_counter()
        if mode == "lockstep":
            outs = qcopt.run(inputs, evaluator=ev)
        else:
            outs = [qcopt.run(inp, evaluator=ev) for inp in inputs]
        wall = time.perf_counter() - t0
        grads = [o.results.extras.get("n_gradients", 0) for o in outs]
        energies = [float(o.results.final_energy) for o in outs]
        record["modes"][mode] = {
            "wall": wall,
            "converged": sum(bool(o.success) for o in outs),
            "gradients": int(sum(grads)),
            "compute_calls": len(calls),
            "largest_call": max(calls) if calls else 0,
            "energies": energies,
        }
        print(
            mode,
            f"wall {wall:.1f} s, {sum(grads)} gradients in {len(calls)} calls, "
            f"{record['modes'][mode]['converged']}/{len(outs)} converged",
            flush=True,
        )
    if {"lockstep", "sequential"} <= set(record["modes"]):
        e1 = np.array(record["modes"]["lockstep"]["energies"])
        e2 = np.array(record["modes"]["sequential"]["energies"])
        record["max_energy_difference"] = float(np.nanmax(np.abs(e1 - e2)))
        print(
            "largest energy difference between modes:", record["max_energy_difference"]
        )
    json.dump(record, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
