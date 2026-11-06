"""Summarize a results directory of ts_bench.py JSON files.

usage: summarize.py RESULTS_DIR [--reference sella] [--json OUT.json]

Per optimizer: converged, first-order saddle (exactly one imaginary mode below -10
cm^-1) or minimum (none), gradients, Hessians, wall and optimizer time. Per pair with
the reference optimizer: the same stationary point (aligned RMSD < 0.01 angstrom and
|dE| < 0.01 kcal/mol).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOHR = 0.529177210544
KCAL = 627.5094740631


def rmsd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt((((a @ r) - b) ** 2).sum(1).mean()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--reference", default="sella")
    p.add_argument("--json")
    a = p.parse_args()
    runs: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in Path(a.results).glob("*.json"):
        name, tag = f.stem.rsplit(".", 1)
        runs[tag][name] = json.loads(f.read_text())
    names = sorted(set().union(*[set(v) for v in runs.values()]))
    task = next(iter(next(iter(runs.values())).values())).get("task", "ts")
    want_imag = 1 if task == "ts" else 0
    summary = {}
    print(f"{len(names)} structures, task {task}\n")
    hdr = f"{'optimizer':<22}{'conv':>6}{'good':>6}{'grad med':>10}{'grad mean':>10}{'hess':>6}{'wall med':>10}{'wall sum':>10}{'opt ms/step':>12}{'calc s':>9}{'opt s':>8}"
    print(hdr)
    for tag in sorted(runs):
        r = runs[tag]
        rows = [r[n] for n in names if n in r]
        conv = [x for x in rows if x.get("converged")]
        good = [x for x in conv if x.get("n_imaginary") == want_imag]
        grads = np.array([x["n_grad"] for x in conv]) if conv else np.array([0])
        hess = np.array([x["n_hess"] for x in conv]) if conv else np.array([0])
        wall = np.array([x["wall"] for x in rows if "wall" in x])
        opt_ms = np.array([1000 * x["t_optimizer"] / max(x["n_grad"], 1) for x in conv])
        calc = sum(x.get("t_grad", 0) + x.get("t_hess", 0) for x in rows)
        opt = sum(x.get("t_optimizer", 0) for x in rows)
        summary[tag] = dict(
            n=len(rows),
            converged=len(conv),
            good=len(good),
            grad_median=float(np.median(grads)),
            grad_mean=float(grads.mean()),
            hess_mean=float(hess.mean()),
            wall_median=float(np.median(wall)),
            wall_sum=float(wall.sum()),
            opt_ms_per_gradient_median=float(np.median(opt_ms))
            if len(opt_ms)
            else None,
            calc_seconds=calc,
            optimizer_seconds=opt,
        )
        print(
            f"{tag:<22}{len(conv):>6}{len(good):>6}{np.median(grads):>10.0f}{grads.mean():>10.1f}{hess.mean():>6.2f}"
            f"{np.median(wall):>10.2f}{wall.sum():>10.0f}{np.median(opt_ms) if len(opt_ms) else float('nan'):>12.1f}{calc:>9.0f}{opt:>8.0f}"
        )
    ref = runs.get(a.reference, {})
    print(f"\nagreement with {a.reference} (both good):")
    for tag in sorted(runs):
        if tag == a.reference:
            continue
        same = lower = higher = differ = both = 0
        for n in names:
            x, y = runs[tag].get(n), ref.get(n)
            ok = lambda z: (
                z and z.get("converged") and z.get("n_imaginary") == want_imag
            )  # noqa: E731
            if not (ok(x) and ok(y)):
                continue
            both += 1
            de = (x["energy"] - y["energy"]) * KCAL
            d = rmsd(x["geometry"], y["geometry"]) * BOHR
            if d < 0.01 and abs(de) < 0.01:
                same += 1
            else:
                differ += 1
                if de < -0.01:
                    lower += 1
                elif de > 0.01:
                    higher += 1
        print(
            f"  {tag:<20} both good {both}, same {same}, different {differ} (lower E {lower}, higher E {higher})"
        )
        summary[tag]["vs_reference"] = dict(
            both=both, same=same, different=differ, lower=lower, higher=higher
        )
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
