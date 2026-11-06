"""Summarize IRC results: cost of the IRC itself and of minimizing its ends."""

import collections
import glob
import json
import sys

import numpy as np

d = collections.defaultdict(dict)
for f in glob.glob(sys.argv[1] + "/*.json"):
    n, t = f.split("/")[-1][:-5].rsplit(".", 1)
    d[t][n] = json.load(open(f))
ref = d.get("sella", {})
print(
    f"{'optimizer':<14}{'done':>5}{'conn':>5}{'same ends':>10}{'IRC grads med':>14}{'+ends med':>10}{'total med':>10}{'total sum':>10}{'hess':>6}{'wall sum':>9}{'opt s':>7}{'opt ms/g':>9}"
)
out = {}
for t, v in sorted(d.items()):
    rows = [r for r in v.values() if r.get("converged") and "end_min_grads" in r]
    if not rows:
        print(
            t,
            "no complete rows",
            collections.Counter((r.get("error") or "")[:50] for r in v.values()),
        )
        continue
    g = np.array([r["n_grad"] for r in rows])
    ge = np.array([r["end_min_grads"] for r in rows])
    he = np.array([r["n_hess"] + r.get("end_min_hess", 0) for r in rows])
    same = sum(
        1
        for n, r in v.items()
        if r.get("end_energy")
        and ref.get(n, {}).get("end_energy")
        and np.allclose(
            sorted(r["end_energy"]), sorted(ref[n]["end_energy"]), atol=2e-4
        )
    )
    conn = sum(bool(r.get("connects")) for r in rows)
    wall = sum(r["wall"] for r in rows)
    opt = sum(r["t_optimizer"] for r in rows)
    ms = np.median([1000 * r["t_optimizer"] / r["n_grad"] for r in rows])
    print(
        f"{t:<14}{len(rows):>5}{conn:>5}{same:>10}{np.median(g):>14.0f}{np.median(ge):>10.0f}{np.median(g + ge):>10.0f}{(g + ge).sum():>10}{he.mean():>6.1f}{wall:>9.0f}{opt:>7.0f}{ms:>9.1f}"
    )
    out[t] = dict(
        n=len(rows),
        connects=conn,
        same_ends_as_sella=same,
        irc_grad_median=float(np.median(g)),
        end_grad_median=float(np.median(ge)),
        total_grad_median=float(np.median(g + ge)),
        total_grad_sum=int((g + ge).sum()),
        hess_mean=float(he.mean()),
        wall_sum=wall,
        opt_sum=opt,
        opt_ms_per_grad=float(ms),
    )
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=2)
