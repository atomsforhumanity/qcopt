"""Run ts_bench.py over a set of structures and optimizers, one process per job.

usage: run_set.py OUTDIR --xyz-glob 'DIR/*/ts_reference.xyz' --optimizers qcopt sella
       [--jobs 16] [--timeout 900] [--task ts] [--extra-qcopt '{}'] [--tag NAME]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("outdir")
    p.add_argument("--xyz-glob", required=True)
    p.add_argument("--optimizers", nargs="+", default=["qcopt", "sella", "geometric"])
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--task", default="ts")
    p.add_argument("--method", default="gxtb-v2.0.1")
    p.add_argument("--accuracy", default="0.01", help="axtb SCF accuracy")
    p.add_argument("--extra", action="append", default=[], help="OPT=JSON")
    p.add_argument(
        "--tag", action="append", default=[], help="OPT=TAG: output name suffix"
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--pairs", help="directory of <name>/start.xyz and end.xyz (IRC checks)"
    )
    a = p.parse_args()
    extras = dict(e.split("=", 1) for e in a.extra)
    tags = dict(t.split("=", 1) for t in a.tag)
    files = sorted(glob.glob(a.xyz_glob))
    if a.limit:
        files = files[: a.limit]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    stems = {Path(f).stem for f in files}
    for f in files:
        name = Path(f).parent.name + (f"_{Path(f).stem}" if len(stems) > 1 else "")
        for opt in a.optimizers:
            tag = opt + (f"-{tags[opt]}" if opt in tags else "")
            dest = out / f"{name}.{tag}.json"
            if dest.exists():
                continue
            jobs.append((f, opt, dest, extras.get(opt, "{}")))

    def run(job: tuple[str, str, Path, str]) -> None:
        f, opt, dest, extra = job
        cmd = [
            sys.executable,
            str(HERE / "ts_bench.py"),
            opt,
            f,
            str(dest),
            "--task",
            a.task,
            "--method",
            a.method,
            "--accuracy",
            a.accuracy,
            "--extra",
            extra,
        ]
        if a.pairs:
            name = Path(f).parent.name
            cmd += [
                "--reactant",
                f"{a.pairs}/{name}/start.xyz",
                "--product",
                f"{a.pairs}/{name}/end.xyz",
            ]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        try:
            subprocess.run(
                cmd,
                timeout=a.timeout,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            dest.write_text(
                json.dumps(
                    {
                        "optimizer": opt,
                        "xyz": f,
                        "converged": False,
                        "error": "timeout",
                        "wall": a.timeout,
                    }
                )
            )
        if not dest.exists():
            dest.write_text(
                json.dumps(
                    {"optimizer": opt, "xyz": f, "converged": False, "error": "crashed"}
                )
            )

    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        for k, _ in enumerate(pool.map(run, jobs), 1):
            if k % 20 == 0:
                print(f"{k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
