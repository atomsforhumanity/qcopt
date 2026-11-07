# Benchmarks

Scripts that compare qcopt with Sella, geomeTRIC and OptKing on the same gradients (axtb,
in-process, one thread per run). They need their own environment:

```sh
cd benchmarks
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .. <axtb wheel> sella==2.6.0 ase==3.29.0 geometric==1.1.1 networkx
VIRTUAL_ENV=.venv uv pip install "optking @ git+https://github.com/psi-rking/optking.git@0.5.0"
```

axtb runs at SCF accuracy 0.01 (`--accuracy`, passed explicitly): at axtb's default (1.0) g-xTB
gradients carry ~1e-5 Eh/bohr of noise, the size of the 0.001 eV/A convergence threshold. Rounds
before 2026-11-06 evening ran at the default.

Pin BLAS to one thread before numpy loads (the scripts do), or small eigendecompositions run
~100x slower next to axtb's threads. Freeze a copy of these scripts and of the code under test
for each round: every job imports them when it starts.

- `ts_bench.py OPTIMIZER XYZ OUT.json`: one optimization, counted and checked with a fresh
  analytic Hessian (frequencies, energy, final gradient).
- `run_set.py OUTDIR --xyz-glob ...`: many, in parallel processes (`--task ts|min|irc`,
  `--extra OPT=JSON`, `--tag OPT=NAME`, `--pairs` for IRC connectivity checks). A run that
  reaches `--timeout` stops inside its own process and is recorded as failed with the gradients
  and Hessians it used. If the final check (a fresh gradient and Hessian at the result) fails,
  the run keeps its counts and records `check_error`.
- `summarize.py RESULTS`: minima and transition states; `irc_summary.py RESULTS`: IRCs, with the
  cost of minimizing their ends.
- OptKing runs through its own driver (`optking.optimize.optimize`) with a computer that calls
  axtb: its step-by-step helper breaks when it rebuilds its coordinates. A request from it for
  an energy alone gets an energy alone, counted as `n_energy`. OptKing 0.5.0 can restart its
  step count indefinitely after failed back-transformations
  ([psi-rking/optking#117](https://github.com/psi-rking/optking/issues/117)), so give it a
  `--timeout`.
- `batch_demo.py OUT.json --data DIR --n 64`: many minimizations with an ML potential through
  qccompute, one at a time and in lockstep (`qcopt.run(list)`), with the time inside
  qccompute (`compute_s`) apart from qcopt's own.

Data (Reaction-QM structures, Jan's F0001 tables) and results live outside the repo.
