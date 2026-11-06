# Benchmarks

Scripts that compare qcopt with Sella and geomeTRIC on the same gradients (axtb,
in-process, one thread per run). They need their own environment:

```sh
cd benchmarks
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .. <axtb wheel> sella==2.6.0 ase==3.29.0 geometric==1.1.1 networkx
VIRTUAL_ENV=.venv uv pip install "optking @ git+https://github.com/psi-rking/optking.git@0.5.0"
```

Pin BLAS to one thread before numpy loads (the scripts do), or small eigendecompositions run
~100x slower next to axtb's threads. Freeze a copy of these scripts and of the code under test
for each round: every job imports them when it starts.

- `ts_bench.py OPTIMIZER XYZ OUT.json`: one optimization, counted and checked with a fresh
  analytic Hessian (frequencies, energy, final gradient).
- `run_set.py OUTDIR --xyz-glob ...`: many, in parallel processes (`--task ts|min|irc`,
  `--extra OPT=JSON`, `--tag OPT=NAME`, `--pairs` for IRC connectivity checks).
- `summarize.py RESULTS`: minima and transition states; `irc_summary.py RESULTS`: IRCs, with the
  cost of minimizing their ends.
- OptKing runs through its own driver (`optking.optimize.optimize`) with a computer that calls
  axtb: its step-by-step helper breaks when it rebuilds its coordinates.

Data (Reaction-QM structures, Jan's F0001 tables) and results live outside the repo.
