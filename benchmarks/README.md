# Benchmarks

Scripts that compare qcopt with Sella and geomeTRIC on the same gradients (axtb,
in-process, one thread per run). They need their own environment:

```sh
cd benchmarks
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .. <axtb wheel> sella==2.6.0 ase==3.29.0 geometric==1.1.1
```

- `ts_bench.py OPTIMIZER XYZ OUT.json`: one optimization, counted and checked with a fresh
  analytic Hessian (frequencies, energy, final gradient).
- `run_set.py OUTDIR --xyz-glob ...`: many, in parallel processes.

Data (Reaction-QM structures, Jan's F0001 tables) and results live outside the repo.
