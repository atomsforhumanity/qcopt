# AGENTS.md

Instructions for AI coding agents (and a quick reference for people) working in `qcopt`.

## This package

`qcopt` optimizes structures: minima, transition states, and IRCs. It is AFH's own optimizer, meant to replace Sella, geomeTRIC, and ASE's optimizers in our stack. Inputs are qcdata `ProgramInput`s with `program="qcopt"` and gradient (and optionally Hessian) subprograms; outputs are qcdata `ProgramOutput`s holding `OptimizationData` or `PathData`.

- **Algorithms are generators** (`qcopt/algorithms.py`): they yield the geometries where they need gradients or Hessians and receive the answers. `qcopt.driver.drive` runs many of them in lockstep and hands each round's requests to an evaluator in one call. Never call a program from inside an algorithm.
- **Evaluators** (`qcopt/evaluators.py`) compute requests: axtb in-process (our own codes come first: no subprocess, wavefunction reuse, threads), qccompute for every other program, or a Python function.
- **Atomic units everywhere** (bohr, Hartree); qcref constants as `from qcref import constants as c` and `c.NAME`, never copied into module variables.
- **Fail loudly**: unknown keywords are errors (pydantic models in `qcopt/keywords.py`). A run that fails or does not converge returns `success=False` with its frames kept.
- **Our own implementation.** Sella (LGPL-3.0), geomeTRIC (BSD-3 with a non-AI clause), OptKing (BSD-3) and pysisyphus may be read as guides; implement from the papers and cite them in docstrings. Never copy their code.

## The qc* suite

Dependencies point one way: qcref -> qcdata -> {qcinf, qccodec} -> qccompute -> qcopt. qccompute is optional (only the `ComputeEvaluator` needs it).

## Commands

```sh
uv sync --all-groups                  # install
uv pip install <axtb wheel>           # axtb is not on PyPI yet; its tests skip without it
uv run pytest                         # tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy
```

Benchmarks against Sella and geomeTRIC live in `benchmarks/` with their own venv (see `benchmarks/README.md`). Benchmark data and results stay outside the repo.

## Code conventions

- Type annotations on all functions; Google-style docstrings with units.
- Performance matters: with axtb a gradient can take under a millisecond, so the optimizer's own time per step is part of the result. Profile before optimizing; keep per-step work vectorized NumPy.
- New behavior has tests; bug fixes have a test that failed before the fix.
- User-visible changes go under `## [unreleased]` in CHANGELOG.md.
- Do not commit, push, tag, or release unless asked to.
