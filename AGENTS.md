# AGENTS.md

Instructions for AI coding agents (and a quick reference for people) working in `qcopt`.

## This package

`qcopt` optimizes structures: minima, transition states, and IRCs. It is AFH's own optimizer, meant to replace Sella, geomeTRIC, and ASE's optimizers in our stack. Its Python interface is `qcopt.compute(inp)` and nothing else: inputs are qcdata `ProgramInput`s with `program="qcopt"` and gradient (and optionally Hessian) subprograms; outputs are qcdata `ProgramOutput`s holding `OptimizationData` or `PathData`; failures raise `qcdata.ProgramError` carrying the failed output.

- **Read [ARCHITECTURE.md](ARCHITECTURE.md) first**: the modules, how a run flows, where to extend, the invariants.
- **Algorithms are generators** (`qcopt/algorithms/`): they yield the geometries where they need gradients or Hessians and receive the answers. `qcopt.driver.drive` runs many of them in lockstep and hands each round's requests to an evaluator in one call. Never call a program from inside an algorithm.
- **Settings** (`qcopt/options.py`): one pydantic model per calculation type, made of groups (`convergence.g_tol`, `hessian.initial`, ...), following the rules shared with axtb and linus (two levels at most, `_tol`, `max_iter`, `"auto"` resolved and reported). A new setting goes in a group with its unit and tier; `scripts/settings_doc.py` regenerates `docs/settings.md` (a test checks it).
- **Evaluators** (`qcopt/evaluators.py`) compute requests through child sessions from the runner's `ctx` (warm-started; qccompute gives programs without sessions one over `compute`). qcopt has no route of its own to programs. Whether a Hessian program computes the gradient program's own surface is read from the children's settings schemas (`x-concern`), never from a table of a program's keywords. Tests that need a hand-made evaluator use `tests/helpers.py` (`FunctionEvaluator`, and `FunctionProgram` to register a Python function as a program).
- **Atomic units everywhere** (bohr, Hartree); qcref constants as `from qcref import constants as c` and `c.NAME`, never copied into module variables.
- **Fail loudly**: unknown settings are errors with a suggestion; so are settings of another calculation type and settings given that cannot affect the calculation (`options.without_effect`). Settings that act only if a condition arises are allowed. A run that fails or does not converge raises `ProgramError` whose `output` keeps its frames and a typed `error` (kind and code; for a failed gradient or Hessian, the program and its own error as `cause`); `compute_batch` returns failures in place.
- **Keep behaviour unless you mean to change it**: a refactor must reproduce the golden trajectories bit for bit (AFH's golden-trajectory harness: record before, record after, compare); a change to results needs the benchmark (`benchmarks/`) and a bump of `NUMERICS_EPOCHS` (`qcopt/program.py`) for every calculation type whose golden trajectories changed: stores reuse a result only while qcopt's `numerics` is unchanged.
- **Our own implementation.** Sella (LGPL-3.0), geomeTRIC (BSD-3 with a non-AI clause), OptKing (BSD-3) and pysisyphus may be read as guides; implement from the papers and cite them in docstrings. Never copy their code.

## The qc* suite

Dependencies point one way: qcref -> qcdata -> {qcinf, qccodec} -> qccompute -> qcopt. qcopt implements qccompute's program protocol (`qccompute.program`) and is found through the `qccompute.programs` entry point.

## Commands

```sh
uv sync --all-groups --all-extras     # install, with axtb and qccompute (their tests skip without them)
uv run pytest                         # tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run python scripts/settings_doc.py   # after changing a setting
```

Benchmarks against Sella and geomeTRIC live in `benchmarks/` with their own venv (see `benchmarks/README.md`). Benchmark data and results stay outside the repo.

## Code conventions

- Type annotations on all functions; Google-style docstrings with units.
- Performance matters: with axtb a gradient can take under a millisecond, so the optimizer's own time per step is part of the result. Profile before optimizing; keep per-step work vectorized NumPy.
- New behavior has tests; bug fixes have a test that failed before the fix.
- User-visible changes go under `## [unreleased]` in CHANGELOG.md.
- Do not commit, push, tag, or release unless asked to.
