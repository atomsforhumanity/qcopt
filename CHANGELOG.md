# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [unreleased]

### Fixed

- Each output of a list reports the threads its own calculation ran with
  (`execution.threads`): the most its children recorded, else its share of the
  list's threads. Every output reported the whole list's threads, so eight IRCs run
  together on 64 threads each said 64, and a store's runs added up to eight times the
  threads used.

## [0.3.0a3] - 2026-11-17

### Changed

- Requires the stack's alpha 3: qcdata 0.24.0a4 and qccompute 0.23.0a3.
- The shared keyword `accuracy` (qcdata's `Accuracy`: energy, gradient and Hessian
  targets) is read as every AFH program reads it: a malformed one fails with kind
  `input`, and a target with kind `unsupported`, since qcopt cannot honour targets
  yet (its `convergence` settings set how tightly it converges; the gradient
  program's own keywords how accurate the gradients are). It was an unknown setting.

### Fixed

- A child whose plan takes the whole grant (`threads=None`) gives qcopt's plan the
  whole grant; it made qcopt's plan fail.
- `numerics` names axtb's version (`"+axtb:0.1.1"`) when qcopt runs its axtb
  children in this process, where no runner sees them, so a store never reuses a
  result made with another axtb.

## [0.3.0a2] - 2026-11-17

### Fixed

- Children on GPUs get them: qcopt declares the `gpu` capability, so a runner grants
  the GPUs its plan asks for (its children's) and passes them on. In 0.3.0a1 an ML
  potential child ran on the CPU under a runner.
- A gradient program without sessions that computes batches (an ML potential) gets
  every calculation's requests of a round as one list (`ctx.compute`), which the
  runner hands it as one batch: 64 AIMNet2 minimizations on a GPU took 16 s as one
  list, against 52 s for 8 in 0.3.0a1 (one worker and one structure per call for
  each calculation).
- The in-process axtb route runs calculations side by side within the granted
  threads (one at a time in 0.3.0a1), and an axtb SCF that does not converge is a
  `convergence` failure of the child (was `internal`).

## [0.3.0a1] - 2026-11-17

The first alpha of qcopt 0.3, on the alphas of AFH's redesigned qc* stack (qcdata
0.24.0a2, qccompute 0.23.0a1). Breaking changes come without compatibility shims.

qcopt 0.3 reorganizes the settings and the results; the algorithms compute exactly what
0.2.2 computed (checked bit for bit on 77 golden trajectories: model surfaces, minima,
transition states and IRCs at GFN2-xTB and g-xTB, constraints, frozen atoms, Hessian
programs, guide modes, batched runs).

### Changed

- **qcopt's Python interface is `qcopt.compute(inp)`** (AFH's program interface): a
  qcdata `ProgramInput` in, a `ProgramOutput` out. `run`, `minimize`, `find_ts`, `irc`,
  the evaluators (`FunctionEvaluator`, `AxtbEvaluator`, `ComputeEvaluator`) and the
  `evaluator=` argument are gone, and so are the top-level exports of the settings
  models, `default_options`, `options_schema`, `frequencies`, `drive` and the driver's
  types (all still in their modules). Many inputs at once go through a runner,
  `qccompute.compute([...])`, which runs them in lockstep. From 0.2:
  - `qcopt.minimize(s, spec, hessian=h, **kw)` →
    `qcopt.compute(ProgramInput(program="qcopt", calctype="optimization", structure=s,
    keywords=kw, subprograms=[{**spec, "calctype": "gradient"},
    {**h, "calctype": "hessian"}]))` (`find_ts`: `transition_state`; `irc`: `irc`);
  - an IRC from a search: `structure=last.structure, given=last.results` with
    `last = search.results.frames[-1]` (the search ran with `freq.check`);
  - a Python function as the surface: register it as a program
    (`qccompute.program.register_program`).
- **A failed calculation raises** `qcdata.ProgramError` (was a `success=False` output):
  `error.kind` and `code` as before, and `output`, the failed output with its frames,
  `record` and `resolved` settings. An input qcopt cannot run raises it with kind
  `input` (was `qcopt.InputError`). `compute_batch` returns failures in place.
- **qcopt has no route of its own to programs.** Every gradient and Hessian comes from
  a child session the runner opens (`ctx.session`); called directly, `qcopt.compute`
  hands itself to qccompute's default runner, so any program qccompute runs is a child.
  Until axtb has sessions of its own, qcopt still runs axtb in this process.
- Requires the alphas qcdata 0.24.0a2 and qccompute 0.23.0a1.
- **Settings come in groups**, one options model per calculation type
  (`OptimizationOptions`, `TransitionStateOptions`, `IRCOptions`), with the rules shared
  with axtb and linus. The 0.2 keywords and where they went:
  - `maxiter`, IRC `max_steps` → `convergence.max_iter`; `convergence.gmax`, `grms`,
    `dmax`, `energy` → `convergence.g_tol`, `g_rms_tol`, `d_tol`, `e_tol`; IRC `gmax` →
    `convergence.g_tol`, `flat_energy` → `convergence.e_tol`;
  - `coordinates` → `coordinate_system` (default `auto`: TRIC, Cartesians with frozen
    atoms; explicit `tric` or `dlc` with frozen atoms is an error);
  - `initial_hessian`, `update`, `hessian_every`, `fd_step` → `hessian.initial`,
    `hessian.update`, `hessian.recompute_every`, `hessian.step`;
    `recompute_hessian_on_loss`, `min_steps_between_hessians`, `lost_curvature` →
    `hessian.recompute_on_loss`, `hessian.recompute_min_steps`, `hessian.lost_curvature`;
  - `trust`, `trust_min`, `trust_max`, `trust_type`, `max_rejections` → `trust.radius`,
    `trust.min`, `trust.max`, `trust.measure`, `trust.max_rejections`;
  - `frozen_atoms`, `constraint_tolerance`, `max_correction` → `constraints.frozen`,
    `constraints.tol`, `constraints.max_correction` (with `constraints.distance`, `angle`,
    `dihedral`);
  - `mode`, `follow` → `mode.guide`, `mode.follow`; `frequencies`, `imaginary_cutoff` →
    `freq.check`, `freq.imaginary_cutoff`; `escape_saddles`, `escape_step` →
    `escape.attempts`, `escape.step`;
  - IRC `direction`, `step_size`, `method` → `irc.direction`, `irc.step`, `irc.method`;
    `minimize_ends`, `end_coordinates`, `end_maxiter`, `end_convergence` →
    `ends.minimize`, `ends.coordinate_system`, `ends.max_iter`, `ends.g_tol` (and
    `ends.g_rms_tol`, `ends.d_tol`, `ends.e_tol`).

  A 0.2 keyword is an error that names its new place; another misspelling suggests the
  closest setting.
- **Results use qcdata's standard slots** (qcdata 0.24): `resolved` (`settings` with
  every `auto` replaced, which reproduce the run, and `reasons`), `record` (the counts
  and diagnostics 0.2 put in `extras`, IRC `branches` and `ends`), `error`
  (`kind`: `input`, `program`, `convergence`, `numerical`, `internal`; `code`, such as
  `max_iter` or `not_first_order_saddle`; for a failed gradient or Hessian calculation
  its `program` and its own error as `cause`) and `execution.timings` (`subprograms`,
  `optimizer`, `total`); `warnings` stays empty. An IRC branch that runs out of steps
  ends with `max_iter` (was `max_steps`).
- Settings of another calculation type, and settings given that cannot affect the
  calculation (IRC `ends.*` without `ends.minimize`, `escape.step` without
  `escape.attempts`, `mode.follow` without `mode.guide`, `hessian.step` with a Hessian
  subprogram, ...), are errors; so are saddle escapes with distance, angle or dihedral
  constraints, which 0.2 skipped silently. Settings that act only if a condition arises
  are allowed. `resolved.settings` leaves out settings that cannot take effect.
- Errors in the input are found before anything is computed.
- **qccompute is a dependency** (it holds AFH's program protocol); the `qcopt[qccompute]`
  extra and qcopt's qccodec codec are gone: qccompute runs qcopt as a native program.

### Added

- **qcopt is an AFH program** (`qccompute.programs` entry point `qcopt`,
  `qcopt.program`): qccompute runs it natively, and its children are computed through
  program sessions (`SessionEvaluator`): one per calculation and program,
  warm-started, with each IRC branch (and its end minimization) its own lineage, the
  displaced gradients of finite-difference Hessians one batch, and calculations side by
  side when the child program is thread-safe. A failed child's own error becomes the
  `cause` of qcopt's `program` error, whose message starts with the child and its
  calculation type (`linus (gradient): ...`), AFH's rule for drivers; every `program`
  error has a cause (`internal` for an evaluator's exception, `numerical` for a
  non-finite result), and the child's traceback follows the message in `traceback`.
  A failed separate Hessian program, or a failed displaced gradient of a
  finite-difference Hessian, is named as the program that failed. `plan` checks the
  settings of the children it can find before anything is computed; a child that
  refuses its settings at run time fails only its own calculation, not the others in
  the batch. Calculations run side by side, as many at a time as the granted threads
  allow (qccompute makes concurrent sessions safe); the displaced gradients of a
  finite-difference Hessian start from the centre's calculation and do not become its
  lineage's history (`commit=False`). Input errors carry AFH's codes (`error.code`:
  `unknown_key`, `not_applicable`, `no_effect`).
- **Starting from a known Hessian** (`ProgramInput.given`): the given
  Hessian, from any program, is an optimization's initial one or an IRC's at the
  transition state (`hessian.initial` `given`, what `auto` becomes when there is one;
  `resolved.reasons` names its program). It is a starting guess: energies and gradients
  are computed, and no frame holds it. Other given fields are left alone; given results
  without a Hessian are an error (`not_applicable`). An IRC from a search with
  `freq.check` passes the search's last frame's `results` as `given`: one gradient at
  the transition state instead of a Hessian, and the same path.
- `numerics(spec)`: what identifies qcopt's numbers for a calculation type, for a
  store's reuse rule: an epoch per calculation type, bumped whenever its results may
  change, and a digest of the qcref data qcopt reads (masses, radii, constants).
- `Ask.lineage` and `Request.lineage`; evaluators may define `forget(task)`, called when
  a task finishes.
- The settings' JSON Schema with units and tiers (`settings_schema(calctype)` of the
  program), `docs/settings.md` (generated from the models) and
  `docs/choosing-settings.md` (the benchmark evidence behind the defaults).
- `ARCHITECTURE.md`.

### Internal

- The rigid-motion basis of recent geometries is kept instead of recomputed (an IRC
  step asked for it three times): 19% less of qcopt's own time on an HCN TS search and
  IRC, results unchanged bit for bit.

- `algorithms.py` is a package: the minimization and transition-state search is one class
  whose phases are short generator methods, the IRC branch a class, the thresholds named
  constants with their rationale; `drive` plans each ask with a small class (direct,
  paired, finite difference).

## [0.2.3] - 2026-11-17

### Changed

- Caps its dependencies below their next minor versions, which change the interfaces
  qcopt 0.2 uses: qcdata <0.24, axtb <0.2 (extra `axtb`), qccompute <0.23 and qccodec
  <0.22 (extra `qccompute`). qcopt 0.3 moves to the new versions.

## [0.2.2] - 2026-11-07

### Fixed

- With axtb 0.1.0, the first release on PyPI, every in-process axtb calculation failed
  (`gradient failed: KeyError: 'gradient'`): its `Calculator.singlepoint` computes the
  gradient only when asked, and qcopt now asks.

### Added

- The `axtb` extra (`pip install "qcopt[axtb]"`, axtb 0.1.0 or newer). CI installs it, so the
  axtb tests (in-process runs, internal coordinates, constraints) run on every push.

## [0.2.1] - 2026-11-07

Documentation and benchmark scripts only; the optimizers are unchanged from 0.2.0.

### Changed

- The IRC `step_size` documentation quotes the 0.2.0 benchmark (same ends as Sella in 94
  of 100 Reaction-QM IRCs, median 349.5 gradients) instead of 0.1.0's.
- The IRC `gmax` documentation says that from a very soft transition state a branch can
  meet the end test after its first step, far above the minimum (minimize the ends).
- Benchmarks (`benchmarks/`, not installed with the package): a run that reaches its time limit
  stops in its own process and keeps its counts; a failed final check is recorded as
  `check_error`; an energy-only request from OptKing gets an energy alone (`n_energy`);
  `batch_demo.py` takes the structures' folder as `--data` and records the time spent in
  qccompute (`compute_s`).

## [0.2.0] - 2026-11-07

### Fixed

- A transition-state search that ends where the Hessian has no negative curvature
  (one started at a minimum, for example), or several in an exact Hessian or in the
  frequencies, fails instead of reporting success; an updated Hessian with a strong
  second negative direction is checked with the exact one. So does a minimization whose
  frequencies (or the exact Hessian it ends with) show a saddle point, or that ends
  higher than it started (a program that changed SCF state, for example).
- An IRC whose branch runs out of steps (`max_steps`), does not go downhill from the
  transition state, or stops with its gradient far above `gmax`, or whose end
  minimizations do not converge, fails (`success=False`, frames kept) instead of
  reporting success.
- IRCs: re-evaluating a point for its exact Hessian no longer takes SCF noise for an
  overshoot and overwrites the previous frame, and its result no longer ends the branch
  as flat regardless of the curvature; a failed optional Hessian (end check,
  `hessian_every`) no longer ends the branch; a point already below `gmax` is no longer
  rejected because its gradient points along the step; a first step that crosses a
  valley is shortened too; Gonzalez-Schlegel steps no longer move sideways along a mode that
  carries no gradient; `minimize_ends` no longer repeats an end frame; a branch whose
  steps shrink to nothing ends ("stalled") instead of crawling to `max_steps`.
- An exception inside an optimization or IRC, a crashing evaluator, and missing, NaN or
  infinite energies, gradients or Hessians fail the calculation and keep its frames,
  instead of losing every frame or (for NaN) converging.
- A reused `AxtbEvaluator` computed a later run with an earlier run's calculator (its
  method, charge and molecule). It now also rejects non-axtb specs and specs without a
  method (it used GFN2).
- A Hessian subprogram with the gradient one's program and method but other keywords that
  define the surface (a solvent, an SCF accuracy; not threads or how the Hessian is
  computed) is a separate calculation: its energies and gradients no longer reach the
  frames. Frequencies from a separate Hessian calculation are reported in
  `extras["frequencies"]`, labelled with it, not on the target method's frame; those of
  a cheaper method (another program or method) no longer decide whether a search
  succeeded.
- `ComputeEvaluator`: one unreadable output no longer crashes the batch, and Hessian
  outputs without a gradient (xtb's codec) are paired with the gradient program's, also
  through `RoutingEvaluator`.
- `frequencies=True` with a failed Hessian, and `escape_saddles` that run out at a
  saddle point, fail instead of reporting success.
- Molecules within about 0.2 degrees of linear count as linear (smallest principal
  moment below 1e-6 of the largest): the frequency check took one of their bends for a
  rotation and could pass a linear second-order saddle as a transition state.
- Delocalized internals (`dlc`) no longer crash when a linear molecule bends; internal
  coordinates are rebuilt before a torsion crosses 180 degrees; coordinates that cannot
  be built (coincident atoms) fall back to Cartesians.
- `hessian_every` recomputes exact Hessians (the subprogram's, else finite differences)
  whatever the starting Hessian.
- The Cartesian fallback after repeated back-transformation failures keeps the Hessian
  instead of restarting from the model one.
- Gradient and Hessian counts are the calculations the evaluator ran, failed ones included
  and none when the evaluator itself failed, on every exit; IRC counts include the end
  minimizations.
- The starting trust radius, also after a saddle escape, is kept between `trust_min`
  and `trust_max`.
- Internal coordinates leave out bends and torsions that are singular when built (an
  angle within 5 degrees of 0 or 180), are rebuilt when a bend or a torsion's angle
  collapses toward 0, and a back-transformation that converges to the wrong point (a
  rotation coordinate flipped by pi on a bent, nearly linear fragment) counts as failed.
- RS-P-RFO keeps its step along the climbed mode when the gradient there is tiny (a
  cancellation in the shift); RS-RFO's shift goes down to 1e-100.
- The SR1 part of Bofill's update and TS-BFGS use scale-free skip tests, so steps down to
  about 1e-7 bohr still meet the secant condition.
- IRC steps by the local quadratic approximation (`method="lqa"`) have the requested arc
  length when curvatures differ widely, and stop at the end of the path without
  searching further.

### Changed

- Keyword combinations that did nothing are errors: a guide `mode` needs
  `coordinates="cartesian"`; an IRC's `initial_hessian` cannot be `model` or `identity`;
  `initial_hessian="finite_difference"` cannot be combined with a Hessian subprogram;
  `trust_min` above `trust_max`. Constraints that name a missing atom, are given twice,
  involve only frozen atoms, or hold an angle above 175 degrees raise `ValueError` before
  anything is computed.
- `escape_saddles` is skipped in constrained optimizations.
- Documentation: in internal coordinates `trust` bounds the norm of the internal step;
  benchmark numbers in the keyword docs updated.
- The release workflow runs the tests before publishing; CI also tests Python 3.14.

## [0.1.0] - 2026-11-06

### Added

- First version: minimizations (RS-RFO), transition-state searches (RS-P-RFO with
  TS-BFGS updates) and IRCs (both branches together), as
  `ProgramInput(program="qcopt")` -> `ProgramOutput` with `OptimizationData` or
  `PathData`.
- Lockstep runs of many calculations with one evaluator call per round (`qcopt.run(list)`).
- Evaluators: axtb in-process, qccompute (any program with a codec), Python functions;
  finite-difference Hessians from gradients computed together.
- Lindh model Hessian; BFGS, SR1, PSB, Bofill and TS-BFGS updates; harmonic frequency
  check of the final structure.
- Internal coordinates, the default: TRIC (per-fragment translation and rotation, with
  analytic rotation derivatives) and delocalized internals; rebuilt when they degrade,
  Cartesians when they cannot describe the structure.
- Constraints on distances, angles and dihedrals; saddle escape for minimizations
  (`escape_saddles`).
- A qccodec codec, so `qccompute.compute(ProgramInput(program="qcopt", ...))` works.
- IRCs by Gonzalez-Schlegel steps on the quadratic model (default, step 0.3) or the
  local quadratic approximation; overshoots retried at half the step; an exact Hessian
  decides when a small-gradient branch shows negative curvature; `minimize_ends`.
- Internal-coordinate values and B matrices are cached per geometry, so a TRIC step
  computes each once (8.0 -> 4.6 ms per step on a 16-atom minimization).
- Hessians from another method than the gradients (`hessian=` a different program or
  model, such as GFN2-xTB Hessians for g-xTB): every Hessian request also gets the
  gradient program's energy and gradient (reused when just computed), and frames keep
  only the gradient program's results. On 100 Reaction-QM transition states at g-xTB,
  GFN2 starting Hessians halved the total time.
- The starting-Hessian keyword is `initial_hessian`; `hessian=` in `minimize`,
  `find_ts` and `irc` names the Hessian program only, and other values are errors
  (a string was ignored).

[unreleased]: https://github.com/atomsforhumanity/qcopt/compare/0.3.0a3...HEAD
[0.3.0a3]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.3.0a3
[0.3.0a2]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.3.0a2
[0.3.0a1]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.3.0a1
[0.2.3]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.2.3
[0.2.2]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.2.2
[0.2.1]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.2.1
[0.2.0]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.2.0
[0.1.0]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.1.0
