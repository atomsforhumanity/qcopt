# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [unreleased]

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

[unreleased]: https://github.com/atomsforhumanity/qcopt/compare/0.2.2...HEAD
[0.2.2]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.2.2
[0.2.1]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.2.1
[0.2.0]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.2.0
[0.1.0]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.1.0
