# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [unreleased]

### Fixed

- A transition-state search that ends where the Hessian has no negative curvature
  (one started at a minimum, for example), or several in an exact Hessian or in the
  frequencies, fails instead of reporting success.
- An IRC whose branch runs out of steps (`max_steps`), or whose end minimizations do not
  converge, fails (`success=False`, frames kept) instead of reporting success.
- IRCs: re-evaluating a point for its exact Hessian no longer takes SCF noise for an
  overshoot and overwrites the previous frame; a failed optional Hessian (end check,
  `hessian_every`) no longer ends the branch; a point already below `gmax` is no longer
  rejected because its gradient points along the step; Gonzalez-Schlegel steps no
  longer move sideways along a mode that carries no gradient.
- An exception inside an optimization or IRC, a crashing evaluator, and NaN or infinite
  energies, gradients or Hessians fail the calculation and keep its frames, instead of
  losing every frame or (for NaN) converging.
- A reused `AxtbEvaluator` computed a later run with an earlier run's calculator (its
  method, charge and molecule). It now also rejects non-axtb specs and specs without a
  method (it used GFN2).
- A Hessian subprogram with the gradient one's program and method but other keywords (a
  solvent, an SCF accuracy) is a separate calculation: its energies and gradients no
  longer reach the frames. Frequencies from a separate Hessian program are reported in
  `extras["frequencies"]`, labelled with it, not on the target method's frame.
- `ComputeEvaluator`: one unreadable output no longer crashes the batch, and Hessian
  outputs without a gradient (xtb's codec) are paired with the gradient program's.
- `frequencies=True` with a failed Hessian, and `escape_saddles` that run out at a
  saddle point, fail instead of reporting success.
- Nearly linear molecules count as linear (smallest principal moment below 1e-3 of the
  largest, as axtb decides): the frequency check took one of their bends for a rotation
  and could pass a linear second-order saddle as a transition state.
- Delocalized internals (`dlc`) no longer crash when a linear molecule bends, and
  internal coordinates are rebuilt before a torsion crosses 180 degrees.
- `hessian_every` recomputes exact Hessians (the subprogram's, else finite differences)
  whatever the starting Hessian.
- The Cartesian fallback after repeated back-transformation failures keeps the Hessian
  instead of restarting from the model one.
- Gradient counts include failed finite-difference Hessians; IRC Hessian counts include
  those of the end minimizations.

### Changed

- Keyword combinations that did nothing are errors: a guide `mode` needs
  `coordinates="cartesian"`; an IRC's `initial_hessian` cannot be `model` or `identity`;
  `initial_hessian="finite_difference"` cannot be combined with a Hessian subprogram.
  Constraints given twice, involving only frozen atoms, or holding an angle above 175
  degrees are rejected.
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

[unreleased]: https://github.com/atomsforhumanity/qcopt/compare/0.1.0...HEAD
[0.1.0]: https://github.com/atomsforhumanity/qcopt/releases/tag/0.1.0
