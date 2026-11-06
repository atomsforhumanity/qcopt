# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [unreleased]

## [0.1.0] - 2026-11-06

### Added

- First version: minimizations (RS-RFO), transition-state searches (RS-P-RFO with
  TS-BFGS updates) and IRCs (local quadratic approximation, both branches together) in
  Cartesian coordinates, as `ProgramInput(program="qcopt")` -> `ProgramOutput` with
  `OptimizationData` or `PathData`.
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
