# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [unreleased]

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
