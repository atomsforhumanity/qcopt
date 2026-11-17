# Architecture

qcopt is built around one idea: **algorithms ask, the driver batches, evaluators answer.**

An optimizer is a Python generator. Where it needs an energy and gradient (or a Hessian)
it yields the geometries, and it receives the answers when they are computed. It never
calls a program. The driver advances every running algorithm one round at a time and
hands all their requests to an evaluator in one call. So separate optimizations, both
branches of an IRC, and the 6N displaced gradients of a finite-difference Hessian are
computed together: batched on a GPU, side by side in threads, or as one qccompute list.
The algorithms do not know which.

```
ProgramInput ──► api._prepare ──► algorithms (generators) ◄──► driver.drive ◄──► evaluator
  keywords         options,          yield [Ask, ...]           plans per ask      axtb in-process,
  subprograms      resolved autos,   receive [Answer, ...]      one call/round     sessions, qccompute,
                   TaskContext       return RunResult                              a function
                                          │
                                          ▼
                                    api._output ──► ProgramOutput (frames, resolved, record,
                                                                    error, timings)
```

## Modules

| module | concept |
|---|---|
| `api.py` | `ProgramInput` in, `ProgramOutput` out: checks inputs (and the Hessian they give, `ProgramInput.given`), resolves every `auto` (with reasons), builds frames and results; `minimize`, `find_ts`, `irc` (from a structure or a search's output), `run` |
| `options.py` | the settings: one pydantic model per calculation type, made of setting groups; suggestions for unknown keys; errors for settings that cannot affect the calculation; `default_options`, `options_schema` |
| `errors.py` | `InputError` and the error kinds results report |
| `algorithms/` | the algorithms as generators: `optimization.py` (minima, transition states), `reaction_path.py` (IRC), `common.py` (their result, counting, failure handling, `lockstep` for nested algorithms) |
| `driver.py` | `Ask`, `Answer`, `Request`, `TaskContext`, the `Evaluator` protocol, and `drive`: per ask a plan (direct, a Hessian paired with the gradient program's gradient, finite differences) |
| `evaluators.py` | what computes requests: `FunctionEvaluator` (a Python function), `AxtbEvaluator` (in-process, warm-started), `ComputeEvaluator` (qccompute), routing between them; `default_evaluator` picks per task (axtb in-process, the program's own sessions, else qccompute) |
| `program.py` | qcopt as an AFH program (`qccompute.programs` entry point): `plan`, `compute`, `compute_batch`; `SessionEvaluator`, which computes every task's requests through program sessions (one per task and program, warm-started, a lineage per IRC branch) |
| `coordinates.py` | the coordinate systems a step is taken in, behind one interface: `CartesianCoordinates`, `InternalCoordinates` (TRIC, DLC) |
| `internal.py`, `primitives.py` | delocalized internal coordinates: primitives (stretches, bends, torsions, fragment rotations), B-matrices, back-transformation |
| `space.py` | the Cartesian space an optimizer moves in: rigid motion and frozen atoms projected out and given a large curvature (mass-weighted for IRCs) |
| `steps.py` | RS-RFO and RS-P-RFO steps in a Hessian's eigenbasis, mode selection, the trust-radius update |
| `hessians.py` | Lindh's model Hessian and the quasi-Newton updates (`UPDATES`: BFGS, Bofill, TS-BFGS, SR1, PSB) |
| `constraints.py` | distances, angles and dihedrals held by projection plus a correction step |
| `vibrations.py` | harmonic frequencies of a Cartesian Hessian, to tell minima from saddle points |
| `geometry.py` | masses, rigid-motion bases, alignment, norms |

## How a run flows

1. **Prepare** (`api._prepare`): check the program, calculation type and subprograms;
   validate the settings (`options.parse_options`); resolve each `auto` and record why;
   build the `TaskContext` (symbols, charge, the gradient and Hessian `ProgramSpec`s).
   Every input is checked before anything is computed.
2. **Run** (`driver.drive`): prime every algorithm; each round, plan every ask, call the
   evaluator once, check the answers (finite energies, gradients, Hessians), and send each
   algorithm its replies. Answers carry what they cost (`n_gradients`, `n_hessians`), so
   counts are right even for finite differences and reused gradients.
3. **Algorithm** (`algorithms.optimization`): one class per run whose `run()` generator
   is short: start, iterate until the convergence criteria are met, verify the
   stationary point (a transition state needs exactly one negative curvature; a minimum
   none), escape saddle points if asked, compute final frequencies if asked, finish.
   Each phase is a method that yields what it needs. The IRC (`reaction_path`) is the
   same shape: branches advance together, each a `_Branch` with its own step length,
   Hessian and end test; the ends are minimized by the minimization algorithm, run in
   lockstep.
4. **Output** (`api._output`): frames from the answers; `resolved` (the settings with
   every `auto` replaced, which reproduce the run), `record` (counts, rebuilds, branch
   ends, the step log), `error` (kind, code, the failed child's own error) on failure,
   and `execution.timings`.

## Where to extend it

- **A quasi-Newton update**: a function `(H, s, y) -> H` in `hessians.py`, added to
  `UPDATES` and to the `update` choices in `options.py`.
- **A coordinate system**: a class with the interface of `CartesianCoordinates` (`at`,
  `gradient`, `hessian`, `eigh`, `step`, `size`, `needs_rebuild`, `n_locked`,
  `jacobian_to_space`), chosen in `coordinates.make_coordinates` and named in
  `CoordinateSystem`.
- **An algorithm** (NEB, a string method, a dimer search): a generator that yields lists of
  `Ask` and returns a `RunResult`, wrapped in `algorithms.common.guarded` so an exception
  keeps its frames; an options model in `options.py` and a calculation type in `api.py`.
  It gets batching, every evaluator, the qcdata results and the error handling for free.
- **An evaluator**: anything with `__call__(requests) -> answers` and
  `supports_hessian(context)` (`driver.Evaluator`). For a program, write a qcdata
  program with sessions instead: `SessionEvaluator` drives every such program, and
  qccompute gives programs without sessions a session over `compute`.

## Invariants

- **Atomic units** (bohr, Hartree) at every boundary; qcref constants as `c.NAME`.
- **Algorithms never call programs**; only evaluators do.
- **Fail loudly, keep the frames**: unknown settings are errors; a run that fails or does
  not converge returns `success=False` with its frames and a typed error.
- **Behaviour is measured**: a change meant to keep results is checked bit for bit
  against golden trajectories (77 runs: model surfaces, minima, transition states and
  IRCs at GFN2/g-xTB, constraints, frozen atoms, Hessian programs, guide modes, batches);
  one meant to change them is benchmarked on the 100 Reaction-QM reactions
  (`benchmarks/`) against Sella, geomeTRIC and OptKing.
- **The optimizer's own time matters**: an axtb gradient can take under a millisecond, so
  per-step work stays vectorized NumPy, and results report qcopt's own time separately.
