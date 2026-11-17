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
qcopt.compute(ProgramInput, ctx)
  │
  ▼
calculation.prepare ──► algorithms (generators) ◄──► driver.drive ◄──► evaluator
  options,                yield [Ask, ...]            plans per ask      child sessions
  resolved autos,         receive [Answer, ...]       one call/round     from ctx (axtb in
  TaskContext             return RunResult                               process until it
                               │                                         has sessions)
                               ▼
                      calculation._output ──► ProgramOutput (frames, resolved, record,
                                                              error, timings); raised
                                                              as ProgramError on failure
```

qcopt is a program with subprograms (`qccompute.program`). Its one Python interface is
`compute(inp, ctx=None)`: a runner passes `ctx` (the resource grant and the way to open
child sessions); called directly, qcopt hands itself to qccompute's default runner,
which passes one. qcopt has no route of its own to programs.

## Modules

| module | concept |
|---|---|
| `program.py` | qcopt as an AFH program (`qccompute.programs` entry point `qcopt`): `compute` (the Python interface; raises `ProgramError` on failure), `plan` (checks the input and its children's settings, resolves every `auto`), `compute_batch` (inputs in lockstep, failures in place) |
| `calculation.py` | one calculation: `prepare` checks an input (and the Hessian it gives, `ProgramInput.given`) and resolves every `auto` (with reasons); `run` runs prepared calculations in lockstep on an evaluator; `_output` builds frames and results |
| `options.py` | the settings: one pydantic model per calculation type, made of setting groups; suggestions for unknown keys; errors for settings that cannot affect the calculation; `default_options`, `options_schema` (the program's `settings_schema`) |
| `errors.py` | `InputError` (inside qcopt; `compute` raises it as `ProgramError` of kind `input`) and the error kinds results report |
| `algorithms/` | the algorithms as generators: `optimization.py` (minima, transition states), `reaction_path.py` (IRC), `common.py` (their result, counting, failure handling, `lockstep` for nested algorithms) |
| `driver.py` | `Ask`, `Answer`, `Request`, `TaskContext`, the `Evaluator` protocol, and `drive`: per ask a plan (direct, a Hessian paired with the gradient program's gradient, finite differences) |
| `evaluators.py` | what computes requests: `SessionEvaluator`, through child sessions opened from `ctx` (one per task and program, warm-started, a lineage per IRC branch, displaced gradients not kept), `ComputeEvaluator` (each round as one list through `ctx.compute`, for batch programs without sessions), with `RoutingEvaluator` between them; `surface_keywords`, the keys that define a child's surface by its schema's `x-concern` |
| `coordinates.py` | the coordinate systems a step is taken in, behind one interface: `CartesianCoordinates`, `InternalCoordinates` (TRIC, DLC) |
| `internal.py`, `primitives.py` | delocalized internal coordinates: primitives (stretches, bends, torsions, fragment rotations), B-matrices, back-transformation |
| `space.py` | the Cartesian space an optimizer moves in: rigid motion and frozen atoms projected out and given a large curvature (mass-weighted for IRCs) |
| `steps.py` | RS-RFO and RS-P-RFO steps in a Hessian's eigenbasis, mode selection, the trust-radius update |
| `hessians.py` | Lindh's model Hessian and the quasi-Newton updates (`UPDATES`: BFGS, Bofill, TS-BFGS, SR1, PSB) |
| `constraints.py` | distances, angles and dihedrals held by projection plus a correction step |
| `vibrations.py` | harmonic frequencies of a Cartesian Hessian, to tell minima from saddle points |
| `geometry.py` | masses, rigid-motion bases, alignment, norms |

## How a run flows

1. **Prepare** (`calculation.prepare`): check the program, calculation type and subprograms;
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
   lockstep. When the caller listens (`ctx.progress`), each algorithm reports every
   frame it keeps as a step event through its `report` callable (the shape is in
   `calculation`'s docstring); nothing is built otherwise, and a step event never
   changes the algorithm's state.
4. **Output** (`calculation._output`): frames from the answers; `resolved` (the settings with
   every `auto` replaced, which reproduce the run), `record` (counts, rebuilds, branch
   ends, the step log, the frames where Hessians were computed), `error` (kind, code,
   the failed child's own error) on failure, and `execution` (the list's timings, this
   calculation's own wall time and threads). `compute` raises a failed output as `ProgramError`, which
   carries it as `output`; `compute_batch` returns failures in place.

## Where to extend it

- **A quasi-Newton update**: a function `(H, s, y) -> H` in `hessians.py`, added to
  `UPDATES` and to the `update` choices in `options.py`.
- **A coordinate system**: a class with the interface of `CartesianCoordinates` (`at`,
  `gradient`, `hessian`, `eigh`, `step`, `size`, `needs_rebuild`, `n_locked`,
  `jacobian_to_space`), chosen in `coordinates.make_coordinates` and named in
  `CoordinateSystem`.
- **An algorithm** (NEB, a string method, a dimer search): a generator that yields lists of
  `Ask` and returns a `RunResult`, wrapped in `algorithms.common.guarded` so an exception
  keeps its frames; an options model in `options.py` and a calculation type in
  `calculation.py`. It gets batching, child sessions, the qcdata results and the error
  handling for free.
- **A program to optimize on**: not here. Write an AFH program (`qccompute.program`),
  with sessions if it carries state from step to step; qcopt reaches it through `ctx`,
  and qccompute gives programs without sessions a session over `compute`.
- **An evaluator** (the seam between the driver and programs, for tests): anything with
  `__call__(requests) -> answers` and `supports_hessian(context)` (`driver.Evaluator`);
  `calculation.run` takes one.

## Invariants

- **Atomic units** (bohr, Hartree) at every boundary; qcref constants as `c.NAME`.
- **Algorithms never call programs**; only evaluators do.
- **Fail loudly, keep the frames**: unknown settings are errors; a run that fails or does
  not converge raises `ProgramError` carrying its output, with the frames and a typed
  error.
- **No route of its own**: children run on sessions from `ctx`, so the runner owns
  resources, placement and the store.
- **Behaviour is measured**: a change meant to keep results is checked bit for bit
  against golden trajectories (77 runs: model surfaces, minima, transition states and
  IRCs at GFN2/g-xTB, constraints, frozen atoms, Hessian programs, guide modes, batches);
  one meant to change them is benchmarked on the 100 Reaction-QM reactions
  (`benchmarks/`) against Sella, geomeTRIC and OptKing.
- **The optimizer's own time matters**: an axtb gradient can take under a millisecond, so
  per-step work stays vectorized NumPy, and results report qcopt's own time separately.
