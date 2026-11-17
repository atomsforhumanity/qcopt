# qcopt

Minimizations, transition-state searches, and IRCs for the qc* stack. Inputs and outputs
are [qcdata](https://github.com/atomsforhumanity/qcdata) objects; gradients and Hessians
come from [axtb](https://github.com/atomsforhumanity/axtb) in-process or from any program
qccompute runs.

A qcopt calculation is a qcdata `ProgramInput`: the calculation type, the starting
structure, qcopt's settings as `keywords`, and the programs that compute gradients (and
Hessians) as `subprograms`. `qcopt.compute` runs it:

```python
import qcopt
from qcdata import ProgramInput, Structure

# accuracy 0.01: at axtb's default SCF accuracy, g-xTB gradients carry ~1e-5 Eh/bohr
# of noise, the size of qcopt's convergence threshold (12% more time per gradient).
gxtb = {"program": "axtb", "model": {"method": "gxtb-v2.0.1"}, "keywords": {"accuracy": 0.01}}

ts = qcopt.compute(
    ProgramInput(
        program="qcopt",
        calctype="transition_state",   # or "optimization", "irc"
        structure=Structure.open("ts_guess.xyz"),
        keywords={"freq": {"check": True}},
        subprograms=[{**gxtb, "calctype": "gradient"}, {**gxtb, "calctype": "hessian"}],
    )
)
ts.results.final_structure      # the saddle point: one imaginary mode
```

The `hessian` subprogram is optional. Without it, minimizations start from Lindh's model
Hessian and transition states from finite differences of gradients. It may be another
program than the gradients', often a cheaper method: the frames keep the gradient
program's energies and gradients.

```python
gfn2 = {"program": "axtb", "model": {"method": "gfn2"}, "keywords": {"accuracy": 0.01}}
subprograms = [{**gxtb, "calctype": "gradient"}, {**gfn2, "calctype": "hessian"}]
```

Any calculation can start from results already known at its structure
(`ProgramInput.given`): their Hessian becomes the starting one, from any program. An IRC
from a search with `freq.check` starts from the search's transition-state Hessian, and
computes only the gradient there:

```python
last = ts.results.frames[-1]
path = qcopt.compute(
    ProgramInput(
        program="qcopt",
        calctype="irc",
        structure=last.structure,
        given=last.results,              # its Hessian: the IRC's at the transition state
        keywords={"ends": {"minimize": True}},
        subprograms=[{**gxtb, "calctype": "gradient"}, {**gxtb, "calctype": "hessian"}],
    )
)                                        # a ProgramOutput holding PathData
```

A calculation that fails or does not converge raises `qcdata.ProgramError`, which
carries the failed output, frames included, as `output` (see Results).

Many calculations at once go through qccompute, the runner of AFH programs:
`qccompute.compute([inp1, inp2, ...])`. They advance in lockstep, and each round's
gradients from all of them are computed together. qcopt computes its gradients and
Hessians through sessions of the child programs, one per calculation and program,
warm-started from geometry to geometry (each IRC branch its own lineage), under the
runner's resources. Called directly, `qcopt.compute` hands itself to qccompute's default
runner, which runs it on this machine. Until axtb has sessions of its own, qcopt runs
axtb in this process.

## Settings

Settings come in groups, under `keywords`:

```python
keywords={"convergence": {"g_tol": 1e-5, "max_iter": 200},
          "constraints": {"frozen": [0, 1], "distance": [[2, 3, 2.9]]}}
keywords={"hessian": {"recompute_every": 5}, "mode": {"guide": tangent},
          "coordinate_system": "cartesian"}            # a transition state
keywords={"irc": {"step": 0.2}, "ends": {"minimize": True}}   # an IRC
```

The common settings ([docs/settings.md](docs/settings.md) documents every one, with its
default and unit):

| group | settings | calculation types |
|---|---|---|
| `coordinate_system` | `auto` (TRIC; Cartesians with frozen atoms), `tric`, `dlc`, `cartesian` | min, TS |
| `convergence` | `max_iter`, `g_tol` (largest atomic gradient, Eh/bohr), `g_rms_tol`, `d_tol`, `e_tol` | all (IRC: per branch) |
| `trust` | `radius`, `min`, `max` (bohr) | min, TS |
| `hessian` | `initial` (`auto`, `exact`, `model`, ...), `update`, `recompute_every`, `step` | all |
| `constraints` | `frozen`, `distance`, `angle`, `dihedral` (degrees) | min, TS (IRC: `frozen`) |
| `mode` | `guide` (a direction to climb), `follow` | TS |
| `freq` | `check` (frequencies at the end decide), `imaginary_cutoff` (cm⁻¹) | min, TS |
| `escape` | `attempts`, `step`: step off saddle points a minimization stops on | min |
| `irc` | `direction`, `step` (bohr amu^1/2), `method` (`gs2`, `lqa`) | IRC |
| `ends` | `minimize` (the IRC's ends), `coordinate_system`, `max_iter`, `g_tol` | IRC |

The rules, shared with AFH's other programs (axtb, linus):

- Atomic units (bohr, Hartree) unless a setting says otherwise.
- `"auto"` means qcopt decides, and every result records what it decided and why.
- A misspelled or unknown setting is an error that says what was meant. So is a setting
  of another calculation type, and one given that cannot affect the calculation (such as
  `escape.step` without `escape.attempts`): you always know what is computed. Settings
  that act only if a condition arises during the run are allowed.
- `qcopt.program.program.settings_schema(calctype)` is the JSON Schema of each calculation
  type's settings.

[docs/choosing-settings.md](docs/choosing-settings.md) has the benchmark evidence behind
the defaults and when to change them.

## Results

A `ProgramOutput` holding `OptimizationData` (minima, transition states) or `PathData`
(IRCs). Every frame is a qcdata `Frame` with the energy and gradient computed there. A run
that fails or does not converge raises `ProgramError`; its `output` has `success=False`
and keeps its frames (`qccompute.compute(..., raise_exc=False)` returns it in place).
The output's standard slots, the same in every AFH program:

- `resolved`: `settings`, the settings with every `auto` replaced (pass them back as
  `keywords` to reproduce the run), and `reasons`;
- `record`: what happened: gradients and Hessians computed, coordinate rebuilds, rejected
  steps, IRC branch ends, the step log;
- `error`, on failure: `kind` (`input`, `program`: a gradient or Hessian calculation
  failed, with its `program` and its own error as `cause`, and a message that starts
  with the program and calculation type, `axtb (gradient): ...`; `convergence`, with
  `code` such as `max_iter` or `not_first_order_saddle`; `numerical`; `internal`),
  `message` and `code`. `traceback` holds the message, then the failed calculation's
  own traceback;
- `execution.timings`: `subprograms` (in the gradient and Hessian programs), `optimizer`
  (qcopt's own), `total`.

## What it does

| Calculation | Method |
|---|---|
| `optimization` | RS-RFO in TRIC coordinates (Wang and Song, 2016: each molecule's internal coordinates plus its translation and rotation), or delocalized internals, or Cartesians; Lindh model Hessian or the program's exact one; BFGS updates (Bofill while the Hessian is indefinite); optional frequency check that steps off saddle points |
| `transition_state` | RS-P-RFO climbing the lowest mode (or, in Cartesians, one following a guide direction); exact (or a cheaper method's) or finite-difference starting Hessian; TS-BFGS updates; recomputed when the negative mode is lost; fails when it ends without negative curvature |
| `irc` | Mass-weighted steepest descent by Gonzalez-Schlegel steps (or the local quadratic approximation), both branches together, frames in the input's frame; overshoots retried at half the step; optionally both ends minimized together |

How it is built, and where to extend it: [ARCHITECTURE.md](ARCHITECTURE.md).

## Benchmarks

`benchmarks/` compares qcopt with Sella, geomeTRIC and OptKing on the same axtb
gradients; results are summarized in AFH's findings.

## Install

```sh
pip install --pre "qcopt==0.3.0a3"         # on the stack's alpha 3 (qcdata 0.24.0a4, qccompute 0.23.0a3)
pip install --pre "qcopt[axtb]==0.3.0a3"   # with axtb: in-process g-xTB and GFN2-xTB, analytic Hessians
```

This README describes the 0.3 alphas. Without `--pre`, pip installs qcopt 0.2, whose
interface differs.

[axtb](https://github.com/atomsforhumanity/axtb) 0.1.0 or newer. Its wheels are for x86-64
Linux; elsewhere pip builds it from source, which needs Rust.

## Status

0.x: the API may still change between minor versions.

## License

MIT.
