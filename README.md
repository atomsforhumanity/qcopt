# qcopt

Minimizations, transition-state searches, and IRCs for the qc* stack. Inputs and outputs
are [qcdata](https://github.com/atomsforhumanity/qcdata) objects; gradients and Hessians
come from [axtb](https://github.com/atomsforhumanity/axtb) in-process or from any program
qccompute runs.

```python
import qcopt
from qcdata import Structure

gxtb = {"program": "axtb", "model": {"method": "gxtb-v2.0.1"}}
guess = Structure.open("ts_guess.xyz")

ts = qcopt.find_ts(guess, gxtb, hessian=True, frequencies=True)  # ProgramOutput
ts.results.extras["n_imaginary"]                                   # 1
path = qcopt.irc(ts.results.final_structure, gxtb, hessian=True)   # PathData
```

Or as a qcdata input, alone or many at once (they advance together, and each round's
gradients from all of them are computed in one call):

```python
from qcdata import ProgramInput, ProgramSpec

inp = ProgramInput(
    program="qcopt",
    calctype="transition_state",
    structure=guess,
    keywords={"maxiter": 300},
    subprograms=[
        ProgramSpec(program="axtb", calctype="gradient", model={"method": "gxtb-v2.0.1"}),
        ProgramSpec(program="axtb", calctype="hessian", model={"method": "gxtb-v2.0.1"}),
    ],
)
out = qcopt.run(inp)            # or qcopt.run([inp1, inp2, ...])
```

## What it does

| Calculation | Method |
|---|---|
| `optimization` | RS-RFO in TRIC coordinates (Wang and Song, 2016: each molecule's internal coordinates plus its translation and rotation), or delocalized internals, or Cartesians; Lindh model Hessian or the program's exact one; BFGS updates (Bofill while the Hessian is indefinite); optional frequency check that steps off saddle points |
| `transition_state` | RS-P-RFO climbing the lowest mode (or one following a guide direction); exact or finite-difference starting Hessian; TS-BFGS updates; recomputed when the negative mode is lost |
| `irc` | Mass-weighted steepest descent by Gonzalez-Schlegel steps (or the local quadratic approximation), both branches together, frames in the input's frame; overshoots retried at half the step; optionally both ends minimized together (`minimize_ends`) |

Distances, angles and dihedrals can be held at target values (`constraints`), atoms
frozen (`frozen_atoms`). Every frame is a qcdata `Frame` with the energy and gradient
computed there.

With qccompute installed, qcopt is also a program it runs:

```python
from qccompute import compute
out = compute(inp)   # the ProgramInput above
```

## Benchmarks

`benchmarks/` compares qcopt with Sella, geomeTRIC and OptKing on the same axtb
gradients; results are summarized in AFH's findings.

## Status

Early development (0.1.0.dev0). Not yet released to PyPI.

## License

MIT.
