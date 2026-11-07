# qcopt

Minimizations, transition-state searches, and IRCs for the qc* stack. Inputs and outputs
are [qcdata](https://github.com/atomsforhumanity/qcdata) objects; gradients and Hessians
come from [axtb](https://github.com/atomsforhumanity/axtb) in-process or from any program
qccompute runs.

```python
import qcopt
from qcdata import Structure

# accuracy 0.01: at axtb's default SCF accuracy, g-xTB gradients carry ~1e-5 Eh/bohr
# of noise, the size of qcopt's convergence threshold (12% more time per gradient).
gxtb = {"program": "axtb", "model": {"method": "gxtb-v2.0.1"}, "keywords": {"accuracy": 0.01}}
guess = Structure.open("ts_guess.xyz")

ts = qcopt.find_ts(guess, gxtb, hessian=True, frequencies=True)  # ProgramOutput
ts.results.extras["n_imaginary"]                                   # 1
path = qcopt.irc(ts.results.final_structure, gxtb, hessian=True)   # ProgramOutput (PathData)
```

`hessian` names the program computing Hessians: `True` for the gradient program, or another
one, often a cheaper method. Frames keep the gradient program's energies and gradients:

```python
gfn2 = {"program": "axtb", "model": {"method": "gfn2"}, "keywords": {"accuracy": 0.01}}
ts = qcopt.find_ts(guess, gxtb, hessian=gfn2)   # g-xTB saddle from a GFN2 starting Hessian
```

Keyword arguments are qcopt's keywords (`qcopt.keywords`), such as `maxiter`,
`initial_hessian` (`"model"`, `"exact"`, ...) or `hessian_every`.

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
        ProgramSpec(program="axtb", calctype="gradient", model={"method": "gxtb-v2.0.1"},
                    keywords={"accuracy": 0.01}),
        ProgramSpec(program="axtb", calctype="hessian", model={"method": "gxtb-v2.0.1"},
                    keywords={"accuracy": 0.01}),
    ],
)
out = qcopt.run(inp)            # or qcopt.run([inp1, inp2, ...])
```

## What it does

| Calculation | Method |
|---|---|
| `optimization` | RS-RFO in TRIC coordinates (Wang and Song, 2016: each molecule's internal coordinates plus its translation and rotation), or delocalized internals, or Cartesians; Lindh model Hessian or the program's exact one; BFGS updates (Bofill while the Hessian is indefinite); optional frequency check that steps off saddle points |
| `transition_state` | RS-P-RFO climbing the lowest mode (or, in Cartesians, one following a guide direction); exact (or a cheaper method's) or finite-difference starting Hessian; TS-BFGS updates; recomputed when the negative mode is lost; fails when it ends without negative curvature (pass `frequencies=True` for a check by the exact Hessian's frequencies) |
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

## Install

```sh
pip install "qcopt[axtb]"       # with axtb: in-process g-xTB and GFN2-xTB, analytic Hessians
pip install "qcopt[qccompute]"  # any program qccompute runs
pip install qcopt               # your own gradient function
```

[axtb](https://github.com/atomsforhumanity/axtb) 0.1.0 or newer. Its wheels are for x86-64
Linux; elsewhere pip builds it from source, which needs Rust.

## Status

0.x: the API may still change between minor versions.

## License

MIT.
