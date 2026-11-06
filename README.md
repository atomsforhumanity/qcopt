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
| `optimization` | RS-RFO in Cartesian coordinates with rigid motion projected out; Lindh model Hessian or the program's exact one; BFGS updates (Bofill while the Hessian is indefinite) |
| `transition_state` | RS-P-RFO climbing the lowest mode (or the one following a guide direction); exact or finite-difference starting Hessian; TS-BFGS updates; recomputed when the negative mode is lost |
| `irc` | Mass-weighted steepest descent by the local quadratic approximation, both branches together, frames in the input's frame |

Frozen atoms, convergence criteria, trust radii and Hessian policies are keywords
(`qcopt.keywords`). Every frame is a qcdata `Frame` with the energy and gradient computed
there.

## Status

Early development (0.1.0.dev0). Not yet released to PyPI.

## License

MIT.
