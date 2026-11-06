"""Benchmark transition-state optimizers on one structure with axtb gradients.

Every optimizer gets the same counted axtb calculator (in-process, 1 thread), the
same exact starting Hessian, and the same convergence threshold on the largest
atomic force. The result is checked with a fresh analytic Hessian.

usage: ts_bench.py OPTIMIZER XYZ OUT.json [--task ts|min] [--method gxtb-v2.0.1]
       [--fmax-ev 0.001] [--maxiter 1000]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # before numpy: one BLAS thread, as each run gets one core

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

import axtb  # noqa: E402
from qcref import constants as c  # noqa: E402
from qcref import periodic_table as pt  # noqa: E402

BOHR = float(c.BOHR_TO_ANGSTROM)
EV = float(c.HARTREE_TO_EV)


class Counted:
    """axtb with counts and timings of every gradient and Hessian."""

    def __init__(
        self,
        method: str,
        symbols: list[str],
        pos_bohr: np.ndarray,
        charge: int = 0,
        mult: int = 1,
    ):
        self.symbols = symbols
        numbers = [int(getattr(pt, s.capitalize()).number) for s in symbols]
        self.calc = axtb.Calculator(
            method, numbers, pos_bohr, charge=float(charge), uhf=mult - 1
        )
        self.n_grad = 0
        self.n_hess = 0
        self.t_grad = 0.0
        self.t_hess = 0.0

    def gradient(self, pos_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        t = time.perf_counter()
        r = self.calc.singlepoint(np.asarray(pos_bohr, dtype=float).reshape(-1, 3))
        self.t_grad += time.perf_counter() - t
        self.n_grad += 1
        return float(r["energy"]), np.asarray(r["gradient"], dtype=float).reshape(-1, 3)

    def hessian(self, pos_bohr: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        r = self.calc.hessian(
            np.asarray(pos_bohr, dtype=float).reshape(-1, 3), vibrations=False
        )
        self.t_hess += time.perf_counter() - t
        self.n_hess += 1
        return np.asarray(r["hessian"], dtype=float)


def read_xyz(path: str) -> tuple[list[str], np.ndarray]:
    lines = open(path).read().splitlines()
    n = int(lines[0])
    sym, pos = [], []
    for line in lines[2 : 2 + n]:
        f = line.split()
        sym.append(f[0])
        pos.append([float(v) for v in f[1:4]])
    return sym, np.array(pos) / BOHR


# ---------------------------------------------------------------- optimizers


def run_qcopt(calc: Counted, sym, x0, task, gmax, maxiter, extra):
    from qcdata import Structure

    import qcopt

    ev = qcopt.FunctionEvaluator(calc.gradient, calc.hessian, name="axtb")
    s = Structure(symbols=sym, geometry=x0)
    kw = dict(maxiter=maxiter, convergence={"gmax": gmax}, **extra)
    fn = qcopt.find_ts if task == "ts" else qcopt.minimize
    out = fn(s, {"program": "axtb"}, hessian=True, evaluator=ev, **kw)
    x = out.results.final_structure.geometry
    return (
        bool(out.results.extras["converged"]),
        x,
        len(out.results.frames) - 1,
        out.traceback,
    )


def run_sella(calc: Counted, sym, x0, task, gmax, maxiter, extra):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from sella import Sella

    class AseAxtb(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            e, g = calc.gradient(self.atoms.positions / BOHR)
            self.results = {"energy": e * EV, "forces": -g * EV / BOHR}

    atoms = Atoms(sym, positions=x0 * BOHR)
    atoms.calc = AseAxtb()

    def hess(a):
        return calc.hessian(a.positions / BOHR) * EV / BOHR**2

    opt = Sella(
        atoms,
        order=1 if task == "ts" else 0,
        internal=bool(extra.get("internal", False)),
        hessian_function=hess,
        logfile=None,
    )
    fmax = gmax * EV / BOHR
    conv = bool(opt.run(fmax=fmax, steps=maxiter))
    return conv, atoms.positions / BOHR, opt.nsteps, None


def run_geometric(calc: Counted, sym, x0, task, gmax, maxiter, extra):
    import logging

    import geometric.engine
    import geometric.molecule
    import geometric.optimize

    logging.getLogger("geometric").setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("geometric"):
            logging.getLogger(name).disabled = True

    class Eng(geometric.engine.Engine):
        def calc_new(self, coords, dirname):
            e, g = calc.gradient(np.asarray(coords).reshape(-1, 3))
            return {"energy": e, "gradient": g.ravel()}

    mol = geometric.molecule.Molecule()
    mol.elem = list(sym)
    mol.xyzs = [x0 * BOHR]
    with tempfile.TemporaryDirectory() as d:
        hfile = os.path.join(d, "hessian.txt")
        np.savetxt(hfile, calc.hessian(x0))
        try:
            m = geometric.optimize.run_optimizer(
                customengine=Eng(mol),
                check=1,
                input=os.path.join(d, "x"),
                transition=(task == "ts"),
                hessian=f"file:{hfile}",
                coordsys=extra.get("coordsys", "tric"),
                maxiter=maxiter,
                convergence_gmax=gmax,
                convergence_grms=1.0,
                convergence_energy=1.0,
                convergence_drms=1.0,
                convergence_dmax=1.0,
            )
            x = np.asarray(m.xyzs[-1]) / BOHR
            return True, x, len(m.xyzs) - 1, None
        except Exception as exc:  # geomeTRIC raises when it does not converge
            return False, None, None, f"{type(exc).__name__}: {exc}"


RUNNERS = {"qcopt": run_qcopt, "sella": run_sella, "geometric": run_geometric}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("optimizer", choices=sorted(RUNNERS))
    p.add_argument("xyz")
    p.add_argument("out")
    p.add_argument("--task", default="ts", choices=["ts", "min"])
    p.add_argument("--method", default="gxtb-v2.0.1")
    p.add_argument("--fmax-ev", type=float, default=0.001)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--extra", default="{}", help="JSON of optimizer-specific options")
    a = p.parse_args()
    extra = json.loads(a.extra)
    sym, x0 = read_xyz(a.xyz)
    gmax = a.fmax_ev / EV * BOHR
    calc = Counted(a.method, sym, x0)
    rec = {
        "optimizer": a.optimizer,
        "xyz": a.xyz,
        "task": a.task,
        "method": a.method,
        "fmax_ev": a.fmax_ev,
        "n_atoms": len(sym),
        "extra": extra,
    }
    t0 = time.perf_counter()
    try:
        conv, x, nsteps, err = RUNNERS[a.optimizer](
            calc, sym, x0, a.task, gmax, a.maxiter, extra
        )
    except Exception as exc:
        conv, x, nsteps, err = (
            False,
            None,
            None,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
    wall = time.perf_counter() - t0
    rec.update(
        converged=conv,
        steps=nsteps,
        error=err,
        wall=wall,
        n_grad=calc.n_grad,
        n_hess=calc.n_hess,
        t_grad=calc.t_grad,
        t_hess=calc.t_hess,
        t_optimizer=wall - calc.t_grad - calc.t_hess,
    )
    if x is not None:
        import qcopt

        check = Counted(a.method, sym, x)
        e, g = check.gradient(x)
        h = check.hessian(x)
        f = qcopt.frequencies(h, sym, x)
        rec.update(
            energy=e,
            gmax_final=float(np.sqrt((g * g).sum(1)).max()),
            n_imaginary=f.n_imaginary(10.0),
            lowest_cm=[float(v) for v in f.wavenumbers[:3]],
            geometry=np.asarray(x).tolist(),
            symbols=sym,
        )
    json.dump(rec, open(a.out, "w"))


if __name__ == "__main__":
    sys.exit(main())
