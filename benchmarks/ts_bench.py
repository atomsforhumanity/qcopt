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
        accuracy: float | None = None,
    ):
        self.symbols = symbols
        numbers = [int(getattr(pt, s.capitalize()).number) for s in symbols]
        kw = {} if accuracy is None else {"accuracy": accuracy}
        self.calc = axtb.Calculator(
            method, numbers, pos_bohr, charge=float(charge), uhf=mult - 1, **kw
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

    extra = dict(extra)
    use_h = not extra.pop("no_hessian", False)
    ev = qcopt.FunctionEvaluator(
        calc.gradient, calc.hessian if use_h else None, name="axtb"
    )
    s = Structure(symbols=sym, geometry=x0)
    if task == "irc":
        out = qcopt.irc(
            s, {"program": "axtb"}, hessian=use_h or None, evaluator=ev, **extra
        )
        frames = [f.structure.geometry for f in out.results.frames]
        return bool(out.success), frames, len(frames), out.traceback
    kw = dict(maxiter=maxiter, convergence={"gmax": gmax}, **extra)
    fn = qcopt.find_ts if task == "ts" else qcopt.minimize
    out = fn(
        s, {"program": "axtb"}, hessian=True if use_h else None, evaluator=ev, **kw
    )
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
    from ase.io import Trajectory
    from sella import IRC, Sella

    class AseAxtb(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            e, g = calc.gradient(self.atoms.positions / BOHR)
            self.results = {"energy": e * EV, "forces": -g * EV / BOHR}

    extra = dict(extra)
    use_h = not extra.pop("no_hessian", False)

    def hess(a):
        return calc.hessian(a.positions / BOHR) * EV / BOHR**2

    if task == "irc":
        frames = []
        for direction in ("reverse", "forward"):
            atoms = Atoms(sym, positions=x0 * BOHR)
            atoms.calc = AseAxtb()
            with tempfile.TemporaryDirectory() as d:
                traj = os.path.join(d, "irc.traj")
                kw = {
                    "keep_going": True,
                    **{k: v for k, v in extra.items() if k not in ("fmax", "steps")},
                }
                if use_h:
                    kw["hessian_function"] = hess
                opt = IRC(atoms, logfile=None, trajectory=traj, **kw)
                try:
                    opt.run(
                        fmax=extra.get("fmax", 0.1),
                        steps=extra.get("steps", 1000),
                        direction=direction,
                    )
                except Exception as exc:
                    return False, None, None, f"{type(exc).__name__}: {exc}"
                branch = [a.positions / BOHR for a in Trajectory(traj)]
            frames += list(reversed(branch)) if direction == "reverse" else branch
        return True, frames, len(frames), None

    atoms = Atoms(sym, positions=x0 * BOHR)
    atoms.calc = AseAxtb()
    opt = Sella(
        atoms,
        order=1 if task == "ts" else 0,
        internal=bool(extra.get("internal", False)),
        hessian_function=hess if use_h else None,
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
    mol.build_topology()  # geomeTRIC's IRC asks for the molecule's fragments
    use_h = not extra.get("no_hessian", False)
    with tempfile.TemporaryDirectory() as d:
        kw = {}
        if use_h:
            hfile = os.path.join(d, "hessian.txt")
            np.savetxt(hfile, calc.hessian(x0))
            kw["hessian"] = f"file:{hfile}"
        if task == "irc":
            kw["irc"] = True
        else:
            kw.update(
                transition=(task == "ts"),
                convergence_gmax=gmax,
                convergence_grms=1.0,
                convergence_energy=1.0,
                convergence_drms=1.0,
                convergence_dmax=1.0,
            )
        try:
            m = geometric.optimize.run_optimizer(
                customengine=Eng(mol),
                check=1,
                input=os.path.join(d, "x"),
                coordsys=extra.get("coordsys", "tric"),
                maxiter=maxiter,
                **kw,
            )
            if task == "irc":
                frames = [np.asarray(xyz) / BOHR for xyz in m.xyzs]
                return True, frames, len(frames), None
            x = np.asarray(m.xyzs[-1]) / BOHR
            return True, x, len(m.xyzs) - 1, None
        except Exception as exc:  # geomeTRIC raises when it does not converge
            return False, None, None, f"{type(exc).__name__}: {exc}"


def bonds(sym, x):
    from qcref import periodic_table as pt

    r = np.array([float(getattr(pt, e.capitalize()).cov_radius) for e in sym])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    adj = d < 1.25 * (r[:, None] + r[None])
    np.fill_diagonal(adj, False)
    return adj


def irc_check(a, sym, frames):
    """Minimize both IRC ends (qcopt, exact Hessian) and compare their bonds."""
    from qcdata import Structure

    import qcopt

    out = {"n_frames": len(frames)}
    ends = []
    for x in (frames[0], frames[-1]):
        c2 = Counted(a.method, sym, x, accuracy=a.accuracy)
        ev = qcopt.FunctionEvaluator(c2.gradient, c2.hessian, name="axtb")
        from qcopt.keywords import OptimizationKeywords

        same = (
            {"coordinates": "cartesian"}
            if "coordinates" in OptimizationKeywords.model_fields
            else {}
        )
        m = qcopt.minimize(
            Structure(symbols=sym, geometry=x),
            {"program": "axtb"},
            hessian=True,
            evaluator=ev,
            frequencies=True,
            maxiter=2000,
            **same,  # the same end minimization for every optimizer and round
        )
        ends.append(m.results.final_structure.geometry)
        out["end_min_grads"] = out.get("end_min_grads", 0) + c2.n_grad
        out["end_min_hess"] = out.get("end_min_hess", 0) + c2.n_hess
        out["end_min_seconds"] = out.get("end_min_seconds", 0.0) + c2.t_grad + c2.t_hess
        out.setdefault("end_n_imaginary", []).append(
            m.results.extras.get("n_imaginary")
        )
        out.setdefault("end_energy", []).append(float(m.results.final_energy))
        out.setdefault("end_drop_kcal", []).append(
            float(
                (
                    Counted(a.method, sym, x, accuracy=a.accuracy).gradient(x)[0]
                    - m.results.final_energy
                )
                * 627.5094740631
            )
        )
    graphs = [bonds(sym, e) for e in ends]
    raw = [bonds(sym, np.asarray(f)) for f in (frames[0], frames[-1])]
    out["ends_changed_on_minimization"] = [
        bool((g != r).any()) for g, r in zip(graphs, raw, strict=False)
    ]
    if a.reactant and a.product:
        import networkx as nx

        def graph(symbols, adj):
            g = nx.Graph()
            for i, e in enumerate(symbols):
                g.add_node(i, element=e)
            g.add_edges_from(zip(*np.nonzero(np.triu(adj)), strict=False))
            return g

        def same(g1, g2):
            return nx.is_isomorphic(
                g1, g2, node_match=lambda u, v: u["element"] == v["element"]
            )

        rs, rx = read_xyz(a.reactant)
        ps, px = read_xyz(a.product)
        gr, gp = graph(rs, bonds(rs, rx)), graph(ps, bonds(ps, px))
        ge = [graph(sym, g) for g in graphs]
        out["ends_match"] = [same(g, gr) or same(g, gp) for g in ge]
        out["connects"] = bool(
            (same(ge[0], gr) and same(ge[1], gp))
            or (same(ge[0], gp) and same(ge[1], gr))
        )
    out["ends"] = [np.asarray(e).tolist() for e in ends]
    return out


def run_optking(calc: Counted, sym, x0, task, gmax, maxiter, extra):
    """OptKing's own driver (optimize.optimize) with a computer that calls axtb.

    Its step-by-step CustomHelper breaks when OptKing rebuilds its coordinates
    (the Hessian keeps the old size), so the full driver is used.
    """
    import logging

    from optking import molsys as ok_molsys
    from optking.compute_wrappers import UserComputer
    from optking.optimize import optimize as ok_optimize
    from optking.optwrapper import initialize_options

    logging.getLogger("optking").setLevel(logging.CRITICAL)
    extra = dict(extra)
    use_h = not extra.pop("no_hessian", False)

    class AxtbComputer(UserComputer):
        def _compute(self, driver):
            g = np.asarray(self.molecule["geometry"], dtype=float).reshape(-1, 3)
            e, grad = calc.gradient(g)
            self.external_energy = e
            self.external_gradient = grad.ravel()
            if driver == "hessian":
                self.external_hessian = calc.hessian(g)
            return super()._compute(driver)

    mol = {
        "symbols": list(sym),
        "geometry": np.asarray(x0).ravel().tolist(),
        "fix_com": True,
        "fix_orientation": True,
    }
    initialize_options(
        {
            "opt_type": "TS" if task == "ts" else "MIN",
            "geom_maxiter": maxiter,
            "max_force_g_convergence": gmax,  # on internal-coordinate forces
            "full_hess_every": 0 if use_h else -1,
            **extra,
        },
        silent=True,
    )
    molsys = ok_molsys.Molsys.from_schema(mol)
    computer = AxtbComputer(
        dict(mol), {"method": "gxtb", "basis": "none"}, {}, "axtb", {}, 1
    )
    out = ok_optimize(molsys, computer)
    ok = bool(out.get("success", False))
    traj = out.get("trajectory") or []
    if not traj:
        return False, None, None, str(out.get("error"))
    x = np.asarray(traj[-1]["molecule"]["geometry"], dtype=float).reshape(-1, 3)
    err = None if ok else str(out.get("error"))
    return ok, x, len(traj), err


RUNNERS = {
    "qcopt": run_qcopt,
    "sella": run_sella,
    "geometric": run_geometric,
    "optking": run_optking,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("optimizer", choices=sorted(RUNNERS))
    p.add_argument("xyz")
    p.add_argument("out")
    p.add_argument("--task", default="ts", choices=["ts", "min", "irc"])
    p.add_argument("--reactant")
    p.add_argument("--product")
    p.add_argument("--method", default="gxtb-v2.0.1")
    p.add_argument("--fmax-ev", type=float, default=0.001)
    p.add_argument(
        "--accuracy",
        type=float,
        default=0.01,
        help="axtb SCF accuracy, passed explicitly (its default, 1.0, leaves ~1e-5 "
        "Eh/bohr of gradient noise, the size of the 0.001 eV/A threshold)",
    )
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--extra", default="{}", help="JSON of optimizer-specific options")
    a = p.parse_args()
    extra = json.loads(a.extra)
    sym, x0 = read_xyz(a.xyz)
    gmax = a.fmax_ev / EV * BOHR
    calc = Counted(a.method, sym, x0, accuracy=a.accuracy)
    rec = {
        "optimizer": a.optimizer,
        "xyz": a.xyz,
        "task": a.task,
        "method": a.method,
        "fmax_ev": a.fmax_ev,
        "accuracy": a.accuracy,
        "n_atoms": len(sym),
        "extra": extra,
    }
    # import the optimizer before timing it
    if a.optimizer == "qcopt":
        import qcdata  # noqa: F401

        import qcopt  # noqa: F401
    elif a.optimizer == "sella":
        import ase  # noqa: F401
        import sella  # noqa: F401
    elif a.optimizer == "optking":
        import optking  # noqa: F401
    else:
        import geometric.optimize  # noqa: F401
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
    if x is not None and a.task == "irc":
        rec.update(irc_check(a, sym, x))
    elif x is not None:
        import qcopt

        check = Counted(a.method, sym, x, accuracy=a.accuracy)
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
