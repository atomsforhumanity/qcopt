"""Delocalized internal coordinates built from redundant primitives.

Primitives (Bakken and Helgaker, J. Chem. Phys. 117, 9160 (2002)): bonds from covalent
radii, bonds joining separate fragments and hydrogen bonds, bends, linear bends (two
components) where a bend is near 180 degrees, and torsions (through linear chains).
Delocalized internals (Baker, Kessi and Delley, J. Chem. Phys. 105, 192 (1996)) are
the combinations ``s = U^T q`` of primitives ``q`` along the non-redundant
eigenvectors ``U`` of ``G = B B^T``; there are exactly as many as internal degrees of
freedom (3N - 6, or 3N - 5 for a linear structure), so steps, gradients and Hessians
in them are ordinary vectors and matrices.

Values are in bohr and radians. Cartesian steps are recovered from internal steps by
iterating ``x += B_s^+ (s_target - s(x))``; the generalized inverse gives the
smallest Cartesian move, so no rigid motion is added.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qcref import periodic_table as pt

from . import primitives as prim
from .geometry import Array, is_linear, rigid_basis

#: Bends at or above this angle (degrees) become linear bends.
LINEAR_DEGREES = 175.0


class CoordinateError(RuntimeError):
    """The internal coordinates cannot describe this structure or step."""


def _wrap(angle: Array) -> Array:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class Fragment:
    """A covalently bonded fragment's rigid-body coordinates (TRIC).

    Translation: the centroid of its atoms (3). Rotation: the rotation vector taking
    its reference geometry onto the current one, times ``rg`` so it is a length (3),
    or for a linear fragment the tilt of its axis along two fixed perpendicular
    directions, times ``rg`` (2). One atom: no rotation.

    Attributes:
        atoms: Atom indices.
        ref: Reference positions, centered.
        rg: Radius of gyration of the reference, at least 1 bohr (bohr).
        linear: Whether the fragment is linear.
        axes: For a linear fragment, the reference axis and two perpendicular
            directions ``(3, 3)``.
    """

    atoms: np.ndarray
    ref: np.ndarray
    rg: float
    linear: bool
    axes: np.ndarray

    @property
    def n_rotation(self) -> int:
        """Number of rotation coordinates."""
        if len(self.atoms) == 1:
            return 0
        return 2 if self.linear else 3

    @property
    def size(self) -> int:
        """Number of coordinates."""
        return 3 + self.n_rotation


def _horn_matrix(s: Array) -> Array:
    """Horn's 4x4 quaternion matrix of correlation matrices ``s`` ``(m, 3, 3)``."""
    sxx, sxy, sxz = s[:, 0, 0], s[:, 0, 1], s[:, 0, 2]
    syx, syy, syz = s[:, 1, 0], s[:, 1, 1], s[:, 1, 2]
    szx, szy, szz = s[:, 2, 0], s[:, 2, 1], s[:, 2, 2]
    f = np.empty((len(s), 4, 4))
    f[:, 0, 0] = sxx + syy + szz
    f[:, 0, 1] = f[:, 1, 0] = syz - szy
    f[:, 0, 2] = f[:, 2, 0] = szx - sxz
    f[:, 0, 3] = f[:, 3, 0] = sxy - syx
    f[:, 1, 1] = sxx - syy - szz
    f[:, 1, 2] = f[:, 2, 1] = sxy + syx
    f[:, 1, 3] = f[:, 3, 1] = szx + sxz
    f[:, 2, 2] = -sxx + syy - szz
    f[:, 2, 3] = f[:, 3, 2] = syz + szy
    f[:, 3, 3] = -sxx - syy + szz
    return f


def _quaternion_to_vector(q: Array) -> Array:
    """Rotation vectors (axis times angle) of quaternions ``(m, 4)``, any norm."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)
    qv = q[:, 1:]
    sn = np.linalg.norm(qv, axis=1)
    angle = 2.0 * np.arctan2(sn, q[:, 0])
    scale = np.where(sn > 1e-12, angle / np.maximum(sn, 1e-300), 2.0 / q[:, 0])
    return qv * scale[:, None]


def _rotation_vectors(ref: Array, cur: Array) -> Array:
    """Rotation vectors best mapping ``ref`` onto each ``cur``.

    Horn's quaternion solution of the superposition problem (J. Opt. Soc. Am. A 4,
    629 (1987)). ``ref`` is ``(n, 3)`` and centered; ``cur`` is ``(m, n, 3)``.
    """
    c = cur - cur.mean(axis=1, keepdims=True)
    _, vecs = np.linalg.eigh(_horn_matrix(np.einsum("ni,mnj->mij", ref, c)))
    return _quaternion_to_vector(vecs[:, :, -1])


def _rotation_jacobian(ref: Array, cur: Array) -> Array:
    """d(rotation vector)/d(positions) ``(3, 3n)``, analytically.

    The quaternion's derivative comes from first-order perturbation theory of
    Horn's eigenproblem (only its correlation matrix depends on the positions, and
    linearly); the rotation vector's from finite differences in the quaternion.
    """
    n = len(ref)
    c = cur - cur.mean(axis=0)
    vals, vecs = np.linalg.eigh(_horn_matrix((ref.T @ c)[None])[0])
    q = vecs[:, -1]
    if q[0] < 0:
        q = -q
    # dS/dx_{j,b} = a_j e_b^T (ref is centered, so the centroid drops out).
    ds = np.zeros((n, 3, 3, 3))
    for b in range(3):
        ds[:, b, :, b] = ref
    df = _horn_matrix(ds.reshape(3 * n, 3, 3))  # (3n, 4, 4)
    others, lams = vecs[:, :-1], vals[:-1]
    dq = (df @ q) @ others / (vals[-1] - lams)  # (3n, 3)
    dq = dq @ others.T  # (3n, 4)
    h = 1e-6
    shifted = q + np.concatenate([np.eye(4), -np.eye(4)]) * h
    v = _quaternion_to_vector(shifted)
    jac_q = ((v[:4] - v[4:]) / (2 * h)).T  # (3, 4)
    return jac_q @ dq.T


def _all_fragment_values(frags: list[Fragment], x: Array) -> Array:
    """Every fragment's coordinates, with one eigen-decomposition for all rotations."""
    rot = [f for f in frags if f.n_rotation == 3]
    vectors: dict[int, Array] = {}
    if rot:
        s = np.stack([f.ref.T @ (x[f.atoms] - x[f.atoms].mean(axis=0)) for f in rot])
        _, vecs = np.linalg.eigh(_horn_matrix(s))
        v = _quaternion_to_vector(vecs[:, :, -1])
        vectors = {id(f): v[i] for i, f in enumerate(rot)}
    out = []
    for f in frags:
        xs = x[f.atoms]
        out.append(xs.mean(axis=0))
        if f.n_rotation == 3:
            out.append(f.rg * vectors[id(f)])
        elif f.n_rotation == 2:
            u = xs[-1] - xs[0]
            out.append(f.rg * (f.axes[1:] @ (u / np.linalg.norm(u))))
    return np.concatenate(out)


def _fragment_b(frag: Fragment, x: Array, n_atoms: int, h: float = 1e-5) -> Array:
    na = len(frag.atoms)
    rows = np.zeros((frag.size, 3 * n_atoms))
    cols = (3 * frag.atoms[:, None] + np.arange(3)).ravel()
    for k in range(3):
        rows[k, 3 * frag.atoms + k] = 1.0 / na
    if frag.n_rotation == 3:
        rows[3:, cols] = frag.rg * _rotation_jacobian(frag.ref, x[frag.atoms])
    elif frag.n_rotation == 2:
        xs = x[frag.atoms]
        d = xs[-1] - xs[0]
        length = np.linalg.norm(d)
        u = d / length
        for a in range(2):
            e = frag.axes[1 + a]
            du = (e - (u @ e) * u) / length  # d(u.e)/d(x_last)
            rows[3 + a, 3 * frag.atoms[-1] : 3 * frag.atoms[-1] + 3] = frag.rg * du
            rows[3 + a, 3 * frag.atoms[0] : 3 * frag.atoms[0] + 3] = -frag.rg * du
    return rows


def make_fragment(
    atoms: Sequence[int], x: Array, linear: bool | None = None
) -> Fragment:
    """The rigid-body coordinates of the atoms ``atoms`` at geometry ``x``.

    Args:
        atoms: The fragment's atoms.
        x: Geometry (bohr).
        linear: Whether the fragment is linear; None: decide from its shape.
    """
    a = np.asarray(sorted(atoms), dtype=np.intp)
    xs = np.asarray(x, dtype=float).reshape(-1, 3)[a]
    ref = xs - xs.mean(axis=0)
    rg = float(np.sqrt((ref * ref).sum(axis=1).mean())) if len(a) > 1 else 0.0
    rg = max(rg, 1.0) if len(a) > 1 else 0.0
    axes = np.eye(3)
    if linear is None:
        linear = len(a) == 2 or (len(a) > 2 and _is_linear(xs, tol=0.05))
    if len(a) == 2:
        linear = True
    if linear:
        u = xs[-1] - xs[0]
        u /= np.linalg.norm(u)
        trial = np.eye(3)[int(np.argmin(np.abs(u)))]
        e1 = trial - (trial @ u) * u
        e1 /= np.linalg.norm(e1)
        axes = np.array([u, e1, prim.cross(u, e1)])
    return Fragment(atoms=a, ref=ref, rg=rg, linear=linear, axes=axes)


@dataclass
class Primitives:
    """Index arrays of each kind of primitive coordinate.

    Attributes:
        bonds: ``(m, 2)``.
        bends: ``(m, 3)``, vertex in the middle.
        linear: ``(m, 3)`` near-linear bends; each gives two coordinates.
        linear_axes: ``(m, 2, 3)`` fixed unit vectors perpendicular to each linear
            bend's axis when it was built.
        torsions: ``(m, 4)``.
    """

    bonds: np.ndarray
    bends: np.ndarray
    linear: np.ndarray
    linear_axes: np.ndarray
    torsions: np.ndarray
    n_atoms: int
    fragments: list[Fragment] = field(default_factory=list)
    _slices: dict[str, slice] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Lay out the primitives in one vector."""
        start = 0
        for name, n in (
            ("bonds", len(self.bonds)),
            ("bends", len(self.bends)),
            ("linear", 2 * len(self.linear)),
            ("torsions", len(self.torsions)),
            ("fragments", sum(f.size for f in self.fragments)),
        ):
            self._slices[name] = slice(start, start + n)
            start += n
        self.size = start
        self.periodic = np.zeros(start, dtype=bool)
        self.periodic[self._slices["torsions"]] = True

    def values(self, x: Array) -> Array:
        """All primitive values at ``x`` ``(n_atoms, 3)``."""
        x = np.asarray(x, dtype=float).reshape(-1, 3)
        q = np.empty(self.size)
        q[self._slices["bonds"]] = prim.stretch_values(x, self.bonds)
        q[self._slices["bends"]] = prim.bend_values(x, self.bends)
        q[self._slices["linear"]] = self._linear_values(x)
        q[self._slices["torsions"]] = prim.torsion_values(x, self.torsions)
        if self.fragments:
            q[self._slices["fragments"]] = _all_fragment_values(self.fragments, x)
        return q

    def b_matrix(self, x: Array) -> Array:
        """Dense B matrix ``(n_primitives, 3 n_atoms)``."""
        x = np.asarray(x, dtype=float).reshape(-1, 3)
        b = np.zeros((self.size, 3 * self.n_atoms))
        for name, idx, fn in (
            ("bonds", self.bonds, prim.stretch_derivatives),
            ("bends", self.bends, prim.bend_derivatives),
            ("torsions", self.torsions, prim.torsion_derivatives),
        ):
            if len(idx):
                b[self._slices[name]] = prim.scatter_b(fn(x, idx), idx, self.n_atoms)
        if len(self.linear):
            blocks, idx = self._linear_derivatives(x)
            b[self._slices["linear"]] = prim.scatter_b(blocks, idx, self.n_atoms)
        if self.fragments:
            b[self._slices["fragments"]] = np.vstack(
                [_fragment_b(f, x, self.n_atoms) for f in self.fragments]
            )
        return b

    def max_rotation(self, x: Array) -> float:
        """Largest fragment rotation (radians) away from its reference."""
        xx = np.asarray(x, dtype=float).reshape(-1, 3)
        out = 0.0
        for f in self.fragments:
            if f.n_rotation == 3:
                out = max(
                    out,
                    float(
                        np.linalg.norm(_rotation_vectors(f.ref, xx[f.atoms][None])[0])
                    ),
                )
            elif f.n_rotation == 2:
                u = xx[f.atoms[-1]] - xx[f.atoms[0]]
                out = max(
                    out,
                    float(np.arccos(np.clip(u @ f.axes[0] / np.linalg.norm(u), -1, 1))),
                )
        return out

    def difference(self, q1: Array, q0: Array) -> Array:
        """``q1 - q0`` with torsions wrapped into (-pi, pi]."""
        d = q1 - q0
        d[self.periodic] = _wrap(d[self.periodic])
        return d

    # Linear bend: q_a = (u_ji + u_jk) . e_a for unit bond vectors u from the vertex j
    # and two fixed directions e_a perpendicular to the axis; zero when linear.
    def _linear_values(self, x: Array) -> Array:
        if not len(self.linear):
            return np.zeros(0)
        i, j, k = self.linear.T
        u = x[i] - x[j]
        v = x[k] - x[j]
        su = (
            u / np.linalg.norm(u, axis=1)[:, None]
            + v / np.linalg.norm(v, axis=1)[:, None]
        )
        return np.einsum("md,mad->ma", su, self.linear_axes).ravel()

    def _linear_derivatives(self, x: Array) -> tuple[Array, np.ndarray]:
        i, j, k = self.linear.T
        u = x[i] - x[j]
        v = x[k] - x[j]
        lu = np.linalg.norm(u, axis=1)
        lv = np.linalg.norm(v, axis=1)
        eu, ev = u / lu[:, None], v / lv[:, None]
        m = len(self.linear)
        blocks = np.zeros((2 * m, 3, 3))
        idx = np.repeat(self.linear, 2, axis=0)
        for a in range(2):
            e = self.linear_axes[:, a, :]
            di = (e - (eu * e).sum(1)[:, None] * eu) / lu[:, None]
            dk = (e - (ev * e).sum(1)[:, None] * ev) / lv[:, None]
            blocks[a::2, 0] = di
            blocks[a::2, 1] = -(di + dk)
            blocks[a::2, 2] = dk
        return blocks, idx


def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return list(groups.values())


def build_primitives(
    symbols: Sequence[str], positions: Array, *, tric: bool = False
) -> Primitives:
    """Choose redundant primitives for a structure.

    Args:
        symbols: Element symbols.
        positions: Geometry (bohr).
        tric: Give each covalently bonded fragment its own translation and
            rotation coordinates (Wang and Song, J. Chem. Phys. 144, 214108 (2016))
            instead of joining fragments with extra bonds and hydrogen bonds.
    """
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(x)
    rc = np.array([float(getattr(pt, s.capitalize()).cov_radius) for s in symbols])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    bonded = d < 1.3 * (rc[:, None] + rc[None])
    np.fill_diagonal(bonded, False)
    edges = {
        (int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(bonded)), strict=True)
    }

    # Join fragments: closest pairs, as a spanning tree over fragments, plus any
    # pair within 1.3 times the closest distance between the two fragments.
    frags = _components(n, list(edges))
    while len(frags) > 1 and not tric:
        best = None
        for fa in range(len(frags)):
            for fb in range(fa + 1, len(frags)):
                sub = d[np.ix_(frags[fa], frags[fb])]
                imin = np.unravel_index(np.argmin(sub), sub.shape)
                if best is None or sub[imin] < best[0]:
                    best = (float(sub[imin]), fa, fb)
        assert best is not None
        dmin, fa, fb = best
        for a in frags[fa]:
            for b in frags[fb]:
                if d[a, b] < max(1.3 * dmin, dmin + 0.5):
                    edges.add((min(a, b), max(a, b)))
        frags = _components(n, list(edges))

    # Hydrogen bonds: H on N/O/F/S/Cl near another N/O/F/S/Cl.
    acceptors = {"N", "O", "F", "S", "Cl"}
    rv = []
    for s in symbols:
        v = getattr(pt, s.capitalize()).vdw_radius
        rv.append(
            float(v)
            if v is not None
            else 2.0 * float(getattr(pt, s.capitalize()).cov_radius)
        )
    rvdw = np.array(rv)
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    for h in range(n if not tric else 0):
        if symbols[h] != "H" or not any(
            symbols[nb] in acceptors for nb in neighbors[h]
        ):
            continue
        donor = next(nb for nb in neighbors[h] if symbols[nb] in acceptors)
        for acc in range(n):
            if (
                acc in (h, donor)
                or symbols[acc] not in acceptors
                or acc in neighbors[h]
            ):
                continue
            if d[h, acc] < 0.9 * (rvdw[h] + rvdw[acc]):
                ang = np.degrees(prim.bend_values(x, np.array([[donor, h, acc]]))[0])
                if ang > 90.0:
                    edges.add((min(h, acc), max(h, acc)))
    neighbors = [set() for _ in range(n)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    bends: list[tuple[int, int, int]] = []
    linear: list[tuple[int, int, int]] = []
    for j in range(n):
        nb = sorted(neighbors[j])
        for p in range(len(nb)):
            for q in range(p + 1, len(nb)):
                i, k = nb[p], nb[q]
                ang = np.degrees(prim.bend_values(x, np.array([[i, j, k]]))[0])
                (linear if ang >= LINEAR_DEGREES else bends).append((i, j, k))

    def is_linear(i: int, j: int, k: int) -> bool:
        ang = np.degrees(prim.bend_values(x, np.array([[i, j, k]]))[0])
        return bool(ang >= LINEAR_DEGREES or ang <= 180.0 - LINEAR_DEGREES)

    torsions: set[tuple[int, int, int, int]] = set()
    for j0, k0 in edges:
        for j, k in ((j0, k0), (k0, j0)):
            # Walk outward through linear chains so the torsion's end bonds are bent.
            ends_i = _ends(j, k, neighbors, is_linear)
            ends_l = _ends(k, j, neighbors, is_linear)
            for i, jj in ends_i:
                for l_, kk in ends_l:
                    if len({i, jj, kk, l_}) < 4:
                        continue
                    t = (i, jj, kk, l_)
                    if (l_, kk, jj, i) not in torsions:
                        torsions.add(t)

    # Out-of-plane motion at atoms with three neighbors, as an improper torsion
    # (n1, center, n2, n3): bends cannot see a planar center's wag, and torsions
    # miss it next to linear chains (ketene).
    for c in range(n):
        nb = sorted(neighbors[c])
        if len(nb) != 3:
            continue
        pick: tuple[float, int, int, int] | None = None
        for a_, b_, d_ in (
            (nb[0], nb[1], nb[2]),
            (nb[0], nb[2], nb[1]),
            (nb[1], nb[2], nb[0]),
        ):
            ang = np.degrees(prim.bend_values(x, np.array([[a_, c, b_]]))[0])
            if ang < LINEAR_DEGREES and (pick is None or ang > pick[0]):
                pick = (float(ang), a_, b_, d_)
        if pick is not None:
            _, a_, b_, d_ = pick
            torsions.add((a_, c, b_, d_))

    lin = np.array(linear, dtype=np.intp).reshape(-1, 3)
    axes = np.zeros((len(lin), 2, 3))
    for m, (i, _, k) in enumerate(lin):
        u = x[k] - x[i]
        u /= np.linalg.norm(u)
        trial = np.eye(3)[int(np.argmin(np.abs(u)))]
        e1 = trial - (trial @ u) * u
        e1 /= np.linalg.norm(e1)
        axes[m] = [e1, prim.cross(u, e1)]
    return Primitives(
        bonds=np.array(sorted(edges), dtype=np.intp).reshape(-1, 2),
        bends=np.array(bends, dtype=np.intp).reshape(-1, 3),
        linear=lin,
        linear_axes=axes,
        torsions=np.array(sorted(torsions), dtype=np.intp).reshape(-1, 4),
        n_atoms=n,
        fragments=[
            # Linear exactly when the fragment has no ordinary bend, so its rotations
            # and its bends never both leave a motion out (H-C-N at 171 degrees).
            make_fragment(
                f,
                x,
                linear=len(f) == 2
                or (len(f) > 2 and not any(set(t) <= set(f) for t in bends)),
            )
            for f in _components(n, list(edges))
        ]
        if tric
        else [],
    )


def _ends(
    j: int, k: int, neighbors: list[set[int]], is_linear: object
) -> list[tuple[int, int]]:
    """Atoms ``i`` bonded to ``j`` (not ``k``) with i-j-k bent, as (i, j) pairs.

    When i-j-k is linear the chain is followed outward: (i', i) for neighbors i' of
    i, and so on, so a torsion can be defined across linear segments.
    """
    out: list[tuple[int, int]] = []
    frontier = [(i, j) for i in neighbors[j] if i != k]
    seen = {j, k}
    while frontier:
        nxt: list[tuple[int, int]] = []
        for i, jj in frontier:
            if i in seen:
                continue
            if not is_linear(i, jj, k):  # type: ignore[operator]
                out.append((i, jj))
            else:
                seen.add(i)
                nxt.extend((i2, i) for i2 in neighbors[i] if i2 not in seen)
        frontier = nxt
        if out:
            break
    return out


class DelocalizedInternals:
    """Delocalized internals on a fixed basis of primitive combinations.

    Args:
        symbols: Element symbols.
        positions: Geometry (bohr) at which the primitives and basis are chosen.
        primitives: Primitives to use instead of building them.
        tric: Translation and rotation coordinates for each fragment (TRIC); the
            coordinates then also span rigid motion (3N of them).

    Raises:
        CoordinateError: The primitives do not span all internal motions.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        positions: Array,
        primitives: Primitives | None = None,
        tric: bool = False,
    ) -> None:
        """Build primitives and the delocalized basis at ``positions``."""
        x = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.symbols = list(symbols)
        self.tric = tric
        self.prims = (
            primitives
            if primitives is not None
            else build_primitives(symbols, x, tric=tric)
        )
        b = self._b_prim(x)
        vals, vecs = np.linalg.eigh(b @ b.T)
        keep = vals > 1e-6
        n = len(x)
        expected = (
            3 * n
            if tric
            else (3 * n - (5 if is_linear(x) else 6) if n > 2 else (1 if n == 2 else 0))
        )
        if int(keep.sum()) < expected:
            raise CoordinateError(
                f"primitives span {int(keep.sum())} of {expected} internal motions"
            )
        self.u = np.ascontiguousarray(vecs[:, keep])
        # Rigid motions projected out of the DLC B matrix: when this changes (a linear
        # molecule bends), the coordinates no longer span the internal motions.
        self.n_rigid = rigid_basis(x).shape[1]
        self.q_ref = self.prims.values(x)
        self.n = self.u.shape[1]
        # A step evaluates the same geometry several times (its start, the gradient
        # and rigid-motion directions at its end); keep the last few results.
        self._cache: dict[tuple[str, bytes], Array] = {}

    def _cached(self, kind: str, x: Array, compute: Any) -> Array:
        xx = np.ascontiguousarray(x, dtype=float).reshape(-1, 3)
        key = (kind, xx.tobytes())
        hit = self._cache.get(key)
        if hit is None:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            hit = self._cache[key] = compute(xx)
            hit.setflags(write=False)  # shared between callers
        return hit

    def primitive_values(self, x: Array) -> Array:
        """Primitives at ``x``, torsions unwrapped relative to the reference."""

        def compute(xx: Array) -> Array:
            q = self.prims.values(xx)
            return self.q_ref + self.prims.difference(q, self.q_ref)

        return self._cached("q", x, compute)

    def values(self, x: Array) -> Array:
        """Delocalized internals ``s = U^T q`` at ``x``."""
        return self.u.T @ self.primitive_values(x)

    def _b_prim(self, x: Array) -> Array:
        """Primitive B matrix; without TRIC, with rigid motion projected out.

        Linear bends are measured against fixed directions, so they change under
        rotation; projecting keeps every DLC an internal motion (exactly 3N - 6).
        """
        b = self.prims.b_matrix(x)
        if self.tric:
            return b
        rb = rigid_basis(np.asarray(x, dtype=float).reshape(-1, 3))
        return b - (b @ rb) @ rb.T

    def b_matrix(self, x: Array) -> Array:
        """``ds/dx``, ``(n, 3 n_atoms)``."""
        return self._cached("b", x, lambda xx: self.u.T @ self._b_prim(xx))

    def gradient(self, x: Array, gradient: Array) -> Array:
        """Internal gradient from a Cartesian one: ``(B B^T)^-1 B g``."""
        b = self.b_matrix(x)
        return np.linalg.solve(b @ b.T, b @ np.asarray(gradient, dtype=float).ravel())

    def hessian(self, x: Array, hessian: Array, gradient: Array | None = None) -> Array:
        """Internal Hessian from a Cartesian one.

        With the Cartesian gradient given, the curvature of the coordinates
        themselves (``sum_i g_i d2 s_i / dx2``, from finite differences of B) is
        removed first, so the result is exact away from stationary points too.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 3)
        b = self.b_matrix(x)
        g_inv = np.linalg.pinv(b @ b.T)  # degenerate coordinates are rebuilt later
        b_pinv = b.T @ g_inv  # (3N, n)
        h = np.asarray(hessian, dtype=float)
        if gradient is not None:
            gs = g_inv @ (b @ np.asarray(gradient, dtype=float).ravel())
            h = h - self._k_term(x, gs)
        hs = b_pinv.T @ h @ b_pinv
        return 0.5 * (hs + hs.T)

    def to_cartesian_hessian(self, x: Array, hs: Array) -> Array:
        """``B^T H_s B``: the Cartesian Hessian of an internal one (no K term)."""
        b = self.b_matrix(x)
        return b.T @ hs @ b

    def _k_term(self, x: Array, gs: Array, h: float = 1e-4) -> Array:
        flat = x.ravel()
        cols = []
        for j in range(flat.size):
            y = flat.copy()
            y[j] += h
            plus = self.b_matrix(y.reshape(-1, 3)).T @ gs
            y[j] -= 2 * h
            minus = self.b_matrix(y.reshape(-1, 3)).T @ gs
            cols.append((plus - minus) / (2 * h))
        k = np.array(cols).T
        return 0.5 * (k + k.T)

    def step(
        self, x: Array, ds: Array, *, max_iter: int = 50, tol: float = 1e-6
    ) -> tuple[Array, Array, bool]:
        """Cartesian geometry for the internal step ``ds`` from ``x``.

        Returns:
            The new geometry, the internal step actually taken, and whether the
            back-transformation converged to ``tol`` (bohr, largest change).
        """
        x0 = np.asarray(x, dtype=float).reshape(-1, 3)
        s0 = self.values(x0)
        target = s0 + np.asarray(ds, dtype=float)
        xk = x0.copy()
        best = (np.inf, x0, s0)
        b = self.b_matrix(x0)
        solve = np.linalg.cholesky(b @ b.T)
        fresh = 0  # iterations since B was last computed
        prev = np.inf
        for _ in range(max_iter):
            sk = self.values(xk)
            err = target - sk
            nerr = float(np.linalg.norm(err))
            if nerr < best[0]:
                best = (nerr, xk.copy(), sk)
            if nerr > 0.5 * prev and fresh > 0:
                # converging slowly with a stale B: recompute it here
                b = self.b_matrix(xk)
                solve = np.linalg.cholesky(b @ b.T)
                fresh = 0
            y = np.linalg.solve(solve, err)
            dx = b.T @ np.linalg.solve(solve.T, y)
            xk = xk + dx.reshape(-1, 3)
            fresh += 1
            prev = nerr
            if np.abs(dx).max() < tol:
                sk = self.values(xk)
                return xk, sk - s0, True
            if nerr > 1e3 * max(best[0], 1e-12) and nerr > 1.0:
                break
        _, xb, sb = best
        if best[0] < 1e-5 * max(1.0, float(np.linalg.norm(ds))):
            return xb, sb - s0, True
        return xb, sb - s0, False


def _is_linear(x: Array, tol: float = 1e-6) -> bool:
    """Whether the points lie on a line, within ``tol`` (ratio of spreads)."""
    if len(x) < 3:
        return True
    c = x - x.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    return bool(sv[1] < tol * sv[0])
