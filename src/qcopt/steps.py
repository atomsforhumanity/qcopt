"""Quasi-Newton steps in the eigenbasis of a Hessian: RS-RFO and RS-P-RFO.

The rational-function step for a given scaling ``alpha`` is a shifted Newton step,
``s_i = -g_i / (h_i - lambda)``, with one shift for the modes being minimized
(below their lowest eigenvalue) and, for a transition state, another for the mode
being maximized (above its eigenvalue). ``alpha = 1`` is plain (P-)RFO. A step longer
than the trust radius is shortened by raising ``alpha`` until it fits (Besalu and
Bofill, Theor. Chem. Acc. 100, 265 (1998)), which keeps its direction meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Array

_TINY = 1e-300


@dataclass
class Step:
    """A step in the eigenbasis and in the space it was computed in.

    Attributes:
        dx: The step.
        predicted: Energy change predicted by the quadratic model (Hartree).
        mode: Index of the maximized eigenvector (transition states), else None.
        alpha: The RFO scaling used (1 when the trust radius was not reached).
        restricted: Whether the trust radius shortened the step.
    """

    dx: Array
    predicted: float
    mode: int | None
    alpha: float
    restricted: bool


def _shift_below(d: Array, g2: Array, hmin: float, alpha: float) -> float:
    """Return mu > 0 with lambda = hmin - mu solving lambda = alpha sum g2/(lambda - h).

    ``d`` holds ``h - hmin`` (>= 0). F(mu) = hmin - mu + alpha sum g2/(d + mu) decreases
    from +inf to -inf; safeguarded Newton on log(mu).
    """
    total = float(g2.sum())
    if total <= _TINY:
        return 1.0

    def f(mu: float) -> tuple[float, float]:
        q = g2 / (d + mu)
        return (
            hmin - mu + alpha * float(q.sum()),
            -1.0 - alpha * float((q / (d + mu)).sum()),
        )

    lo, hi = 1e-100, max(0.0, hmin) + np.sqrt(alpha * total) + 1.0
    if f(lo)[0] <= 0.0:
        return lo
    mu = float(np.clip(max(hmin, 0.0) + 1e-3, lo * 10, hi))
    for _ in range(200):
        val, der = f(mu)
        if val > 0:
            lo = mu
        else:
            hi = mu
        if abs(val) <= 1e-14 * max(1.0, abs(hmin) + mu) or hi - lo <= 1e-15 * hi:
            break
        new = mu - val / der
        if not (lo < new < hi):
            new = np.sqrt(lo * hi) if hi / lo > 10.0 else 0.5 * (lo + hi)
        mu = float(new)
    return mu


def _above_denominator(h: float, g2: float, alpha: float) -> float:
    """``h - lambda`` for the highest root of lambda = alpha g2 / (lambda - h).

    The maximized mode's denominator, in a form without cancellation: for h > 0
    and a tiny gradient the textbook root 0.5 (h + sqrt(h^2 + 4 alpha g2)) equals h
    to machine precision, and the step loses its component along the mode.
    """
    root = float(np.sqrt(h * h + 4.0 * alpha * g2))
    if h > 0:
        return -2.0 * alpha * g2 / (h + root)
    return 0.5 * (h - root)


def _rfo(vals: Array, gt: Array, alpha: float, mode: int | None) -> tuple[Array, float]:
    """Step components in the eigenbasis for one ``alpha``; also lambda_min."""
    s = np.zeros_like(gt)
    keep = np.ones(len(vals), dtype=bool)
    if mode is not None:
        keep[mode] = False
        denom = _above_denominator(float(vals[mode]), float(gt[mode] ** 2), alpha)
        s[mode] = -gt[mode] / denom if abs(denom) > _TINY else 0.0
    v, g = vals[keep], gt[keep]
    if v.size:
        hmin = float(np.min(v))
        mu = _shift_below(v - hmin, g * g, hmin, alpha)
        s[keep] = -g / ((v - hmin) + mu)
        return s, hmin - mu
    return s, 0.0


def rfo_step(
    vals: Array,
    vecs: Array,
    gradient: Array,
    trust: float,
    *,
    mode: int | None = None,
    atomic: bool = False,
    max_alpha: float = 1e12,
) -> Step:
    """RS-RFO (``mode=None``) or RS-P-RFO step within ``trust``.

    Args:
        vals: Hessian eigenvalues, ascending.
        vecs: Eigenvectors as columns.
        gradient: Gradient in the same space.
        trust: Largest allowed step: its norm, or with ``atomic`` its largest
            per-atom displacement (the space must then be Cartesian, 3 per atom).
        mode: Eigenvector to maximize along (a transition-state search).
        atomic: Measure the step by its largest atomic displacement.
        max_alpha: Largest RFO scaling tried before the step is simply scaled.
    """
    gt = vecs.T @ gradient

    def size(s_eig: Array) -> float:
        if atomic:
            d = (vecs @ s_eig).reshape(-1, 3)
            return float(np.sqrt((d * d).sum(axis=1)).max())
        return float(np.linalg.norm(s_eig))

    s, _ = _rfo(vals, gt, 1.0, mode)
    n1 = size(s)
    alpha, restricted = 1.0, False
    if n1 > trust:
        restricted = True
        # Regula falsi (Illinois) on log(size) - log(trust) over log(alpha).
        u_lo, f_lo = 0.0, np.log(n1 / trust)
        u_hi = float(np.log(max_alpha))
        s_hi, _ = _rfo(vals, gt, max_alpha, mode)
        n_hi = size(s_hi)
        if n_hi >= trust:
            s, alpha = s_hi * (trust / n_hi), max_alpha
        else:
            f_hi = np.log(max(n_hi, 1e-300) / trust)
            s, side = s_hi, 0
            for _ in range(60):
                u = (u_lo * f_hi - u_hi * f_lo) / (f_hi - f_lo)
                s_u, _ = _rfo(vals, gt, float(np.exp(u)), mode)
                f_u = np.log(max(size(s_u), 1e-300) / trust)
                if f_u > 0:
                    u_lo, f_lo = u, f_u
                    if side == -1:
                        f_hi *= 0.5
                    side = -1
                else:
                    u_hi, f_hi, s = u, f_u, s_u
                    if side == 1:
                        f_lo *= 0.5
                    side = 1
                if abs(f_u) < 1e-4:
                    s = s_u
                    break
            alpha = float(np.exp(u_hi))
            n = size(s)
            if n > trust:
                s = s * (trust / n)
    predicted = float(gt @ s + 0.5 * (vals * s * s).sum())
    return Step(
        dx=vecs @ s, predicted=predicted, mode=mode, alpha=alpha, restricted=restricted
    )


def select_mode(
    vecs: Array, vals: Array, reference: Array | None, n_lowest: int = 0
) -> int:
    """The eigenvector to follow uphill: most overlap with ``reference``, else lowest.

    Args:
        vecs: Eigenvectors as columns, eigenvalues ascending.
        vals: The eigenvalues.
        reference: The previous mode or a guide direction (same space).
        n_lowest: Only the lowest ``n_lowest`` eigenvectors are candidates (0: all).
    """
    if reference is None:
        return 0
    r = np.asarray(reference, dtype=float).ravel()
    nr = np.linalg.norm(r)
    if nr < 1e-12:
        return 0
    cand = vecs if not n_lowest else vecs[:, :n_lowest]
    return int(np.argmax(np.abs(cand.T @ (r / nr))))


def update_trust(
    trust: float,
    ratio: float,
    step_size: float,
    *,
    minimum: float,
    maximum: float,
    ts: bool = False,
) -> float:
    """New trust radius from the ratio of actual to predicted energy change.

    Minimizations: shrink below a ratio of 0.25, grow above 0.75 when the step used
    the trust radius. Transition states (where the energy is not minimized, so
    the ratio only says how good the quadratic model was): shrink to 0.65 of the
    step when the ratio leaves [0.2, 5], grow to 1.15 of the step when it is
    within 3.5% of 1 (the scheme of Sella, Hermes et al., JCTC 2022).
    """
    if ts:
        if ratio < 0.2 or ratio > 5.0:
            trust = 0.65 * step_size
        elif abs(ratio - 1.0) < 0.035:
            trust = max(1.15 * step_size, trust)
    elif ratio < 0.25:
        trust = 0.25 * min(trust, step_size)
    elif ratio > 0.75 and step_size > 0.8 * trust:
        trust = 2.0 * trust
    return float(min(maximum, max(minimum, trust)))
