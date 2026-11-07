"""Model surfaces with known stationary points, for tests."""

import numpy as np

_A = np.array([-200.0, -100.0, -170.0, 15.0])
_a = np.array([-1.0, -1.0, -6.5, 0.7])
_b = np.array([0.0, 0.0, 11.0, 0.6])
_c = np.array([-10.0, -10.0, -6.5, 0.7])
_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_y0 = np.array([0.0, 0.5, 1.5, 1.0])
SCALE = 1e-3

MB_MINIMA = [(-0.558224, 1.441726), (0.623499, 0.028038), (-0.050011, 0.466694)]
MB_SADDLES = [(-0.822002, 0.624313), (0.212487, 0.292988)]


def muller_brown(pos):
    """Müller-Brown energy and gradient for one 'atom' at (x, y, z); z is free."""
    x, y = float(pos[0, 0]), float(pos[0, 1])
    dx, dy = x - _x0, y - _y0
    e = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    gx = (e * (2 * _a * dx + _b * dy)).sum()
    gy = (e * (_b * dx + 2 * _c * dy)).sum()
    return SCALE * e.sum(), SCALE * np.array([[gx, gy, 0.0]])


def muller_brown_hessian(pos):
    x, y = float(pos[0, 0]), float(pos[0, 1])
    dx, dy = x - _x0, y - _y0
    e = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    px = 2 * _a * dx + _b * dy
    py = _b * dx + 2 * _c * dy
    hxx = (e * (px**2 + 2 * _a)).sum()
    hyy = (e * (py**2 + 2 * _c)).sum()
    hxy = (e * (px * py + _b)).sum()
    h = np.zeros((3, 3))
    h[0, 0], h[1, 1], h[0, 1], h[1, 0] = hxx, hyy, hxy, hxy
    return SCALE * h


# A path with a shoulder: along x, a maximum at 0 (the transition state), a flat
# inflection at 1 and the minimum at 2; harmonic in y and z.
_C, _K = 0.02, 0.05


def shoulder(pos):
    """Energy and gradient of the shoulder surface for one 'atom'."""
    x, y, z = (float(v) for v in pos[0])
    e = _C * (x**5 / 5 - x**4 + 5 * x**3 / 3 - x**2) + _K * (y * y + z * z)
    gx = _C * (x**4 - 4 * x**3 + 5 * x**2 - 2 * x)
    return e, np.array([[gx, 2 * _K * y, 2 * _K * z]])


def shoulder_hessian(pos):
    x = float(pos[0, 0])
    h = np.zeros((3, 3))
    h[0, 0] = _C * (4 * x**3 - 12 * x**2 + 10 * x - 2)
    h[1, 1] = h[2, 2] = 2 * _K
    return h


def triatomic(pos, *, r0=2.0, k_r=0.5, k_theta=0.2, theta0=150.0):
    """Two harmonic bonds to the middle atom and a harmonic bend (degrees)."""
    x = np.asarray(pos, dtype=float).reshape(3, 3)
    u, v = x[0] - x[1], x[2] - x[1]
    ru, rv = np.linalg.norm(u), np.linalg.norm(v)
    cos = float(np.clip(u @ v / (ru * rv), -1.0, 1.0))
    theta = np.arccos(cos)
    t0 = np.radians(theta0)
    e = k_r * ((ru - r0) ** 2 + (rv - r0) ** 2) + k_theta * (theta - t0) ** 2
    g = np.zeros((3, 3))
    g[0] += 2 * k_r * (ru - r0) * u / ru
    g[2] += 2 * k_r * (rv - r0) * v / rv
    g[1] -= g[0] + g[2]
    sin = max(np.sqrt(1.0 - cos * cos), 1e-12)
    dcos_du = (v / (ru * rv)) - cos * u / ru**2
    dcos_dv = (u / (ru * rv)) - cos * v / rv**2
    dtheta = -2 * k_theta * (theta - t0) / sin
    g[0] += dtheta * dcos_du
    g[2] += dtheta * dcos_dv
    g[1] -= dtheta * (dcos_du + dcos_dv)
    return e, g
