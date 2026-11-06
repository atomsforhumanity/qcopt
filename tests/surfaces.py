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
