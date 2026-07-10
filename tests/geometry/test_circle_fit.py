import numpy as np
from networkgeometry.geometry.circle_fit import fit_circle, circular_correlation, angular_order_score

def _circle_points(n, order):
    t = 2 * np.pi * np.asarray(order) / n
    return np.column_stack([np.cos(t), np.sin(t)])

def test_perfect_circle_low_residual():
    pts = _circle_points(12, range(12))
    fit = fit_circle(pts)
    assert fit.normalized_residual < 1e-6
    assert abs(fit.r - 1.0) < 1e-6

def test_angular_order_high_when_in_calendar_order():
    idx = np.arange(12)
    pts = _circle_points(12, idx)
    assert angular_order_score(pts, idx, 12) > 0.99

def test_angular_order_low_when_scrambled():
    idx = np.arange(12)
    scrambled = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
    pts = _circle_points(12, scrambled)
    assert angular_order_score(pts, idx, 12) < 0.5

def test_angular_order_invariant_to_rotation():
    n = 12
    idx = np.arange(n)
    pts = _circle_points(n, idx)
    fit_angles = np.arctan2(pts[:, 1], pts[:, 0])
    for phase_offset in [0, 0.3, np.pi / 2, np.pi, 4.5]:
        rotated_pts = np.column_stack(
            [np.cos(fit_angles + phase_offset), np.sin(fit_angles + phase_offset)]
        )
        assert angular_order_score(rotated_pts, idx, n) > 0.99

def test_angular_order_invariant_to_reflection():
    n = 12
    idx = np.arange(n)
    reversed_order = (n - idx) % n
    pts = _circle_points(n, reversed_order)
    assert angular_order_score(pts, idx, n) > 0.99
