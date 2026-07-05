from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CircleFit:
    cx: float
    cy: float
    r: float
    normalized_residual: float
    angles: np.ndarray

def fit_circle(points: np.ndarray) -> CircleFit:
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack([x, y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, x**2 + y**2, rcond=None)
    cx, cy = solution[0] / 2, solution[1] / 2
    r = float(np.sqrt(solution[2] + cx**2 + cy**2))
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    normalized_residual = float(np.sqrt(np.mean((distances - r) ** 2)) / r)
    angles = np.arctan2(y - cy, x - cx)
    return CircleFit(float(cx), float(cy), r, normalized_residual, angles)

def circular_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation-invariant alignment between two angle series, in [0, 1].

    Uses the mean resultant length of pairwise angle differences (a
    phase-locking-value style statistic). This is exactly invariant to a
    global rotation applied to either series: shifting every element of
    `a` by a constant c multiplies every pairwise-difference unit vector
    by exp(ic), which does not change the magnitude of the mean.

    The naive circular-mean-centered formula (subtract each series' own
    circular mean before correlating) is NOT safe for a full, evenly-spaced
    ring: the resultant vector (mean cos, mean sin) is exactly zero by
    symmetry for any rotation, so the "circular mean" is a degenerate
    atan2(~0, ~0) that does not track the rotation and does not achieve
    invariance. That degeneracy is exactly this project's primary case
    (7-day, 12-month full cycles), which is why this formula is used
    instead.
    """
    diff = a - b
    return float(np.abs(np.mean(np.exp(1j * diff))))


def angular_order_score(points, canonical_index, n_states) -> float:
    """Rotation- and reflection-invariant score for whether fitted circle
    angles follow the states' canonical (e.g. calendar) order, in [0, 1].

    Reflection invariance (accepting a mirror-reversed traversal direction
    as equivalent) is intentional: PCA/SVD singular vectors have an
    arbitrary sign convention, so a fitted circle can come out mirrored
    for reasons unrelated to whether the underlying cyclic structure is
    correctly represented.
    """
    target = 2 * np.pi * np.asarray(canonical_index) / n_states
    angles = fit_circle(points).angles
    forward = circular_correlation(angles, target)
    mirrored = circular_correlation(-angles, target)
    return max(forward, mirrored)
