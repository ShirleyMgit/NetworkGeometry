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

def _circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

def circular_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = np.sin(a)
    b0 = np.sin(b)
    denom = np.sqrt(np.sum(a0**2) * np.sum(b0**2))
    return float(np.sum(a0 * b0) / denom) if denom else 0.0

def angular_order_score(points, canonical_index, n_states) -> float:
    target = 2 * np.pi * np.asarray(canonical_index) / n_states
    return abs(circular_correlation(fit_circle(points).angles, target))
