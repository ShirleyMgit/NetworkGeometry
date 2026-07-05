import numpy as np
from scipy.integrate import trapezoid
from networkgeometry.geometry.linalg import mean_center, source_pcs

def cumulative_variance_curve(U_source, A_target, centering: str = "mean") -> np.ndarray:
    target = mean_center(A_target, centering)
    total = float(np.sum(target**2))
    if total == 0.0:
        return np.zeros(U_source.shape[1])
    projected = U_source.T @ target                 # (r, n_states_target)
    per_pc = np.sum(projected**2, axis=1)
    return np.cumsum(per_pc) / total

def auc_from_curve(cum: np.ndarray) -> float:
    y = np.concatenate([[0.0], np.asarray(cum, dtype=float)])
    x = np.linspace(0.0, 1.0, len(y))
    return float(trapezoid(y, x))

def cross_auc(A_source, A_target, centering: str = "mean") -> float:
    return auc_from_curve(
        cumulative_variance_curve(source_pcs(A_source, centering), A_target, centering)
    )
