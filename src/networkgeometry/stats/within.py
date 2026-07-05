from dataclasses import dataclass
import numpy as np
from networkgeometry.geometry.linalg import mean_center
from networkgeometry.subspace.auc import cumulative_variance_curve, auc_from_curve

@dataclass(frozen=True)
class WithinResult:
    mean: float
    sem: float
    per_fold: np.ndarray

def _held_in_mean(runs, held_out):
    kept = [r for i, r in enumerate(runs) if i != held_out]
    return np.mean(kept, axis=0)

def _summarize(per_fold) -> WithinResult:
    per_fold = np.asarray(per_fold, dtype=float)
    sem = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0
    return WithinResult(float(np.mean(per_fold)), sem, per_fold)

def within_structure_auc(runs, centering: str = "mean") -> WithinResult:
    from networkgeometry.geometry.linalg import source_pcs
    folds = []
    for j in range(len(runs)):
        u = source_pcs(_held_in_mean(runs, j), centering)
        folds.append(auc_from_curve(cumulative_variance_curve(u, runs[j], centering)))
    return _summarize(folds)

def v_side_stability(runs, centering: str = "mean") -> WithinResult:
    folds = []
    for j in range(len(runs)):
        held_in = mean_center(_held_in_mean(runs, j), centering)
        _u, _s, vt = np.linalg.svd(held_in, full_matrices=False)
        target = mean_center(runs[j], centering)
        total = float(np.sum(target**2)) or 1.0
        per_mode = np.sum((vt @ target.T) ** 2, axis=1)   # project states onto held-in state modes
        folds.append(auc_from_curve(np.cumsum(per_mode) / total))
    return _summarize(folds)
