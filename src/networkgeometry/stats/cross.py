from dataclasses import dataclass
import numpy as np
from networkgeometry.geometry.linalg import source_pcs
from networkgeometry.subspace.auc import cumulative_variance_curve, auc_from_curve

@dataclass(frozen=True)
class CrossResult:
    mean: float
    sem: float
    per_run: np.ndarray

def cross_structure_auc(source_runs, target_runs, centering: str = "mean") -> CrossResult:
    u = source_pcs(np.mean(source_runs, axis=0), centering)
    per_run = np.array(
        [auc_from_curve(cumulative_variance_curve(u, t, centering)) for t in target_runs]
    )
    sem = float(np.std(per_run, ddof=1) / np.sqrt(len(per_run))) if len(per_run) > 1 else 0.0
    return CrossResult(float(np.mean(per_run)), sem, per_run)
