from dataclasses import dataclass
import numpy as np
from networkgeometry.types import group_runs
from networkgeometry.geometry.linalg import mean_center, source_pcs
from networkgeometry.geometry.circle_fit import fit_circle, angular_order_score


@dataclass(frozen=True)
class LayerCircularity:
    layer: int
    normalized_residual: float
    angular_order: float
    top2_variance_ratio: float


def circularity_by_layer(dms_by_layer: dict, excluded: tuple = (), centering: str = "mean") -> list:
    results = []
    for layer in sorted(dms_by_layer):
        dms = dms_by_layer[layer]
        all_states = sorted(dms[0].states, key=lambda s: s.canonical_index)
        keep_mask = np.array([s.label not in excluded for s in all_states])

        mean_matrix = np.mean(group_runs(dms), axis=0)          # (d, n_states), same order as all_states
        basis_matrix = mean_matrix[:, keep_mask]
        canonical = np.array([s.canonical_index for s, keep in zip(all_states, keep_mask) if keep])

        centered = mean_center(basis_matrix, centering)
        u = source_pcs(basis_matrix, centering)
        scores = (u[:, :2].T @ centered).T                      # (n_kept_states, 2)
        energy = np.sum((u.T @ centered) ** 2, axis=1)
        ratio = float(energy[:2].sum() / energy.sum())
        fit = fit_circle(scores)
        order = angular_order_score(scores, canonical, len(canonical))
        results.append(LayerCircularity(layer, fit.normalized_residual, order, ratio))
    return results
