import numpy as np
from networkgeometry.geometry.linalg import source_pcs
from networkgeometry.subspace.auc import cumulative_variance_curve, auc_from_curve

def identity_shuffle_null(A_source, A_target, n_perm, rng, centering="mean") -> np.ndarray:
    u = source_pcs(A_source, centering)
    out = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = A_target[rng.permutation(A_target.shape[0])]
        out[i] = auc_from_curve(cumulative_variance_curve(u, shuffled, centering))
    return out

def p_value(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))

def bonferroni(pvals: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(pvals, dtype=float) * len(pvals), 1.0)

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank, idx in enumerate(reversed(order)):
        k = n - rank
        running = min(running, p[idx] * n / k)
        adjusted[idx] = running
    return adjusted

def gate(within_p_by_layer: dict, alpha: float = 0.05) -> set:
    return {layer for layer, pval in within_p_by_layer.items() if pval < alpha}
