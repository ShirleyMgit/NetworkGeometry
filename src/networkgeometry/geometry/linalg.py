import numpy as np

def mean_center(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    if centering == "none":
        return A
    if centering != "mean":
        raise ValueError(f"unknown centering {centering!r}")
    return A - A.mean(axis=1, keepdims=True)

def source_pcs(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    # full_matrices=True (not the economy SVD) is deliberate: it makes U span the
    # complete d-dimensional ambient space rather than being capped at n_states
    # columns. This matters for cross_auc's chance-level behavior. Projecting an
    # unrelated target onto a COMPLETE orthonormal basis is guaranteed to reach
    # 100% cumulative variance by the last component, spread roughly evenly when
    # source and target are truly unrelated -- so an uninformative/orthogonal
    # source now correctly yields AUC near the classical chance level (~0.5).
    # With the economy SVD (full_matrices=False), U is only (d, n_states) wide;
    # once n_states << d (our real regime -- e.g. Gemma 2 2B has d=2304 against
    # 7-30 cycle states), the missing d - n_states columns are an arbitrary,
    # only-partially-uninformative completion, and chance-level AUC breaks down
    # (verified numerically: ~0.07 instead of ~0.5 at n_states=7, d=48).
    # Tradeoff: this is O(d) in curve length instead of O(n_states), so it costs
    # more compute. This is a v1 choice and may be revisited in a later version,
    # e.g. by truncating to a smaller common k shared across compared structures.
    centered = mean_center(A, centering)
    U, _s, _vt = np.linalg.svd(centered, full_matrices=True)
    return U

def state_gram(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    centered = mean_center(A, centering)
    return centered.T @ centered

def state_correlation(A: np.ndarray) -> np.ndarray:
    """n_states x n_states Pearson correlation matrix between state vectors.

    `A` is (d, n_states) with columns = states. Entry (i, j) is the Pearson
    correlation coefficient between state i's and state j's activation vectors,
    computed across the d feature dimensions (each column centered by its own
    feature-mean and scaled by its own feature-std). Diagonal is 1 by
    construction; the matrix is symmetric with entries in [-1, 1]. This is the
    magnitude-normalized companion of `state_gram`, more legible as a heatmap
    (fixed scale, unit diagonal). For a cyclic structure it shows circulant
    banding: each state most correlated with its cyclic neighbours.
    """
    return np.corrcoef(A, rowvar=False)

def is_toeplitz(G: np.ndarray, atol: float = 1e-8) -> bool:
    n = G.shape[0]
    for offset in range(-n + 1, n):
        diag = np.diagonal(G, offset)
        if not np.allclose(diag, diag[0], atol=atol):
            return False
    return True
