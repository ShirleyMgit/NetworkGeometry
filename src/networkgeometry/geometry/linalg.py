import numpy as np

def mean_center(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    if centering == "none":
        return A
    if centering != "mean":
        raise ValueError(f"unknown centering {centering!r}")
    return A - A.mean(axis=1, keepdims=True)

def source_pcs(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    centered = mean_center(A, centering)
    U, _s, _vt = np.linalg.svd(centered, full_matrices=False)
    return U

def state_gram(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    centered = mean_center(A, centering)
    return centered.T @ centered

def is_toeplitz(G: np.ndarray, atol: float = 1e-8) -> bool:
    n = G.shape[0]
    for offset in range(-n + 1, n):
        diag = np.diagonal(G, offset)
        if not np.allclose(diag, diag[0], atol=atol):
            return False
    return True
