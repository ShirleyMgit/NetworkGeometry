import numpy as np

def ring_matrix(d=64, n_states=12, noise=0.0, rng=None, embed_rot=True):
    """(d, n_states) whose columns lie on a circle in a 2D plane of R^d."""
    rng = rng or np.random.default_rng(0)
    theta = 2 * np.pi * np.arange(n_states) / n_states
    plane = np.zeros((d, n_states))
    plane[0] = np.cos(theta)
    plane[1] = np.sin(theta)
    if embed_rot:
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        plane = q @ plane
    return plane + noise * rng.standard_normal((d, n_states))

def shared_subspace_pair(d=64, n_a=7, n_b=12, rng=None):
    """Two ring matrices sharing the SAME 2D plane (different sizes)."""
    rng = rng or np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    def ring(n):
        t = 2 * np.pi * np.arange(n) / n
        m = np.zeros((d, n)); m[0] = np.cos(t); m[1] = np.sin(t)
        return q @ m
    return ring(n_a), ring(n_b)

def orthogonal_subspace_pair(d=64, n_a=7, n_b=12, rng=None):
    """Two rings in ORTHOGONAL planes of R^d."""
    rng = rng or np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    def ring(n, i, j):
        t = 2 * np.pi * np.arange(n) / n
        m = np.zeros((d, n)); m[i] = np.cos(t); m[j] = np.sin(t)
        return q @ m
    return ring(n_a, 0, 1), ring(n_b, 2, 3)
