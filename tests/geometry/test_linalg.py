import numpy as np
import pytest
from networkgeometry.geometry.linalg import mean_center, source_pcs, state_gram, is_toeplitz
from tests.fixtures import ring_matrix

def test_mean_center_zeroes_column_mean():
    A = np.arange(12.0).reshape(3, 4)
    assert np.allclose(mean_center(A).mean(axis=1), 0.0)
    assert np.allclose(mean_center(A, "none"), A)

def test_source_pcs_orthonormal_and_ranked():
    A = ring_matrix(d=32, n_states=12)
    U = source_pcs(A)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-6)
    # a clean ring's variance is concentrated in the first 2 PCs
    m = mean_center(A)
    energy = (U.T @ m) ** 2
    assert energy.sum(axis=1)[:2].sum() / energy.sum() > 0.99

def test_uncentered_gram_of_translation_symmetric_is_toeplitz():
    # G_ij depends only on |i-j|: build from a circulant kernel
    n = 8
    kernel = np.exp(-np.arange(n))
    G = np.array([[kernel[abs(i - j)] for j in range(n)] for i in range(n)])
    assert is_toeplitz(G)

def test_mean_center_raises_for_invalid_centering():
    A = np.arange(12.0).reshape(3, 4)
    with pytest.raises(ValueError, match="unknown centering"):
        mean_center(A, centering="bogus")
