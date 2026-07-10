import numpy as np
from networkgeometry.subspace.auc import cumulative_variance_curve, auc_from_curve, cross_auc
from tests.fixtures import ring_matrix, shared_subspace_pair, orthogonal_subspace_pair

def test_auc_uniform_is_half():
    assert abs(auc_from_curve(np.array([0.25, 0.5, 0.75, 1.0])) - 0.5) < 1e-9

def test_auc_concentrated_is_near_one():
    assert auc_from_curve(np.array([1.0, 1.0, 1.0, 1.0])) > 0.85

def test_self_projection_high_auc():
    A = ring_matrix(d=48, n_states=12)
    assert cross_auc(A, A) > 0.9

def test_shared_subspace_high_orthogonal_chance():
    a, b = shared_subspace_pair(d=48, n_a=7, n_b=12)
    oa, ob = orthogonal_subspace_pair(d=48, n_a=7, n_b=12)
    assert cross_auc(a, b) > 0.85
    assert abs(cross_auc(oa, ob) - 0.5) < 0.15

def test_curve_normalized_and_monotone():
    a, b = shared_subspace_pair()
    from networkgeometry.geometry.linalg import source_pcs
    cum = cumulative_variance_curve(source_pcs(a), b)
    assert cum[-1] <= 1.0 + 1e-9 and np.all(np.diff(cum) >= -1e-9)
