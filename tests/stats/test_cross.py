import numpy as np
from networkgeometry.stats.cross import cross_structure_auc
from tests.fixtures import shared_subspace_pair, orthogonal_subspace_pair

def _runs_from(matrix, n_runs=5, jitter=0.02):
    rng = np.random.default_rng(5)
    return [matrix + jitter * rng.standard_normal(matrix.shape) for _ in range(n_runs)]

def test_cross_high_for_shared_subspace():
    a, b = shared_subspace_pair(d=48, n_a=7, n_b=12)
    res = cross_structure_auc(_runs_from(a), _runs_from(b))
    assert res.mean > 0.8
    assert res.per_run.shape == (5,)

def test_cross_chance_for_orthogonal():
    a, b = orthogonal_subspace_pair(d=48, n_a=7, n_b=12)
    assert abs(cross_structure_auc(_runs_from(a), _runs_from(b)).mean - 0.5) < 0.15
