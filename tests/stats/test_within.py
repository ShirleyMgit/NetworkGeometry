import numpy as np
from networkgeometry.stats.within import within_structure_auc, v_side_stability
from tests.fixtures import ring_matrix

def _runs(n_runs=6, d=48, n_states=12, jitter=0.02):
    rng = np.random.default_rng(3)
    base = ring_matrix(d=d, n_states=n_states)
    return [base + jitter * rng.standard_normal(base.shape) for _ in range(n_runs)]

def test_within_high_for_consistent_runs():
    res = within_structure_auc(_runs())
    assert res.mean > 0.85
    assert res.per_fold.shape == (6,)
    assert res.sem >= 0.0

def test_within_chance_for_noise_runs():
    rng = np.random.default_rng(4)
    runs = [rng.standard_normal((48, 12)) for _ in range(6)]
    assert within_structure_auc(runs).mean < 0.7

def test_v_side_stability_high_for_consistent_runs():
    assert v_side_stability(_runs()).mean > 0.8
