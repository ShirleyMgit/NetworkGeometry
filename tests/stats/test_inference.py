import numpy as np
from networkgeometry.stats.inference import (
    identity_shuffle_null, p_value, bonferroni, benjamini_hochberg, gate)
from tests.fixtures import shared_subspace_pair

def test_null_centers_near_chance_and_observed_is_significant():
    a, b = shared_subspace_pair(d=48, n_a=7, n_b=12)
    rng = np.random.default_rng(7)
    null = identity_shuffle_null(a, b, n_perm=200, rng=rng)
    assert abs(np.mean(null) - 0.5) < 0.15
    from networkgeometry.subspace.auc import cross_auc
    assert p_value(cross_auc(a, b), null) < 0.05

def test_bonferroni_and_bh_bounds():
    p = np.array([0.01, 0.02, 0.5])
    assert np.allclose(bonferroni(p), [0.03, 0.06, 1.0])
    assert np.all(benjamini_hochberg(p) >= p)

def test_gate_selects_significant_layers():
    assert gate({0: 0.5, 5: 0.001, 9: 0.2}, alpha=0.05) == {5}

def test_benjamini_hochberg_hand_derived_case():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    expected = np.array([0.02, 0.04, 0.04, 0.02])
    assert np.allclose(benjamini_hochberg(p), expected)
