import numpy as np
from networkgeometry.types import State, DataMatrix
from networkgeometry.geometry.part1 import circularity_by_layer, manifold_scores
from networkgeometry.geometry.circle_fit import fit_circle
from tests.fixtures import ring_matrix


def _dms_for_layer(layer, n_runs=4, n_states=12):
    states = tuple(State(f"s{i}", i + 1) for i in range(n_states))
    rng = np.random.default_rng(8)
    base = ring_matrix(d=40, n_states=n_states)
    return [DataMatrix("month", layer, r, base + 0.01 * rng.standard_normal(base.shape), states)
            for r in range(n_runs)]


def test_clean_ring_scores_high():
    result = circularity_by_layer({3: _dms_for_layer(3)})
    lc = result[0]
    assert lc.layer == 3
    assert lc.normalized_residual < 0.1
    assert lc.angular_order > 0.9
    assert lc.top2_variance_ratio > 0.9


def test_excluding_an_outlier_state_restores_high_circularity():
    # 12-state clean ring, but state "s5" is replaced with an unrelated outlier
    # vector (simulating May's polysemy). Left in, it should corrupt the score;
    # excluded from the basis, circularity should be high again.
    rng = np.random.default_rng(11)
    states = tuple(State(f"s{i}", i + 1) for i in range(12))
    base = ring_matrix(d=40, n_states=12)
    outlier_col = 10.0 * rng.standard_normal((40, 1))
    dms = []
    for r in range(4):
        matrix = (base + 0.01 * rng.standard_normal(base.shape)).copy()
        matrix[:, [5]] = outlier_col + 0.01 * rng.standard_normal((40, 1))
        dms.append(DataMatrix("month", 3, r, matrix, states))

    contaminated = circularity_by_layer({3: dms})[0]
    assert contaminated.angular_order < 0.9

    cleaned = circularity_by_layer({3: dms}, excluded=("s5",))[0]
    assert cleaned.angular_order > 0.9
    assert cleaned.top2_variance_ratio > 0.9


def test_manifold_scores_returns_ring_coords_and_kept_labels():
    labels = [f"s{i}" for i in range(12)]
    base = ring_matrix(d=40, n_states=12)
    scores, kept = manifold_scores(base, labels)
    assert scores.shape == (12, 2)
    assert kept == labels
    # a clean ring: the top-2 PC scores lie tightly on a fitted circle
    assert fit_circle(scores).normalized_residual < 0.1


def test_manifold_scores_drops_excluded_states():
    labels = [f"s{i}" for i in range(12)]
    base = ring_matrix(d=40, n_states=12)
    scores, kept = manifold_scores(base, labels, excluded=("s5",))
    assert scores.shape == (11, 2)
    assert "s5" not in kept and len(kept) == 11
