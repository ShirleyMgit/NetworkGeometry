import numpy as np
from networkgeometry.sae.gemma_scope import cluster_decoder, cluster_circle_score

def test_cluster_groups_aligned_columns():
    base = np.eye(6)[:, :2]                       # two orthogonal directions
    cluster_a = base[:, [0]] + 0.01 * np.random.default_rng(0).standard_normal((6, 4))
    cluster_b = base[:, [1]] + 0.01 * np.random.default_rng(1).standard_normal((6, 4))
    decoder = np.hstack([cluster_a, cluster_b])
    clusters = cluster_decoder(decoder, threshold=0.9)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [4, 4]

def test_circle_score_low_for_ring():
    from tests.fixtures import ring_matrix
    recon = ring_matrix(d=20, n_states=16).T       # (n_points, d)
    assert cluster_circle_score(recon) < 0.1
