import numpy as np
from networkgeometry.analysis.stage2 import run_stage2_ladder
from tests.fixtures import shared_subspace_pair, orthogonal_subspace_pair


def _runs(matrix, n=5, jitter=0.02, seed=9):
    rng = np.random.default_rng(seed)
    return [matrix + jitter * rng.standard_normal(matrix.shape) for _ in range(n)]


def _controls(d=48, n_years=10, n_states=12):
    _, years = shared_subspace_pair(d=d, n_a=7, n_b=n_years, rng=np.random.default_rng(3))
    _, hierarchy = orthogonal_subspace_pair(d=d, n_a=7, n_b=n_states)
    _, flat = orthogonal_subspace_pair(d=d, n_a=7, n_b=n_states, rng=np.random.default_rng(4))
    return years, hierarchy, flat


EXPECTED_LABELS = {
    "day (within)", "month (within)",
    "day -> month (matched)", "month -> day (matched)",
    "day -> month (specific)", "month -> day (specific)",
    "day -> years", "day -> hierarchy", "day -> flat",
}


def test_stage2_ladder_produces_all_comparison_rows_at_gate_passing_layer():
    day, month = shared_subspace_pair(d=48, n_a=7, n_b=12)
    years, hierarchy, flat = _controls()

    shared_runs = {
        ("day", 5): _runs(day), ("month", 5): _runs(month),
        ("years", 5): _runs(years), ("hierarchy", 5): _runs(hierarchy), ("flat", 5): _runs(flat),
    }
    specific_runs = {
        ("day", 5): _runs(day, seed=11), ("month", 5): _runs(month, seed=12),
    }

    rows = run_stage2_ladder(shared_runs, specific_runs, layers=[5])
    labels = {r.label for r in rows}
    assert labels == EXPECTED_LABELS

    by_label = {r.label: r for r in rows}
    assert by_label["day -> month (matched)"].auc > by_label["day -> flat"].auc
    assert by_label["day -> month (matched)"].fdr_p is not None
    assert by_label["day -> month (matched)"].bonferroni_p is not None
    assert by_label["day (within)"].fdr_p is None
    assert by_label["day (within)"].bonferroni_p is None


def test_stage2_ladder_gates_each_source_independently():
    _, month = shared_subspace_pair(d=48, n_a=7, n_b=12)
    day = month  # placeholder, overwritten by noise below for the failing-gate case
    years, hierarchy, flat = _controls()

    def _noise_runs(d=48, n_states=7, n=5, seed=12):
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((d, n_states)) for _ in range(n)]

    shared_runs = {
        ("day", 3): _noise_runs(),          # noise source -> day fails its own gate at layer 3
        ("month", 3): _runs(month),         # structured -> month passes its own gate
        ("years", 3): _runs(years), ("hierarchy", 3): _runs(hierarchy), ("flat", 3): _runs(flat),
    }
    specific_runs = {("day", 3): _noise_runs(seed=13), ("month", 3): _runs(month, seed=14)}

    rows = run_stage2_ladder(shared_runs, specific_runs, layers=[3])
    labels = {r.label for r in rows}

    assert "day (within)" in labels
    assert "month (within)" in labels
    assert "day -> years" not in labels
    assert "day -> hierarchy" not in labels
    assert "day -> flat" not in labels
    assert "day -> month (matched)" not in labels
    assert "day -> month (specific)" not in labels
    assert "month -> day (matched)" in labels
    assert "month -> day (specific)" in labels


def test_stage2_ladder_without_specific_pool_emits_single_cross_rows():
    day, month = shared_subspace_pair(d=48, n_a=7, n_b=12)
    years, hierarchy, flat = _controls()
    probe_runs = {
        ("day", 5): _runs(day), ("month", 5): _runs(month),
        ("years", 5): _runs(years), ("hierarchy", 5): _runs(hierarchy), ("flat", 5): _runs(flat),
    }

    rows = run_stage2_ladder(probe_runs, None, layers=[5])
    labels = {r.label for r in rows}

    assert labels == {
        "day (within)", "month (within)",
        "day -> month", "month -> day",
        "day -> years", "day -> hierarchy", "day -> flat",
    }
    assert not any("(matched)" in lbl or "(specific)" in lbl for lbl in labels)
    by_label = {r.label: r for r in rows}
    assert by_label["day -> month"].fdr_p is not None
    assert by_label["day -> month"].bonferroni_p is not None
