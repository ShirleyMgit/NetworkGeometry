import numpy as np
from networkgeometry.analysis.ladder import run_ladder, to_json
from tests.fixtures import shared_subspace_pair, orthogonal_subspace_pair

def _runs(matrix, n=5, jitter=0.02, seed=9):
    rng = np.random.default_rng(seed)
    return [matrix + jitter * rng.standard_normal(matrix.shape) for _ in range(n)]

def test_ladder_flags_shared_high_orthogonal_chance():
    day, month = shared_subspace_pair(d=48, n_a=7, n_b=12)
    _, flat = orthogonal_subspace_pair(d=48, n_a=7, n_b=12)
    runs = {("day", 5): _runs(day), ("month", 5): _runs(month), ("flat", 5): _runs(flat)}
    out = run_ladder(runs, layers=[5], source="day", targets=("month", "flat"))
    layer5 = out[0]
    assert layer5.gate_passed
    assert layer5.crosses["month"]["auc"] > layer5.crosses["flat"]["auc"]
    assert "fdr_p" in layer5.crosses["month"] and "bonferroni_p" in layer5.crosses["month"]
    assert isinstance(to_json(out), dict)

def test_ladder_excludes_gate_failing_layer():
    day, month = shared_subspace_pair(d=48, n_a=7, n_b=12)
    _, flat = orthogonal_subspace_pair(d=48, n_a=7, n_b=12)

    def _noise_runs(d=48, n_states=7, n=5, seed=12):
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((d, n_states)) for _ in range(n)]

    runs = {
        ("day", 3): _noise_runs(),          # pure noise source -> should FAIL the gate at layer 3
        ("month", 3): _runs(month),
        ("flat", 3): _runs(flat),
        ("day", 5): _runs(day),             # structured source -> should PASS the gate at layer 5
        ("month", 5): _runs(month),
        ("flat", 5): _runs(flat),
    }
    out = run_ladder(runs, layers=[3, 5], source="day", targets=("month", "flat"))
    layer3 = next(r for r in out if r.layer == 3)
    layer5 = next(r for r in out if r.layer == 5)

    assert layer3.gate_passed is False
    assert layer3.crosses == {}
    assert layer5.gate_passed is True
    assert layer5.crosses != {}
