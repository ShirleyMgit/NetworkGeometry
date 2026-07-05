# LLM Cycle-Geometry v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 pipeline that (Part 1) reproduces day/month circular manifolds in Gemma 2 2B and (Part 2) tests whether cycles share an abstract subspace via the Samborska/Mark AUC subspace-generalization method.

**Architecture:** A model-agnostic Python package. All scientific math (centering/PCA, Gram, circle-fit, the AUC engine, LOO, permutation nulls, gating) is pure-NumPy and unit-tested on **synthetic fixtures with known answers**, independent of any model. A thin `extraction` layer (TransformerLens) turns prompts into the shared `DataMatrix` contract; every downstream module consumes only `DataMatrix`, so activity paths are interchangeable. Part 1 is a validation gate that must reproduce before Part 2 is interpreted.

**Tech Stack:** Python 3.11, uv, NumPy, SciPy, TransformerLens + PyTorch (Gemma 2 2B), sae_lens (Gemma Scope), matplotlib, PyYAML, pytest.

## Global Constraints

- **Package manager:** uv only — `uv add` / `uv run` / `uv sync`. Never `pip install`. (Global CLAUDE.md.)
- **Python:** 3.11 (align to the project venv; do not use 3.12+-only syntax).
- **Clean Code:** self-explanatory names; comments only for non-obvious "why"; single-responsibility modules. (Global CLAUDE.md.)
- **Prefer `@dataclass`** (frozen where immutability fits) over hand-written `__init__`/`__eq__`. (Global CLAUDE.md.)
- **Pandas:** never `.copy()`, never `inplace=True`, never `.apply()/.iterrows()/.itertuples()/.items()`; vectorize or list-comprehend over `zip`. (Global CLAUDE.md.)
- **Determinism:** every stochastic function takes an explicit `numpy.random.Generator` (`np.random.default_rng(seed)`). No global RNG, no `Math.random`-style calls.
- **Model is frozen:** read hooks in v1; extraction backend must also expose write hooks (unused now, required for later Phase 4). No LLM training.
- **Canonical matrix orientation:** activation matrix `A` is `(d, n_states)` — rows = hidden units, columns = states. `W = Aᵀ` only where a states×states Gram reads more naturally.
- **AUC chance level = 0.5** (diagonal cumulative-variance curve).
- **Spec of record:** `docs/superpowers/specs/2026-07-04-llm-cycle-geometry-design.md`. v1 = Parts 1–2 only; Phases 3–6 are out of scope.

---

## File Structure

```
pyproject.toml
networkgeometry/
  __init__.py
  types.py                 # State, DataMatrix, run-grouping helpers
  geometry/
    __init__.py
    linalg.py              # mean_center, source_pcs, state_gram, is_toeplitz
    circle_fit.py          # fit_circle, circular_correlation, angular_order_score
    part1.py               # circularity_by_layer (May handling, strict/generalized)
  subspace/
    __init__.py
    auc.py                 # cumulative_variance_curve, auc_from_curve
  stats/
    __init__.py
    within.py              # within_structure_auc (LOO), v_side_stability
    cross.py               # cross_structure_auc (no LOO), across-run SEM
    inference.py           # identity_shuffle_null, p_value, bonferroni, benjamini_hochberg, gate
  analysis/
    __init__.py
    ladder.py              # run_ladder -> LadderResult, to_json
  stimuli/
    __init__.py
    definitions.py         # load structures/templates from YAML
    data/
      structures.yaml
      templates.yaml
  extraction/
    __init__.py
    activations.py         # extract(model, prompts_by_run, layers) -> [DataMatrix]
  sae/
    __init__.py
    gemma_scope.py         # cluster_decoder, cluster_circle_score
  figures/
    __init__.py
    part1_plots.py         # manifold, gram, circularity-vs-layer
    part2_plots.py         # auc-vs-layer, ladder, cumulative curves, null hist
  report/
    __init__.py
    build.py               # build_memo(summary, template, out)
    findings_template.md
  run.py                   # orchestration entrypoints (strict leg, generalized leg, ladder)
tests/
  ... mirrors package ...
  fixtures.py              # synthetic ring / shared-subspace / orthogonal matrices
```

**Milestone A (Part 1 deliverable):** Tasks 1–5 + 11 + 13(part1 plots) → reproduces circles, produces circularity metrics/figures.
**Milestone B (Part 2 deliverable):** Tasks 6–10 + 12 + 13(part2 plots) + 14 → full AUC ladder, stats, figures, findings memo.

---

### Task 1: Project scaffolding + core types

**Files:**
- Create: `pyproject.toml`, `networkgeometry/__init__.py`, `networkgeometry/types.py`
- Test: `tests/test_types.py`, `tests/fixtures.py`

**Interfaces:**
- Produces:
  - `State(label: str, canonical_index: int)` — frozen dataclass.
  - `DataMatrix(structure: str, layer: int, run: int, matrix: np.ndarray, states: tuple[State, ...])` — frozen; `matrix` is `(d, n_states)`; `.n_states` property; validates `matrix.shape[1] == len(states)`.
  - `group_runs(dms: list[DataMatrix]) -> list[np.ndarray]` — returns each run's matrix with columns reordered to a common canonical-index order; raises if state sets differ.

- [ ] **Step 1: Initialize the project with uv**

Run:
```bash
uv init --package --name networkgeometry --python 3.11
uv add numpy scipy pyyaml matplotlib
uv add --dev pytest
```
Expected: `pyproject.toml` and `networkgeometry/` created; deps resolve.

- [ ] **Step 2: Write the failing test**

Create `tests/test_types.py`:
```python
import numpy as np
import pytest
from networkgeometry.types import State, DataMatrix, group_runs

def test_datamatrix_validates_shape():
    states = (State("Monday", 1), State("Tuesday", 2))
    with pytest.raises(ValueError):
        DataMatrix("day", 0, 0, np.zeros((4, 3)), states)

def test_n_states_and_construction():
    states = (State("Monday", 1), State("Tuesday", 2))
    dm = DataMatrix("day", 0, 0, np.zeros((4, 2)), states)
    assert dm.n_states == 2

def test_group_runs_aligns_by_canonical_index():
    a = State("Monday", 1); b = State("Tuesday", 2)
    m1 = np.array([[1.0, 2.0]])                      # columns (Mon, Tue)
    m2 = np.array([[20.0, 10.0]])                    # columns (Tue, Mon)
    dm1 = DataMatrix("day", 0, 0, m1, (a, b))
    dm2 = DataMatrix("day", 0, 1, m2, (b, a))
    grouped = group_runs([dm1, dm2])
    np.testing.assert_allclose(grouped[0], [[1.0, 2.0]])
    np.testing.assert_allclose(grouped[1], [[10.0, 20.0]])  # reordered to (Mon, Tue)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: networkgeometry.types`.

- [ ] **Step 4: Implement `types.py`**

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class State:
    label: str
    canonical_index: int

@dataclass(frozen=True)
class DataMatrix:
    structure: str
    layer: int
    run: int
    matrix: np.ndarray
    states: tuple[State, ...]

    def __post_init__(self):
        if self.matrix.ndim != 2 or self.matrix.shape[1] != len(self.states):
            raise ValueError(
                f"matrix columns {self.matrix.shape} must equal n_states {len(self.states)}"
            )

    @property
    def n_states(self) -> int:
        return len(self.states)

def group_runs(dms: list[DataMatrix]) -> list[np.ndarray]:
    reference = tuple(sorted(s.canonical_index for s in dms[0].states))
    aligned = []
    for dm in dms:
        if tuple(sorted(s.canonical_index for s in dm.states)) != reference:
            raise ValueError("runs do not share the same state set")
        order = np.argsort([s.canonical_index for s in dm.states])
        aligned.append(dm.matrix[:, order])
    return aligned
```

- [ ] **Step 5: Add shared synthetic fixtures**

Create `tests/fixtures.py`:
```python
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
```

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_types.py -v` → Expected: PASS.
```bash
git add pyproject.toml networkgeometry/ tests/
git commit -m "feat: project scaffolding, DataMatrix contract, synthetic fixtures"
```

---

### Task 2: Linear-algebra core (centering, source PCs, Gram, Toeplitz)

**Files:**
- Create: `networkgeometry/geometry/__init__.py`, `networkgeometry/geometry/linalg.py`
- Test: `tests/geometry/test_linalg.py`

**Interfaces:**
- Consumes: `tests.fixtures`.
- Produces:
  - `mean_center(A: np.ndarray, centering: str = "mean") -> np.ndarray` — subtract mean over columns (states) when `centering=="mean"`, else identity. `centering ∈ {"mean","none"}`.
  - `source_pcs(A: np.ndarray, centering: str = "mean") -> np.ndarray` — left singular vectors `U` `(d, d)` (full ambient basis via `full_matrices=True`, NOT capped at `min(d, n_states)` — see the AMENDED note below this interface list), ordered by variance (real signal directions first, then an arbitrary-but-orthonormal completion of the remaining ambient dimensions).
  - `state_gram(A: np.ndarray, centering: str = "mean") -> np.ndarray` — `(n_states, n_states)` Gram over columns.
  - `is_toeplitz(G: np.ndarray, atol: float = 1e-8) -> bool`.

> **AMENDED (post-Task-4 fix, applies from Task 4 onward):** `source_pcs` originally used `full_matrices=False` (economy SVD, `U` capped at `min(d, n_states)`), as shown in Step 3 below. Task 4's own test (`test_shared_subspace_high_orthogonal_chance`) revealed this breaks the classical "chance-level AUC ≈ 0.5" guarantee whenever `n_states ≪ d` — exactly this project's real regime (Gemma 2 2B: `d=2304`, cyclic structures have only 7-30 states). With economy SVD, the PC columns beyond the source's true signal rank are an arbitrary, only-partially-uninformative completion, so an unrelated/orthogonal source's cross-AUC came out far below 0.5 (verified ~0.07 at `d=48, n_states=7`) instead of near it. **Fix: use `full_matrices=True`** so `U` is always the full `(d,d)` ambient basis — this guarantees any target's cumulative variance reaches 100% by the last component, restoring genuine chance-level behavior (verified ~0.45-0.52 across seeds) for uninformative sources while leaving real shared structure unaffected (still ~0.98). **This is an explicit v1 choice, not final** — it increases `source_pcs`'s cost to `O(d²)` per call (~42MB dense array at Gemma 2B's `d=2304`); revisit later (e.g. truncating to a smaller common `k` across compared structures) once the pipeline runs end-to-end. Every later task that calls `source_pcs` (Tasks 5, 6, 9, 10) inherits this automatically — no other code changes. **`Vt` (right singular vectors) is unaffected by this flag when `d > n_states`** (verified numerically: identical shape and values either way) — so Task 5's separate direct `np.linalg.svd(..., full_matrices=False)` call for `v_side_stability` (which only uses `vt`) is correct as written and needs no change.

- [ ] **Step 1: Write the failing test**

Create `tests/geometry/test_linalg.py`:
```python
import numpy as np
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
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/geometry/test_linalg.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `linalg.py`**

```python
import numpy as np

def mean_center(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    if centering == "none":
        return A
    if centering != "mean":
        raise ValueError(f"unknown centering {centering!r}")
    return A - A.mean(axis=1, keepdims=True)

def source_pcs(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    # NOTE: as originally written below (full_matrices=False) this was the
    # Task 2 implementation. It was corrected to full_matrices=True during
    # Task 4 — see the AMENDED note above the interface list for why. The
    # actual, current implementation lives in the codebase; this code block
    # is kept as historical context for how the task was originally planned.
    centered = mean_center(A, centering)
    U, _s, _vt = np.linalg.svd(centered, full_matrices=False)  # AMENDED to True; see note above
    return U

def state_gram(A: np.ndarray, centering: str = "mean") -> np.ndarray:
    centered = mean_center(A, centering)
    return centered.T @ centered

def is_toeplitz(G: np.ndarray, atol: float = 1e-8) -> bool:
    n = G.shape[0]
    for offset in range(-n + 1, n):
        diag = np.diagonal(G, offset)
        if not np.allclose(diag, diag[0], atol=atol):
            return False
    return True
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/geometry/test_linalg.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/geometry/ tests/geometry/
git commit -m "feat: linalg core (centering, source PCs, Gram, Toeplitz check)"
```

---

### Task 3: Circle fit + angular-order correlation

**Files:**
- Create: `networkgeometry/geometry/circle_fit.py`
- Test: `tests/geometry/test_circle_fit.py`

**Interfaces:**
- Produces:
  - `CircleFit(cx, cy, r, normalized_residual, angles)` — frozen dataclass; `angles` is `(n,)` radians.
  - `fit_circle(points: np.ndarray) -> CircleFit` — `points` is `(n, 2)`; algebraic (Kåsa) fit.
  - `circular_correlation(a: np.ndarray, b: np.ndarray) -> float` — circular–circular correlation in `[-1, 1]`.
  - `angular_order_score(points: np.ndarray, canonical_index: np.ndarray, n_states: int) -> float` — `|circular_correlation(fit angles, 2π·index/n)|`.

- [ ] **Step 1: Write the failing test**
```python
import numpy as np
from networkgeometry.geometry.circle_fit import fit_circle, circular_correlation, angular_order_score

def _circle_points(n, order):
    t = 2 * np.pi * np.asarray(order) / n
    return np.column_stack([np.cos(t), np.sin(t)])

def test_perfect_circle_low_residual():
    pts = _circle_points(12, range(12))
    fit = fit_circle(pts)
    assert fit.normalized_residual < 1e-6
    assert abs(fit.r - 1.0) < 1e-6

def test_angular_order_high_when_in_calendar_order():
    idx = np.arange(12)
    pts = _circle_points(12, idx)
    assert angular_order_score(pts, idx, 12) > 0.99

def test_angular_order_low_when_scrambled():
    idx = np.arange(12)
    scrambled = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
    pts = _circle_points(12, scrambled)
    assert angular_order_score(pts, idx, 12) < 0.5
```

- [ ] **Step 2: Run to verify fail** — FAIL (module missing).

- [ ] **Step 3: Implement `circle_fit.py`**
```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CircleFit:
    cx: float
    cy: float
    r: float
    normalized_residual: float
    angles: np.ndarray

def fit_circle(points: np.ndarray) -> CircleFit:
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack([x, y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, x**2 + y**2, rcond=None)
    cx, cy = solution[0] / 2, solution[1] / 2
    r = float(np.sqrt(solution[2] + cx**2 + cy**2))
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    normalized_residual = float(np.sqrt(np.mean((distances - r) ** 2)) / r)
    angles = np.arctan2(y - cy, x - cx)
    return CircleFit(float(cx), float(cy), r, normalized_residual, angles)

def _circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

def circular_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = np.sin(a - _circular_mean(a))
    b0 = np.sin(b - _circular_mean(b))
    denom = np.sqrt(np.sum(a0**2) * np.sum(b0**2))
    return float(np.sum(a0 * b0) / denom) if denom else 0.0

def angular_order_score(points, canonical_index, n_states) -> float:
    target = 2 * np.pi * np.asarray(canonical_index) / n_states
    return abs(circular_correlation(fit_circle(points).angles, target))
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/geometry/circle_fit.py tests/geometry/test_circle_fit.py
git commit -m "feat: circle fit + angular-order correlation"
```

---

### Task 4: Subspace AUC engine

**Files:**
- Create: `networkgeometry/subspace/__init__.py`, `networkgeometry/subspace/auc.py`
- Test: `tests/subspace/test_auc.py`

**Interfaces:**
- Consumes: `geometry.linalg.mean_center`, `source_pcs`; `tests.fixtures`.
- Produces:
  - `cumulative_variance_curve(U_source: np.ndarray, A_target: np.ndarray, centering: str = "mean") -> np.ndarray` — `M_k`, length `= U_source.shape[1]`, values in `[0, 1]`, non-decreasing.
  - `auc_from_curve(cum: np.ndarray) -> float` — area under `(0,0)→…→(1,1)`; uniform curve → 0.5.
  - `cross_auc(A_source: np.ndarray, A_target: np.ndarray, centering: str = "mean") -> float` — convenience: `auc_from_curve(cumulative_variance_curve(source_pcs(A_source), A_target))`.

- [ ] **Step 1: Write the failing test**
```python
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
```

- [ ] **Step 2: Run to verify fail** — FAIL (module missing).

- [ ] **Step 3: Implement `auc.py`**
```python
import numpy as np
from networkgeometry.geometry.linalg import mean_center, source_pcs

def cumulative_variance_curve(U_source, A_target, centering: str = "mean") -> np.ndarray:
    target = mean_center(A_target, centering)
    total = float(np.sum(target**2))
    if total == 0.0:
        return np.zeros(U_source.shape[1])
    projected = U_source.T @ target                 # (r, n_states_target)
    per_pc = np.sum(projected**2, axis=1)
    return np.cumsum(per_pc) / total

def auc_from_curve(cum: np.ndarray) -> float:
    y = np.concatenate([[0.0], np.asarray(cum, dtype=float)])
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.trapz(y, x))

def cross_auc(A_source, A_target, centering: str = "mean") -> float:
    return auc_from_curve(
        cumulative_variance_curve(source_pcs(A_source, centering), A_target, centering)
    )
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/subspace/ tests/subspace/
git commit -m "feat: subspace AUC engine (cumulative variance + AUC)"
```

---

### Task 5: Within-structure AUC (leave-one-run-out) + V-side stability

**Files:**
- Create: `networkgeometry/stats/__init__.py`, `networkgeometry/stats/within.py`
- Test: `tests/stats/test_within.py`

**Interfaces:**
- Consumes: `geometry.linalg`, `subspace.auc`.
- Produces:
  - `WithinResult(mean: float, sem: float, per_fold: np.ndarray)` — frozen dataclass.
  - `within_structure_auc(runs: list[np.ndarray], centering: str = "mean") -> WithinResult` — leave-one-run-out: `U` from the **mean of held-in runs**, project the held-out run, one AUC per fold.
  - `v_side_stability(runs: list[np.ndarray], centering: str = "mean") -> WithinResult` — same LOO, but projects the held-out run onto the held-in **right** singular vectors (state modes); AUC of that cumulative curve.

- [ ] **Step 1: Write the failing test**
```python
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
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `within.py`**
```python
from dataclasses import dataclass
import numpy as np
from networkgeometry.geometry.linalg import mean_center
from networkgeometry.subspace.auc import cumulative_variance_curve, auc_from_curve

@dataclass(frozen=True)
class WithinResult:
    mean: float
    sem: float
    per_fold: np.ndarray

def _held_in_mean(runs, held_out):
    kept = [r for i, r in enumerate(runs) if i != held_out]
    return np.mean(kept, axis=0)

def _summarize(per_fold) -> WithinResult:
    per_fold = np.asarray(per_fold, dtype=float)
    sem = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0
    return WithinResult(float(np.mean(per_fold)), sem, per_fold)

def within_structure_auc(runs, centering: str = "mean") -> WithinResult:
    from networkgeometry.geometry.linalg import source_pcs
    folds = []
    for j in range(len(runs)):
        u = source_pcs(_held_in_mean(runs, j), centering)
        folds.append(auc_from_curve(cumulative_variance_curve(u, runs[j], centering)))
    return _summarize(folds)

def v_side_stability(runs, centering: str = "mean") -> WithinResult:
    folds = []
    for j in range(len(runs)):
        held_in = mean_center(_held_in_mean(runs, j), centering)
        _u, _s, vt = np.linalg.svd(held_in, full_matrices=False)
        target = mean_center(runs[j], centering)
        total = float(np.sum(target**2)) or 1.0
        per_mode = np.sum((vt @ target.T) ** 2, axis=1)   # project states onto held-in state modes
        folds.append(auc_from_curve(np.cumsum(per_mode) / total))
    return _summarize(folds)
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/stats/__init__.py networkgeometry/stats/within.py tests/stats/test_within.py
git commit -m "feat: within-structure LOO AUC + V-side stability"
```

---

### Task 6: Cross-structure AUC (no LOO) + across-run SEM

**Files:**
- Create: `networkgeometry/stats/cross.py`
- Test: `tests/stats/test_cross.py`

**Interfaces:**
- Consumes: `geometry.linalg.source_pcs`, `subspace.auc`.
- Produces:
  - `CrossResult(mean: float, sem: float, per_run: np.ndarray)` — frozen dataclass.
  - `cross_structure_auc(source_runs: list[np.ndarray], target_runs: list[np.ndarray], centering: str = "mean") -> CrossResult` — `U` from **all** source runs (mean), one AUC per target run, mean ± SEM. No leave-one-out (different words ⇒ no leakage).

- [ ] **Step 1: Write the failing test**
```python
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
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `cross.py`**
```python
from dataclasses import dataclass
import numpy as np
from networkgeometry.geometry.linalg import source_pcs
from networkgeometry.subspace.auc import cumulative_variance_curve, auc_from_curve

@dataclass(frozen=True)
class CrossResult:
    mean: float
    sem: float
    per_run: np.ndarray

def cross_structure_auc(source_runs, target_runs, centering: str = "mean") -> CrossResult:
    u = source_pcs(np.mean(source_runs, axis=0), centering)
    per_run = np.array(
        [auc_from_curve(cumulative_variance_curve(u, t, centering)) for t in target_runs]
    )
    sem = float(np.std(per_run, ddof=1) / np.sqrt(len(per_run))) if len(per_run) > 1 else 0.0
    return CrossResult(float(np.mean(per_run)), sem, per_run)
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/stats/cross.py tests/stats/test_cross.py
git commit -m "feat: cross-structure AUC (all source runs, across-run SEM)"
```

---

### Task 7: Inference — permutation null, gating, multiple comparisons

**Files:**
- Create: `networkgeometry/stats/inference.py`
- Test: `tests/stats/test_inference.py`

**Interfaces:**
- Consumes: `geometry.linalg.source_pcs`, `subspace.auc`.
- Produces:
  - `identity_shuffle_null(A_source, A_target, n_perm: int, rng, centering="mean") -> np.ndarray` — shuffle target feature rows before projecting; returns `(n_perm,)` AUCs.
  - `p_value(observed: float, null: np.ndarray) -> float` — `(1 + #{null ≥ observed}) / (1 + n_perm)`.
  - `bonferroni(pvals: np.ndarray) -> np.ndarray`.
  - `benjamini_hochberg(pvals: np.ndarray) -> np.ndarray`.
  - `gate(within_p_by_layer: dict[int, float], alpha: float = 0.05) -> set[int]` — layers whose within-structure AUC beats chance.

- [ ] **Step 1: Write the failing test**
```python
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
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `inference.py`**
```python
import numpy as np
from networkgeometry.geometry.linalg import source_pcs
from networkgeometry.subspace.auc import cumulative_variance_curve, auc_from_curve

def identity_shuffle_null(A_source, A_target, n_perm, rng, centering="mean") -> np.ndarray:
    u = source_pcs(A_source, centering)
    out = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = A_target[rng.permutation(A_target.shape[0])]
        out[i] = auc_from_curve(cumulative_variance_curve(u, shuffled, centering))
    return out

def p_value(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))

def bonferroni(pvals: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(pvals, dtype=float) * len(pvals), 1.0)

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank, idx in enumerate(reversed(order)):
        k = n - rank
        running = min(running, p[idx] * n / k)
        adjusted[idx] = running
    return adjusted

def gate(within_p_by_layer: dict, alpha: float = 0.05) -> set:
    return {layer for layer, pval in within_p_by_layer.items() if pval < alpha}
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/stats/inference.py tests/stats/test_inference.py
git commit -m "feat: permutation null, gating, FDR + Bonferroni"
```

---

### Task 8: Stimuli definitions + loader

**Files:**
- Create: `networkgeometry/stimuli/__init__.py`, `networkgeometry/stimuli/definitions.py`, `networkgeometry/stimuli/data/structures.yaml`, `networkgeometry/stimuli/data/templates.yaml`
- Test: `tests/stimuli/test_definitions.py`

**Interfaces:**
- Produces:
  - `Structure(name, states: tuple[State,...], excluded: tuple[str,...])` — frozen.
  - `load_structures(path=None) -> dict[str, Structure]` — day, month, years, hierarchy, flat.
  - `load_templates(path=None) -> dict[str, dict]` — `{"shared": [...], "specific": {structure: [...]}}`; each template has a `{X}` slot.
  - `prompts_for(structure: Structure, templates: list[str]) -> dict[int, list[str]]` — `{run_index: [prompt per state in canonical order]}`.

- [ ] **Step 1: Write the failing test**
```python
from networkgeometry.stimuli.definitions import load_structures, load_templates, prompts_for

def test_day_and_month_states():
    s = load_structures()
    assert [st.label for st in s["day"].states][:2] == ["Monday", "Tuesday"]
    assert s["day"].n_states == 7 if hasattr(s["day"], "n_states") else len(s["day"].states) == 7
    assert len(s["month"].states) == 12
    assert "May" in s["month"].excluded

def test_templates_have_slot_and_pools():
    t = load_templates()
    assert all("{X}" in tpl for tpl in t["shared"])
    assert "day" in t["specific"] and "month" in t["specific"]

def test_prompts_fill_slot_in_canonical_order():
    s = load_structures(); t = load_templates()
    runs = prompts_for(s["day"], t["shared"])
    assert runs[0][0].endswith("Monday") or "Monday" in runs[0][0]
    assert len(runs[0]) == 7
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Create the YAML data**

`networkgeometry/stimuli/data/structures.yaml`:
```yaml
day:
  states: [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
month:
  states: [January, February, March, April, May, June, July, August, September, October, November, December]
  excluded: [May, March, August]   # polysemy: excluded from PCA basis, projected back for plots
years:
  states: ["1996","1998","2000","2002","2004","2006","2008","2010","2012","2014","2016","2018","2020","2022","2024"]
hierarchy:
  states: [dog, cat, horse, robin, eagle, owl, oak, pine, maple, rose, tulip, daisy]
flat:
  states: [river, cloud, book, stone, road, key, ship, bridge, window, candle, mirror, ticket]
```

`networkgeometry/stimuli/data/templates.yaml`:
```yaml
shared:
  - "{X} is my favorite."
  - "Honestly, I love {X}."
  - "Let's talk about {X}."
  - "Nothing beats {X}."
specific:
  day:
    - "We'll meet on {X}."
    - "The concert is on {X}."
  month:
    - "We'll meet in {X}."
    - "The concert is in {X}."
  years:
    - "It happened in {X}."
    - "Back in {X}, things changed."
  hierarchy:
    - "I saw a {X} today."
    - "Look, a {X}."
  flat:
    - "I saw a {X} today."
    - "Look, a {X}."
```

- [ ] **Step 4: Implement `definitions.py`**
```python
from dataclasses import dataclass, field
from importlib import resources
import yaml
from networkgeometry.types import State

@dataclass(frozen=True)
class Structure:
    name: str
    states: tuple
    excluded: tuple = ()

    @property
    def n_states(self) -> int:
        return len(self.states)

def _data_path(filename: str, path):
    if path is not None:
        return open(path, encoding="utf-8")
    return resources.files("networkgeometry.stimuli.data").joinpath(filename).open(encoding="utf-8")

def load_structures(path=None) -> dict:
    with _data_path("structures.yaml", path) as handle:
        raw = yaml.safe_load(handle)
    out = {}
    for name, spec in raw.items():
        labels = [lbl for lbl in spec["states"] if lbl not in spec.get("excluded", [])]
        states = tuple(State(lbl, i + 1) for i, lbl in enumerate(labels))
        out[name] = Structure(name, states, tuple(spec.get("excluded", [])))
    return out

def load_templates(path=None) -> dict:
    with _data_path("templates.yaml", path) as handle:
        return yaml.safe_load(handle)

def prompts_for(structure: Structure, templates: list) -> dict:
    ordered = sorted(structure.states, key=lambda s: s.canonical_index)
    return {run: [tpl.replace("{X}", s.label) for s in ordered] for run, tpl in enumerate(templates)}
```

> Note: `excluded` labels are dropped from the PCA-basis state set here. Projecting excluded tokens back for visualization (Task 9) uses a separate call that includes them.

- [ ] **Step 5: Run to verify pass** — PASS.

- [ ] **Step 6: Commit**
```bash
git add networkgeometry/stimuli/ tests/stimuli/
git commit -m "feat: stimuli structures + template pools + prompt builder"
```

---

### Task 9: Part 1 geometry pipeline (circularity by layer, May handling)

**Files:**
- Create: `networkgeometry/geometry/part1.py`
- Test: `tests/geometry/test_part1.py`

> **AMENDED (pre-dispatch, discovered during Task 8 review):** the original interface below took no `excluded` parameter, so it could not implement the "May handling" this task is named for (spec §4.3(b): compute the PCA basis / circle-fit metrics from only the non-polysemous states, so an outlier like May doesn't corrupt the quantitative circularity score). Task 8 changed `Structure.states` to always include ALL labels (with `Structure.excluded` as separate metadata, not pre-filtered) specifically so this filtering happens here, not in the loader. Added an `excluded: tuple[str, ...] = ()` parameter, filtering by label before mean-centering/PCA/circle-fit. Projecting excluded states *back* onto the fitted plane for visualization is explicitly OUT of this task's scope (no test requires it) — that belongs to Task 13's plotting code, which can re-derive a small projection from the same basis if/when needed. Default `excluded=()` leaves prior behavior (and the original test) unchanged.

**Interfaces:**
- Consumes: `types.DataMatrix`, `types.group_runs`, `geometry.linalg`, `geometry.circle_fit`.
- Produces:
  - `LayerCircularity(layer, normalized_residual, angular_order, top2_variance_ratio)` — frozen.
  - `circularity_by_layer(dms_by_layer: dict[int, list[DataMatrix]], excluded: tuple[str, ...] = (), centering="mean") -> list[LayerCircularity]` — averages runs per layer; **filters out any state whose label is in `excluded` before** mean-centering, PCA, top-2 plane, circle-fit, angular-order, and top-2 variance ratio.

- [ ] **Step 1: Write the failing test**
```python
import numpy as np
from networkgeometry.types import State, DataMatrix
from networkgeometry.geometry.part1 import circularity_by_layer
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
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `part1.py`**
```python
from dataclasses import dataclass
import numpy as np
from networkgeometry.types import group_runs
from networkgeometry.geometry.linalg import mean_center, source_pcs
from networkgeometry.geometry.circle_fit import fit_circle, angular_order_score

@dataclass(frozen=True)
class LayerCircularity:
    layer: int
    normalized_residual: float
    angular_order: float
    top2_variance_ratio: float

def circularity_by_layer(dms_by_layer: dict, excluded: tuple = (), centering: str = "mean") -> list:
    results = []
    for layer in sorted(dms_by_layer):
        dms = dms_by_layer[layer]
        all_states = sorted(dms[0].states, key=lambda s: s.canonical_index)
        keep_mask = np.array([s.label not in excluded for s in all_states])

        mean_matrix = np.mean(group_runs(dms), axis=0)          # (d, n_states), same order as all_states
        basis_matrix = mean_matrix[:, keep_mask]
        canonical = np.array([s.canonical_index for s, keep in zip(all_states, keep_mask) if keep])

        centered = mean_center(basis_matrix, centering)
        u = source_pcs(basis_matrix, centering)
        scores = (u[:, :2].T @ centered).T                      # (n_kept_states, 2)
        energy = np.sum((u.T @ centered) ** 2, axis=1)
        ratio = float(energy[:2].sum() / energy.sum())
        fit = fit_circle(scores)
        order = angular_order_score(scores, canonical, len(canonical))
        results.append(LayerCircularity(layer, fit.normalized_residual, order, ratio))
    return results
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/geometry/part1.py tests/geometry/test_part1.py
git commit -m "feat: Part 1 circularity-by-layer pipeline"
```

---

### Task 10: Comparison ladder orchestration + results serialization

**Files:**
- Create: `networkgeometry/analysis/__init__.py`, `networkgeometry/analysis/ladder.py`
- Test: `tests/analysis/test_ladder.py`

**Interfaces:**
- Consumes: `stats.within`, `stats.cross`, `stats.inference`, `types.group_runs`.
- Produces:
  - `LadderResult(layer, within, crosses: dict[str, dict], gate_passed: bool)`.
  - `run_ladder(runs_by_structure_layer: dict[tuple[str,int], list[np.ndarray]], layers: list[int], source="day", targets=("month","years","hierarchy","flat"), n_perm=500, alpha=0.05, seed=0) -> list[LadderResult]` — Stage 1 within-gate per layer, then Stage 2 cross-structure AUC + null p-values only at gate-passing layers; applies FDR + Bonferroni across the surviving (layer×target) set.
  - `to_json(results: list[LadderResult]) -> dict`.

> **AMENDED (post-implementation cleanup):** an earlier draft of this interface list also named a `Contrast(name, source, target, context)` dataclass. It was never used by Step 3's actual code, never referenced by any test, and no later task (11-15) consumes it — confirmed via a repo-wide search finding this single mention. Removed as a stale artifact; `LadderResult.crosses: dict[str, dict]` (keyed by target name) already carries everything the comparison ladder (spec §5.3) needs per layer.

- [ ] **Step 1: Write the failing test**
```python
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
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `ladder.py`**
```python
from dataclasses import dataclass, asdict
import numpy as np
from networkgeometry.stats.within import within_structure_auc
from networkgeometry.stats.cross import cross_structure_auc
from networkgeometry.stats.inference import (
    identity_shuffle_null, p_value, bonferroni, benjamini_hochberg, gate)

@dataclass(frozen=True)
class LadderResult:
    layer: int
    within: float
    crosses: dict
    gate_passed: bool

def run_ladder(runs_by_structure_layer, layers, source="day",
               targets=("month", "years", "hierarchy", "flat"),
               n_perm=500, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    within_p, within_mean, cross_raw = {}, {}, {}
    for layer in layers:
        src = runs_by_structure_layer[(source, layer)]
        wr = within_structure_auc(src)
        within_mean[layer] = wr.mean
        src_mean = np.mean(src, axis=0)
        null = identity_shuffle_null(src_mean, src_mean, n_perm, rng)
        within_p[layer] = p_value(wr.mean, null)

    passed = gate(within_p, alpha)
    flat_pvals, flat_keys = [], []
    for layer in layers:
        cross_raw[layer] = {}
        if layer not in passed:
            continue
        src = runs_by_structure_layer[(source, layer)]
        for target in targets:
            tgt = runs_by_structure_layer[(target, layer)]
            cr = cross_structure_auc(src, tgt)
            null = identity_shuffle_null(np.mean(src, axis=0), np.mean(tgt, axis=0), n_perm, rng)
            pval = p_value(cr.mean, null)
            cross_raw[layer][target] = {"auc": cr.mean, "sem": cr.sem, "raw_p": pval}
            flat_pvals.append(pval); flat_keys.append((layer, target))

    if flat_pvals:
        fdr = benjamini_hochberg(np.array(flat_pvals))
        bonf = bonferroni(np.array(flat_pvals))
        for (layer, target), f, b in zip(flat_keys, fdr, bonf):
            cross_raw[layer][target]["fdr_p"] = float(f)
            cross_raw[layer][target]["bonferroni_p"] = float(b)

    return [LadderResult(layer, within_mean[layer], cross_raw[layer], layer in passed)
            for layer in layers]

def to_json(results) -> dict:
    return {"layers": [asdict(r) for r in results]}
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/analysis/ tests/analysis/
git commit -m "feat: comparison ladder with gate + FDR/Bonferroni + JSON"
```

---

### Task 11: Extraction (TransformerLens → DataMatrix)

**Files:**
- Create: `networkgeometry/extraction/__init__.py`, `networkgeometry/extraction/activations.py`
- Test: `tests/extraction/test_activations.py`

**Interfaces:**
- Consumes: `types.State`, `types.DataMatrix`.
- Produces:
  - `load_model(name="google/gemma-2-2b", device="cpu")` — returns a `HookedTransformer` (uses `HookedTransformer.from_pretrained`, which registers read+write hooks).
  - `extract(model, prompts_by_run: dict[int, list[str]], states: tuple[State,...], structure: str, layers: list[int]) -> list[DataMatrix]` — runs each prompt, reads `blocks.{l}.hook_resid_post` at the final token, assembles one `DataMatrix` per (layer, run) with `matrix` shape `(d, n_states)`.

- [ ] **Step 1: Write the failing test (tiny public model, CPU)**
```python
import numpy as np
import pytest
from networkgeometry.types import State

transformer_lens = pytest.importorskip("transformer_lens")

@pytest.mark.integration
def test_extract_shapes_with_gpt2():
    from networkgeometry.extraction.activations import load_model, extract
    model = load_model("gpt2", device="cpu")
    states = (State("Monday", 1), State("Tuesday", 2), State("Wednesday", 3))
    prompts = {0: ["A day: Monday", "A day: Tuesday", "A day: Wednesday"],
               1: ["See you Monday", "See you Tuesday", "See you Wednesday"]}
    dms = extract(model, prompts, states, "day", layers=[0, 5])
    by_layer = {dm.layer for dm in dms}
    assert by_layer == {0, 5}
    for dm in dms:
        assert dm.matrix.shape[1] == 3 and dm.matrix.shape[0] == model.cfg.d_model
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/extraction -v -m integration` → FAIL (module missing). *(Requires `uv add transformer-lens torch`.)*

- [ ] **Step 3: Implement `activations.py`**
```python
import numpy as np
from networkgeometry.types import DataMatrix

def load_model(name: str = "google/gemma-2-2b", device: str = "cpu"):
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained(name, device=device)

def _final_token_resid(model, prompt: str, layers: list) -> dict:
    names = [f"blocks.{l}.hook_resid_post" for l in layers]
    _logits, cache = model.run_with_cache(prompt, names_filter=lambda n: n in names)
    return {l: cache[f"blocks.{l}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            for l in layers}

def extract(model, prompts_by_run, states, structure, layers) -> list:
    dms = []
    for run, prompts in prompts_by_run.items():
        per_layer = {l: [] for l in layers}
        for prompt in prompts:
            resid = _final_token_resid(model, prompt, layers)
            for l in layers:
                per_layer[l].append(resid[l])
        for l in layers:
            matrix = np.stack(per_layer[l], axis=1)     # (d, n_states)
            dms.append(DataMatrix(structure, l, run, matrix, states))
    return dms
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/extraction -v -m integration` → PASS (downloads gpt2 once).

- [ ] **Step 5: Register the `integration` marker + commit**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
markers = ["integration: requires downloading a model"]
```
```bash
git add networkgeometry/extraction/ tests/extraction/ pyproject.toml
git commit -m "feat: TransformerLens extraction to DataMatrix (final-token resid)"
```

---

### Task 12: SAE reproduction check (Gemma Scope cluster → circle)

**Files:**
- Create: `networkgeometry/sae/__init__.py`, `networkgeometry/sae/gemma_scope.py`
- Test: `tests/sae/test_gemma_scope.py`

**Interfaces:**
- Produces:
  - `cluster_decoder(decoder: np.ndarray, threshold: float) -> list[list[int]]` — `decoder` is `(d, m)` columns = dictionary directions; cosine-similarity graph, connected components above `threshold`.
  - `cluster_circle_score(reconstructed: np.ndarray) -> float` — top-2 PCA of a cluster's reconstructed activations → `angular_order`-agnostic circularity via `fit_circle(...).normalized_residual` (lower = more circular).
  - `load_sae(release, sae_id)` — thin wrapper over `sae_lens.SAE.from_pretrained` (integration).

- [ ] **Step 1: Write the failing test (pure-numpy parts only)**
```python
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
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `gemma_scope.py`**
```python
import numpy as np
from networkgeometry.geometry.linalg import source_pcs, mean_center
from networkgeometry.geometry.circle_fit import fit_circle

def cluster_decoder(decoder: np.ndarray, threshold: float) -> list:
    normed = decoder / (np.linalg.norm(decoder, axis=0, keepdims=True) + 1e-12)
    similarity = normed.T @ normed
    m = decoder.shape[1]
    adjacency = (similarity >= threshold)
    seen, clusters = set(), []
    for start in range(m):
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node); component.append(node)
            stack.extend(np.where(adjacency[node])[0])
        clusters.append(sorted(component))
    return clusters

def cluster_circle_score(reconstructed: np.ndarray) -> float:
    matrix = reconstructed.T                        # (d, n_points)
    u = source_pcs(matrix)
    scores = (u[:, :2].T @ mean_center(matrix)).T
    return fit_circle(scores).normalized_residual

def load_sae(release: str, sae_id: str):
    from sae_lens import SAE
    return SAE.from_pretrained(release, sae_id)
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/sae/ tests/sae/
git commit -m "feat: SAE decoder clustering + cluster circularity score"
```

---

### Task 13: Figures (Part 1 + Part 2)

**Files:**
- Create: `networkgeometry/figures/__init__.py`, `networkgeometry/figures/part1_plots.py`, `networkgeometry/figures/part2_plots.py`
- Test: `tests/figures/test_plots.py`

**Interfaces:**
- Produces (each saves a PNG and returns the path):
  - `plot_manifold(scores: np.ndarray, labels: list[str], out_path) -> str`
  - `plot_circularity_by_layer(results: list, out_path) -> str`
  - `plot_auc_by_layer(ladder: list, targets: list[str], out_path) -> str`
  - `plot_ladder(ladder_result, out_path) -> str`
  - `plot_cumulative_curves(curves: dict[str, np.ndarray], out_path) -> str`
  - `plot_null_hist(null: np.ndarray, observed: float, out_path) -> str`

- [ ] **Step 1: Write the failing test (smoke: files are produced)**
```python
import numpy as np
from pathlib import Path
from networkgeometry.figures.part1_plots import plot_manifold, plot_circularity_by_layer
from networkgeometry.figures.part2_plots import plot_auc_by_layer, plot_null_hist
from networkgeometry.geometry.part1 import LayerCircularity
from networkgeometry.analysis.ladder import LadderResult

def test_part1_and_part2_plots_write_files(tmp_path):
    scores = np.column_stack([np.cos(np.linspace(0, 6, 12)), np.sin(np.linspace(0, 6, 12))])
    p1 = plot_manifold(scores, [str(i) for i in range(12)], tmp_path / "m.png")
    lc = [LayerCircularity(l, 0.1, 0.9, 0.95) for l in range(3)]
    p2 = plot_circularity_by_layer(lc, tmp_path / "c.png")
    ladder = [LadderResult(5, 0.9, {"month": {"auc": 0.8, "sem": 0.02}}, True)]
    p3 = plot_auc_by_layer(ladder, ["month"], tmp_path / "a.png")
    p4 = plot_null_hist(np.random.default_rng(0).normal(0.5, 0.05, 500), 0.8, tmp_path / "n.png")
    assert all(Path(p).exists() for p in [p1, p2, p3, p4])
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement the two plot modules**

`part1_plots.py`:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_manifold(scores, labels, out_path) -> str:
    fig, ax = plt.subplots()
    ax.plot(scores[:, 0], scores[:, 1], "-o")
    for (x, y), label in zip(scores, labels):
        ax.annotate(label, (x, y))
    ax.set_aspect("equal"); ax.set_title("PC1–PC2 manifold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_circularity_by_layer(results, out_path) -> str:
    layers = [r.layer for r in results]
    fig, ax = plt.subplots()
    ax.plot(layers, [r.angular_order for r in results], "-o", label="angular order")
    ax.plot(layers, [r.top2_variance_ratio for r in results], "-s", label="top-2 var ratio")
    ax.set_xlabel("layer"); ax.set_ylabel("score"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)
```

`part2_plots.py`:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_auc_by_layer(ladder, targets, out_path) -> str:
    layers = [lr.layer for lr in ladder]
    fig, ax = plt.subplots()
    for target in targets:
        ys = [lr.crosses.get(target, {}).get("auc", np.nan) for lr in ladder]
        ax.plot(layers, ys, "-o", label=target)
    ax.axhline(0.5, ls="--", color="gray", label="chance")
    ax.set_xlabel("layer"); ax.set_ylabel("cross AUC"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_ladder(ladder_result, out_path) -> str:
    targets = list(ladder_result.crosses)
    aucs = [ladder_result.crosses[t]["auc"] for t in targets]
    sems = [ladder_result.crosses[t].get("sem", 0.0) for t in targets]
    fig, ax = plt.subplots()
    ax.bar(targets, aucs, yerr=sems); ax.axhline(0.5, ls="--", color="gray")
    ax.set_ylabel("AUC"); ax.set_title(f"layer {ladder_result.layer}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_cumulative_curves(curves, out_path) -> str:
    fig, ax = plt.subplots()
    for name, cum in curves.items():
        ax.plot(range(1, len(cum) + 1), cum, "-o", label=name)
    ax.set_xlabel("k (source PCs)"); ax.set_ylabel("cumulative variance"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_null_hist(null, observed, out_path) -> str:
    fig, ax = plt.subplots()
    ax.hist(null, bins=30); ax.axvline(observed, color="red", label="observed")
    ax.set_xlabel("AUC"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**
```bash
git add networkgeometry/figures/ tests/figures/
git commit -m "feat: Part 1 + Part 2 figure functions"
```

---

### Task 14: Findings-memo builder + orchestration entrypoints

**Files:**
- Create: `networkgeometry/report/__init__.py`, `networkgeometry/report/build.py`, `networkgeometry/report/findings_template.md`, `networkgeometry/run.py`
- Test: `tests/report/test_build.py`

**Interfaces:**
- Consumes: `analysis.ladder.to_json`.
- Produces:
  - `build_memo(summary: dict, template_path, out_path) -> str` — fills the template's `{{tl_dr}}`, `{{part1_table}}`, `{{part2_table}}` placeholders from `summary`, writes markdown.
  - `run.py`: `run_part1(...)`, `run_part2(...)`, `main()` — orchestrate extraction → analysis → figures → memo. (Integration-level; unit-tested pieces already covered.)

- [ ] **Step 1: Write the failing test**
```python
from pathlib import Path
from networkgeometry.report.build import build_memo

def test_build_memo_fills_placeholders(tmp_path):
    template = tmp_path / "t.md"
    template.write_text("# Findings\n{{tl_dr}}\n{{part2_table}}\n", encoding="utf-8")
    summary = {"tl_dr": "Cycles share a subspace at layer 5.",
               "part1_table": "| day | 0.95 |",
               "part2_table": "| month | 0.82 | 0.001 |"}
    out = build_memo(summary, template, tmp_path / "out.md")
    text = Path(out).read_text(encoding="utf-8")
    assert "Cycles share a subspace" in text and "0.82" in text and "{{" not in text
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `build.py` + template**

`build.py`:
```python
from pathlib import Path

def build_memo(summary: dict, template_path, out_path) -> str:
    text = Path(template_path).read_text(encoding="utf-8")
    for key, value in summary.items():
        text = text.replace("{{" + key + "}}", str(value))
    Path(out_path).write_text(text, encoding="utf-8")
    return str(out_path)
```

`findings_template.md`:
```markdown
# LLM Cycle-Geometry — Findings

## TL;DR
{{tl_dr}}

## Part 1 — circle reproduction
{{part1_table}}

## Part 2 — abstract cycle code (AUC ladder)
{{part2_table}}

## Scope & limitations
Single frozen model (Gemma 2 2B). Inference generalizes across prompt contexts,
not across a population of models. See spec §5.5.
```

- [ ] **Step 4: Implement `run.py` orchestration (no unit test; smoke via `--help`)**
```python
import argparse
from networkgeometry.stimuli.definitions import load_structures, load_templates, prompts_for

def run_part1(model, layers, out_dir):
    structures, templates = load_structures(), load_templates()
    from networkgeometry.extraction.activations import extract
    from networkgeometry.geometry.part1 import circularity_by_layer
    results = {}
    for name in ("day", "month"):
        dms = extract(model, prompts_for(structures[name], templates["shared"]),
                      structures[name].states, name, layers)
        by_layer = {}
        for dm in dms:
            by_layer.setdefault(dm.layer, []).append(dm)
        results[name] = circularity_by_layer(by_layer)
    return results

def main():
    parser = argparse.ArgumentParser(description="LLM cycle-geometry v1 runner")
    parser.add_argument("--part", choices=["part1", "part2"], required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(26)))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    from networkgeometry.extraction.activations import load_model
    model = load_model()
    if args.part == "part1":
        run_part1(model, args.layers, args.out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests + smoke, then commit**

Run: `uv run pytest tests/report -v` → PASS.
Run: `uv run python -m networkgeometry.run --help` → prints usage.
```bash
git add networkgeometry/report/ networkgeometry/run.py tests/report/
git commit -m "feat: findings-memo builder + orchestration entrypoints"
```

---

### Task 15: Full-model integration run (manual gate)

**Files:**
- Modify: none (uses `run.py`)
- Create: `docs/findings/cycle-geometry-findings.md` (generated)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Install model deps**

Run: `uv add transformer-lens torch sae_lens` (and authenticate to Hugging Face for the gated `google/gemma-2-2b`: `uv run huggingface-cli login`).

- [ ] **Step 2: Part 1 strict-leg reproduction (the gate)**

Run: `uv run python -m networkgeometry.run --part part1 --layers $(seq 0 25)` (all 26 layers; `--layers` defaults to `range(26)` if omitted, per `run.py`'s `argparse` default in Task 14 — omitting the flag is equivalent and simpler)
Expected: month + years circularity high at some layers (angular_order > ~0.9). **If not, treat as a bug** (check token position, layer indexing, centering, tokenization, May exclusion) per spec §4.5 — do not proceed to Part 2.

- [ ] **Step 3: Part 2 ladder at gate-passing layers**

Run: `uv run python -m networkgeometry.run --part part2 --layers <passing layers>`
Expected: `results/summary.json` with the AUC ladder; `day↔month > day→hierarchy > day→flat ≈ 0.5`.

- [ ] **Step 4: Build figures + findings memo**

Expected: figures in `figures/`, memo at `docs/findings/cycle-geometry-findings.md`.

- [ ] **Step 5: Commit results**
```bash
git add results/ figures/ docs/findings/
git commit -m "results: Part 1 reproduction + Part 2 AUC ladder (Gemma 2 2B)"
```

---

## Self-Review

**Spec coverage:**
- §3 stimuli (structures, states, two template pools, canonical index) → Tasks 1, 8. ✓
- §4 Part 1 (extraction, PCA manifold, Gram, circle-fit, layer sweep, May/centering, SAE check) → Tasks 2, 3, 9, 11, 12; Gram helper in Task 2 (`state_gram`, `is_toeplitz`). ✓
- §5 Part 2 (AUC engine, within-LOO, cross no-LOO, V-side, gate, nulls, FDR+Bonferroni, ΔAUC/SEM) → Tasks 4, 5, 6, 7, 10. ✓
- §8 deliverables (figures, findings memo, results) → Tasks 13, 14, 15. ✓
- §7 DataMatrix contract → Task 1. ✓
- Out of scope (Phases 3–6, community control, bootstrap, multi-model) → correctly absent. ✓

**Notes for the implementer:**
- `ΔAUC = within.mean − cross.mean` at a gate-passing layer is a trivial subtraction of Task 5 and Task 6 outputs; compute it in the memo/`to_json` step rather than a separate module (YAGNI).
- The strict-leg vs generalized-leg distinction (§4.3a) is a *stimulus* choice (`templates["shared"][:1]` vs the full pool) fed to the existing `prompts_for`/`extract`/`circularity_by_layer` path — no separate code.
- `centering="none"` for years is already threaded through every linalg/AUC function; pass it per-structure in orchestration.

**Type consistency:** `source_pcs`, `mean_center`, `cumulative_variance_curve`, `auc_from_curve`, `WithinResult`, `CrossResult`, `LadderResult`, `DataMatrix`, `State`, `Structure` are referenced with identical signatures across tasks. ✓
