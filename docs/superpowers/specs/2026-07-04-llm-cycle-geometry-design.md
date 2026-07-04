# Design Spec — Do LLMs represent cycles as an abstract, flexible code?

**Date:** 2026-07-04
**Status:** Design approved for v1 (Parts 1–2); Phases 3–6 specified as architected extensions.
**Author context:** Reproduces + extends two lines of work — Karkada et al. *Symmetry in Language Statistics Shapes the Geometry of Model Representations* (calendar-cycle geometry in Gemma 2 2B) and Mark et al. 2026 eLife / Samborska et al. 2022 (subspace-generalization AUC for abstract structural knowledge in entorhinal cortex).

---

## 0. One-paragraph summary

We test whether a frozen LLM represents cyclic concepts (day-of-week, month-of-year) not as isolated idiosyncratic manifolds but as a **shared, reusable low-dimensional subspace** — an *abstract cycle code* — the way entorhinal grid cells occupy a common subspace across environments. **Part 1** reproduces the circular calendar manifolds in Gemma 2 2B (a positive-control validation gate). **Part 2** adapts the Samborska/Mark **subspace-generalization AUC** method to the SVD of the LLM's activation matrix, asking whether one cycle's principal subspace explains another cycle's variance, with a graded ladder of controls. Later phases (boundary code, causal intervention, compositional torus, intrinsic ring topology) are specified but out of v1 scope.

---

## 1. Scientific questions

- **Q1 (Part 1 — reproduction):** Does frozen `google/gemma-2-2b` arrange day-of-week (7) and month (12) into circular manifolds in its residual stream, and at which layers? *Geometry only — no natural-language-statistics theory.*
- **Q2 (Part 2 — the real question):** Are cycles carried by a shared, reusable low-dimensional subspace (abstract cycle code)? Tested with subspace-generalization AUC on the activation-matrix SVD.
- **Q3–Q6 (later phases):** boundary/start-point code; causal use + cross-cycle causal transfer; compositional (torus) factorization; intrinsic ring topology.

**Which model is "clever enough" to hold such a representation is itself an open empirical question.** A negative Part-2 result (cycles represented but *not* shared) is a real finding; a negative Part-1 result on Gemma 2 2B is a **pipeline bug** (see §4.5).

---

## 2. Model & framing

- **Model:** `google/gemma-2-2b`, **frozen**. We never train the LLM; we only read (and, in Phase 4, write) residual-stream activations. Matches the Symmetry paper's primary model exactly.
- **SAE:** Gemma Scope (open SAEs for every Gemma 2 2B layer). Used **only** for the Engels-style reproduction check in Part 1 (§4.4). *Not* an AUC substrate.
- **Model-agnostic pipeline:** the model is a config value. Swapping in GPT-2-small (cheap CPU dev check) or Llama 3 8B / Mistral 7B (needed if Phase 4 requires strong modular-arithmetic accuracy) is a config change, not a refactor.
- **Extraction backend:** TransformerLens, with **read *and* write hooks** enabled from day one (write-hooks unused in Parts 1–2, required for Phase 4 causal patching).

---

## 3. Stimuli

### 3.1 Structures

| Role | Structure | States (v1) | Topology |
|---|---|---|---|
| Cycle | Day-of-week | 7 (Mon…Sun) | Ring (periodic) |
| Cycle | Month | 12 (Jan…Dec) | Ring (periodic) |
| Near-control (boundary axis) | Years, single-token (e.g. 1990–2025) | ~30 | Open 1D sequence |
| Non-seq control (primary) | Community / cluster set | ~12 | Clustered, non-ordered |
| Non-seq control (floor) | Flat unrelated set | ~12 | Unstructured |

- Years is **not** a "non-cyclic foil": per the Symmetry paper, months and years are the *same* translation-symmetric 1D object differing only in **boundary conditions** (months = closed Fourier series → ring; years = open-boundary Fourier series → rippled line). It is a **near-control on the periodicity/boundary axis** — we may find it *shares* the cycle subspace, which motivates Phase 3.
- The **non-sequential** controls (community, flat) are the true "should not share the sequence subspace" controls. Community has genuine (non-1D) internal structure, so a non-generalization result is meaningful ("represented, but not in the sequence subspace"); flat is an unstructured floor.
- **Matching:** controls matched to cycles on set-size band, single-token-ness, corpus-frequency band, and template slots, so a null reflects *structure*, not surface confounds.

### 3.2 States are a config parameter (not constants)

v1 uses the canonical sets above. A documented extension **densifies** cycles with phase-modifier states ("Monday morning", "late Monday", "early April") that place additional points *between* the canonical ones (Engels continuity dataset). All downstream analysis is agnostic to state count. Densification feeds Phases 5–6 (torus / topology need many points).

### 3.3 Four-level stimulus hierarchy

| Level | Definition | Example |
|---|---|---|
| **State** | one item of a structure | `Monday` |
| **Template / context** | a sentence frame with a `{X}` slot | `"We'll meet on {X}"` |
| **Prompt** | template × state | `"We'll meet on Monday"` |
| **Run** | one template applied to **all** states of the structure → the `d × n_states` activation matrix | context `r` over Mon…Sun → `X[day, r] ∈ ℝ^{d×7}` |

**Hard constraint:** every run spans the **full, identical state set** of its structure (never a partial set) — this is what makes each run a complete `d × n_states` matrix and makes runs comparable.

### 3.4 Two template pools

1. **Shared/neutral pool** — natural frames whose slot works **identically** for any single-token state (used for matched-context comparisons):
   - `"{X} is my favorite."` → "Monday is my favorite." / "January is my favorite."
   - `"Honestly, I love {X}."`, `"Let's talk about {X}."`, `"Nothing beats {X}."` (≈8 templates)
2. **Structure-specific pool** — natural per-structure frames that necessarily differ in wording (used for different-context comparisons):
   - day: `"We'll meet on {day}."`, `"The concert is on {day}."`
   - month: `"We'll meet in {month}."`, `"The concert is in {month}."` (≈8 per structure)

Runs are **paired by index across structures** (run `r` = the r-th template of each structure) so within- and cross-structure analyses share a fold structure and are directly comparable. Target ≈16 templates/runs per structure total.

---

## 4. PART 1 — Reproduce the cyclic manifolds (validation gate)

**Question.** Does Gemma 2 2B arrange day-of-week and month into circular manifolds, and where? Geometry only.

### 4.1 Extraction
Extract `blocks.{l}.hook_resid_post` at the **final token**, **all 26 layers**, for every (structure, state, template). Store keyed by (structure, state, template, layer).

### 4.2 Geometry analyses (per structure, per layer)
1. **PCA manifold** — mean-centered state matrix `W̄` (averaged over templates); plot PC1–PC2 (expect ring) and PC1–PC3.
2. **Gram matrix** `G = W̄ W̄ᵀ ∈ ℝ^{n_states × n_states}` heatmap — expect circulant/Toeplitz banding (each state most similar to cyclic neighbors). Here `W ∈ ℝ^{n_states × d}` (rows = states); note `W = Aᵀ` where `A ∈ ℝ^{d × n_states}` is Part-2's orientation. Standardize on this in code.
3. **Circle-fit metric** (quantitative "is it a circle"): fit the best circle in the top-2 PCA plane; report (a) fit residual, (b) **angular-order correlation** — circular–circular correlation between each state's fitted angle θ and its **canonical index** (Mon=1…Sun=7 / Jan=1…Dec=12), i.e. does the ring preserve calendar order, (c) variance captured by the top-2 PCs.
4. **Layer sweep** — circularity-vs-layer curve; selects the layers Part 2 focuses on.

### 4.3 Reproduction-fidelity details (mirror the Symmetry paper)
- **Exclude "May"** from the month PCA basis (polysemy: also a verb).
- Use the **uncentered Gram** variant for years (preserves Toeplitz structure); mean-centered for cycles. Keep both as options.

### 4.4 SAE reproduction check (secondary, Engels-style)
Load the Gemma Scope SAE at a clean layer, encode day/month activations, cluster decoder directions by cosine similarity, and confirm a cluster whose reconstructed activations form a circle in PCA. Qualitative confirmation the circle lives in the SAE feature dictionary. **SAE appears only here.**

### 4.5 Part 1 is a validation gate, not an open-sign question
- **Month + years geometry were produced on `gemma-2-2b` specifically** in the Symmetry paper → strict reproduction. **Day-of-week** was shown by Engels on other models → tiny extension, near-certain by cycle-family similarity.
- **Reproduction-failure protocol:** if month/years geometry does not reproduce, treat it as a **pipeline bug** and debug: read/token position, layer indexing, centering convention, tokenization of state words, template design, May-exclusion, uncentered-Gram-for-years. **Do not proceed to Part 2 until it reproduces.**
- **Expected non-bug negatives:** community and flat controls should *not* form clean ordered circles — the specificity check.
- **Gate:** Part 2 is interpreted only once Part 1 reproduces. This de-risks Part 2 — a null there can't be blamed on broken extraction.

### 4.6 Outputs
Per-layer manifold plots, Gram heatmaps, circularity-vs-layer curves, SAE cluster figure.

---

## 5. PART 2 — Is the cycle code abstract? (subspace generalization)

**Question.** Are cycles carried by a shared, reusable low-dimensional subspace? Method: subspace-generalization AUC on the SVD of the **original activation matrix**, using its **two sides** for the comparisons each is valid for.

### 5.1 Data matrix
For structure `s`, layer `l`, run (template) `r`: `A[s,l,r] ∈ ℝ^{d × n_states}` (columns = states, `d = 2304`), mean-centered across states. Runs provide replicates for leave-one-run-out CV. (Uncentered variant kept as a robustness option for open sequences.)

SVD: `A = U Σ Vᵀ`.
- **`U` (activation side)** — hidden-unit modes ("cell assemblies"), directions in the shared `ℝ^d`. Cross-projection well-defined across any state count → **used for cross-structure**.
- **`V` (feature side)** — mode-shapes over that structure's own states. State axes don't correspond across different-size structures → **used within-structure only**.

### 5.2 The AUC engine (leave-one-run-out)
For source structure `s`, target structure `s′`, held-out run `j`:
1. Source PCs `U_s` = left singular vectors of the **mean of held-in runs** of `s` (mirrors eLife held-out-runs averaging).
2. Project the **held-out** target run: `P = U_sᵀ A[s′,l,j]`.
3. Cumulative variance of the target captured by the top-`k` source PCs, normalized by target total variance:
   `M_k = Σ_{i≤k} ‖U_s[:,i]ᵀ A[s′,l,j]‖² / ‖A[s′,l,j]‖²`.
4. **AUC** = area under `M_k` vs `k`. Average over folds `j`. (Chance ≈ 0.5 = diagonal curve = variance spread uniformly across modes.)

Within a fixed-source contrast the source PC count (curve length) is constant → AUCs are directly comparable.

### 5.3 Comparison ladder + two-stage gating

**Stage 1 — within-structure reference (the gate).**
For each structure and layer, compute **within-structure, different-context AUC**: leave-one-run-out over that structure's runs (held-in runs build `U`, the held-out run is projected). The held-out run always uses a template unseen when building `U`, so it is automatically a *different-context* test. This is the **ceiling** ("is this structure robustly represented at this layer?").
- **Gate:** only layers where within-structure AUC is **significantly > chance** proceed to Stage 2.
- **Why the gate is required (not just convenient):** (a) it shrinks the number of cross-structure tests → gentler multiple-comparison correction → real effects easier to detect; (b) `ΔAUC = within − cross` uses the same-layer within AUC as the ceiling — if a layer failed the gate, its within AUC is near-chance noise and `ΔAUC` is meaningless. So `ΔAUC` is only interpretable at gate-passing layers.

**Stage 2 — cross-structure comparisons (only at gate-passing layers).**

| # | Comparison | Source → Target | Context | Pool | Purpose / prediction |
|---|---|---|---|---|---|
| 1 | Within-structure, diff. context | day → day | different | either | **Reference/gate.** Must be ≫ chance. |
| 2 | Across circular structures | day ↔ month | matched | shared | **Core abstraction test.** High ⇒ shared cycle code. |
| 3 | Across circular structures, diff. context | day ↔ month | different | structure-specific | **Robustness.** Survives context change ⇒ not phrasing-bound. |
| 4 | Circular → non-circular (control) | day → years / community / flat | any | either | **Specificity.** years (near) > community > flat ≈ chance. |

**Predicted ordering if an abstract cycle code exists:**
`AUC₁(within) ≳ AUC₂ ≈ AUC₃ (cross-cycle) > AUC₄(years) > AUC₄(community) > AUC₄(flat) ≈ chance`

Both cross-cycle directions are run (day→month and month→day).

### 5.4 `V`-side within-structure stability
Same leave-one-run-out setup as comparison 1, read off the **`V` side**: does the *pattern over states* recur across contexts? A representational-stability score and the within reference for the feature side. **Never run across different cycles.**

### 5.5 Statistics
- **CV:** leave-one-run-out.
- **Null (chance):** shuffle the `d` feature-identity rows of the target matrix before projection → AUC null ≈ 0.5; per-contrast p-values.
- **Effect:** `ΔAUC = within − cross`, bootstrap CIs over folds; graded-tier comparison across controls.
- **Multiple comparisons:** correction applied only over the surviving `(layer × comparison)` set from Stage 2 (kept small by the gate).
- **Repeated-measures units:** template folds (later, models). No subject dimension — reported honestly as such.

### 5.6 Outputs
AUC-vs-layer curves per contrast (`U` side), within-structure stability scores (`V` side), null distributions, `ΔAUC` table with CIs.

---

## 6. Architected extensions (out of v1 scope, no refactor required)

- **Phase 3 — Boundary / start-point code (conditional).** Runs only if years shares the `U` subspace with cycles. Learn a linear "boundary vs interior" direction on one sequence (years: endpoints; cycles: candidate seam Jan / Monday) and test **cross-sequence transfer** vs shuffled null. Significant transfer ⇒ an abstract "edge-of-sequence" code shared across sequences; null ⇒ cycles are true seamless rings.
- **Phase 4 — Causality (runtime intervention).** Engels-style: fit a circular probe, patch the 2D circular subspace toward a target angle, read the **logit-difference** toward the target token; off-distribution `(r,θ)` sweep confirms angular (identity) vs radial coding. **Cross-cycle causal transfer** (your extension): steer a *month* arithmetic prompt using the circular plane derived from *day* activations (and vice versa), with **control planes** (years / community / random). Predicted: `self ≈ cross-cycle ≫ years > community/random`. Requires the model to actually do day/month modular arithmetic (Gemma 2 2B capability TBD; may need Llama/Mistral). Needs write-hooks (already required).
- **Phase 5 — Compositional cycles (torus / factorization).** Compositional date stimuli (day × week/month, e.g. "first Monday of April"). Test whether the day-of-week subspace is **invariant across month contexts** (factorized, torus-like) via the same `U`-side engine (extract day subspace within April, project onto day variance within December). Tightest grid-cell analogy (grid modules are toroidal).
- **Phase 6 — Intrinsic ring topology in *activation* space (distinct from Part 1's *representation* space).** Part 1 works in **representation space**: it averages over templates to get one representation per state (7 day-points, 12 month-points) and studies how those concept means are arranged. Phase 6 works in **activation space**: it keeps the **individual activation samples** (many prompts/tokens, *not* averaged into n_states points) and asks whether that **population-activity manifold** intrinsically has ring (S¹) topology — the same question the neuronal-data papers ask of head-direction population activity (a ring) or grid-cell activity (a torus), where the topology is of the activity cloud itself, not of averaged condition representations.
  - Methods over the raw sample cloud: **PCA embedding** (linear first pass) → **graph Laplacian / Laplacian eigenmaps** (k-NN graph over the individual activation vectors; the two lowest non-trivial eigenvectors form a `cos`/`sin` pair iff the manifold is circular) → (later) **persistent homology** for a topology-invariant test.
  - Requires **many points**, so it pairs with densification (continuity-modifier states + many templates) — 7–12 averaged means are far too few. Input: a forthcoming paper summary on ring topology in neuronal data.

---

## 7. Repository layout (Python)

The two reference repos (MATLAB `ShirleyMgit/code_subspace_generalization_fMRI_paper`, notebooks `veronikasamborska1994/notebooks_paper`, and `JoshEngels/MultiDimensionalFeatures`) are **inspiration only** — we write cleaner, tested Python and ground the exact SVD orientation / AUC normalization / permutation details against them.

```
networkgeometry/
  stimuli/            # declarative structures, states, templates (YAML/JSON). No model code.
  extraction/         # (model, prompts) -> residual activations, all layers, final token. Read+write hooks.
  sae/                # Gemma Scope load + encode (Part-1 reproduction check only)
  geometry/           # Part 1: PCA, Gram, circle-fit, layer sweep, plots
  subspace/           # Part 2 core: SVD two-sided AUC engine (space-agnostic)
  stats/              # LOO-CV, identity-permutation null, gating, ΔAUC, bootstrap
  boundary/           # Phase 3 (stub)
  causality/          # Phase 4 (stub; write-hook interventions)
  compositional/      # Phase 5 (stub)
  topology/           # Phase 6 (stub; PCA + graph Laplacian)
  figures/ results/   # outputs
  tests/              # unit tests per module
```

**Data contract** between modules: `DataMatrix = {structure, layer, run, matrix[d × n_states], states}` so activity paths are interchangeable and the AUC engine is agnostic to what filled the matrix.

---

## 8. Deliverables

1. The module pipeline above (v1 = `stimuli`, `extraction`, `sae`, `geometry`, `subspace`, `stats`).
2. `docs/SAE-explainer.md` — plain-language explanation of SAEs (superposition, dictionary learning, Gemma Scope, how `z` is used here), written alongside this spec.
3. Part-1 figures (manifolds, Gram, circularity-vs-layer, SAE cluster).
4. Part-2 results (AUC-vs-layer per contrast, `ΔAUC` table + CIs, null distributions, `V`-side stability).
5. Short findings write-up.

---

## 9. Success criteria

- **Part 1:** the reproduction adjudicates — a quantitative yes/no on clean, correctly-ordered day/month circles and where they live. Failure ⇒ debug (bug), not a finding.
- **Part 2:** calibrated AUC contrasts with nulls, gated by layer, that give a clear yes/no on abstract cycle sharing (either sign), plus which layers carry it, plus the graded control ordering.

---

## 10. Explicit non-goals (YAGNI for v1)

- No natural-language co-occurrence / spectral-embedding theory (excluded per request).
- No SAE **training** (Gemma Scope only); SAE used only for the Part-1 reproduction check.
- No causal intervention / patching in v1 (Phase 4).
- No compositional, boundary, topology, or densified-state analyses in v1 (Phases 3, 5, 6).
- No multi-model sweep in v1 (pipeline stays model-agnostic so it's a later config add).
