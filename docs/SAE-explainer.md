# Sparse Autoencoders (SAEs), explained for this project

A companion to the cycle-geometry spec. This explains what an SAE is, the problem it solves, what Gemma Scope gives us, and *exactly* how we use it here — where it appears (the Part-1 reproduction check) and, just as importantly, where it does **not** (the Part-2 AUC analysis).

---

## 1. The problem: superposition

A transformer, at a given layer and token position, produces an **activation vector** `x ∈ ℝ^d` (for Gemma 2 2B, `d = 2304`). We'd like to read this vector as a list of human-interpretable "concepts that are currently active." But we can't just read off individual neurons (individual axes of `x`), because of **superposition**:

- A model needs to represent **far more concepts than it has dimensions** (`d`).
- So it packs many concepts into **overlapping directions** spread across many neurons.
- Consequently, **one neuron participates in many concepts**, and **one concept is smeared across many neurons**. A single axis is not "the Monday neuron."

Superposition works because, at any given moment, only a **small fraction** of all possible concepts are active — the code is *sparse*. That sparsity is the lever an SAE pulls on.

---

## 2. The idea: learn an overcomplete dictionary of directions

A **Sparse Autoencoder** learns to rewrite each activation vector as a **sparse combination of a large dictionary of directions**, each direction meant to correspond to one interpretable concept ("feature").

A single-hidden-layer autoencoder:

```
x ∈ ℝ^d                     one activation vector (layer l, token position i)
   │
   ▼  encode
z = ReLU(E · x) ∈ ℝ^m        sparse code: mostly zeros; nonzeros = "which features are active"
   │
   ▼  decode
x̂ = D · z ∈ ℝ^d              reconstruction of x
```

- `E ∈ ℝ^{m × d}` — **encoder**; each row detects one feature.
- `D ∈ ℝ^{d × m}` — **decoder**; each **column `d_k` is a dictionary element**: a direction in activation space standing for one concept. The reconstruction is `x̂ = Σ_k z_k · d_k`.
- `z ∈ ℝ^m` — the **sparse code**: most entries zero; the nonzero `z_k` say which features fired and how strongly.
- `m ≫ d` — the dictionary is **overcomplete** (many more features than dimensions), which is how it can name more concepts than the model has axes.

**Loss = reconstruct faithfully + stay sparse:**

```
L = Σ_x [ ‖x − D·ReLU(E·x)‖²   +   λ · (number of active features in ReLU(E·x)) ]
        └── reconstruction ──┘        └──────── sparsity penalty ────────┘
```

The sparsity term (an `L0` count, in practice relaxed to `L1`/`Lp` or handled by a JumpReLU/TopK variant) forces each activation to be explained by only a few dictionary directions — which is what pressures each direction toward a single, interpretable concept. `λ` trades reconstruction against sparsity.

**Key intuition for us:** if a concept like "day of the week" is stored by the model as a **2D circle**, an SAE won't capture it with one dictionary element — it uses **several nearby dictionary directions** to tile that circle. Finding a *cluster* of co-aligned dictionary elements is how you rediscover a multi-dimensional (e.g. circular) feature.

---

## 3. Gemma Scope — SAEs we don't have to train

**Gemma Scope** is a public suite of pretrained SAEs for **every layer** of Gemma 2 2B (and larger Gemma models), at various dictionary widths and sparsity levels. Because we use the frozen Gemma 2 2B and Gemma Scope's ready-made SAEs, **we never train an SAE** in this project — we just load one and run its encoder to get `z`, or read its decoder columns `D`.

---

## 4. How we use SAEs in *this* project (and how we don't)

This is the part that matters for the spec, and it changed during design — so it's worth stating sharply.

### 4.1 Where the SAE IS used — Part 1 reproduction check (§4.4)

We reproduce the Engels et al. finding that the day/month **circle lives in the SAE's feature dictionary**:

1. Load the Gemma Scope SAE at a clean layer (chosen from the Part-1 layer sweep).
2. Encode the day/month activations → look at which dictionary elements participate.
3. **Cluster decoder columns `d_k` by cosine similarity**; extract connected components.
4. Reconstruct the activations attributable to a cluster, run PCA, and confirm a cluster whose points trace a **circle**.

This is a **qualitative confirmation** that the circular structure is present at the level of the model's disentangled "concepts," not just in the raw activation basis. It is secondary and self-contained.

### 4.2 Where the SAE is NOT used — Part 2 AUC (important)

The abstract-cycle-code test (Part 2) runs the subspace-generalization AUC on the **SVD of the original activation matrix** `A ∈ ℝ^{d × n_states}`, **not** on SAE latents. The SVD already yields two complementary "sides":

- the **`U` (activation) side** — hidden-unit modes / "cell assemblies", used for cross-structure generalization;
- the **`V` (feature) side** — mode-shapes over states, used within-structure.

These two sides give us the two comparisons we need directly from the raw activations, exactly as in the Samborska/Mark method (where they are the "cellular" and "temporal" modes). **We do not build an `m`-dimensional SAE-latent data matrix for the AUC.** (An earlier design draft considered using `z ∈ ℝ^m` as a "features = neurons" substrate; that was dropped in favor of the two-sided SVD of the activation matrix, which is more parsimonious and is what the original neuroscience method actually does.)

**Why relegate the SAE?** SAE features tend to be **content-specific** — a "Monday" feature and a "January" feature can be *disjoint* directions. If we ran cross-cycle generalization on SAE latents, disjoint content features could make day and month look near-orthogonal **even when an abstract cycle geometry exists**, i.e. the SAE could *under-detect* the very sharing we're testing for. The raw-activation SVD sidesteps that. (Whether shared *phase* features exist is an interesting question, but it's a later add, not the v1 test.)

---

## 5. Glossary

| Term | Meaning |
|---|---|
| **Neuron** | one axis of the activation vector `x ∈ ℝ^d` |
| **Feature** | a *direction* in activation space corresponding to an interpretable concept — generally a combination across many neurons, not one neuron |
| **Dictionary element** | a decoder column `d_k` of the SAE = one learned feature direction |
| **Sparse code `z`** | the SAE encoder output; which features are active and how strongly |
| **Overcomplete** | `m ≫ d`: more dictionary elements than activation dimensions |
| **Superposition** | the model packing more concepts than dimensions into overlapping directions |
| **Multi-dimensional feature** | a concept needing a subspace, not a single direction (e.g. the day-of-week circle) — recovered as a *cluster* of dictionary elements |
| **Gemma Scope** | public pretrained SAEs for every Gemma 2 2B layer; we load, never train |
