# Not All Language Model Features Are One-Dimensionally Linear

**Paper:** [arXiv 2405.14860](https://arxiv.org/abs/2405.14860) (Accepted at ICLR 2025)  
**Authors:** Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, Max Tegmark  
**Code:** https://github.com/JoshEngels/MultiDimensionalFeatures

---

## 1. Core Claim

Language models encode some concepts as **irreducible multi-dimensional features** (e.g., 2D circular representations of days-of-the-week and months-of-the-year), not just as one-dimensional linear directions. These multi-dimensional features are causally used by the model for computation (modular arithmetic).

---

## 2. Formal Framework (Section 3)

### 2.1 Definitions

> **Clarification: "Feature" ≠ "Neuron"**
>
> In this paper (and in mechanistic interpretability broadly), a **feature** is a **direction in activation space** that corresponds to an interpretable concept. It generally does **not** align with a single neuron.
>
> The reason is **superposition**: a model has far more concepts to represent than it has neurons/dimensions ($d$), so it packs multiple concepts into overlapping directions across many neurons. A single neuron participates in many features; a single feature is spread across many neurons.
>
> This is why Sparse Autoencoders are needed: the SAE learns $m \gg d$ dictionary elements (directions), each activating sparsely on a small fraction of inputs. Each dictionary element is one "feature."
>
> - **Neuron** = one axis of the activation vector (dimension $j$ of $\mathbf{x} \in \mathbb{R}^d$)
> - **1D Feature** = a learned direction in activation space (a column $\mathbf{d}_k$ of the SAE decoder $\mathbf{D}$), generally a linear combination across many neurons
> - **Multi-dimensional feature** (this paper's contribution) = a concept that requires a $d_f$-dimensional subspace, not just a single direction — e.g., the 2D circular representation of days-of-the-week, spanned by multiple SAE dictionary elements

**Feature (formal):** A function $\mathbf{f}: \text{inputs} \to \mathbb{R}^{d_f}$ mapping a subset of inputs to $d_f$-dimensional vectors.

**Reducibility:** A feature $\mathbf{f}$ is **reducible** into features $\mathbf{a}$ and $\mathbf{b}$ if there exists an orthonormal rotation $\mathbf{R} \in \mathbb{R}^{d_f \times d_f}$ and constant $\mathbf{c}$ such that:

$$\mathbf{R}\mathbf{f} + \mathbf{c} = \begin{pmatrix} \mathbf{a} \\ \mathbf{b} \end{pmatrix}$$

and the joint distribution $p(\mathbf{a}, \mathbf{b})$ satisfies **at least one** of:

1. **Separability:** $p(\mathbf{a}, \mathbf{b}) = p(\mathbf{a}) \cdot p(\mathbf{b})$ (statistically independent components)
2. **Mixture:** $p(\mathbf{a}, \mathbf{b}) = w \cdot p(\mathbf{a})\delta(\mathbf{b}) + (1-w) \cdot p'(\mathbf{a}, \mathbf{b})$ with $0 < w < 1$

> **Intuition for the Mixture condition:** This captures sub-features that **don't always fire together**. The joint distribution is a weighted mix of two regimes:
>
> - With probability $w$: sub-feature $\mathbf{a}$ is active but $\mathbf{b} = 0$ (that's $\delta(\mathbf{b})$ — a point mass at zero). The two sub-features don't co-occur.
> - With probability $(1-w)$: both $\mathbf{a}$ and $\mathbf{b}$ can take any values according to some joint distribution $p'$.
>
> **Example:** Suppose an SAE cluster has 4 dictionary elements that look like a single 4D feature. But after rotating, dimensions (1,2) activate for "cat" contexts and dimensions (3,4) activate for "dog" contexts, never at the same time. Then a fraction $w$ of the time $\mathbf{a} = \text{(cat stuff)}, \mathbf{b} = \mathbf{0}$, and the rest of the time $\mathbf{a} = \mathbf{0}, \mathbf{b} = \text{(dog stuff)}$. The "4D feature" is actually two independent 2D features — it's reducible.
>
> **Contrast with Separability:** Separability catches features that are *always both active* but statistically independent (two unrelated co-occurring concepts). Mixture catches features that *take turns* being active. Together, the two conditions cover the ways a multi-dimensional feature could secretly be multiple lower-dimensional ones.
>
> **Link to $\varepsilon$-Mixture Index:** $M_\varepsilon$ operationalizes this by asking: "Is there any direction $\mathbf{v}$ along which a large fraction of the data lands near zero?" If yes, that direction likely separates an inactive sub-feature from an active one, indicating a mixture decomposition exists.

A feature is **irreducible** if no such decomposition exists.

### 2.2 Empirical Irreducibility Metrics

**Separability Index** — minimal mutual information across all rotations:

$$S(\mathbf{f}) = \min_{\mathbf{R}} I(\mathbf{a}; \mathbf{b})$$

- Lower $S$ → more separable → more reducible
- High $S$ → components are entangled → likely irreducible

**$\varepsilon$-Mixture Index** — fraction of data projectable near zero along some direction:

$$M_\varepsilon(\mathbf{f}) = \max_{\mathbf{v} \in \mathbb{R}^{d_f},\; c \in \mathbb{R}} \; \mathbb{P}_{\mathbf{t} \in \mathcal{T}} \left( |\mathbf{v} \cdot \mathbf{f}(\mathbf{t}) + c| < \varepsilon \sqrt{\mathbb{E}[(\mathbf{v} \cdot \mathbf{f}(\mathbf{t}) + c)^2]} \right)$$

- Higher $M_\varepsilon$ → more mixture-like → more reducible
- Low $M_\varepsilon$ → unlikely to be a mixture of non-co-occurring sub-features

**Combined Irreducibility Score** (used for ranking):

$$\text{Score} = (1 - M_\varepsilon(\mathbf{f})) \times S(\mathbf{f})$$

**Key hyperparameter:** $\varepsilon = 0.1$

---

## 3. Discovery Method (Section 4)

A four-step pipeline to find multi-dimensional features from trained sparse autoencoders:

### Step 1: Sparse Autoencoder (SAE) Training

> **What problem does the SAE solve?**
> A transformer's activation vector $\mathbf{x} \in \mathbb{R}^d$ at a given token position and layer is a dense superposition of many concepts. We want to decompose it into a sparse set of interpretable directions ("dictionary elements"). The SAE learns an overcomplete dictionary of $m \gg d$ directions, such that any activation can be reconstructed as a sparse linear combination of a few of those directions.

**Architecture — single hidden-layer autoencoder:**

```
Input:  x ∈ ℝ^d          (one activation vector from layer l, token position i)
                           ↓
Encode: z = ReLU(E · x)   (z ∈ ℝ^m, mostly zeros due to sparsity pressure)
                           ↓
Decode: x̂ = D · z         (x̂ ∈ ℝ^d, reconstruction)
```

- $\mathbf{E} \in \mathbb{R}^{m \times d}$: **encoder** — projects the $d$-dimensional activation into the $m$-dimensional overcomplete space. Each row of $\mathbf{E}$ is a "detector" for one dictionary element.
- $\mathbf{D} \in \mathbb{R}^{d \times m}$: **decoder** — each column $\mathbf{d}_k$ is a **dictionary element** (a direction in activation space representing one learned concept). The reconstruction is $\hat{\mathbf{x}} = \sum_k z_k \, \mathbf{d}_k$.
- $\mathbf{z} = \text{ReLU}(\mathbf{E} \cdot \mathbf{x}) \in \mathbb{R}^m$: **sparse code** — the coefficient vector. Most entries are zero; the nonzero $z_k$ tell us which dictionary elements are "active" for this input and with what magnitude.
- $m \gg d$: the dictionary is **overcomplete** — many more dictionary elements than activation dimensions, allowing the SAE to represent more concepts than the model has dimensions (this is the superposition hypothesis).

**Loss function:**

$$\mathcal{L} = \sum_{\mathbf{x}} \left[ \underbrace{\left\| \mathbf{x} - \mathbf{D} \cdot \text{ReLU}(\mathbf{E} \cdot \mathbf{x}) \right\|_2^2}_{\text{reconstruction error}} + \lambda \underbrace{\left\| \text{ReLU}(\mathbf{E} \cdot \mathbf{x}) \right\|_0}_{\text{sparsity penalty}} \right]$$

- **Reconstruction term:** the SAE should faithfully reconstruct the original activation — minimizes information loss.
- **Sparsity term ($L_0$):** penalizes the number of active (nonzero) dictionary elements — forces the SAE to explain each activation using only a few directions, making each direction more likely to correspond to a single concept.
- $\lambda$: controls the reconstruction-vs-sparsity tradeoff. Higher $\lambda$ → fewer active elements per input → more interpretable but potentially lossy.
- In practice, $L_0$ is non-differentiable, so it is **relaxed to $L_p$** with $0 < p \leq 1$ (e.g., $L_1$ penalty $\|\mathbf{z}\|_1$).

> **Why does this matter for finding multi-dimensional features?**
> The standard assumption is that each dictionary element $\mathbf{d}_k$ is one feature (1D direction). But if the model encodes a concept like "day of the week" as a 2D circle, then the SAE will use **multiple dictionary elements pointing in nearby directions** to approximate that circle. Step 2 (clustering) detects these groups of co-aligned elements and recovers the multi-dimensional structure.

**Models & SAEs used:**
| Model | SAE Source | Layers |
|-------|-----------|--------|
| GPT-2-small | Pre-trained (Bloom 2024) | All layers |
| Mistral 7B | Trained by authors | Layers 8, 16, 24 |

### Step 2: Cluster Dictionary Elements

1. Build a complete graph over the $m$ dictionary element column vectors in $\mathbf{D}$
2. Edge weight = **cosine similarity** between dictionary elements
3. Prune all edges with similarity below threshold $T$
4. Extract **connected components** as candidate multi-dimensional feature clusters
5. Alternative: spectral clustering (details in Appendix F)

*Note: Exact threshold $T$ is not specified in the paper.*

### Step 3: Reconstruct Cluster Activations

For each cluster:
1. Zero out (ablate) all dictionary elements **not** in the cluster
2. Collect the reconstructed activations on data points where at least one cluster element is active
3. This gives the activation subspace attributable to that cluster

### Step 4: PCA + Irreducibility Testing

1. Apply PCA to the reconstructed cluster activations
2. Test consecutive 2D PCA projections: (PC1, PC2), (PC2, PC3), (PC3, PC4), (PC4, PC5)
3. For each 2D projection, compute $S(\mathbf{f})$ and $M_\varepsilon(\mathbf{f})$
4. Average scores across PCA planes
5. Rank clusters by the combined irreducibility score

**Result:** ~1000 candidate clusters found across models. Days-of-week circle ranked 9th, months ranked 28th, years ranked 15th by irreducibility score.

---

## 4. Causal Validation via Intervention (Section 5)

### 4.1 Circular Probe Training

A linear probe $\mathbf{P} \in \mathbb{R}^{2 \times k}$ maps PCA-reduced activations to a unit circle:

$$\mathbf{P} = \arg\min_{\mathbf{P}'} \sum_{\mathbf{x}^j_{i,l}} \left\| \mathbf{P}' \cdot \mathbf{W}_{i,l} \cdot \mathbf{x}^j_{i,l} - \texttt{circle}(\alpha) \right\|_2^2$$

where:
- $\mathbf{W}_{i,l}$: PCA projection matrix (top $k=5$ components)
- $\texttt{circle}(\alpha) = [\cos(2\pi\alpha/m), \sin(2\pi\alpha/m)]$
- $m = 7$ for days, $m = 12$ for months
- Solved in closed form (least squares)

### 4.2 Activation Intervention

Replace the model's circular subspace with a target representation:

$$\mathbf{x}^{j*}_{i,l} = \overline{\mathbf{x}_{i,l}} + \mathbf{W}_{i,l}^T \mathbf{P}^+ \left( \texttt{circle}(\alpha_{j'}) - \overline{\mathbf{x}_{i,l}} \right)$$

- $\overline{\mathbf{x}_{i,l}}$: average activation across all prompts
- $\mathbf{P}^+$: pseudoinverse of the probe
- $\alpha_{j'}$: target label (the day/month we want to steer toward)
- Components orthogonal to the circular subspace are average-ablated

### 4.3 SAE-Based Intervention (Alternative)

Instead of layer-level PCA, use the SAE cluster's own 2D plane (from Step 4 of discovery) for intervention. At Mistral 7B layer 8, the weekdays circle projects onto PCA dimensions 2-3 of the discovered cluster.

### 4.4 Off-Distribution Probing

Grid sweep over $(r, \theta)$ within the circular manifold:
- $r \in \{0, 0.1, 0.2, \ldots, 2.0\}$ (21 radial steps)
- $\theta \in \{0, 2\pi/100, \ldots, 198\pi/100\}$ (100 angular steps)
- Tests whether the model uses angular position (concept identity) vs. radial magnitude

---

## 5. Experimental Setup

### 5.1 Task Datasets

**Weekday modular arithmetic:**
- Template: `"Let's do some day of the week math. Two days from Monday is"`
- 7 days $\times$ 7 durations = **49 prompts**

**Month modular arithmetic:**
- Template: `"Let's do some calendar math. Four months from January is"`
- 12 months $\times$ 12 durations = **144 prompts**

**Continuity dataset:**
- Synthetic: `"[very early/very late/morning/evening] on [Day]"`
- Tests interpolation along the circular manifold

### 5.2 Models Evaluated

| Model | Weekdays Acc. | Months Acc. |
|-------|:---:|:---:|
| GPT-2-small | 8/49 (16%) | 10/144 (7%) |
| Mistral 7B | 31/49 (63%) | 125/144 (87%) |
| Llama 3 8B | 29/49 (59%) | 143/144 (99%) |

### 5.3 Intervention Baselines

1. Replace top 5 PCA dimensions from a clean run, leave remainder unchanged
2. Replace entire layer from a clean run
3. Replace entire layer with task-averaged activation

### 5.4 Key Evaluation Metric

**Average logit difference** between the original correct token and the intervened target token, across all patching experiments.

---

## 6. Key Results

1. **Circular features exist and are irreducible** — days and months form clear circles in 2D PCA projections of SAE cluster activations
2. **Circular intervention outperforms baselines** — patching the 2D circular subspace is more effective than patching top-5 PCA components for steering model output
3. **Early-layer interventions are most effective** — approaching full-layer patching performance
4. **Angular position encodes concept identity** — off-distribution sweep shows the model responds to $\theta$ (which day/month), not $r$ (magnitude)
5. **SAE-based features are more robust** — circular probe trained on SAE plane at layer 8 generalizes better across layers than per-layer probes
6. **Months task shows stronger effects** than weekdays (larger modulus, more training data coverage)

---

## 7. Information Needed for Reproduction

### Available
- Code: https://github.com/JoshEngels/MultiDimensionalFeatures
- GPT-2 SAEs: from Bloom (2024), publicly available
- Models: GPT-2-small, Mistral 7B, Llama 3 8B (all public)
- Framework definitions and metrics are fully specified

### Gaps to Fill from Code/Appendix
- Exact SAE dictionary size $m$ for Mistral 7B
- Exact sparsity coefficient $\lambda$
- Cosine similarity threshold $T$ for clustering
- SAE training data composition and size
- Learning rate / optimizer for SAE training on Mistral
- Spectral clustering details (Appendix F)
- Mutual information estimation method for separability index

---

## 8. Reproduction Plan Outline

1. **Phase 1 — SAE Training/Loading:** Load pre-trained SAEs for GPT-2; train SAEs for Mistral 7B on layers 8, 16, 24
2. **Phase 2 — Feature Discovery:** Cluster dictionary elements by cosine similarity → extract connected components → reconstruct cluster activations → PCA → compute irreducibility scores
3. **Phase 3 — Identify Circular Features:** Rank clusters by irreducibility score, inspect top candidates for circular structure (days, months)
4. **Phase 4 — Causal Validation:** Train circular probes → run activation interventions on weekday/month prompts → compare with baselines
5. **Phase 5 — Off-Distribution Analysis:** Grid sweep over $(r, \theta)$ to confirm angular encoding
