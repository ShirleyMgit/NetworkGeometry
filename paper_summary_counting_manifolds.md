# When Models Manipulate Manifolds: The Geometry of a Counting Task

**Paper:** [arXiv 2601.04480](https://arxiv.org/abs/2601.04480) (January 2026)  
**Authors:** Wes Gurnee, Emmanuel Ameisen, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, Joshua Batson  
**Model studied:** Claude 3.5 Haiku

---

## 1. Core Claim

Claude 3.5 Haiku solves fixed-width linebreaking (deciding where to insert `\n` to keep lines under $k$ characters) through a sequence of geometric transformations: it represents character counts on **low-dimensional curved manifolds** discretized by sparse features that behave like biological place cells, then uses attention heads to rotate and combine these manifolds into linear decision boundaries.

---

## 2. Motivation & Gap

- Language models receive only token sequences (integers) yet perform spatially-aware tasks like linebreaking in fixed-width text (code, emails, legal documents).
- Prior interpretability work studied token-position encodings but not **character-position within lines** — a naturally learned scalar quantity.
- The paper bridges two views of mechanistic interpretability: **discrete feature attribution** (sparse features, circuits) and **continuous geometric** (manifolds, subspaces). It argues both are needed.

---

## 3. Dataset Construction

Synthetic dataset from diverse prose corpus:
1. Strip all newlines
2. Reinsert newlines every $k$ characters at the nearest word boundary $\leq k$
3. Vary $k \in \{15, 20, 25, \ldots, 150\}$

> **Why synthetic:** Gives programmatic control over line-width constraints and enables systematic study across all character counts 1–150 and line widths. The model adapts quickly — by the 3rd line it performs at high accuracy, confirming it can learn the task in-context.

---

## 4. Methodology

The paper follows a progression: discover features → find manifold structure → trace computation through attention → verify causally → derive theoretical optimality.

### 4.1 Feature Discovery (Sparse Dictionary Learning)

**Tool:** 10-million-feature **Weakly Causal Crosscoder (WCC)** trained on Claude 3.5 Haiku's residual streams.

> **Why WCC over standard SAE:** Weakly Causal Crosscoders enable cross-layer feature discovery, connecting embeddings through to output layers. A standard SAE is trained per-layer and cannot track a feature's identity across layers.

**Feature identification procedure:**
1. For each of the 10M features, compute its mean activation **binned by line character count** (how many characters into the current line the token is)
2. Select features with smooth, bell-curve-like tuning profiles and large between-count variance
3. Result: **10 character-count features**, each with a tuning curve peaked at a different character count, with overlapping support (always ~2 features active at any count)

> **Analogy to neuroscience:** These features behave like **place cells** — each fires maximally for a specific "location" (character count) and tapers off to either side, collectively tiling the full range 1–150.

### 4.2 Subspace Discovery via PCA

1. Compute the **mean residual stream at layer 2** for each of the 150 character-count values → 150 vectors in $\mathbb{R}^d$
2. Run PCA on these 150 vectors
3. **Top 6 PCs capture 95% of variance** → defines the "character count subspace"

> **Why PCA:** Provides an orthogonal low-dimensional embedding that reveals the manifold's shape while remaining computationally simple and interpretable. The key finding is that the manifold has **intrinsic dimension 1** (a curve) embedded in ~6D space.

**What the manifold looks like:** A helix-like curve with "rippling" — high-frequency oscillations superimposed on the smooth curve. Neighboring character counts are close on the manifold; distant counts are far apart.

### 4.3 Feature Reconstruction

For each token, reconstruct its activation using only the 10 identified features:

$$\hat{\mathbf{x}} = \sum_{i=1}^{10} f_i(\mathbf{x}) \cdot \mathbf{d}_i$$

where $f_i(\mathbf{x})$ is feature $i$'s activation and $\mathbf{d}_i$ is its decoder direction. This reconstruction faithfully traces the manifold with mild kinks at feature maxima (spline-like interpolation between feature peaks).

> **Why reconstruct from features:** Validates that the 10 discrete features are sufficient to explain the continuous manifold structure — bridges the discrete (feature) and continuous (geometric) views.

### 4.4 Supervised Probe Training

**150 independent logistic regression probes** (one per character-count value $c \in \{1, \ldots, 150\}$):

$$\text{logit}_c(\mathbf{x}) = \mathbf{W}_c \cdot \pi(\mathbf{x}) + b_c$$

where $\pi(\mathbf{x})$ is the projection onto the 6D character-count subspace.

- Loss: binary cross-entropy per probe
- Result: linear probe $R^2 = 0.985$ at layer 1

**Post-hoc analysis of probe weights:** PCA on the 150 weight vectors $\{\mathbf{W}_c\}$ reveals a "ringing" pattern in their cosine similarity matrix — off-diagonal bands indicating that the representation of count $c$ has systematic interference with counts $c \pm \Delta$ for specific $\Delta$ values.

### 4.5 Causal Interventions

Two types of intervention to verify the character-count subspace is causally used:

**Ablation** — remove the subspace:
$$\mathbf{a}_{\text{ablated}} = \mathbf{a} - \pi_k(\mathbf{a})$$

where $\pi_k$ projects onto the top-$k$ PCA dimensions. Compare loss impact vs. ablating a random $k$-dimensional subspace.

- Result: ablation specifically increases loss on **newline tokens** while leaving non-newline tokens unaffected → the subspace is specifically used for linebreak decisions.

**Patching** — substitute a different character count:
$$\mathbf{a}_{\text{patched}} = \mathbf{a}_{\text{original}} - \boldsymbol{\mu}_{\text{original}} + \boldsymbol{\mu}_c$$

Replace the mean activation at the original character count with the mean activation at target count $c$. Applied to layers 0–2, tokens $t$ and $t-1$.

> **Why both intervention types:** Ablation shows **necessity** (removing the subspace breaks the task). Patching shows **sufficiency** (substituting a specific count value steers the output predictably).

---

## 5. How the Model Counts Characters

### 5.1 Embedding Geometry (Layer 0 Input)

Token character lengths are already geometrically structured in the embedding layer:
- First 2 PCs of embeddings show a circular pattern (different lengths distributed around a circle)
- PC3 adds an oscillating component (rippling)
- Top 3 PCs capture 70% of variance

> **Why pre-represented:** Token length is intrinsic to token identity. The model learns this structure from the embedding table before any transformer computation.

### 5.2 Layer 0: Distributed Counting via Attention Heads

**5 key attention heads in layer 0** each contribute a low-rank approximation to the character count.

**QK circuit (where to attend):**
Each head $h$ has:
- **Sink position $s_h$**: for tokens 0 to $s_h$ after a newline, attend purely to the previous newline (attention sink)
- **Receptive field $r_h$**: for tokens $s_h$ to $s_h + r_h$, attention smears over a window of preceding tokens

**OV circuit (what to output):**
$$\text{output}_h \approx (\text{tokens in sink}) \times \mu_c + \text{correction}$$

where $\mu_c \approx 4$ (average token length in characters). The correction adjusts for tokens that are above or below average length within the receptive field.

> **Concrete example (Head L0H1):** sink size = 1 token, receptive field ≈ 8 tokens. Output ≈ $1 \times 4 + \text{adjustments}$ ≈ 4-character estimate per token scanned.

Each head produces a **1D representation** (a rough estimate). Their **sum** produces the curved manifold.

> **Why multiple heads are necessary:** A single head outputs a low-rank (roughly 1D) approximation. The manifold's curvature requires multiple independent 1D contributions combined — just as a helix in 3D requires at least 3 independent sinusoidal components.

### 5.3 Layer 1: Refinement

6 additional heads receive the layer-0 output (coarse estimate) and produce higher-curvature corrections by leveraging the existing manifold structure.

**Combined result:** 11 heads total (5 in L0 + 6 in L1) yield $R^2 = 0.97$ for character count prediction.

### 5.4 Line Width Computation (Separate Pathway)

A separate set of "width counting" heads activate on **newline tokens** (not current-line tokens), estimating the line width $k$.

- Different attention anchor: current newline → looks back at line content
- Mechanism partially disjoint from character-count heads
- **Open question:** How does the model aggregate line widths globally (max? moving average? exponential decay?)

---

## 6. Boundary Detection: Attention Head Geometry

### 6.1 QK/OV Circuit Decomposition

Standard transformer attention:
$$\text{scores} = \mathbf{Q}\mathbf{K}^T / \sqrt{d}, \quad \text{weights} = \text{softmax}(\text{scores}), \quad \text{output} = \text{weights} \cdot \mathbf{V}$$

The paper decomposes this into:
- **QK circuit** (attention pattern): determines *where* to attend
- **OV circuit** (value transformation): determines *what* information flows

**Boundary head analysis method:**
1. Multiply line-width probes through $\mathbf{W}_K$ (key weights)
2. Multiply character-count probes through $\mathbf{W}_Q$ (query weights)
3. Compute cosine similarities in QK space
4. Visualize in joint PCA

**Key finding — geometric rotation:**

| Space | Max cosine similarity | Alignment |
|-------|:---:|---|
| Residual stream | ~0.25 | count $i$ aligns weakly with width $k = i$ |
| QK space | ~1.0 | count $i$ aligns strongly with width $k = i + \varepsilon$ |

The boundary head **rotates** the character-count manifold to align with the line-width manifold at an offset $\varepsilon$. This means "character count $i$" attends maximally to "line width $k$ slightly larger than $i$" — detecting when you're approaching the boundary.

### 6.2 Stereoscopic Algorithm (Multiple Boundary Heads)

The model uses ~3 boundary heads per layer, each with a different offset $\varepsilon_h$, sink position $s_h$, and receptive field $r_h$.

**Why multiple heads:** A single head cannot distinguish adjacent counts across the full 150-character range without norm explosion. Multiple heads with overlapping, offset tuning curves achieve fine resolution everywhere:
- Head 0: high variance in [0,10] and [15,20] ranges
- Head 1: peaks in [10,20] range  
- Head 2: peaks in [5,15] range
- **Sum:** 2D manifold with consistent resolution across all values

> **Analogy:** Like stereoscopic vision — each eye gives a slightly different offset view; combining them yields depth (fine distance resolution).

---

## 7. Final Linebreak Decision

### 7.1 Orthogonal Geometry

At ~90% model depth, two quantities are arranged in **nearly orthogonal subspaces**:
- Characters remaining until line boundary ($i$)
- Next-token character length ($j$)

**Decision rule** (linear separability):
$$v_{\text{remaining},i} + v_{\text{nextword},j} \quad \Rightarrow \quad \text{break if } (i - j) \geq 0$$

A single separating hyperplane achieves **AUC = 0.91** on ground truth linebreak decisions.

> **Why orthogonal arrangement:** Enables a simple linear computation. The alternative — redistributing probability mass from all over-length words to the newline token — would require complex nonlinear computation. The model instead arranges representations so a linear readout suffices.

---

## 8. Rippling: Why the Manifold Is Curved

### 8.1 The Problem

Represent 150 discrete ordinal values in a low-dimensional space (~6D). In 150D, the ideal solution is orthogonal vectors (cosine similarity = $\delta_{ij}$). But in 6D, orthogonality is impossible.

### 8.2 Theoretical Result

The **optimal** low-dimensional embedding of an ordinal sequence under capacity constraints exhibits "rippling" — high-frequency oscillations:
- Main diagonal band: neighboring counts are similar (desired)
- Off-diagonal bands: systematic interference at regular intervals (unavoidable)

**Toy model verification:**
1. Start with 150 orthogonal unit vectors in $\mathbb{R}^{150}$
2. Keep only the top 5 eigenvectors → 5D embedding
3. Result: the embedding traces a circle with rippling — matching what's observed in the model

**Physical simulation:**
- 100 points on a 6D hypersphere with circular topology
- Attractive forces to 6 nearest neighbors on each side
- Repulsive forces to all others
- Optimize → in 3D, produces a baseball-seam topology; in 6D, produces the observed rippled helix

### 8.3 Connection to Fourier Features

The rippled representations are related to **discrete Fourier transform modes**. Cosine basis functions with different frequencies naturally produce the interference pattern seen in the off-diagonal bands of the probe similarity matrix.

> **Why this matters for reproduction:** Rippling is not noise or a training artifact — it is the **theoretically optimal** solution. Any reproduction should expect to see it. It also connects to the sinusoidal tuning curves of the sparse features (Section 4.1).

---

## 9. Visual Illusions (Adversarial Examples)

### 9.1 Discovery

Layer 0 counting heads attend to `@@` (git diff delimiter) as an attention sink in some contexts, disrupting the counting mechanism.

### 9.2 Construction & Evaluation

- Insert `@@` into prompts where it shouldn't trigger attention redirection
- Newline probability drops from 0.79 → ~0.5
- Tested 180 two-character sequences systematically
- **Most disruptive:** code/delimiter-related (`@@`, `''`, `>>`, `}}`, `;|`, `||`)
- **Least disruptive:** random letter pairs (`zx`, `qp`)

> **Why "visual illusions":** Analogous to human visual illusions that exploit learned priors (e.g., perspective cues applied incorrectly). The model's learned prior that `@@` signals a code context overrides its counting mechanism.

---

## 10. Key Quantitative Results

| Measurement | Value |
|---|---|
| Linear probe $R^2$ (layer 1, char count) | 0.985 |
| PCA variance explained (6D subspace) | 95% |
| Logistic probe RMSE | 5 characters |
| Layer 0 heads alone $R^2$ | 0.93 |
| Layer 0+1 heads $R^2$ | 0.97 |
| Boundary head cosine sim (QK space) | ~1.0 |
| Ablation impact | Large on `\n` tokens, negligible elsewhere |
| Final decision hyperplane AUC | 0.91 |

---

## 11. What's Needed for Reproduction

### Available
- Task is fully specified (synthetic linebreaking with variable $k$)
- Probe architectures and intervention formulas are given
- Theoretical models (Fourier, physical simulation) are self-contained

### Requires Access / Approximation
- **Claude 3.5 Haiku internals** — residual streams, attention weights, WCC features (not publicly available)
- **10M-feature WCC** — proprietary dictionary learning tool
- Partial replication possible with open models: paper briefly mentions **Gemma 2 2B** and **Qwen 3 4B** in appendix as showing similar structure

### Reproduction Strategy for Open Models
1. Pick an open model capable of linebreaking (Gemma 2 2B, Qwen 3 4B, or similar)
2. Replace WCC with a standard **SAE** or **TopK SAE** trained per-layer on residual streams
3. Follow the same pipeline: feature identification by char-count binning → PCA subspace → probes → interventions → attention head analysis
4. Expect: similar manifold structure and rippling (if the model solves the task); details of head count and offsets will differ

---

## 12. Connections to the Multi-Dimensional Features Paper (Engels et al.)

Both papers study **curved manifolds in neural network representations**:

| Aspect | Engels et al. (2405.14860) | Gurnee et al. (2601.04480) |
|---|---|---|
| Manifold type | 2D circles (days, months) | 1D helix (character counts 1–150) |
| Modular structure | mod 7, mod 12 | mod $k$ (variable line width) |
| Discovery method | SAE → clustering → PCA | WCC → feature binning → PCA |
| Causal validation | Circular probe + intervention | Ablation + patching |
| Theoretical grounding | Irreducibility framework | Fourier optimality of rippling |
| Key shared insight | Discrete concepts encoded as continuous periodic geometry | Same |

---

## 13. Key Referenced Papers (Short Summaries)

**[7] Chang, Tu & Bergen (2022)** — *The Geometry of Multilingual Language Model Representations.* Shows that in multilingual models (e.g., XLM-R), languages occupy similar linear subspaces after mean-centering, with language-specific and universal features arranged along orthogonal axes. Relevant here as an early example of geometric subspace analysis in representations.

**[11] Coenen, Reif, Yuan, Kim, Pearce, Viégas & Wattenberg (2019)** — *Visualizing and Measuring the Geometry of BERT.* Finds that semantic and syntactic features in BERT occupy distinct geometric subspaces, with fine-grained word sense representations forming interpretable clusters. Pioneered geometric visualization of transformer internals.

**[31] Hewitt & Manning (2019)** — *A Structural Probe for Finding Syntax in Word Representations.* NAACL 2019. Trains linear probes that recover parse tree distances and depths from BERT/ELMo embeddings, showing that syntax is encoded as a geometric structure (tree metric) in a linear subspace. Foundational work for the probing methodology used throughout this paper.

**[32] Hindupur, Lubana, Fel et al. (2026)** — *Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry.* NeurIPS 2026. Formalizes the relationship between SAE dictionary elements and the geometric structure of concepts in activation space — shows that SAE features and concept subspaces are dual views of the same underlying structure. Directly relevant to understanding why SAE-discovered features trace manifolds.

**[42] Li, Michaud, Baek, Engels, Sun & Tegmark (2024)** — *The Geometry of Concepts: Sparse Autoencoder Feature Structure.* Shows that SAE-extracted features exhibit multi-scale geometric structure: parallelogram analogies at the micro level, spatially clustered feature lobes at the meso level, and power-law distributed eigenvalues at the macro level. Provides the geometric lens through which this paper's manifold discoveries are interpreted.

**[48] Modell, Rubin-Delanchy & Whiteley (2025)** — *The Origins of Representation Manifolds in Large Language Models.* Demonstrates that cosine similarity in representation space encodes the intrinsic geometry of feature manifolds via shortest paths — connecting distance in activation space to conceptual relatedness. Provides theoretical grounding for why the character-count manifold has the specific curved shape observed.

**[59] Park, Choe, Jiang & Veitch (2024)** — *The Geometry of Categorical and Hierarchical Concepts in Large Language Models.* ICLR 2025 (Best Paper, ICML 2024 MechInterp Workshop). Extends the linear representation hypothesis by showing categorical concepts are encoded as polytopes and hierarchical concepts as nested vector structures, validated on 900+ WordNet concepts in Gemma/LLaMA-3. Provides the theoretical framework for how discrete concept categories become geometric objects.

**[73] Wattenberg & Viégas (2024)** — *Relational Composition in Neural Networks: A Survey and Call to Action.* Argues that interpretability must move beyond discovering individual feature directions to understanding how models **compose** features to represent relationships. Directly motivates this paper's focus on how multiple attention heads combine 1D outputs into higher-dimensional manifolds.

**[74] Wollschläger, Elstner, Geisler et al. (2025)** — *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence.* Shows that LLM refusal mechanisms use multiple independent directions and multi-dimensional "concept cones" rather than a single refusal direction. Demonstrates that multi-dimensional geometric structure (not just 1D linear directions) is functionally important — the same principle underlying this paper's character-count manifolds.
