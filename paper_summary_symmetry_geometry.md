# Symmetry in Language Statistics Shapes the Geometry of Model Representations

**Paper:** [arXiv 2602.15029](https://arxiv.org/abs/2602.15029) (ICML 2026)  
**Authors:** Dhruva Karkada, Daniel J. Korchinski, Andres Nava, Matthieu Wyart, Yasaman Bahri  
**Subjects:** Machine Learning (cs.LG), Disordered Systems (cond-mat.dis-nn), Computation and Language (cs.CL)

---

## 1. Core Claim

The striking geometric structure observed in LLM representations (months forming circles, years forming curved manifolds, cities linearly decodable by coordinates) is **not an accident of training** — it is a mathematical consequence of **translation symmetry** in word co-occurrence statistics. The paper derives these geometries analytically and validates them across word embeddings, text embeddings, and LLM internal representations.

---

## 2. Motivation & Gap

**Empirical observation** (from prior work): language models arrange semantic concepts into interpretable geometric structures — months on circles, years on smooth 1D curves, geographic locations on spatial lattices.

**Open question prior to this paper:** *Why* does this happen? Is it a training artifact? A model architecture effect?

**This paper's answer:** It is a consequence of **statistical symmetries in language itself** — specifically, translation symmetry in co-occurrence, which forces the learned representations to take specific Fourier-mode structures regardless of the model or training procedure.

---

## 3. LLMs Analysed

### 3.1 Gemma 2 2B

- **Identifier:** `google/gemma-2-2b`
- **Parameters:** 2 billion
- **Architecture:** Transformer, 26 blocks (layers 0–25)
- **Hidden size:** $d = 2304$
- **Vocabulary:** 256K tokens
- **Role in paper:** Primary LLM used for internal representation analysis. Residual stream activations extracted at all 26 layers for calendar months, historical years, and US states.

### 3.2 EmbeddingGemma

- **Type:** 308M-parameter text embedding model from the Gemma 3 family
- **Output dimension:** 768 (unit-normalized)
- **Library:** SentenceTransformers (Hugging Face)
- **Role in paper:** Used for geographic semantic testing (US states). Produces fixed-length sentence-level embeddings.

### 3.3 Word Embedding Baseline (Spectral / Word2Vec approximation)

- **Not a neural LLM** — trained via spectral decomposition (closed-form eigendecomposition of the PMI matrix)
- **Corpus:** November 2023 English Wikipedia dump (3.37M articles, 2.72B tokens, top 25K vocabulary)
- **Embedding dimension:** $d = 1000$
- **Role in paper:** Main controlled baseline where theory predictions can be tested exactly, since the training procedure is analytically tractable.

> **Why these models?** They span the spectrum from fully tractable (spectral word embeddings, where training is analytically understood) to modern LLMs (Gemma 2 2B, where only activations are accessible). Validation across all three supports the claim that the geometry is driven by statistics, not by any particular architecture.

---

## 4. Analysis Methods for the LLM

### 4.1 Activation Extraction

**Library:** TransformerLens (with Hugging Face weights)

**What is extracted:** Residual stream activations at hook points `blocks.{l}.hook_resid_post` for all 26 layers.

**Where in the token sequence:** Final token position ($T-1$), where $T$ is the tokenized prompt length.

**Prompt templates used:**
```
Calendar months : "The month of the year is [X]"
Historical years: "In the year [X]"
US states       : "The location of the US state [X]"
```

> **Why the final token?** It aggregates information from the full prompt and is the position the model uses to generate the next token — most likely to contain the fully-processed semantic representation of the concept.

> **Why TransformerLens?** Provides clean hooks into residual stream positions at each layer without modifying the model, and is the community standard for mechanistic interpretability work on transformer internals.

**Semantic categories:**
- Calendar months: 12 prompts (January–December), "May" excluded from PCA basis (polysemy: also a verb)
- Historical years: years 1700–2020
- US states: 48 contiguous US states

### 4.2 PCA (Principal Component Analysis)

Applied to the mean-centered matrix of extracted activations.

- For months: "May" excluded from PCA computation to avoid polysemy artifacts
- For years: uncentered Gram matrix used (to preserve Toeplitz structure)
- Visualize top 2–3 PCs to reveal geometric manifold shape

> **Why PCA?** Standard, parameter-free method for revealing dominant geometric structure. The theoretical predictions are stated in terms of eigenvectors, so PCA directly tests them.

### 4.3 Gram Matrix Construction and Comparison

$$G = \bar{W}_S \bar{W}_S^T$$

where $\bar{W}_S$ is the matrix of mean-centered representation vectors for words in semantic set $S$.

The paper's core theoretical prediction is:

$$G = P \cdot M_S^* \cdot P^T$$

where $M_S^*$ is the submatrix of the PMI co-occurrence matrix restricted to the semantic set $S$, and $P$ is a rotation. Gram matrix analysis directly tests this prediction by comparing the predicted and empirical similarity patterns.

> **Why Gram matrices?** They capture all pairwise similarities between representations — a direct readout of geometry — and correspond exactly to the co-occurrence matrix under the theoretical model.

### 4.4 Linear Coordinate Decoding Probe

A linear transformation $\Omega \in \mathbb{R}^{r \times D}$ is trained to decode semantic coordinates (e.g., month index, year, latitude/longitude) from rank-$r$ PCA projections $\bar{w}_{i,r}$:

**Training:** Ridgeless linear regression (closed form)

**Evaluation protocol:** 100 random trials with 60/40 train-test splits

**Metric — normalized MSE:**

$$\varepsilon^2(r) = \frac{\mathbb{E}_{i \in S} \left\| \bar{w}_{i,r}^T \Omega - x_i \right\|^2}{\mathbb{E}\|x_i\|^2}$$

where $x_i$ is the true semantic coordinate (e.g., month number, year, geographic lat/lon).

**Theoretical error bound:**

$$\varepsilon^2(r) \lesssim \left(\frac{r}{\text{Vol}_D}\right)^{-1/D}$$

Asymptotic scaling: $\varepsilon^2 \sim r^{-1/D}$, where $D$ is the intrinsic dimension of the latent semantic space.

> **Why linear probes?** The theoretical prediction is that coordinates are encoded in linear subspaces (Fourier modes). A linear probe is the minimal test of linear decodability — if even a linear probe can recover coordinates, the representation is geometrically organized as predicted.

### 4.5 Eigenmode Visualization

For non-lattice data (US states), the theoretical prediction is that top PCA modes should correspond to slow spatial variation (low-frequency eigenvectors of the geographic co-occurrence matrix).

Method:
1. Numerically diagonalize the theoretical co-occurrence matrix $M_S^*$
2. Display top eigenvectors as heatmaps over the US map
3. Compare to top PCs of the empirical EmbeddingGemma representations

---

## 5. Theoretical Framework

### 5.1 Co-occurrence Matrix

$$M^*_{ij} = \frac{P_{ij} - P_i P_j}{\frac{1}{2}(P_{ij} + P_i P_j)}$$

Approximation (PMI): $M^* \approx \log(P_{ij} / P_i P_j)$, bounded $-2 \leq M^*_{ij} \leq 2$.

### 5.2 Learned Embeddings

Under spectral training:
$$W_{i\mu} = \Phi_{i\mu} \sqrt{|\Lambda^*_{\mu\mu}|}$$

where $\Phi$ are eigenvectors and $\Lambda^*$ are eigenvalues of $M^*$.

### 5.3 Translation Symmetry (Key Assumption)

For words $i, j$ in a semantic set $S$ on a latent continuum:

$$M^*_{ij} = C\big(\text{dist}(x_i, x_j)\big)$$

where $x_i \in \mathbb{R}^D$ are latent coordinates (e.g., position on a calendar, geographic coordinates), $\text{dist}$ is Euclidean distance (with periodic or open boundary conditions), and $C(\cdot)$ is a co-occurrence kernel.

> **Intuition:** Words that are semantically close (nearby months, nearby years) co-occur more. The co-occurrence depends only on the distance between their latent positions — this is translation symmetry, the same principle behind Fourier analysis.

### 5.4 Fourier Embedding Geometry (Main Result)

**For periodic boundary conditions** (closed loop — e.g., months):

$$\bar{w}_i = \sqrt{\frac{2}{|S|}} \left[ a_1 \cos(k_1 x_i),\; a_1 \sin(k_1 x_i),\; a_2 \cos(k_2 x_i),\; \ldots \right]$$

For exponential kernel $C(\Delta x) = \sum_{n \in \mathbb{Z}} e^{-|\Delta x + 2n|/\sigma}$:
- Wavenumbers: $k_n = \pi n$
- Amplitudes: $a_n = \sqrt{2\sigma / (1 + \sigma^2 k_n^2)}$
- $\sigma$: decay parameter estimated from empirical co-occurrence data

**For open boundary conditions** (open curve — e.g., years):

$$\bar{w}_i = \left[ a_1 \sin_{k_1}(k_1 x_i),\; a_2 \cos_{k_2}(k_2 x_i),\; \ldots \right]$$

Wavenumbers satisfy self-consistent quantization:
- Odd $n$: $k_n = \frac{(n+1)\pi}{2} - \arctan(\sigma k_n)$
- Even $n$: $k_n = \frac{n\pi}{2} + \arctan\!\left(\frac{k_n}{1 + \sigma(1+\sigma)k_n^2}\right)$

> **Consequence:** Periodic concepts (months) → circle (closed Fourier series). Non-periodic concepts (years) → rippled open curve (open-boundary Fourier series). Both match exactly what is observed in LLM activations.

### 5.5 Latent Variable Mechanism

The translation symmetry emerges from a latent variable $t$ (e.g., season, time of year) that modulates word occurrence:

$$P(i \mid t) = P(i)\big(1 + g(t - t_i)\big)$$

where $t_i$ is word $i$'s semantic center and $g$ is a zero-mean symmetric function.

Under conditional independence, the pairwise PMI becomes:

$$\text{PMI}(i,j) = \log\!\left(1 + \tilde{K}(t_i - t_j)\right) =: K(t_i - t_j)$$

This makes the PMI matrix **circulant** — diagonalized exactly by discrete Fourier modes, producing the Fourier embedding geometry analytically.

### 5.6 Robustness (Eigenvalue Stability)

Using Weyl/Davis-Kahan perturbation bounds: for fixed embedding dimension $d$ and large vocabulary $N$, perturbations affecting $O(1)$ matrix entries are negligible relative to eigenvalue gaps (proportional to $N$). This explains why month geometry is preserved even when month-month co-occurrences are completely removed — the geometry is maintained by "seasonal helper words" that co-occur with multiple months.

---

## 6. Experimental Setup

### 6.1 Word Embedding Training

| Parameter | Value |
|---|---|
| Corpus | English Wikipedia, Nov 2023 |
| Articles | 3.37M (filtered >200 tokens) |
| Tokens | 2.72B |
| Vocabulary $V$ | 25,000 most frequent words |
| Context window $L$ | 16 |
| Distance weighting $f(d)$ | $L + 1 - d$ |
| Embedding dimension | $d = 1000$ |
| Training method | Spectral decomposition of $M^*$ |

Co-occurrence probability:
$$P_{ij} = \frac{1}{Z} \sum_\nu \delta_{C[\nu],i} \sum_{d=1}^{L} f(d)\big(\delta_{C[\nu+d],j} + \delta_{C[\nu-d],j}\big)$$

### 6.2 LLM Extraction Details

| Parameter | Value |
|---|---|
| Gemma 2 2B layers | All 26 (0–25) |
| Extraction hook | `blocks.{l}.hook_resid_post` |
| Token position | Final token $T-1$ |
| Probe splits | 60/40 train-test, 100 random trials |
| Ridge regularization | Tuned per appendix Figure 13 |
| Year range | 1700–2020 |
| US states | 48 contiguous |

---

## 7. Key Results

| Finding | Evidence |
|---|---|
| Months form a circle in PCA | Closed Lissajous curve in PC1-PC2; matches periodic Fourier prediction |
| Years form rippled open curve | Open-boundary Fourier modes; "rippling" matches prediction |
| US states follow geographic modes | Top PCA modes = low-frequency geographic eigenvectors |
| Geometry preserved without direct co-occurrence | Month circle intact even when $M_S^* = 0$ (month-month entries zeroed) |
| Theory matches across all model types | Word embeddings, EmbeddingGemma, Gemma 2 2B all show predicted structure |
| Linear decoding error scales as $\varepsilon^2 \sim r^{-1/D}$ | Validated empirically with linear probes |
| Layer 26 Gemma 2B resolves "May" polysemy | Context-dependent disambiguation visible in activation space (Appendix Figure 14) |

---

## 8. Connection to Related Papers

| Paper | Connection |
|---|---|
| Engels et al. (2405.14860) — *Not All Features Are 1D Linear* | Empirically observed the circular month/day representations this paper explains theoretically |
| Gurnee et al. (2601.04480) — *When Models Manipulate Manifolds* | Observed rippled manifolds for character counts; this paper provides the statistical origin story |
| Park et al. (2024) — *Geometry of Categorical Concepts* | Context-dependent manifolds; compatible with latent variable view |
| Saxe et al. (2019) | Showed linear models on periodic lattice data produce circular representations (synthetic); this paper extends to real language |
| Neuroscience grid cells | Fourier-mode interference patterns in mammalian cortex — same mathematical structure |

---

## 9. Reproduction Notes

### What is fully specified
- Co-occurrence matrix construction formula and all hyperparameters
- Spectral embedding training procedure
- Prompt templates for LLM activation extraction
- TransformerLens hook names and extraction protocol
- Linear probe evaluation procedure (100 trials, 60/40 split, ridgeless regression)
- All theoretical derivations and proofs (Appendices B–E)

### What requires access / careful replication
- Gemma 2 2B weights: publicly available via Hugging Face (`google/gemma-2-2b`)
- EmbeddingGemma: available via SentenceTransformers
- Wikipedia corpus: publicly available via Hugging Face datasets
- Ridge regularization hyperparameter: see Appendix Figure 13

### Reproduction steps (open-model path)
1. Build spectral word embeddings from Wikipedia corpus using the co-occurrence formula above
2. Compute $M^*$ for semantic subsets (months, years, states)
3. Compare empirical Gram matrix to theoretical prediction $G = P M_S^* P^T$
4. Extract Gemma 2 2B residual stream activations using TransformerLens hooks
5. Apply PCA → compare to predicted Fourier mode geometry
6. Train linear coordinate decoding probes and verify $\varepsilon^2 \sim r^{-1/D}$ scaling
7. Robustness check: zero out $M_S^*$ month-month entries and verify month geometry is preserved
