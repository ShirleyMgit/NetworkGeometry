# Flexible Neural Representations of Abstract Structural Knowledge in the Human Entorhinal Cortex

**Paper:** [eLife 101134](https://elifesciences.org/articles/101134) (Published February 25, 2026)  
**Authors:** Shirley Mark, Philipp Schwartenbeck, Avital Hahamy, Veronika Samborska, Alon Boaz Baram, Timothy E Behrens  
**Journal:** eLife (Neuroscience Research Article)  
**DOI:** https://doi.org/10.7554/eLife.101134.3

---

## 1. Core Claim

Humans generalize knowledge across tasks that share only statistical structural form (hexagonal graph topology) even when the tasks differ in size and sensory content. The human entorhinal cortex (EC) supports this via **flexible, generalizable neural representations** that preserve coactivation structure across non-isomorphic tasks — extending findings from spatial grid cells to abstract non-spatial domains.

---

## 2. Motivation & Gap

**Background:**
- Rodent EC grid cells maintain their coactivation structure across different spatial environments — the relative firing of cell assemblies is preserved even when the fields shift and rotate. This means grid cell activity lies in the **same low-dimensional subspace** across environments.
- Prior work (Baram et al. 2021) showed EC generalizes over non-spatial tasks with **identical** graph structure (isomorphic graphs with the same number of nodes and stimuli).

**Gap:** What happens when tasks share only the structural *form* (e.g., both hexagonal) but differ in **size** (36 vs. 42 nodes) and **sensory stimuli**? These are non-isomorphic graphs with no one-to-one node correspondence between tasks.

**Why this matters:** Traditional analyses (RSA, repetition suppression) require explicit state-to-state correspondence across tasks. They cannot be applied when tasks have different numbers of states — which is exactly the case for real-world generalization.

---

## 3. Neural Data Recorded

### 3.1 Rodent Electrophysiology (Validation Dataset)

- **Source:** Chen et al. (2018) publicly available recordings
- **Species:** Mice
- **Brain regions:**
  - dmEC: grid cells
  - CA1: place cells
- **Recording technique:** Tetrode arrays (17 mm platinum-iridium, buffer amplified)
- **Number of neurons per animal:**

| Animal | Grid cells | Place cells |
|--------|:---:|:---:|
| 1 | 14 | 14 |
| 2 | 21 | 25 |
| 3 | 21 | 25 |

- **Environments:** Two square arenas — real arena (60×60 cm) and VR arena (60×60 or 90×90 cm, air-suspended ball)
- **Both cell types recorded concurrently within each animal**

### 3.2 Human fMRI

- **Modality:** BOLD fMRI
- **Scanner:** Siemens Prisma 3T, 32-channel head coil
- **Acquisition:**
  - T2\*-weighted EPI: TR=1.450s, TE=35ms, flip=70°, 2×2×2mm voxels, multi-band acceleration=4
  - Dual-echo field map for distortion correction
  - T1 MPRAGE structural: 1×1×1mm
  - Discard first 6 volumes
- **Primary ROI:** Bilateral entorhinal cortex (Jülich atlas mask)
- **Validation ROI:** Bilateral lateral occipital cortex (LOC, Harvard-Oxford atlas)
- **Exploratory ROI:** vmPFC
- **Sample:** N=28 (60 recruited → 34 scanned for high behavioral performance → 6 excluded for motion/sleepiness)
- **Session:** 4 fMRI runs × 5 blocks each; ~10 sequence trials per graph per run

---

## 4. Experimental Design

### 4.1 Graph Structures

Participants learned **4 graphs** total:

| Graph | Type | Nodes | Structure |
|---|---|---|---|
| Hex-large (Hl) | Hexagonal | 42 | Triangular lattice |
| Hex-small (Hs) | Hexagonal | 36 | Triangular lattice |
| Comm-large (Cl) | Community | 42 (6×7) | 6 clusters, degree-6 within |
| Comm-small (Cs) | Community | 35 (5×7) | 5 clusters, degree-6 within |

- Same-sized graphs across types **shared the same image set** (controls for visual content when comparing Hex vs. Community of same size)
- Training schedule: Days 1–2 Hexagonal, Days 3–4 Community, Day 5 fMRI

### 4.2 Behavioral Training Tasks (Days 1–4)

1. **Random walk sequences** (120 steps hex; 180 steps community): Passive learning; participants invented stories linking images
2. **Can It Be in Middle:** Does image X fit between Y and Z along a graph path? (16 questions/block)
3. **Extending Sequences:** Which 3-image sequence can a target image extend? (16 questions/block)
4. **Distance Estimation:** Which of two images is closer to a target? No direct neighbors shown, no feedback (45 questions/block)
5. **Navigation:** Navigate from start to target, choosing between 2 neighbors at each step; distances 2, 3, 4 links (3 games/block)

**Behavioral results (all above-chance, one-sample t-tests against 50%):**

| Task | Example statistic |
|---|---|
| Can It Be Middle | $t(27)=31.2,\; p<10^{-22}$ (Hex Day 1) |
| Extending Sequences | $t(27)=39.9,\; p<10^{-25}$ (Hex Day 1) |
| Distance Estimation | $t(27)=12.6,\; p<10^{-12}$ (Hex Day 1) |
| Day 1→2 improvement | Hex: $t(27)=4.78, p<10^{-5}$; Comm: $t(27)=3.49, p<10^{-3}$ |

**Community awareness check:** 26/28 participants explicitly identified the community structure (conscious learning). Participants were NOT aware of hexagonal structure (implicit learning).

### 4.3 fMRI Task (Day 5)

Per block:
1. **Random walk (70s):** Self-paced graph navigation; participant infers image set
2. **Sequence viewing:** 3-image sequences presented rapidly (~1.4s each); 800ms inter-sequence interval; order pseudo-randomized to ensure coverage and eliminate temporal correlations
3. **Catch trials (12.5%):** Single image after sequence; judge if it can follow the sequence's last image
   - Hex: $t(27)=11.3, p<0.001$; Community: $t(27)=10.6, p<0.001$
4. **End-of-block query:** Identify image set / indicate community grouping
   - Hex: $t(27)=3.8, p<0.001$; Community: $t(27)=9.96, p<0.001$

---

## 5. Analysis Methods

### 5.1 Core Method: Subspace Generalization

> **Origin of the method:** Introduced in Samborska et al. (*Nature Neuroscience*, 2022) for electrophysiology data. See `veronlika_paper_summary.md` Section 5 for the full SVD derivation, cumulative variance curve construction, AUC formula, and permutation test details. The Mark et al. (2026) eLife paper adapts this method to fMRI.

**The key insight:** Rather than comparing representations state-by-state (which requires a mapping between tasks), subspace generalization compares the **covariance structure** of representations. If two tasks recruit the same neural patterns (same "cell assemblies"), the principal components of one task will explain variance in the other.

**Why NOT RSA or repetition suppression:**
- Both require explicit state-to-state correspondence (e.g., node 5 in graph A ↔ node 5 in graph B)
- Non-isomorphic graphs (36 vs. 42 nodes) have no such mapping
- Subspace generalization sidesteps this by asking: "Do the same *directions of variance* appear in both tasks?"

**Formal definition:**

Given activations organized as a matrix $A \in \mathbb{R}^{\text{neurons} \times \text{states}}$:

1. Compute PCs of task 1: $PC_1$ = left eigenvectors of $A_1 A_1^T$, ordered by eigenvalue
2. Project task 2 onto these PCs: $V_{12} = \text{diag}(PC_1^T A_2 A_2^T PC_1)$
3. **Summary statistic:** AUC of the cumulative variance explained curve

> If the same assemblies represent both tasks, the first PCs of task 1 will also explain most variance in task 2 → concave cumulative curve → high AUC.  
> If representations are unrelated → variance is spread uniformly across all PCs → diagonal cumulative curve → AUC ≈ 0.5.

**Why AUC over raw variance:**
- Enforces requirement that *early* PCs (low-dimensional structure) generalize — not just any PCs
- Weighted by $(N - q + 1)$, emphasizing the first PCs
- More robust to noise than correlation of correlation matrices (see Section 5.5)

---

### 5.2 Rodent Electrophysiology Analysis Pipeline

**Preprocessing:**
- Firing rate maps on 64×64 bin spatial matrices
- Smoothed with 5-bin boxcar filter

**Subspace generalization on cells:**
1. Build neuron × spatial-bin matrix from environment 1
2. Compute PCs from the neuron-neuron correlation matrix of environment 1
3. Project environment 2 firing maps onto these PCs
4. Compute cumulative variance AUC

**Statistical Test 1 — within cell type (permutation test):**
- $H_0$: Cell assemblies are random; no preservation across environments
- Procedure: Permute cell identities of environment 2 firing maps; compute AUC difference (within-environment AUC − across-environment AUC) under permutation
- Build null distribution; compare observed AUC difference
- **Result:** Grid cells: $p < 0.001$; Place cells: $p < 0.05$ (some but less generalization)

**Statistical Test 2 — between cell types (Kolmogorov-Smirnov test):**
- $H_0$: Grid cells and place cells show equal generalization
- Procedure: Match cell counts across animals (14, 21, or 25 cells sampled); compute AUC difference distributions; compare distributions
- **Result:** Grid cell AUC differences significantly smaller (better generalization): $p < 0.001$ (KS test)

---

### 5.3 Simulated Voxel Analysis

**Why simulate:** Bridge the gap between single-cell recordings (which confirm the method works) and fMRI voxels (which average over thousands of cells). Establishes under what conditions generalization is detectable in low-resolution data.

**Grid cell simulation:**
- Formula: Thresholded sum of three 2D cosines (Burgess et al. 2007)
- 4 grid modules, each with 13,456 cells (116×116 grid covering hexagonal rhombus)
- Different spacings and phases per module; different environment realizations maintain within-module cell relationships

**Voxel grouping strategies:**
1. **Random:** Cells grouped without regard to phase
2. **Phase-organized:** Cells grouped by grid phase (biologically plausible; Yi et al. 2018)
Each module divided into 4 pseudo-voxels = 3,364 cells per voxel

**Variables explored:**
- Fraction of randomly vs. phase-organized cells: 0 → 1
- Noise amplitude: 0 → 0.1 (spatial white noise)

**Key finding:** With realistic noise (amplitude = 0.1):
- Phase-organized voxels: AUC ≈ 0.65–0.70 (significant generalization detectable)
- Random voxels: AUC ≈ 0.50 (chance; generalization not detectable)

**Conclusion:** The method can detect grid generalization from fMRI-like low-resolution data, *provided* there is some phase organization within voxels (biologically expected from columnar organization).

---

### 5.4 Human fMRI Analysis Pipeline

#### Preprocessing (FSL + ICA-FIX)

1. Motion correction (MCFLIRT, per-run)
2. Brain extraction (BET)
3. Temporal high-pass filter: 100s cutoff
4. EPI → structural → MNI registration (FLIRT then FNIRT)
5. **No spatial smoothing during preprocessing** (smoothing done post-analysis, to avoid blurring searchlight results)
6. ICA-FIX cleaning (especially important for EC: susceptibility to breathing artifacts)

#### GLM (SPM12)

Regressors:
- 6 motion parameters + mean CSF signal (nuisance)
- Bias term (mean activity per run)
- 'Start' message (delta function)
- Self-paced random walk (delta per new image)
- **Each 3-image sequence (1.4s duration boxcar) — primary regressors of interest**
- Catch trial onset (delta) + duration (1.4s boxcar)

All non-nuisance regressors convolved with canonical HRF. Analysis in native (non-normalized) space.

#### Searchlight

- **Size:** 100 nearest voxels per searchlight centre
- **Output per searchlight:** $n_\text{voxels} \times n_\text{sequences}$ activation matrix per graph

#### Leave-One-Out Cross-Validation (per run, per graph)

For each run $j$ and graph pair $(a, b)$:

1. Average activation matrices across held-out 3 runs → $\tilde{B}^{\sim j}$
2. Compute left PCs: $U^{\sim j}_{\text{voxel} \times \text{voxel}}$ from held-out runs
3. Project left-out run $j$ onto these PCs:
$$P_{a,b} = U^{\sim j,a} \cdot B^{j,b}$$
4. Cumulative variance:
$$M_k^{a,b} = \frac{\sum_{l=1}^{100} \left(P_{a,b}[l,k]\right)^2}{S_j}, \quad S_j = \text{diag}\!\left(U_j^T B_j^T B_j U_j\right)$$
5. AUC of cumulative $M_k^{a,b}$ over $k$ = alignment between graphs $a$ and $b$ for run $j$

**Output:** 4×4 alignment matrix (Hex-large, Hex-small, Comm-large, Comm-small)

**Notation:**
- $HlHl$ = Hex-large projected onto its own PCs (within-graph, different stimuli across runs)
- $HlHs$ = Hex-large projected onto Hex-small PCs (same structure, different size)
- $HlCl$ = Hex-large projected onto Comm-large PCs (same image set, different structure)

#### Group-Level Statistics

1. Average 4×4 matrix across runs per participant
2. Normalize searchlight maps to MNI space (FLIRT/FNIRT)
3. Smooth 6mm FWHM
4. One-sample t-tests per contrast per voxel
5. **Multiple comparisons correction:** PALM permutation tests (10,000 iterations), TFCE statistic, FWE-corrected p-values from permutation null distribution, within anatomical mask

---

### 5.5 Statistical Contrasts

**Contrast 1 — Visual Encoding Sanity Check (LOC):**
$$\text{Visual} = [HlHl + ClCl + HsHs + CsCs] - [HlHs + HsHl + ClCs + CsCl]$$

Rationale: Penalizes different-stimulus comparisons; detects visual content encoding regardless of structure type.  
Result: $t(27)_\text{peak} = 4.96$, $p_\text{TFCE} < 0.05$ (bilateral LOC, FWE-corrected; peak MNI $[-44, -86, -8]$)

**Contrast 2 — Hexagonal Structural Generalization (Primary, EC):**
$$\text{Hex Struct} = [HlHl + HlHs + HsHl + HsHs] - [HlCl + HlCs + HsCl + HsCs]$$

Rationale: Hexagonal PCs explain hexagonal data better than community PCs — regardless of image set or size.  
Result: $t(27)_\text{peak} = 4.2$, $p_\text{TFCE} < 0.01$ (bilateral EC, FWE-corrected; peak MNI $[28, -10, -40]$, right EC)

**Contrast 3 — Community Structural Generalization (Control, EC):**
$$\text{Comm Struct} = [ClCl + ClCs + CsCl + CsCs] - [ClHl + ClHs + CsHl + CsHs]$$

Result: **Not significant** in EC or whole-brain.

**Robustness Check — Independent ROI:**
Test Hex structural contrast in orthogonal ROI from Baram et al. 2021 (peak MNI $[25, -5, -28]$):
$$t(27) = 3.6,\; p < 0.001 \quad \text{(one-sample t-test)}$$

---

## 6. Why Subspace Generalization vs. Other Methods

| Method | Problem for this study | Why subspace generalization works |
|---|---|---|
| RSA | Requires state-to-state mapping (36 nodes ↔ 42 nodes — impossible) | Compares covariance structure, not individual states |
| Repetition suppression | Requires identical stimuli across conditions | Works on any neural response |
| Correlation of correlation matrices | Does not enforce low-dimensionality; less noise-robust | AUC weights early PCs; noise orthogonal to signal cancelled by first PC |
| Univariate GLM contrasts | Single-voxel, misses distributed patterns | Multivariate searchlight captures assembly-level structure |

---

## 7. Key Results Summary

| Finding | Statistic | Interpretation |
|---|---|:---:|
| Grid cells generalize better than place cells | $p < 0.001$ (permutation) | Grid assemblies preserved across environments |
| Place cells partially remap | $p < 0.05$ (permutation) | Partial but not full remapping |
| Low-res grid data retains generalization | $p < 0.001$ (KS test) | Method valid for fMRI-scale resolution |
| Visual encoding detected in LOC | $t(27)=4.96$, $p_\text{TFCE}<0.05$ | Method captures image content |
| Hex structural generalization in EC | $t(27)=4.2$, $p_\text{TFCE}<0.01$ | EC generalizes hex structure across size/stimuli |
| Hex generalization in independent ROI | $t(27)=3.6$, $p<0.001$ | Robust, replicates Baram et al. 2021 |
| Community generalization in EC | **Not significant** | EC does not detectably generalize community structure |
| Community (weak) in vmPFC | Weak trend (cautious) | Explicit structure may be coded in PFC |

---

## 8. Discussion: Why No Community Generalization?

Three non-exclusive explanations:

1. **Technical:** The nested ring topology of community graphs may not produce detectable subspace structure (circular arrangement within communities = potentially flat representation)

2. **Representational:** A useful binary representation ("within-community node" vs. "connecting node") would show little variance across states — subspace method needs variance to detect generalization

3. **Implicit vs. Explicit learning:**
   - 26/28 participants explicitly identified communities → conscious, deliberate knowledge
   - Participants were NOT aware of hexagonal structure → implicit, latent knowledge
   - Hypothesis: MTL/EC preferentially encodes **latent implicit structure**; mPFC encodes **explicit structural knowledge**
   - Supporting evidence: Garvert et al. (2017) — structure learned without awareness represented in MTL but not mPFC; exploratory vmPFC analysis in this paper shows weak community generalization

---

## 9. Connection to the LLM Papers

This paper directly inspired the representation geometry work in:

| This paper | LLM geometry papers |
|---|---|
| EC grid cells → same subspace across environments | Months form circles, years form manifolds (Engels et al.; Gurnee et al.) |
| Subspace = low-dimensional covariance structure | Manifold = low-dimensional curved geometry in activation space |
| Translation symmetry in graph structure → generalizable representations | Translation symmetry in co-occurrence statistics → Fourier geometry (Karkada et al.) |
| Non-spatial tasks represented in spatial-like manifolds | Non-spatial concepts (days, counts) represented in spatial-like geometry |
| Hexagonal graph → 2D hexagonal manifold | Day-of-week → 2D circular manifold |
