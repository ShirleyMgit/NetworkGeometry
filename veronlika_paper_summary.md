# Samborska et al. (2022) — Paper Summary

**"Complementary task representations in hippocampus and prefrontal cortex for generalizing the structure of problems"**
Veronika Samborska, James L. Butler, Mark E. Walton, Timothy E. J. Behrens, Thomas Akam
*Nature Neuroscience*, 25, 1314–1326.

---

## Short Summary

Mice performed reversal learning across multiple problems that shared the same abstract structure (choose A or B with probabilistic reward, reversals) but used different physical port configurations. Mice transferred knowledge across problems, learning faster over time. Simultaneous recordings from mPFC and hippocampal CA1 revealed a fundamental dissociation: **mPFC** maintained abstract, sensorimotor-invariant representations of task structure that generalized across problems, while **CA1** representations were remapped to encode the sensorimotor specifics of each problem. The two regions play complementary roles — PFC abstracts the common structure, hippocampus maps it onto current-problem specifics.

---

## Analysis Methods (Detailed)

### 1. Temporal Alignment (Time Warping)

Because the task was self-paced, the time between trial initiation and choice varied across trials. To align neural activity, the interval was linearly warped to match the median interval across all trials. Firing rates were computed with a Gaussian kernel (40-ms SD) sampled at 40-ms intervals, spanning 1 s pre-initiation to 1 s post-outcome.

### 2. Linear Regression & Coefficient of Partial Determination (CPD)

Single-neuron activity at each timepoint was regressed against task variables: **choice** (A vs B), **outcome** (reward vs omission), and **choice x outcome** interaction. The key metric is the **coefficient of partial determination (CPD)**, which quantifies the fraction of variance *uniquely* explained by each regressor:

$$CPD(RDM_i) = \frac{SSE_{\sim i} - SSE_{full}}{SSE_{\sim i}}$$

where $SSE_{full}$ is the sum of squared errors of the full model and $SSE_{\sim i}$ is the error when regressor $i$ is removed.

**Key finding:** CA1 encoded choice more strongly (8.4% vs 4.8%); mPFC encoded outcome more strongly (12.9% vs 7.1%).

### 3. Representational Similarity Analysis (RSA)

Neural activity was extracted around port-entry events (±20 ms windows) and grouped into conditions defined by: problem number, trial stage (initiation/choice), choice type (A/B), and outcome. Pairwise Pearson correlations between population activity vectors for all condition pairs formed the observed similarity matrix.

This was modeled as a linear combination of **representational design matrices (RDMs)**, each encoding a hypothesized feature (port location, abstract A-vs-B choice, outcome, trial stage, problem-specific A choice):

$$r_{i,j} = \beta_0 + \sum_{n=1}^{9} \beta_n \cdot RDM_n(i,j) + \varepsilon_{i,j}$$

The regression weights $\beta_n$ quantify how strongly each feature shapes neural similarity. CPD was again used to assess the unique contribution of each RDM.

**Key finding:** mPFC similarity was dominated by abstract trial stage and outcome; CA1 similarity was dominated by physical port identity. CA1 also showed problem-specific remapping of the A choice even when its physical location was unchanged.

### 4. Cross-Problem Decoding

A support vector classifier was trained to decode trial stage (initiation vs choice) from population activity on one problem and tested on a different problem. If representations generalize abstractly, the decoder transfers well. If they are sensorimotor-specific, the decoder confuses ports that swap roles between problems.

**Key finding:** mPFC decoding transferred cleanly; CA1 decoding showed systematic errors reflecting physical-port encoding.

### 5. Singular Value Decomposition (SVD) — Core Geometric Analysis

This is the central method for characterizing the geometry of neural representations and testing cross-problem generalization.

#### 5.1 Building the Data Matrix

For each problem, construct matrix $D$ where:
- **Each row** = demeaned, trial-aligned firing rate of one neuron, concatenated across timepoints for 4 trial types: rewarded A choice, non-rewarded A, rewarded B, non-rewarded B
- **Shape:** $[n_\text{neurons} \times (4 \times n_\text{timepoints})]$
- Movement-related variance (2D nose position, velocity, acceleration) is regressed out first using Gaussian radial basis functions + PCA occupancy components before building $D$

To control for representational drift within a session, each problem is split into **first** ($f$) and **second** ($s$) halves:

$$D_{i,h} = U_{i,h} \Sigma_{i,h} V^T_{i,h}$$

#### 5.2 SVD Decomposition

$$D = U \Sigma V^T$$

- $U$ — **cellular modes** (columns): unit vectors over neurons — groups of co-activating neurons ("cell assemblies"). Each column $u_k$ has a weight per neuron indicating how strongly that neuron participates.
- $V^T$ — **temporal modes** (rows): unit vectors over timepoints and trial types — the time course of each cellular mode across trial events.
- $\Sigma$ — diagonal matrix of singular values $\sigma_k$, sorted $\sigma_1 \geq \sigma_2 \geq \ldots$ Each $\sigma_k^2$ is the variance explained by mode pair $k$.

**Relationship to covariance eigendecomposition:**
$$DD^T = U\Sigma^2 U^T \quad \text{(neuron–neuron covariance)}$$
$$D^TD = V\Sigma^2 V^T \quad \text{(timepoint–timepoint covariance)}$$

So $U$ are eigenvectors of the neuron–neuron covariance matrix and $V$ are eigenvectors of the timepoint–timepoint covariance matrix — SVD is equivalent to PCA in both neuron space and time space simultaneously.

#### 5.3 Three Types of Cross-Problem Generalization

**1. Temporal mode generalization** — do the same trial events recur, regardless of which neurons encode them?

$$M_V^\text{cross} = D_{2,f} V_{1,s}$$

Variance explained by temporal mode $k$ in problem 2 = sum of squared elements of column $k$ of $M_V^\text{cross}$, normalized by total variance in $D_{2,f}$.

Within-problem control:
$$M_V^\text{same} = D_{1,f} V_{1,s}$$

**2. Cellular mode generalization** — do the same cell assemblies co-activate, regardless of what they represent?

$$M_U^\text{cross} = U_{1,s}^T D_{2,f}$$

Variance explained by cellular mode $k$ = sum of squared elements of row $k$ of $M_U^\text{cross}$, normalized by total variance.

Within-problem control:
$$M_U^\text{same} = U_{1,s}^T D_{1,f}$$

**3. Joint (paired) mode generalization** — do the same cell assemblies perform the same roles?

$$\Sigma^\text{cross} = U_{1,s}^T D_{2,f} V_{1,s}$$

$\Sigma^\text{cross}$ is not constrained to be diagonal. If the same cell assemblies perform the same roles across problems, the temporal and cellular modes will align and $\Sigma^\text{cross}$ will have high weights on the diagonal. The cumulative sum of squared **diagonal elements** is plotted (normalized by peak within-problem cumulative weight from $\Sigma^\text{same} = U_{1,s}^T D_{1,f} V_{1,s}$).

#### 5.4 Cumulative Variance Curve and AUC

For each analysis type (temporal, cellular, joint), plot the **cumulative variance explained** as a function of the number of modes $k$, where modes are **ordered by variance explained in problem 1** (not problem 2):

$$\text{CumVE}(k) = \sum_{k'=1}^{k} \frac{\text{variance explained by mode } k'}{\text{total variance in } D_2} \times 100\%$$

This ordering is the key design choice: it asks "how quickly does problem 2's variance accumulate when sorted by problem 1's importance ordering?" If the same structure exists in both problems, the early (high-variance) modes of problem 1 will also explain a lot of variance in problem 2 → the curve rises steeply → concave shape.

**AUC as the summary statistic:**

$$\text{AUC} = \sum_{k=1}^{K} \text{CumVE}(k)$$

The key test quantity is the **AUC difference**:

$$\Delta\text{AUC} = \text{AUC}_\text{within} - \text{AUC}_\text{cross}$$

| $\Delta\text{AUC}$ value | Interpretation |
|---|---|
| $\approx 0$ | Cross-problem curve matches within-problem → strong generalization |
| Large | Cross-problem curve is much flatter → representations don't generalize |

> **Why AUC rather than variance at a fixed $k$?**
> - Integrates over the full curve shape → single robust scalar
> - Naturally penalizes solutions where only a few modes generalize but the bulk does not
> - Has a natural chance level: a flat/diagonal cumulative curve gives AUC = 0.5 × total possible area
> - Emphasizes *low-dimensional* structure: if the first few modes already explain most variance in both problems, the curve is strongly concave and AUC is high; noise modes at the tail contribute little

#### 5.5 Statistical Test — Permutation Across Sessions

To determine whether the difference in $\Delta\text{AUC}$ between PFC and CA1 is significant:

1. Compute $\Delta\text{AUC}_\text{PFC}$ and $\Delta\text{AUC}_\text{CA1}$ from the real data
2. Compute the observed test statistic: $\Delta\text{AUC}_\text{CA1} - \Delta\text{AUC}_\text{PFC}$
3. Build null distribution by **shuffling recording sessions between brain regions** (reassigning sessions randomly to "PFC" or "CA1" labels), repeating many times
4. Compare observed statistic against the **95th percentile** of the null distribution

This is non-parametric, accounts for different neuron counts per region, and handles the fact that recordings came from different animals. Individual animal plots (Extended Data Fig. 6d–f) show AUC differences per animal to confirm the effect is not driven by a single animal.

**Results:**

| Mode type | PFC vs CA1 | $p$ |
|---|---|---|
| Temporal modes | No difference | $> 0.05$ |
| Cellular modes | PFC generalizes significantly more | $< 0.001$ |
| Cellular + Temporal pairs | PFC generalizes significantly more | $< 0.001$ |

The temporal/cellular dissociation is the paper's main finding: *what* events are encoded (temporal structure) is the same in both regions, but *which neurons* encode them is conserved only in mPFC.

### 6. Policy Representation Analysis

The mouse's internal "policy" (belief about the best option) was estimated by logistic regression of choices on the preceding 12-trial history of choices and outcomes. The resulting policy signal was then decoded from neural activity.

Cross-problem generalization of policy encoding was assessed by training the decoder on one problem and testing on another. For B choices (which change physical port across problems), mPFC policy representations transferred well (abstract encoding), while CA1 representations did not (sensorimotor-bound encoding).

### 7. Surprise Measure (Detecting Representational Transitions)

To distinguish abrupt remapping from gradual drift at problem transitions, a surprise metric was computed:

$$s(x_{ij}) = \frac{\left(\frac{1}{n} \sum x_{ij} - \mu_{kl}\right)^2}{\sigma^2_{kl}}$$

This quantifies how much each neuron's firing rate deviates from its baseline. An abrupt spike in surprise at a problem boundary indicates discrete remapping rather than slow drift.

**Key finding:** CA1 showed abrupt remapping at problem transitions; mPFC did not.

### 8. Statistical Methods

- **Permutation testing:** Null distributions built by shuffling sessions between brain regions; family-wise error correction across timepoints using peak statistics.
- **Animal random effects:** Robustness tested by shuffling animal identities between regions.

---

## Key Results Summary

| Finding | mPFC | CA1 |
|---|---|---|
| Dominant coding | Outcome, trial stage | Choice, port location |
| Cross-problem representation | Abstract, invariant | Sensorimotor-specific, remapped |
| SVD cellular modes | Generalize across problems | Problem-specific |
| Policy encoding | Abstract, transferable | Bound to physical ports |
| Transition dynamics | Gradual/continuous | Abrupt remapping |

## Interpretation

The two regions implement a **complementary system**: mPFC extracts and maintains a low-dimensional, transferable abstraction of task structure (the "schema"), while CA1 flexibly maps this structure onto the sensorimotor particulars of each specific problem instance. This division of labor enables both generalization (via PFC) and context-appropriate behavior (via hippocampus).
