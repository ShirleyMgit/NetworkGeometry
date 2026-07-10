# Hierarchical Concept Geometry in Language Models Emerges from Word Co-occurrence

**Authors:** Andres Nava (Johns Hopkins), Matthieu Wyart (EPFL & Johns Hopkins)
**arXiv:** 2605.23821 (May 2026)

## Core claim

Hierarchical "is-a" (hypernymy) structure in LLM representation geometry — e.g., a *bird* direction nested inside an *animal* direction — does not require a hierarchy-specific functional mechanism (as proposed by Park et al., 2024/2025's linear representation hypothesis). Instead, it emerges as a byproduct of ordinary word co-occurrence statistics, the same statistics that drive word2vec-style embeddings.

## Theoretical argument (brief)

- Start from the empirical fact that words closer on the WordNet hypernym graph co-occur more often (verified via a normalized co-occurrence/PMI-like matrix M\*, which decays roughly exponentially with WordNet graph distance).
- If co-occurrence is purely a function of tree distance, then on any regular (they focus on binary) subtree, the Gram matrix M\* decomposes exactly into a "scaling" block (functions constant per depth level) and "wavelet/split" blocks (Haar-like contrasts between sibling subtrees), each block invariant under M\*.
- Under a positivity + monotone-decay assumption on the co-occurrence kernel, they prove (Theorems 1–2) that eigenvectors organize hierarchically and coarse-to-fine: the top eigenvector is a roughly constant "scaling" mode, the next eigenvectors split the broadest branches (e.g., plant vs. animal), and progressively lower eigenvectors resolve finer sub-splits (bird vs. fish, daisy vs. poppy, etc.), mirroring the tree structure. They call this **hierarchical splitting geometry**.
- This reframes Park et al.'s postulate that child concept vectors = parent vector + an orthogonal innovation: instead of that orthogonality being a designed/functional property, it falls out naturally from the spectral structure of co-occurrence statistics.

## Methodology for the LLM analysis (Section 5, Appendix C.7)

**Model studied:** Gemma 2B (`google/gemma-2b`, via Hugging Face `transformers`), a 2B-parameter decoder-only model, vocab size 256,128, hidden/representation dimension 2048. Robustness checks additionally used Llama 3.2-1B (16 layers, dim 2048, vocab 128,256), via TransformerLens.

**Representations analyzed:**
- Primary: Gemma's **unembedding vectors** (output embedding matrix), globally **centered and whitened** (v% = Σ^(-1/2)(v − μ), fit on the full vocabulary) to remove anisotropy, following Park et al.'s preprocessing.
- Robustness controls: unwhitened (only centered) Gemma unembeddings; internal residual-stream activations from Gemma (middle layer, ℓ=9) and Llama (ℓ=8), extracted at the final token of a raw title-cased prompt (e.g., "Animal", "Sea Turtle") using TransformerLens, then globally centered.

**Vocabulary construction:** WordNet noun lemmas were filtered to (i) have a matching single Gemma token, (ii) have co-occurrence statistics from the corpus, and (iii) be **monosemous** in the candidate set (using a WordNet-count-based monosemy score) so each word maps unambiguously to one synset. This yielded 17,566 lemmas → 11,735 unique eligible synset–lemma pairs. The WordNet hypernym DAG was converted into a rooted arborescence (max-depth parent selection) and contracted through ineligible nodes to define tree distances.

**Co-occurrence statistics:** Built from the November 2023 English Wikipedia dump, tokenized to lowercase alphabetic spans, with a symmetric weighted skip-gram window (L=16, weight 1/d). This produced the M\* matrix (bounded transform of PMI) used to fit an exponential decay kernel f(d) = α·e^(−βd) (also tested: a shifted power-law kernel as robustness check).

**word2vec baseline embeddings:** Constructed directly from the top d=2048 positive eigenmodes of the (restricted) M\* matrix, matching Gemma's representation dimension for apples-to-apples comparison.

**Core test — binary subtree sampling + eigenspace alignment (Section 4.3/5.2, the main quantitative result):**
1. Sample perfect binary WordNet subtrees of depth L=3 (15 nodes) uniformly (via dynamic programming over valid tree structures), rooted at ~21 different high-level roots (entity, organism, animal, cognition, etc.), up to 5,000 trees per root.
2. For each sampled tree, build (a) a **theoretical** Gram matrix from the fitted exponential-decay kernel evaluated on tree distances, and (b) **empirical** Gram matrices from word2vec and from (whitened) Gemma unembeddings restricted to that tree's tokens.
3. Quantify agreement using **top-k eigenspace alignment**: g(k) = (1/k)‖U_k^T V_k‖²_F, comparing the top-k eigenvectors of empirical (U_k) vs. theoretical (V_k) Gram matrices. This subspace-level (not eigenvector-level) comparison is used because degenerate eigenspaces (same-depth sibling splits) have no canonical individual basis.
4. Summarize each root via an **Alignment Area** (area between the top-15 alignment curve and the k/15 null diagonal).
5. Compare against a **shuffled-label baseline** (randomly permute which empirical vector attaches to which synset) and a stricter **within-tree shuffle baseline** (permute vectors only within each sampled tree, preserving that tree's exact vector multiset/spectrum).

**Concept-vector diagnostic (Section 5, connecting to Park et al.):** They re-run Park et al.'s parent–child diagnostic — cos(ℓ̄_w − ℓ̄_parent, ℓ̄_parent), which should be ≈0 if child concept vectors are parent + orthogonal innovation — on both Gemma unembeddings and on the theoretical co-occurrence embeddings, using concept vectors estimated from 70% of each synset's descendant tokens (Ledoit–Wolf-shrunk covariance, following Park et al.'s estimator).

## Results of the LLM analysis

- **Qualitative match:** For the illustrative organism taxonomy, Gemma's Gram matrix and its top eigenvectors visually reproduce the predicted hierarchical splitting geometry: the first eigenvector is roughly constant, the second splits animals from plants, and degenerate third/fourth eigenvectors split bird-vs-fish and flower-vs-tree — closely matching both the theoretical kernel and word2vec (Figure 3).
- **Quantitative alignment:** Across sampled organism-rooted and cognition-rooted binary trees, **both word2vec and Gemma unembeddings show top-k eigenspace alignment with the theoretical co-occurrence-based prediction substantially above the shuffled-label null** (Figure 4a-b). This holds broadly across ~21 eligible root concepts spanning very different semantic domains (entity, object, person, animal, artifact, cognition, event, quality, part, chemical, etc.) — Gemma's alignment area is consistently above baseline for essentially all of them (Figure 4c).
- **Robustness:** The alignment signal survives (a) an additional, stricter within-tree shuffle baseline; (b) an alternative power-law kernel instead of exponential; (c) smaller depth-2 trees with a much larger and less curated set of eligible roots (144 vs. 21); and (d) alternative representations — unwhitened (only centered) Gemma unembeddings, and internal residual-stream activations from both Gemma (mid-layer) and Llama-3.2-1B (mid-layer) — though the internal-activation alignment is noticeably weaker than for unembeddings.
- **Concept-vector diagnostic:** Both Gemma unembeddings and the purely co-occurrence-derived synthetic embeddings show parent–child innovation vectors concentrated near-orthogonal to the parent vector (cosine ≈ 0), while a shuffled-parent baseline is clearly displaced. Since the *synthetic* (co-occurrence-only) embeddings reproduce this same near-orthogonality pattern that Park et al. treated as a signature of a hierarchy-specific functional mechanism, the authors argue it need not indicate functional design — it can arise purely from spectral/statistical structure.

## Interpretation / conclusion

Hierarchical concept geometry (coarse-to-fine, roughly-orthogonal splitting directions) is a **generic consequence of the statistics of word co-occurrence** interacting with tree-structured semantic distance, not evidence that LLMs have a special hypernymy-specific circuit. This complements related work (Korchinski et al. on discrete-attribute parallelograms like king−man+woman=queen; Karkada et al. on continuous-attribute manifolds like seasons/geography) into a unified view: words share attributes (discrete, continuous, or hierarchical), similar-attribute words co-occur more, and this alone generates the elegant geometric organization observed in embeddings — useful for downstream function, but not necessarily *driven by* function.

## Limitations noted by authors

- Theory is derived for co-occurrence-driven, word2vec-style embeddings; extending it to full transformer training dynamics remains open.
- Analysis excludes context-dependent representation of ambiguous (polysemous) words by design (monosemy filtering).
