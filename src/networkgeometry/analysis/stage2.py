from dataclasses import dataclass
import numpy as np
from networkgeometry.stats.within import within_structure_auc
from networkgeometry.stats.cross import cross_structure_auc
from networkgeometry.stats.inference import (
    identity_shuffle_null, p_value, bonferroni, benjamini_hochberg, gate)

CYCLES = ("day", "month")
CONTROL_SOURCE = "day"
CONTROLS = ("years", "hierarchy", "flat")


@dataclass(frozen=True)
class Stage2Row:
    label: str
    layer: int
    auc: float
    sem: float
    raw_p: float | None = None
    fdr_p: float | None = None
    bonferroni_p: float | None = None


def run_stage2_ladder(shared, specific=None, layers=None, n_perm=500, alpha=0.05, seed=0):
    """Stage-2 comparison ladder (spec §5.3).

    shared: {(structure, layer): [run, ...]} for the within gate, the cross-cycle
        source/target, and the controls.
    specific: optional per-structure pool; when given, each cross-cycle test yields
        both a '(matched)' row (from shared) and a '(specific)' row (from specific).
        When None (e.g. the comprehension-probe pool, which has no shared frame),
        each cross-cycle test yields a single row in the `shared` pool's context.
    """
    rng = np.random.default_rng(seed)

    within = {}    # (structure, layer) -> WithinResult
    passed = {}    # structure -> set of gate-passing layers
    rows = []
    for structure in CYCLES:
        pvals = {}
        for layer in layers:
            runs = shared[(structure, layer)]
            wr = within_structure_auc(runs)
            within[(structure, layer)] = wr
            src_mean = np.mean(runs, axis=0)
            null = identity_shuffle_null(src_mean, src_mean, n_perm, rng)
            pvals[layer] = p_value(wr.mean, null)
            rows.append(Stage2Row(f"{structure} (within)", layer, wr.mean, wr.sem))
        passed[structure] = gate(pvals, alpha)

    pending = []  # (label, layer, CrossResult, raw_p) awaiting multiple-comparison correction

    def _cross_row(label, source, target, runs, layer):
        src_runs, tgt_runs = runs[(source, layer)], runs[(target, layer)]
        cr = cross_structure_auc(src_runs, tgt_runs)
        null = identity_shuffle_null(np.mean(src_runs, axis=0), np.mean(tgt_runs, axis=0), n_perm, rng)
        pval = p_value(cr.mean, null)
        pending.append((label, layer, cr, pval))

    for source, target in ((CYCLES[0], CYCLES[1]), (CYCLES[1], CYCLES[0])):
        for layer in layers:
            if layer not in passed[source]:
                continue
            if specific is None:
                # single-context pool (e.g. the per-structure probe pool): no shared
                # frame, so there is no matched/specific contrast — one cross row.
                _cross_row(f"{source} -> {target}", source, target, shared, layer)
            else:
                _cross_row(f"{source} -> {target} (matched)", source, target, shared, layer)
                _cross_row(f"{source} -> {target} (specific)", source, target, specific, layer)

    for target in CONTROLS:
        for layer in layers:
            if layer not in passed[CONTROL_SOURCE]:
                continue
            _cross_row(f"{CONTROL_SOURCE} -> {target}", CONTROL_SOURCE, target, shared, layer)

    if pending:
        raw_pvals = np.array([p[3] for p in pending])
        fdr = benjamini_hochberg(raw_pvals)
        bonf = bonferroni(raw_pvals)
        for (label, layer, cr, raw_p), f, b in zip(pending, fdr, bonf):
            rows.append(Stage2Row(label, layer, cr.mean, cr.sem, float(raw_p), float(f), float(b)))

    return rows
