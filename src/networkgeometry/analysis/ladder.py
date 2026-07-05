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
