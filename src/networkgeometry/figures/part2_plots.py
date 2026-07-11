import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_AUC_DEF = (
    "AUC = area under the cumulative-variance curve M_k (fraction of the TARGET structure's "
    "across-state variance captured by the top-k SOURCE principal components, plotted vs k), "
    "normalized so chance = 0.5. Higher = the source subspace explains more of the target's "
    "variance = more shared/reusable geometry.")

def _caption(fig, text):
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=7, wrap=True)

def plot_auc_by_layer(ladder, targets, out_path) -> str:
    layers = [lr.layer for lr in ladder]
    fig, ax = plt.subplots()
    for target in targets:
        ys = [lr.crosses.get(target, {}).get("auc", np.nan) for lr in ladder]
        ax.plot(layers, ys, "-o", label=target)
    ax.axhline(0.5, ls="--", color="gray", label="chance")
    ax.set_xlabel("layer"); ax.set_ylabel("cross AUC"); ax.legend()
    ax.set_title("Cross-structure AUC vs layer (source = day)")
    _caption(fig, _AUC_DEF)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_ladder(ladder_result, out_path) -> str:
    targets = list(ladder_result.crosses)
    aucs = [ladder_result.crosses[t]["auc"] for t in targets]
    sems = [ladder_result.crosses[t].get("sem", 0.0) for t in targets]
    fig, ax = plt.subplots()
    ax.bar(targets, aucs, yerr=sems); ax.axhline(0.5, ls="--", color="gray")
    ax.set_ylabel("AUC"); ax.set_title(f"AUC ladder at layer {ladder_result.layer}")
    _caption(fig, _AUC_DEF + " Error bars = SEM across template runs.")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_cumulative_curves(curves, out_path) -> str:
    fig, ax = plt.subplots()
    for name, cum in curves.items():
        ax.plot(range(1, len(cum) + 1), cum, "-o", label=name)
    ax.set_xlabel("k (number of source PCs)"); ax.set_ylabel("cumulative target variance M_k")
    ax.legend(); ax.set_title("Cumulative-variance curves")
    _caption(fig,
        "M_k = fraction of the target structure's variance captured by the top-k source principal "
        "components. The AREA under this curve (normalized) is the AUC: concave/above-diagonal = "
        "generalizing (variance concentrated in the leading shared modes); straight diagonal = "
        "chance (variance spread evenly across all modes).")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def _prompt_summary(templates) -> str:
    """One caption sentence spelling out the concrete prompt each comparison uses.

    within / matched / control rows all use the shared frame (one phrase); 'specific'
    rows use each cycle's own frame (two phrases). Returns "" when templates is None.
    """
    if templates is None:
        return ""
    shared = templates["shared"][0]
    day = templates["specific"]["day"][0]
    month = templates["specific"]["month"][0]
    return (
        f'Prompts used — matched (shared frame, identical for both cycles): "{shared}".  '
        f'specific (each cycle\'s own frame): day "{day}", month "{month}".  '
        f'within-structure rows and the day -> years/hierarchy/flat controls use the shared frame.')

def plot_stage2_ladder(rows, out_path, context_note=None, templates=None, model_name=None) -> str:
    """One AUC-vs-layer curve per Stage-2 comparison label (spec §5.3 table).

    templates: optional loaded templates.yaml dict; when given, the caption spells out
    the concrete prompt phrase(s) each comparison row actually uses.
    context_note: optional extra string appended to the caption.
    model_name: optional model tag appended to the figure title.
    """
    by_label = {}
    for r in rows:
        by_label.setdefault(r.label, []).append((r.layer, r.auc))
    fig, ax = plt.subplots()
    for label, points in by_label.items():
        points.sort()
        layers, aucs = zip(*points)
        ax.plot(layers, aucs, "-o", label=label)
    ax.axhline(0.5, ls="--", color="gray", label="chance")
    ax.set_xlabel("layer"); ax.set_ylabel("AUC"); ax.legend(fontsize="small")
    title = "Stage-2 comparison ladder: AUC vs layer"
    if model_name:
        title += f"  ·  model: {model_name}"
    ax.set_title(title)
    caption = (
        _AUC_DEF + " One line per comparison row of spec §5.3: '<s> (within)' is the leave-one-"
        "run-out ceiling for structure s; 'a -> b (matched)' / '(specific)' are cross-cycle tests "
        "in the shared / structure-specific template pools; 'day -> years/hierarchy/flat' are the "
        "non-cyclic controls. Cross rows appear only at layers passing the source's within gate.")
    prompt_summary = _prompt_summary(templates)
    if prompt_summary:
        caption += "\n" + prompt_summary
    if context_note:
        caption += "\n" + context_note
    _caption(fig, caption)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_null_hist(null, observed, out_path) -> str:
    fig, ax = plt.subplots()
    ax.hist(null, bins=30); ax.axvline(observed, color="red", label="observed AUC")
    ax.set_xlabel("AUC"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("Permutation null vs observed AUC")
    _caption(fig,
        "Null distribution: the AUC recomputed many times with the target matrix's d feature-rows "
        "randomly permuted, which destroys feature identity while preserving marginal statistics. "
        "Red line = observed AUC; its rank in this null gives the permutation p-value.")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)
