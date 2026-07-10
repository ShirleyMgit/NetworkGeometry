import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_auc_by_layer(ladder, targets, out_path) -> str:
    layers = [lr.layer for lr in ladder]
    fig, ax = plt.subplots()
    for target in targets:
        ys = [lr.crosses.get(target, {}).get("auc", np.nan) for lr in ladder]
        ax.plot(layers, ys, "-o", label=target)
    ax.axhline(0.5, ls="--", color="gray", label="chance")
    ax.set_xlabel("layer"); ax.set_ylabel("cross AUC"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_ladder(ladder_result, out_path) -> str:
    targets = list(ladder_result.crosses)
    aucs = [ladder_result.crosses[t]["auc"] for t in targets]
    sems = [ladder_result.crosses[t].get("sem", 0.0) for t in targets]
    fig, ax = plt.subplots()
    ax.bar(targets, aucs, yerr=sems); ax.axhline(0.5, ls="--", color="gray")
    ax.set_ylabel("AUC"); ax.set_title(f"layer {ladder_result.layer}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_cumulative_curves(curves, out_path) -> str:
    fig, ax = plt.subplots()
    for name, cum in curves.items():
        ax.plot(range(1, len(cum) + 1), cum, "-o", label=name)
    ax.set_xlabel("k (source PCs)"); ax.set_ylabel("cumulative variance"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_null_hist(null, observed, out_path) -> str:
    fig, ax = plt.subplots()
    ax.hist(null, bins=30); ax.axvline(observed, color="red", label="observed")
    ax.set_xlabel("AUC"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)
