import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkgeometry.geometry.linalg import state_correlation

def _caption(fig, text):
    """Attach a defining caption beneath the axes so each figure states exactly
    what it plots and how the plotted quantity is computed."""
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=7, wrap=True)

def plot_manifold(scores, labels, out_path, title=None) -> str:
    fig, ax = plt.subplots()
    ax.plot(scores[:, 0], scores[:, 1], "-o")
    for (x, y), label in zip(scores, labels):
        ax.annotate(label, (x, y))
    ax.set_aspect("equal")
    ax.set_title(title or "State manifold in the top-2 PC plane")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    _caption(fig,
        "Each point is one state's mean-centered residual-stream activation (averaged over "
        "templates) projected onto the top 2 principal components (the 2 directions of greatest "
        "variance across states). A cyclic structure traces a ring in canonical (calendar) order.")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_circularity_by_layer(results, out_path, title=None) -> str:
    layers = [r.layer for r in results]
    fig, ax = plt.subplots()
    ax.plot(layers, [r.angular_order for r in results], "-o", label="angular order")
    ax.plot(layers, [r.top2_variance_ratio for r in results], "-s", label="top-2 variance ratio")
    ax.set_xlabel("layer"); ax.set_ylabel("score (0–1)"); ax.set_ylim(0, 1.02); ax.legend()
    ax.set_title(title or "Circularity vs layer")
    _caption(fig,
        "Angular order (0–1): fit a circle to the states in the top-2 PC plane, read each state's "
        "angle on it, and take the rotation- and reflection-invariant circular correlation between "
        "those fitted angles and the states' canonical order (Mon=1..Sun=7 / Jan=1..Dec=12); "
        "1 = a cleanly ordered ring, ~0 = no ordered ring.   "
        "Top-2 variance ratio (0–1): fraction of total across-state variance captured by the first "
        "2 principal components; high = the state geometry is essentially 2-D, as a ring must be.")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_correlation_matrix(matrix, labels, out_path, title=None) -> str:
    """Heatmap of the n_states x n_states state correlation matrix at one layer."""
    C = state_correlation(matrix)
    fig, ax = plt.subplots()
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(title or "State correlation matrix")
    _caption(fig,
        "Entry (i,j) = Pearson correlation between states i and j's residual-stream activation "
        "vectors (across all d feature dimensions), averaged over templates, at this layer. "
        "Diagonal = 1 by construction. A cyclic structure shows circulant banding: each state most "
        "correlated with its cyclic neighbours, correlation falling with cyclic distance.")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)
