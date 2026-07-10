import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_manifold(scores, labels, out_path) -> str:
    fig, ax = plt.subplots()
    ax.plot(scores[:, 0], scores[:, 1], "-o")
    for (x, y), label in zip(scores, labels):
        ax.annotate(label, (x, y))
    ax.set_aspect("equal"); ax.set_title("PC1–PC2 manifold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)

def plot_circularity_by_layer(results, out_path) -> str:
    layers = [r.layer for r in results]
    fig, ax = plt.subplots()
    ax.plot(layers, [r.angular_order for r in results], "-o", label="angular order")
    ax.plot(layers, [r.top2_variance_ratio for r in results], "-s", label="top-2 var ratio")
    ax.set_xlabel("layer"); ax.set_ylabel("score"); ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(out_path)
