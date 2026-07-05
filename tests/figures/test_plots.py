import numpy as np
from pathlib import Path
from networkgeometry.figures.part1_plots import plot_manifold, plot_circularity_by_layer
from networkgeometry.figures.part2_plots import plot_auc_by_layer, plot_null_hist
from networkgeometry.geometry.part1 import LayerCircularity
from networkgeometry.analysis.ladder import LadderResult

def test_part1_and_part2_plots_write_files(tmp_path):
    scores = np.column_stack([np.cos(np.linspace(0, 6, 12)), np.sin(np.linspace(0, 6, 12))])
    p1 = plot_manifold(scores, [str(i) for i in range(12)], tmp_path / "m.png")
    lc = [LayerCircularity(l, 0.1, 0.9, 0.95) for l in range(3)]
    p2 = plot_circularity_by_layer(lc, tmp_path / "c.png")
    ladder = [LadderResult(5, 0.9, {"month": {"auc": 0.8, "sem": 0.02}}, True)]
    p3 = plot_auc_by_layer(ladder, ["month"], tmp_path / "a.png")
    p4 = plot_null_hist(np.random.default_rng(0).normal(0.5, 0.05, 500), 0.8, tmp_path / "n.png")
    assert all(Path(p).exists() for p in [p1, p2, p3, p4])
