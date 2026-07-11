import numpy as np
from pathlib import Path
from networkgeometry.figures.part1_plots import (
    plot_manifold, plot_circularity_by_layer, plot_correlation_matrix)
from networkgeometry.figures.part2_plots import (
    plot_auc_by_layer, plot_null_hist, plot_stage2_ladder, _prompt_summary)
from networkgeometry.geometry.part1 import LayerCircularity
from networkgeometry.analysis.ladder import LadderResult
from networkgeometry.analysis.stage2 import Stage2Row
from tests.fixtures import ring_matrix

def test_part1_and_part2_plots_write_files(tmp_path):
    scores = np.column_stack([np.cos(np.linspace(0, 6, 12)), np.sin(np.linspace(0, 6, 12))])
    p1 = plot_manifold(scores, [str(i) for i in range(12)], tmp_path / "m.png")
    lc = [LayerCircularity(l, 0.1, 0.9, 0.95) for l in range(3)]
    p2 = plot_circularity_by_layer(lc, tmp_path / "c.png")
    ladder = [LadderResult(5, 0.9, {"month": {"auc": 0.8, "sem": 0.02}}, True)]
    p3 = plot_auc_by_layer(ladder, ["month"], tmp_path / "a.png")
    p4 = plot_null_hist(np.random.default_rng(0).normal(0.5, 0.05, 500), 0.8, tmp_path / "n.png")
    assert all(Path(p).exists() for p in [p1, p2, p3, p4])

def test_plot_correlation_matrix_writes_file(tmp_path):
    A = ring_matrix(d=32, n_states=7)
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    path = plot_correlation_matrix(A, labels, tmp_path / "corr.png", title="day (layer 6)")
    assert Path(path).exists()

def test_plot_correlation_matrix_accepts_caption_override(tmp_path):
    A = ring_matrix(d=32, n_states=7)
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    path = plot_correlation_matrix(
        A, labels, tmp_path / "corr_strict.png", title="day (strict, layer 14)",
        caption="single paper-exact template")
    assert Path(path).exists()

def test_plot_stage2_ladder_writes_one_line_per_comparison_row(tmp_path):
    rows = [
        Stage2Row("day (within)", 5, 0.9, 0.02),
        Stage2Row("day (within)", 6, 0.92, 0.01),
        Stage2Row("day -> month (matched)", 5, 0.8, 0.03, 0.01, 0.02, 0.05),
        Stage2Row("day -> month (matched)", 6, 0.82, 0.02, 0.01, 0.02, 0.05),
        Stage2Row("day -> flat", 6, 0.5, 0.05, 0.4, 0.4, 1.0),
    ]
    path = plot_stage2_ladder(rows, tmp_path / "stage2.png")
    assert Path(path).exists()

def test_plot_stage2_ladder_accepts_context_note(tmp_path):
    rows = [Stage2Row("day -> month (matched)", 5, 0.8, 0.03, 0.01, 0.02, 0.05)]
    path = plot_stage2_ladder(
        rows, tmp_path / "stage2_note.png",
        context_note="matched = shared frame; specific = per-structure frame")
    assert Path(path).exists()

_TEMPLATES = {
    "shared": ["Tell me about {X}", "This is about {X}"],
    "specific": {"day": ["We'll meet on {X}"], "month": ["We'll meet in {X}"]},
}

def test_prompt_summary_names_the_shared_and_specific_prompts():
    summary = _prompt_summary(_TEMPLATES)
    assert "Tell me about {X}" in summary       # shared / matched frame
    assert "We'll meet on {X}" in summary        # day specific frame
    assert "We'll meet in {X}" in summary        # month specific frame

def test_prompt_summary_is_empty_without_templates():
    assert _prompt_summary(None) == ""

def test_plot_stage2_ladder_accepts_templates(tmp_path):
    rows = [
        Stage2Row("day (within)", 5, 0.9, 0.02),
        Stage2Row("day -> month (specific)", 5, 0.8, 0.03, 0.01, 0.02, 0.05),
    ]
    path = plot_stage2_ladder(rows, tmp_path / "stage2_t.png", templates=_TEMPLATES)
    assert Path(path).exists()
