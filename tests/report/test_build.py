from pathlib import Path
from networkgeometry.report.build import build_memo

def test_build_memo_fills_placeholders(tmp_path):
    template = tmp_path / "t.md"
    template.write_text("# Findings\n{{tl_dr}}\n{{part2_table}}\n", encoding="utf-8")
    summary = {"tl_dr": "Cycles share a subspace at layer 5.",
               "part1_table": "| day | 0.95 |",
               "part2_table": "| month | 0.82 | 0.001 |"}
    out = build_memo(summary, template, tmp_path / "out.md")
    text = Path(out).read_text(encoding="utf-8")
    assert "Cycles share a subspace" in text and "0.82" in text and "{{" not in text
