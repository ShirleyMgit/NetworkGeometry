from pathlib import Path

def build_memo(summary: dict, template_path, out_path) -> str:
    text = Path(template_path).read_text(encoding="utf-8")
    for key, value in summary.items():
        text = text.replace("{{" + key + "}}", str(value))
    Path(out_path).write_text(text, encoding="utf-8")
    return str(out_path)
