import argparse
import numpy as np
from networkgeometry.stimuli.definitions import load_structures, load_templates, prompts_for

def run_part1(model, layers, out_dir):
    structures, templates = load_structures(), load_templates()
    from networkgeometry.extraction.activations import extract
    from networkgeometry.geometry.part1 import circularity_by_layer
    from dataclasses import asdict
    import json
    from pathlib import Path

    results = {}
    for name in ("day", "month"):
        structure = structures[name]
        dms = extract(model, prompts_for(structure, templates["shared"]),
                      structure.states, name, layers)
        by_layer = {}
        for dm in dms:
            by_layer.setdefault(dm.layer, []).append(dm)
        results[name] = circularity_by_layer(by_layer, excluded=structure.excluded)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {name: [asdict(lc) for lc in layer_circularity_list]
               for name, layer_circularity_list in results.items()}
    (out_path / "part1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results

def run_part2(model, layers, out_dir):
    structures, templates = load_structures(), load_templates()
    from networkgeometry.extraction.activations import extract
    from networkgeometry.analysis.ladder import run_ladder, to_json
    import json
    from pathlib import Path

    runs_by_structure_layer = {}
    for name in ("day", "month", "years", "hierarchy", "flat"):
        structure = structures[name]
        keep_mask = np.array([s.label not in structure.excluded for s in structure.states])
        dms = extract(model, prompts_for(structure, templates["shared"]),
                      structure.states, name, layers)
        for dm in dms:
            runs_by_structure_layer.setdefault((name, dm.layer), []).append(dm.matrix[:, keep_mask])

    results = run_ladder(runs_by_structure_layer, layers, source="day",
                         targets=("month", "years", "hierarchy", "flat"))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "summary.json").write_text(json.dumps(to_json(results), indent=2), encoding="utf-8")
    return results

def main():
    parser = argparse.ArgumentParser(description="LLM cycle-geometry v1 runner")
    parser.add_argument("--part", choices=["part1", "part2"], required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(26)))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    from networkgeometry.extraction.activations import load_model
    model = load_model()
    if args.part == "part1":
        run_part1(model, args.layers, args.out)
    elif args.part == "part2":
        run_part2(model, args.layers, args.out)

if __name__ == "__main__":
    main()
