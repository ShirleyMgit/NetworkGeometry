import argparse
import numpy as np
from networkgeometry.stimuli.definitions import load_structures, load_templates, prompts_for

def mean_state_matrices(model, names, layers, pool="shared"):
    """Per-structure mean (over templates) residual-stream activation matrix at each
    requested layer, plus the canonical-ordered state labels. For descriptive figures
    such as the correlation heatmaps; all states are included (no polysemy exclusion)
    since these plots are purely descriptive.

    Returns: {name: {"labels": [...], "matrices": {layer: (d, n_states) array}}}.
    """
    structures, templates = load_structures(), load_templates()
    from networkgeometry.extraction.activations import extract
    from networkgeometry.types import group_runs

    def _templates_for(name):
        if pool == "shared":
            return templates["shared"]
        if pool == "specific":
            return templates["specific"][name]
        if pool == "strict":
            return [templates["strict"][name]]     # single paper-exact template
        raise ValueError(f"unknown pool {pool!r}")

    out = {}
    for name in names:
        structure = structures[name]
        dms = extract(model, prompts_for(structure, _templates_for(name)), structure.states, name, layers)
        by_layer = {}
        for dm in dms:
            by_layer.setdefault(dm.layer, []).append(dm)
        labels = [s.label for s in sorted(structure.states, key=lambda s: s.canonical_index)]
        out[name] = {
            "labels": labels,
            "matrices": {layer: np.mean(group_runs(group), axis=0)
                         for layer, group in by_layer.items()},
        }
    return out

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

def run_part1_strict(model, layers, out_dir):
    """Strict-reproduction leg (spec §4.3a) — the paper's exact single template per
    structure, no template averaging. Covers day and month (the two circular structures);
    years needs a different (uncentered/Toeplitz) geometry analysis not yet implemented,
    so it is not part of this leg."""
    structures, templates = load_structures(), load_templates()
    from networkgeometry.extraction.activations import extract
    from networkgeometry.geometry.part1 import circularity_by_layer
    from dataclasses import asdict
    import json
    from pathlib import Path

    results = {}
    for name in ("day", "month"):
        structure = structures[name]
        dms = extract(model, prompts_for(structure, [templates["strict"][name]]),
                      structure.states, name, layers)
        by_layer = {}
        for dm in dms:
            by_layer.setdefault(dm.layer, []).append(dm)
        results[name] = circularity_by_layer(by_layer, excluded=structure.excluded)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {name: [asdict(lc) for lc in layer_circularity_list]
               for name, layer_circularity_list in results.items()}
    (out_path / "part1_strict_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results

def run_part2(model, layers, out_dir):
    """Runs the full Stage-2 comparison ladder (spec §5.3): within-structure reference,
    matched- and different-context cross-cycle comparisons (both directions), and the
    circular-to-non-circular controls — one row per (comparison, layer)."""
    structures, templates = load_structures(), load_templates()
    from networkgeometry.extraction.activations import extract
    from networkgeometry.analysis.stage2 import run_stage2_ladder
    from dataclasses import asdict
    import json
    from pathlib import Path

    def _extract_pool(names, template_for):
        runs = {}
        for name in names:
            structure = structures[name]
            keep_mask = np.array([s.label not in structure.excluded for s in structure.states])
            dms = extract(model, prompts_for(structure, template_for(name)),
                          structure.states, name, layers)
            for dm in dms:
                runs.setdefault((name, dm.layer), []).append(dm.matrix[:, keep_mask])
        return runs

    shared_runs = _extract_pool(("day", "month", "years", "hierarchy", "flat"),
                                 lambda name: templates["shared"])
    specific_runs = _extract_pool(("day", "month"),
                                   lambda name: templates["specific"][name])

    results = run_stage2_ladder(shared_runs, specific_runs, layers)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "summary.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return results

def main():
    parser = argparse.ArgumentParser(description="LLM cycle-geometry v1 runner")
    parser.add_argument("--part", choices=["part1", "part1_strict", "part2"], required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(26)))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    from networkgeometry.extraction.activations import load_model
    model = load_model()
    if args.part == "part1":
        run_part1(model, args.layers, args.out)
    elif args.part == "part1_strict":
        run_part1_strict(model, args.layers, args.out)
    elif args.part == "part2":
        run_part2(model, args.layers, args.out)

if __name__ == "__main__":
    main()
