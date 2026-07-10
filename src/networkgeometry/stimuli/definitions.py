from dataclasses import dataclass
from importlib import resources
import yaml
from networkgeometry.types import State


@dataclass(frozen=True)
class Structure:
    name: str
    states: tuple
    excluded: tuple = ()

    @property
    def n_states(self) -> int:
        return len(self.states)


def _data_path(filename: str, path):
    if path is not None:
        return open(path, encoding="utf-8")
    return resources.files("networkgeometry.stimuli.data").joinpath(filename).open(encoding="utf-8")


def load_structures(path=None) -> dict:
    with _data_path("structures.yaml", path) as handle:
        raw = yaml.safe_load(handle)
    out = {}
    for name, spec in raw.items():
        states = tuple(State(lbl, i + 1) for i, lbl in enumerate(spec["states"]))
        out[name] = Structure(name, states, tuple(spec.get("excluded", [])))
    return out


def load_templates(path=None) -> dict:
    with _data_path("templates.yaml", path) as handle:
        return yaml.safe_load(handle)


def prompts_for(structure: Structure, templates: list) -> dict:
    ordered = sorted(structure.states, key=lambda s: s.canonical_index)
    return {run: [tpl.replace("{X}", s.label) for s in ordered] for run, tpl in enumerate(templates)}
