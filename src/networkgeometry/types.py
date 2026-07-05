from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class State:
    label: str
    canonical_index: int

@dataclass(frozen=True)
class DataMatrix:
    structure: str
    layer: int
    run: int
    matrix: np.ndarray
    states: tuple[State, ...]

    def __post_init__(self):
        if self.matrix.ndim != 2 or self.matrix.shape[1] != len(self.states):
            raise ValueError(
                f"matrix columns {self.matrix.shape} must equal n_states {len(self.states)}"
            )

    @property
    def n_states(self) -> int:
        return len(self.states)

def group_runs(dms: list[DataMatrix]) -> list[np.ndarray]:
    reference = tuple(sorted(s.canonical_index for s in dms[0].states))
    aligned = []
    for dm in dms:
        if tuple(sorted(s.canonical_index for s in dm.states)) != reference:
            raise ValueError("runs do not share the same state set")
        order = np.argsort([s.canonical_index for s in dm.states])
        aligned.append(dm.matrix[:, order])
    return aligned
