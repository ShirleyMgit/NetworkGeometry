import numpy as np
import pytest
from networkgeometry.types import State, DataMatrix, group_runs

def test_datamatrix_validates_shape():
    states = (State("Monday", 1), State("Tuesday", 2))
    with pytest.raises(ValueError):
        DataMatrix("day", 0, 0, np.zeros((4, 3)), states)

def test_n_states_and_construction():
    states = (State("Monday", 1), State("Tuesday", 2))
    dm = DataMatrix("day", 0, 0, np.zeros((4, 2)), states)
    assert dm.n_states == 2

def test_group_runs_aligns_by_canonical_index():
    a = State("Monday", 1); b = State("Tuesday", 2)
    m1 = np.array([[1.0, 2.0]])                      # columns (Mon, Tue)
    m2 = np.array([[20.0, 10.0]])                    # columns (Tue, Mon)
    dm1 = DataMatrix("day", 0, 0, m1, (a, b))
    dm2 = DataMatrix("day", 0, 1, m2, (b, a))
    grouped = group_runs([dm1, dm2])
    np.testing.assert_allclose(grouped[0], [[1.0, 2.0]])
    np.testing.assert_allclose(grouped[1], [[10.0, 20.0]])  # reordered to (Mon, Tue)
