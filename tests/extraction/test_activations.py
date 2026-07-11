import numpy as np
import pytest
import torch
from networkgeometry.types import State
from networkgeometry.extraction.activations import _final_token_resid

transformer_lens = pytest.importorskip("transformer_lens")


class _FakeModel:
    """Minimal stand-in whose residual cache holds a chosen final-token value."""
    def __init__(self, final_value):
        self.final_value = final_value

    def run_with_cache(self, prompt, names_filter=None):
        acts = torch.tensor([[[0.0, 0.0, 0.0], [self.final_value, 1.0, 2.0]]])  # (1, seq, d)
        return None, {"blocks.0.hook_resid_post": acts}


def test_non_finite_activation_raises_actionable_error():
    with pytest.raises(ValueError, match="bfloat16"):
        _final_token_resid(_FakeModel(float("inf")), "The month of the year is May", [0])


def test_finite_activations_pass_through():
    out = _final_token_resid(_FakeModel(3.0), "The month of the year is May", [0])
    assert np.isfinite(out[0]).all()
    assert out[0][0] == 3.0

@pytest.mark.integration
def test_extract_shapes_with_gpt2():
    from networkgeometry.extraction.activations import load_model, extract
    model = load_model("gpt2", device="cpu")
    states = (State("Monday", 1), State("Tuesday", 2), State("Wednesday", 3))
    prompts = {0: ["A day: Monday", "A day: Tuesday", "A day: Wednesday"],
               1: ["See you Monday", "See you Tuesday", "See you Wednesday"]}
    dms = extract(model, prompts, states, "day", layers=[0, 5])
    by_layer = {dm.layer for dm in dms}
    assert by_layer == {0, 5}
    for dm in dms:
        assert dm.matrix.shape[1] == 3 and dm.matrix.shape[0] == model.cfg.d_model
