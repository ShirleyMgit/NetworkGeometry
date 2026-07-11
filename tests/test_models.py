import pytest
from networkgeometry.models import MODELS, resolve_model


def test_registry_has_the_four_menu_entries():
    assert set(MODELS) == {"gemma-2-2b", "gemma-3-12b", "llama-3.1-8b", "llama-4-scout"}


def test_resolve_supported_models_gives_hf_id_and_layer_count():
    assert resolve_model("gemma-2-2b").hf_id == "google/gemma-2-2b"
    assert resolve_model("gemma-2-2b").n_layers == 26
    assert resolve_model("gemma-3-12b").hf_id == "google/gemma-3-12b-pt"
    assert resolve_model("gemma-3-12b").n_layers == 48
    assert resolve_model("llama-3.1-8b").hf_id == "meta-llama/Llama-3.1-8B"
    assert resolve_model("llama-3.1-8b").n_layers == 32


def test_large_models_load_in_bfloat16_to_avoid_fp16_overflow():
    # Gemma/Llama have massive outlier activations that overflow fp16 (-> inf -> SVD
    # fails); bfloat16 has fp32's exponent range and is these checkpoints' native dtype.
    assert resolve_model("gemma-3-12b").dtype == "bfloat16"
    assert resolve_model("llama-3.1-8b").dtype == "bfloat16"
    # gemma-2-2b keeps its validated fp16 path (its activations fit in fp16).
    assert resolve_model("gemma-2-2b").dtype == "float16"


def test_unsupported_model_raises_a_clear_message():
    with pytest.raises(ValueError, match="TransformerLens"):
        resolve_model("llama-4-scout")


def test_unknown_model_lists_the_valid_choices():
    with pytest.raises(KeyError, match="gemma-2-2b"):
        resolve_model("gpt-2")
