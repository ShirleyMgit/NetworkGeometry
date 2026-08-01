import pytest
from networkgeometry.models import (
    MODELS,
    base_keys,
    post_trained_keys,
    resolve_model,
    supported_keys,
)


def test_registry_has_every_menu_entry():
    assert set(MODELS) == {
        "gemma-2-2b", "gemma-3-12b", "llama-3.1-8b", "qwen2.5-7b",
        "deepseek-r1-distill-8b", "gemma-4-12b", "jamba-2-mini", "jamba-2-mini-fp8",
        "llama-4-scout",
    }


def test_resolve_supported_models_gives_hf_id_and_layer_count():
    assert resolve_model("gemma-2-2b").hf_id == "google/gemma-2-2b"
    assert resolve_model("gemma-2-2b").n_layers == 26
    assert resolve_model("gemma-3-12b").hf_id == "google/gemma-3-12b-pt"
    assert resolve_model("gemma-3-12b").n_layers == 48
    assert resolve_model("llama-3.1-8b").hf_id == "meta-llama/Llama-3.1-8B"
    assert resolve_model("llama-3.1-8b").n_layers == 32
    assert resolve_model("qwen2.5-7b").hf_id == "Qwen/Qwen2.5-7B"
    assert resolve_model("qwen2.5-7b").n_layers == 28


def test_registry_models_use_the_classic_hooked_loader():
    for key in ("gemma-2-2b", "gemma-3-12b", "llama-3.1-8b", "qwen2.5-7b"):
        assert resolve_model(key).loader == "hooked"


def test_newer_architectures_use_the_transformer_bridge_loader():
    assert resolve_model("deepseek-r1-distill-8b").loader == "bridge"
    assert resolve_model("deepseek-r1-distill-8b").hf_id == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    assert resolve_model("gemma-4-12b").loader == "bridge"
    assert resolve_model("gemma-4-12b").hf_id == "google/gemma-4-12B"
    assert resolve_model("jamba-2-mini").loader == "bridge"
    assert resolve_model("jamba-2-mini").hf_id == "ai21labs/AI21-Jamba2-Mini"
    assert resolve_model("jamba-2-mini-fp8").loader == "bridge"
    assert resolve_model("jamba-2-mini-fp8").hf_id == "ai21labs/AI21-Jamba2-Mini-FP8"


def test_jamba_needs_remote_code_for_its_hybrid_architecture():
    assert resolve_model("jamba-2-mini").trust_remote_code is True
    assert resolve_model("jamba-2-mini-fp8").trust_remote_code is True


def test_models_partition_into_base_and_post_trained():
    # Base (pretrained-only) vs instruction-/reasoning-tuned — the sample split for asking
    # whether the base models' circular concept subspaces survive post-training.
    assert set(base_keys()) == {
        "gemma-2-2b", "gemma-3-12b", "llama-3.1-8b", "qwen2.5-7b", "gemma-4-12b",
    }
    assert set(post_trained_keys()) == {
        "deepseek-r1-distill-8b", "jamba-2-mini", "jamba-2-mini-fp8",
    }
    # The two groups are a disjoint cover of every supported (loadable) model.
    assert set(base_keys()) | set(post_trained_keys()) == set(supported_keys())
    assert not (set(base_keys()) & set(post_trained_keys()))


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
