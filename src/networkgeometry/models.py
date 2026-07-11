"""Model menu for the cycle-geometry pipeline.

Every model here is loaded through TransformerLens `HookedTransformer`, which is what
lets us read `blocks.{l}.hook_resid_post`. Only base (pretrained) variants are listed —
instruct-tuning warps the concept geometry we measure. Memory notes are fp16 weights;
the 8B/12B models need a Colab Pro L4/A100, not a free T4.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    key: str
    hf_id: str | None
    n_layers: int | None
    fits: str
    supported: bool = True


MODELS = {
    "gemma-2-2b": ModelChoice("gemma-2-2b", "google/gemma-2-2b", 26, "free T4 (~4 GB)"),
    "gemma-3-12b": ModelChoice("gemma-3-12b", "google/gemma-3-12b-pt", 48, "A100 (~24 GB)"),
    "llama-3.1-8b": ModelChoice("llama-3.1-8b", "meta-llama/Llama-3.1-8B", 32, "L4/A100 (~16 GB)"),
    # The 12B "Gemma" is officially Gemma 3 — there is no Gemma 4. Llama 4 Scout is a very
    # new MoE model that TransformerLens does not support yet, so it is listed but disabled.
    "llama-4-scout": ModelChoice("llama-4-scout", None, None, "unsupported", supported=False),
}


def supported_keys() -> list:
    return [key for key, choice in MODELS.items() if choice.supported]


def resolve_model(key: str) -> ModelChoice:
    """Look up a menu key, raising a helpful error for unknown or unsupported keys."""
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose one of {supported_keys()}")
    choice = MODELS[key]
    if not choice.supported:
        raise ValueError(
            f"model {key!r} is not supported by TransformerLens 3.5.1 yet "
            f"(no HookedTransformer loader); choose one of {supported_keys()}")
    return choice


def load_model(key: str = "gemma-2-2b", device: str = "cuda", dtype=None):
    """Load a menu model via the memory-safe no-processing path in fp16.

    Resolves (and validates) the key before importing torch/TransformerLens, so an
    unsupported selection fails fast with a clear message instead of after a long import.
    """
    choice = resolve_model(key)
    import torch
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained_no_processing(
        choice.hf_id, device=device, dtype=dtype or torch.float16)
