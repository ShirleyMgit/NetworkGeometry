"""Model menu for the cycle-geometry pipeline.

Every model here is read through a TransformerLens hook on `blocks.{l}.hook_resid_post`.
Two loaders back the menu (see `loader`): the classic `HookedTransformer` no-processing
path for models in the TransformerLens 3.5.1 registry, and `TransformerBridge` for newer
architectures that are not. Base (pretrained) checkpoints are preferred — instruct/reasoning
tuning warps the concept geometry we measure; the post-trained entries (DeepSeek-R1-Distill,
Jamba2 Mini) are included on request and flagged inline. Memory notes are fp16 weights; the
8B/12B models need a Colab Pro L4/A100, not a free T4.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    key: str
    hf_id: str | None
    n_layers: int | None
    fits: str
    dtype: str = "bfloat16"   # torch dtype name; bf16 has fp32's exponent range (overflow-safe)
    supported: bool = True
    loader: str = "hooked"    # "hooked" = classic no-processing path; "bridge" = TransformerBridge
    trust_remote_code: bool = False
    training: str = "base"    # "base" = pretrained-only; "instruct"/"reasoning" = post-trained


MODELS = {
    # gemma-2-2b keeps fp16: its validated reproduction path, and its activations fit in fp16.
    "gemma-2-2b": ModelChoice("gemma-2-2b", "google/gemma-2-2b", 26, "free T4 (~4 GB)", dtype="float16"),
    # Gemma/Llama at 8B+ have massive outlier activations that OVERFLOW fp16 (-> inf ->
    # "SVD did not converge"); bf16 (their native checkpoint dtype) avoids this.
    "gemma-3-12b": ModelChoice("gemma-3-12b", "google/gemma-3-12b-pt", 48, "A100 (~24 GB)", dtype="bfloat16"),
    "llama-3.1-8b": ModelChoice("llama-3.1-8b", "meta-llama/Llama-3.1-8B", 32, "L4/A100 (~16 GB)", dtype="bfloat16"),
    # Qwen2.5-7B base is in the TransformerLens registry, so it uses the classic loader.
    "qwen2.5-7b": ModelChoice("qwen2.5-7b", "Qwen/Qwen2.5-7B", 28, "L4/A100 (~15 GB)", dtype="bfloat16"),
    # --- Loaded via TransformerBridge (not in the TransformerLens 3.5.1 registry). ---
    # enable_compatibility_mode(no_processing=True) restores the legacy blocks.{l}.hook_resid_post
    # names WITHOUT weight processing, matching the no-processing path used by the models above.
    #
    # DeepSeek-R1-Distill is a reasoning-distilled Llama-3.1-8B, NOT a base checkpoint — its
    # geometry is post-trained, unlike every other entry here. Included on request; interpret
    # cross-model comparisons with that caveat.
    "deepseek-r1-distill-8b": ModelChoice(
        "deepseek-r1-distill-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 32,
        "L4/A100 (~16 GB)", dtype="bfloat16", loader="bridge", training="reasoning"),
    # gemma-4-12B postdates the TL 3.5.1 registry and is natively multimodal; the bridge loads
    # its text tower. Gated on HF (accept the license + authenticate a token, like Llama).
    # Bridge support for this architecture is unverified — may need a newer `transformers`.
    "gemma-4-12b": ModelChoice(
        "gemma-4-12b", "google/gemma-4-12B", None, "A100 (~24 GB)", dtype="bfloat16", loader="bridge"),
    # Jamba2 Mini is a hybrid Mamba/attention MoE; the TransformerBridge has a `jamba` adapter,
    # so it loads via the bridge (no_processing=True skips the transformer-specific LN folding
    # that would not apply to its Mamba layers). Caveats: it is instruction-tuned, not a base
    # checkpoint, and it is a 52B-TOTAL MoE (12B active) — all experts stay resident, so bf16
    # needs ~104 GB. The FP8 build (ai21labs/AI21-Jamba2-Mini-FP8) is the only single-GPU-viable
    # option (~52 GB, A100-80GB). trust_remote_code is on because Jamba2 is newer than the
    # pinned `transformers` may know about.
    "jamba-2-mini": ModelChoice(
        "jamba-2-mini", "ai21labs/AI21-Jamba2-Mini", None, "A100-80GB / multi-GPU (52B-total MoE)",
        dtype="bfloat16", loader="bridge", trust_remote_code=True, training="instruct"),
    # FP8 build of the same weights (~52 GB) — the only single-A100-viable Jamba2 Mini. Loading
    # FP8-quantized weights through the bridge with no_processing is unverified; dtype names the
    # compute dtype (activations), the checkpoint carries its own fp8 quantization config.
    "jamba-2-mini-fp8": ModelChoice(
        "jamba-2-mini-fp8", "ai21labs/AI21-Jamba2-Mini-FP8", None, "A100-80GB (FP8, ~52 GB)",
        dtype="bfloat16", loader="bridge", trust_remote_code=True, training="instruct"),
    # Llama 4 Scout is a very new MoE model that TransformerLens does not support yet; disabled.
    "llama-4-scout": ModelChoice("llama-4-scout", None, None, "unsupported", supported=False),
}


def supported_keys() -> list:
    return [key for key, choice in MODELS.items() if choice.supported]


def base_keys() -> list:
    """Supported base (pretrained-only) models — the clean sample for concept geometry."""
    return [key for key in supported_keys() if MODELS[key].training == "base"]


def post_trained_keys() -> list:
    """Supported instruction-/reasoning-tuned models — geometry is shaped by post-training.

    The base-vs-post-trained split is meant for comparison: whether the circular concept
    subspaces the base models expose survive (or warp) after instruction/reasoning tuning.
    """
    return [key for key in supported_keys() if MODELS[key].training != "base"]


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
    """Load a menu model with unprocessed weights, dispatching on its loader.

    Resolves (and validates) the key before importing torch/TransformerLens, so an
    unsupported selection fails fast with a clear message instead of after a long import.
    Both loaders yield the same `blocks.{l}.hook_resid_post` hook on unprocessed activations.
    """
    choice = resolve_model(key)
    import torch
    resolved_dtype = dtype if dtype is not None else getattr(torch, choice.dtype)
    if choice.loader == "bridge":
        return _load_via_bridge(choice, device, resolved_dtype)
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained_no_processing(
        choice.hf_id, device=device, dtype=resolved_dtype)


def _load_via_bridge(choice: ModelChoice, device: str, resolved_dtype):
    """Load an architecture outside the classic registry through TransformerBridge.

    Compatibility mode with `no_processing=True` re-exposes the legacy hook names the
    extraction path expects while overriding weight folding/centering off, so the residual
    stream matches the classic no-processing loader used for the registry models.
    """
    from transformer_lens.model_bridge import TransformerBridge
    bridge = TransformerBridge.boot_transformers(
        choice.hf_id, device=device, dtype=resolved_dtype,
        trust_remote_code=choice.trust_remote_code)
    bridge.enable_compatibility_mode(no_processing=True, disable_warnings=True)
    return bridge
