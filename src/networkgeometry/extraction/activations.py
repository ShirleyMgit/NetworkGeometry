import numpy as np
from networkgeometry.types import DataMatrix

def load_model(name: str = "google/gemma-2-2b", device: str = "cpu"):
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained(name, device=device)

def _final_token_resid(model, prompt: str, layers: list) -> dict:
    names = [f"blocks.{l}.hook_resid_post" for l in layers]
    _logits, cache = model.run_with_cache(prompt, names_filter=lambda n: n in names)
    resids = {}
    for l in layers:
        vec = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].detach().float().cpu().numpy()
        if not np.isfinite(vec).all():
            raise ValueError(
                f"non-finite activation at layer {l} for prompt {prompt!r}: the forward pass "
                f"overflowed (fp16 caps at 65504, and large models like Gemma/Llama have massive "
                f"outlier activations). Load the model in bfloat16 (dtype=torch.bfloat16) — it has "
                f"fp32's exponent range. Left unchecked this surfaces later as 'SVD did not converge'.")
        resids[l] = vec
    return resids

def extract(model, prompts_by_run, states, structure, layers) -> list:
    dms = []
    for run, prompts in prompts_by_run.items():
        per_layer = {l: [] for l in layers}
        for prompt in prompts:
            resid = _final_token_resid(model, prompt, layers)
            for l in layers:
                per_layer[l].append(resid[l])
        for l in layers:
            matrix = np.stack(per_layer[l], axis=1)     # (d, n_states)
            dms.append(DataMatrix(structure, l, run, matrix, states))
    return dms
