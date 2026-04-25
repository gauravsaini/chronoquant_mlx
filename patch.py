"""Monkey-patch for ChronoQuant SDPA dispatch."""

import mlx_lm.models.base as _base

from .attention import chronoquant_sdpa
from .cache import ChronoQuantCache

_original_sdpa = _base.scaled_dot_product_attention
_patched = False


import math
import mlx.core as mx

_HADAMARD_CACHE = {}

def get_hadamard_matrix(dim: int, dtype):
    if dim not in _HADAMARD_CACHE:
        H = mx.array([[1.0]], dtype=dtype)
        while H.shape[0] < dim:
            H = mx.concatenate([
                mx.concatenate([H, H], axis=1),
                mx.concatenate([H, -H], axis=1)
            ], axis=0)
        H = H / math.sqrt(dim)
        _HADAMARD_CACHE[dim] = H
    return _HADAMARD_CACHE[dim]

def apply_hadamard(tensor):
    D = tensor.shape[-1]
    H = get_hadamard_matrix(D, tensor.dtype)
    return tensor @ H

def _patched_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
    if isinstance(cache, ChronoQuantCache):
        return chronoquant_sdpa(queries, keys, values, cache, scale, mask)
        
    return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)


def apply_patch():
    """Activate ChronoQuant SDPA patch. Idempotent."""
    global _patched
    if _patched:
        return

    _base.scaled_dot_product_attention = _patched_sdpa

    import sys

    for name, module in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and hasattr(module, "scaled_dot_product_attention"):
            module.scaled_dot_product_attention = _patched_sdpa

    _patched = True
    print("✅ ChronoQuant SDPA patch applied to all loaded model modules.")


def apply():
    apply_patch()


def revert():
    global _patched
    _base.scaled_dot_product_attention = _original_sdpa
    _patched = False


def create_chronoquant_caches(
    model,
    stride_k: int = 32,
    stride_v: int = 8,
    delta_bits_k: int = 4,
    delta_bits_v: int = 3,
    use_fused: bool = True,
    residual_scale: float = 1.0,
    pruning_ratio: float = 0.0,
    pruning_ratio_k: float = 0.0,
    pruning_ratio_v: float = 0.0,
):
    """Create ChronoQuant caches for all KV cache slots in model."""
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    caches = model.make_cache() if hasattr(model, "make_cache") else make_prompt_cache(model)
    for index in range(len(caches)):
        if isinstance(caches[index], KVCache):
            caches[index] = ChronoQuantCache(
                stride_k=stride_k,
                stride_v=stride_v,
                delta_bits_k=delta_bits_k,
                delta_bits_v=delta_bits_v,
                use_fused=use_fused,
                residual_scale=residual_scale,
                pruning_ratio=pruning_ratio,
                pruning_ratio_k=pruning_ratio_k,
                pruning_ratio_v=pruning_ratio_v,
            )
    return caches
