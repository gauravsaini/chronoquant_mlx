"""Monkey-patch for ChronoQuant SDPA dispatch."""

import mlx_lm.models.base as _base

from .attention import chronoquant_sdpa
from .cache import ChronoQuantCache

_original_sdpa = _base.scaled_dot_product_attention
_patched = False


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
    delta_bits: int = 4,
    use_fused: bool = True,
    heterogeneous_layers: bool = False,
):
    """Create ChronoQuant caches for all KV cache slots in model."""
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    caches = model.make_cache() if hasattr(model, "make_cache") else make_prompt_cache(model)
    total_layers = len(caches)
    for index in range(total_layers):
        if isinstance(caches[index], KVCache):
            layer_bits = 2 if (heterogeneous_layers and index < total_layers // 2) else delta_bits
            caches[index] = ChronoQuantCache(
                stride_k=stride_k,
                stride_v=stride_v,
                delta_bits=layer_bits,
                use_fused=use_fused,
            )
    return caches
