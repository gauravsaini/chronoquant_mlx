"""Per-layer bit allocator - finds bits that minimize PPL.

Usage:
    from chronoquant_mlx.allocator import allocate_optimal_bits
    
    model, tokenizer = mlx_lm.load(MODEL_PATH)
    per_layer_v_bits = allocate_optimal_bits(model, tokenizer, calibration_text)
    # per_layer_v_bits -> [4, 4, 4, 4, 4, 4, 4, 4]
    
    caches = chronoquant_mlx.create_chronoquant_caches(
        model, 
        per_layer_v_bits=per_layer_v_bits
    )
"""
import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple

from .cache import ChronoQuantCache


def allocate_optimal_bits(
    model,
    tokenizer,
    calibration_text: str,
    bit_candidates: List[int] = [2, 3, 4],
) -> List[int]:
    """Find bit allocation that minimizes PPL.
    
    Greedy approach:
    1. Start with minimum bits
    2. Add bits where they help most (reduce PPL)
    3. Stop when no bits help
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Find KV layer indices
    caches = model.make_cache()
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    n_kv = len(kv_indices)
    
    # Baseline: minimum bits
    min_bits = min(bit_candidates)
    caches = model.make_cache()
    for i in kv_indices:
        caches[i] = ChronoQuantCache(
            stride_k=32, stride_v=8,
            delta_bits_k=4, delta_bits_v=min_bits,
        )
    
    logits = model(input_ids, cache=caches)
    loss = nn.losses.cross_entropy(logits[:, :-1, :], input_ids[:, 1:], reduction="mean")
    mx.eval(loss)
    best_ppl = mx.exp(loss).item()
    best_allocation = [min_bits] * n_kv
    
    # Greedily add bits
    for bits in sorted(bit_candidates):
        if bits <= min_bits:
            continue
            
        # Test this bit level on a single layer at a time
        for kv_idx in kv_indices:
            test_allocation = best_allocation.copy()
            test_allocation[kv_indices.index(kv_idx)] = bits
            
            # Create cache with test config
            caches = model.make_cache()
            for i, kv_i in enumerate(kv_indices):
                caches[kv_i] = ChronoQuantCache(
                    stride_k=32, stride_v=8,
                    delta_bits_k=4, delta_bits_v=test_allocation[i],
                )
            
            logits = model(input_ids, cache=caches)
            loss = nn.losses.cross_entropy(logits[:, :-1, :], input_ids[:, 1:], reduction="mean")
            mx.eval(loss)
            test_ppl = mx.exp(loss).item()
            
            # If helps, keep it
            if test_ppl < best_ppl - 0.01:  # meaningful improvement
                best_ppl = test_ppl
                best_allocation = test_allocation.copy()
                print(f"  Layer {kv_i}: {min_bits}→{bits} bits improves PPL: {test_ppl:.4f}")
    
    print(f"Optimal: {best_allocation[0]}-bit for all, PPL={best_ppl:.4f}")
    return best_allocation


def allocate_bits_greedy(
    model,
    tokenizer,
    calibration_text: str,
) -> List[int]:
    """Wrapper that returns per_layer_v_bits compatible with create_chronoquant_caches."""
    return allocate_optimal_bits(model, tokenizer, calibration_text)