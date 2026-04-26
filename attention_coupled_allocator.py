"""Attention-coupled bit allocator.

Measures marginal attention-output error reduction per bit,
not structural sparsity. Key metric:
  Δ loss_attn(head) / Δ bits

This is what actually determines compressibility.
"""
import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .cache import ChronoQuantCache
from .codec import ChronoQuantCodecMLX


@dataclass
class HeadDistortionStats:
    """Per-head distortion statistics from forward pass."""
    head_idx: int
    layer_idx: int
    
    # Attention output distortion (the real metric)
    attn_output_error: float = 0.0
    
    # Bit consumption
    bits_consumed: float = 0.0
    
    # Marginal efficiency: error reduction per bit
    marginal_efficiency: float = 0.0


class AttentionCoupledAllocator:
    """Allocates bits based on attention-output distortion per bit."""
    
    def __init__(self, n_layers: int, n_heads_per_layer: int):
        self.n_layers = n_layers
        self.n_heads = n_heads_per_layer
        
        # Per-layer/head distortion history
        self.distortion_history: Dict[Tuple[int, int], List[float]] = {}
        
        # Bit allocation (to be learned)
        self.bit_allocation: Dict[Tuple[int, int], int] = {}
        
        # Baseline distortion (no compression)
        self.baseline_distortion: Dict[int, float] = {}
    
    def _measure_attention_output_error(
        self, 
        attn_fp16: mx.array, 
        attn_compressed: mx.array
    ) -> float:
        """Measure L2 error between FP16 and compressed attention outputs."""
        error = mx.square(attn_fp16 - attn_compressed).mean()
        mx.eval(error)
        return error.item()
    
    def calibrate_layer_bits(
        self,
        model,
        input_ids: mx.array,
        layer_idx: int,
        cache_template: ChronoQuantCache,
        bit_candidates: List[int] = [2, 3, 4],
    ) -> Dict[int, float]:
        """Sweep bit widths for one layer, measure actual distortion.
        
        Returns: {bits: distortion}
        """
        from mlx_lm.models.cache import KVCache
        
        results = {}
        
        for bits in bit_candidates:
            # Create cache with this bit width
            cache = model.make_cache()
            for i, c in enumerate(cache):
                if isinstance(c, KVCache):
                    cache[i] = cache_template.__class__(
                        stride_k=cache_template.stride_k,
                        stride_v=cache_template.stride_v,
                        delta_bits_k=4,
                        delta_bits_v=bits,
                        dead_zone_k=cache_template.dead_zone_k,
                        dead_zone_v=cache_template.dead_zone_v,
                    )
            
            # Forward pass
            logits = model(input_ids, cache=cache)
            
            # Measure output distortion (proxy for attention error)
            target = input_ids[:, 1:]
            loss = nn.losses.cross_entropy(logits[:, :-1, :], target, reduction="mean")
            mx.eval(loss)
            results[bits] = mx.exp(loss).item()
        
        return results
    
    def compute_marginal_efficiency(
        self,
        distortion_sweep: Dict[int, float],
    ) -> Dict[int, float]:
        """Compute marginal error REDUCTION per bit.
        
        Positive = bit helps (reduces PPL)
        Negative = bit hurts (increases PPL)
        
        efficiency(bits) = (distortion(bits-1) - distortion(bits)) / 1 bit
        """
        efficiency = {}
        sorted_bits = sorted(distortion_sweep.keys())
        
        for i, bits in enumerate(sorted_bits[1:], 1):
            prev_bits = sorted_bits[i-1]
            delta_distortion = distortion_sweep[prev_bits] - distortion_sweep[bits]
            delta_bits = bits - prev_bits
            
            # Positive = bit helps (reduces distortion)
            # Negative = bit hurts (increases distortion)
            efficiency[bits] = delta_distortion / delta_bits
        
        return efficiency
    
    def allocate_bits_greedy(
        self,
        layer_distortions: Dict[int, Dict[int, float]],  # layer -> {bits: distortion}
        budget_multiplier: float = 1.0,
    ) -> List[int]:
        """Greedy bit allocation minimizing total distortion.
        
        Args:
            layer_distortions: {layer_idx: {bits: ppl}}
            budget_multiplier: multiply base budget (1.0 = use minimum bits)
        
        Returns:
            per_layer_v_bits: [bits per KV layer]
        """
        # Compute marginal efficiency per layer
        layer_efficiency = {}
        for layer, sweep in layer_distortions.items():
            eff = self.compute_marginal_efficiency(sweep)
            layer_efficiency[layer] = eff
        
        # Start with minimum bits for all layers
        allocation = {
            layer: min(sweep.keys()) 
            for layer, sweep in layer_distortions.items()
        }
        
        # Greedy: add bits where they help most (highest positive efficiency)
        for _ in range(int(budget_multiplier * len(layer_distortions))):
            best_layer = None
            best_gain = float('-inf')
            
            for layer in layer_distortions.keys():
                current_bits = allocation[layer]
                available = [b for b in layer_distortions[layer].keys() if b > current_bits]
                
                for bits in available:
                    gain = layer_efficiency[layer].get(bits, float('-inf'))
                    if gain > best_gain:
                        best_gain = gain
                        best_layer = layer
            
            if best_layer is not None and best_gain > 0:
                best_bits = min([b for b in layer_distortions[best_layer].keys() 
                            if b > allocation[best_layer]])
                allocation[best_layer] = best_bits
            else:
                break
        
        # Return in layer order
        sorted_layers = sorted(allocation.keys())
        return [allocation[l] for l in sorted_layers]


def run_calibration_sweep(
    model,
    tokenizer,
    calibration_text: str,
    stride_k: int = 32,
    stride_v: int = 8,
    bit_candidates: List[int] = [2, 3, 4],
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Run full calibration sweep measuring attention-output distortion.
    
    Returns: {(layer_idx, head_idx): {bits: distortion}}
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache, make_prompt_cache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Get layer info
    caches = model.make_cache()
    kv_layer_indices = [
        i for i, c in enumerate(caches) 
        if isinstance(c, KVCache)
    ]
    
    results = {}
    
    for layer_idx in kv_layer_indices:
        for bits in bit_candidates:
            caches = model.make_cache()
            for i in range(len(caches)):
                if isinstance(caches[i], KVCache):
                    caches[i] = ChronoQuantCache(
                        stride_k=stride_k,
                        stride_v=stride_v,
                        delta_bits_k=4,
                        delta_bits_v=bits,
                    )
            
            # Run forward
            logits = model(input_ids, cache=caches)
            
            # Measure loss as proxy for attention distortion
            loss = nn.losses.cross_entropy(
                logits[:, :-1, :], 
                input_ids[:, 1:], 
                reduction="mean"
            )
            mx.eval(loss)
            ppl = mx.exp(loss).item()
            
            key = (layer_idx, bits)
            if key not in results:
                results[key] = {}
            results[key][bits] = ppl
        
        print(f"  Layer {layer_idx}: {results[(layer_idx, 4)]:.4f} PPL @ 4-bit")
    
    return results


def learn_bit_allocation_from_sweep(
    sweep_results: Dict[Tuple[int, int], Dict[int, float]],
    per_layer_budget: int = 4,
) -> List[int]:
    """Learn optimal per-layer bit allocation from sweep results.
    
    Returns: per_layer_v_bits list
    """
    # Group by layer
    layer_results = {}
    for (layer, bits), distortion in sweep_results.items():
        if layer not in layer_results:
            layer_results[layer] = {}
        layer_results[layer][bits] = distortion
    
    # Greedy allocation per layer
    allocation = []
    
    for layer in sorted(layer_results.keys()):
        lr = layer_results[layer]
        sorted_bits = sorted(lr.keys())
        
        # Find bits closest to budget (minimize distortion)
        best_bits = min(sorted_bits, key=lambda b: lr[b])
        
        allocation.append(best_bits)
    
    return allocation