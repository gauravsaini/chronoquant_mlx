import mlx.core as mx
from typing import Tuple

class ChronoQuantCodecMLX:
    """MLX-native ChronoQuant Codec.
    
    Handles symmetric delta quantization with dead-zone support.
    Dead-zone: values below threshold are zeroed before quantization,
    concentrating bits on informative residual components.
    """
    def __init__(self, stride: int = 32, delta_bits: int = 4, dead_zone_ratio: float = 0.0):
        self.stride = stride
        self.delta_bits = delta_bits
        self.half_levels = (2 ** delta_bits) // 2
        self.dead_zone_ratio = dead_zone_ratio
        
        # Distribution logging (diagnostic)
        self._total_elements = 0
        self._zero_elements = 0
        self._log_interval = 100  # log every N calls
        self._call_count = 0

    def quantize_delta(self, delta: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Quantize delta tensor with optional dead-zone.
        delta: [B, n_kv_heads, S, head_dim]
        Returns:
            codes: [B, n_kv_heads, S, head_dim] (int8)
            scales: [B, n_kv_heads, S, 1] (float16)
        """
        # Per-token, per-head initial scale: max absolute value over head_dim
        amax = mx.abs(delta).max(axis=-1, keepdims=True)
        scale = amax / (self.half_levels - 1)
        
        # Dead-zone: zero out near-zero residuals before quantization
        if self.dead_zone_ratio > 0.0:
            threshold = amax * self.dead_zone_ratio
            delta_clipped = mx.where(mx.abs(delta) < threshold, 0.0, delta)
            
            # Recompute scale after dead-zone clipping (key fix!)
            amax_clipped = mx.abs(delta_clipped).max(axis=-1, keepdims=True)
            # If all values were dead-zoned, keep original scale to avoid div-by-zero
            amax_safe = mx.where(amax_clipped < 1e-10, amax, amax_clipped)
            scale = amax_safe / (self.half_levels - 1)
            delta = delta_clipped
        
        # Avoid division by zero
        scale_safe = mx.where(scale < 1e-10, 1.0, scale)
        
        codes = mx.round(delta / scale_safe)
        codes = mx.clip(codes, -self.half_levels, self.half_levels - 1)
        
        # If scale was exactly 0, codes should be 0
        codes = mx.where(scale < 1e-10, 0, codes)
        
        # Distribution logging
        self._call_count += 1
        if self._call_count % self._log_interval == 0:
            n_total = codes.size
            n_zero = mx.sum(codes == 0).item()
            sparsity = n_zero / max(n_total, 1)
            print(f"  [codec] bits={self.delta_bits} dz={self.dead_zone_ratio:.2f} "
                  f"sparsity={sparsity:.1%} ({n_zero}/{n_total} zeros)")
        
        return codes.astype(mx.int8), scale.astype(mx.float16)

    def dequantize_delta(self, codes: mx.array, scale: mx.array) -> mx.array:
        """
        Dequantize codes.
        codes: [B, n_kv_heads, S, head_dim] (int8)
        scale: [B, n_kv_heads, S, 1] (float16)
        """
        return codes.astype(mx.float16) * scale

