import mlx.core as mx
from typing import Tuple

class ChronoQuantCodecMLX:
    """MLX-native ChronoQuant Codec.
    
    Handles INT4 symmetric delta quantization and dequantization.
    """
    def __init__(self, stride: int = 32, delta_bits: int = 4):
        self.stride = stride
        self.delta_bits = delta_bits
        self.half_levels = (2 ** delta_bits) // 2

    def quantize_delta(self, delta: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Quantize delta tensor to INT4 with per-tensor scale.
        delta: [B, n_kv_heads, S, head_dim]
        Returns:
            codes: [B, n_kv_heads, S, head_dim] (int8)
            scales: [B, n_kv_heads, S, 1] (float16)
        """
        # Per-token, per-head scale: max absolute value over head_dim
        amax = mx.abs(delta).max(axis=-1, keepdims=True)
        scale = amax / (self.half_levels - 1)
        
        # Avoid division by zero
        scale_safe = mx.where(scale < 1e-10, 1.0, scale)
        
        scaled = delta / scale_safe
        codes = mx.where(mx.abs(scaled) < 0.5, mx.zeros_like(scaled), mx.round(scaled))
        codes = mx.clip(codes, -self.half_levels, self.half_levels - 1)
        
        # If scale was exactly 0, codes should be 0
        codes = mx.where(scale < 1e-10, 0, codes)
        
        return codes.astype(mx.int8), scale.astype(mx.float16)

    def dequantize_delta(self, codes: mx.array, scale: mx.array) -> mx.array:
        """
        Dequantize INT4 codes.
        codes: [B, n_kv_heads, S, head_dim] (int8)
        scale: [B, n_kv_heads, S, 1] (float16)
        """
        return codes.astype(mx.float16) * scale
