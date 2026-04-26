"""Simple Gradient-Free Rotation Learning.

Much simpler approach that actually runs:
- Sample random orthogonal matrices
- Pick the one that preserves attention best
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class SimpleRotationLearner:
    """Simple greedy rotation learner."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.R = mx.eye(dim, dtype=mx.float32)
        
    def sample_orthogonal(self, seed: int) -> mx.array:
        """Sample random orthogonal matrix."""
        np.random.seed(seed)
        Q, _ = np.linalg.qr(np.random.randn(self.dim, self.dim))
        return mx.array(Q, dtype=mx.float32)
    
    def fit(
        self,
        Q: mx.array,
        K: mx.array,
        V: mx.array,
        n_candidates: int = 32,
    ) -> Dict:
        """Fit rotation by picking best candidate."""
        
        # Compute baseline attention output
        scores_base = mx.matmul(Q, K.transpose(0, 1, 3, 2))
        attn_base = mx.softmax(scores_base, axis=-1)
        out_base = mx.matmul(attn_base, V)
        
        best_R = self.R
        best_loss = float('inf')
        
        loss_history = []
        
        for cand in range(n_candidates):
            # Sample candidate rotation
            R_cand = self.sample_orthogonal(cand * 1000 + 42)
            
            # Apply and measure
            K_rot = K @ R_cand
            V_rot = V @ R_cand
            
            scores = mx.matmul(Q, K_rot.transpose(0, 1, 3, 2))
            attn = mx.softmax(scores, axis=-1)
            out_rot = mx.matmul(attn, V_rot)
            
            # Attention preservation loss
            loss = float(mx.square(out_base - out_rot).mean())
            loss_history.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_R = R_cand
        
        self.R = best_R
        
        return {
            'best_loss': best_loss,
            'loss_mean': np.mean(loss_history),
            'loss_std': np.std(loss_history),
        }
    
    def get_matrix(self) -> mx.array:
        return self.R


def learn_rotation_simple(
    model,
    tokenizer,
    calibration_text: str,
    n_candidates: int = 16,
) -> Dict[int, mx.array]:
    """Learn rotation for each KV layer (simplified)."""
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Forward pass
    caches = model.make_cache()
    _ = model(input_ids, cache=caches)
    
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    rotations = {}
    
    for layer_idx in kv_indices[:4]:  # First 4 for speed
        keys = caches[layer_idx].keys
        values = caches[layer_idx].values if hasattr(caches[layer_idx], 'values') else keys
        
        if keys is None:
            continue
        
        # Take sample for learning
        Q = keys[:, :, :4, :]  # B, H, S, D
        K_sample = keys[:, :, :8, :]  # shorter seq
        V_sample = values[:, :, :8, :] if values is not None else K_sample
        
        # Learn rotation
        learner = SimpleRotationLearner(keys.shape[-1])
        
        try:
            result = learner.fit(Q, K_sample, V_sample, n_candidates=n_candidates)
            rotations[layer_idx] = learner.get_matrix()
            print(f"Layer {layer_idx}: loss = {result['best_loss']:.6f}")
        except Exception as e:
            print(f"Layer {layer_idx}: failed - {e}")
    
    return rotations


def apply_rotation_compression(
    kv: mx.array,
    R: mx.array,
    drop_fraction: float = 0.5,
) -> Tuple[mx.array, mx.array, Dict]:
    """Apply learned rotation + compress sensitive dims.
    
    Args:
        kv: (B, H, S, D)
        R: rotation matrix (D, D)
        fraction: fraction of dims to compress/quantize
    
    Returns:
        compressed_kv, scale, info
    """
    # Apply rotation
    kv_rot = kv @ R
    
    # Get importance (variance in rotated space)
    importance = mx.abs(kv_rot).mean(axis=(0, 2))  # per dimension
    
    # Sort and identify low-importance dims
    imp_np = np.array(importance)
    sorted_dims = np.argsort(imp_np)
    
    n_keep_high = int(len(imp_np) * 0.2)  # top 20%
    n_keep_med = int(len(imp_np) * 0.3)   # next 30%
    
    high_dims = sorted_dims[:n_keep_high]
    med_dims = sorted_dims[n_keep_high:n_keep_high + n_keep_med]
    low_dims = sorted_dims[n_keep_high + n_keep_med:]
    
    # Quantize high dims at 4-bit
    kv_high = kv_rot[..., high_dims]
    scale_high = mx.abs(kv_high).max(axis=-1, keepdims=True) / 7.0
    codes_high = mx.round(kv_high / (scale_high + 1e-8))
    codes_high = mx.clip(codes_high, -7, 7)
    
    # Quantize med dims at 3-bit  
    kv_med = kv_rot[..., med_dims]
    scale_med = mx.abs(kv_med).max(axis=-1, keepdims=True) / 3.0
    codes_med = mx.round(kv_med / (scale_med + 1e-8))
    codes_med = mx.clip(codes_med, -3, 3)
    
    # Drop low dims
    dropped = len(low_dims)
    
    # Reconstruct with dropped dims
    kv_compressed = mx.zeros_like(kv_rot)
    kv_compressed = mx.concatenate([
        codes_high.astype(mx.float32) * scale_high,
        codes_med.astype(mx.float32) * scale_med,
        mx.zeros((kv.shape[0], kv.shape[1], kv.shape[2], dropped), dtype=mx.float32)
    ], axis=-1)
    
    # Rotate back (inverse is transpose for orthogonal)
    kv_final = kv_compressed @ R.T
    
    info = {
        'n_high': len(high_dims),
        'n_medium': len(med_dims),
        'n_dropped': dropped,
        'compression_ratio': dropped / kv.shape[-1],
    }
    
    return kv_final, None, info


# Test it works
if __name__ == "__main__":
    print("Testing rotation learning...")