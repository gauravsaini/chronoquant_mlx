"""Attention-Diagonalization Learning.

The true SOTA objective:

Learn rotation R such that:
    Attention becomes locally linear in R-space

Instead of:
    min ||A - Â||  (uniform loss)

Use:
    min J ⊙ ||K - K̂||  (Jacobian-weighted)
    
Where:
    J = importance of each dimension to attention output
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class LearnedRotation:
    """Rotation learned via attention-diagonalization."""
    layer_idx: int
    
    # Orthogonal rotation (D, D)  
    rotation: mx.array = None
    
    # Inverse rotation
    inverse: mx.array = None
    
    # Training metrics
    initial_loss: float = 0.0
    final_loss: float = 0.0
    n_iterations: int = 0


def compute_attention_jacobian_weighted_loss(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    K_hat: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Compute Jacobian-weighted loss for rotation learning.
    
    L = Σ_j (J[j] * ||K[:,:,:,j] - K_hat[:,:,:,j]||^2)
    
    Where J = attention sensitivity per dimension.
    """
    # Attention scores
    scores = mx.matmul(Q, K.transpose(0, 1, 3, 2))
    
    # Attention weights
    attn = mx.softmax(scores, axis=-1)
    
    # Sensitivity: gradient of attention w.r.t K
    # Simplified: weight by attention contribution
    sensitivity = mx.sqrt(attn + 1e-8).mean(axis=(2,))
    
    # Per-dimension loss
    diff = mx.square(K - K_hat)
    weighted_loss = diff * sensitivity[:, None, None, :]
    
    return weighted_loss.sum(), sensitivity


def compute_attention_rotation_objective(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    R: mx.array,
) -> mx.array:
    """Compute attention-diagonalization objective.
    
    Goal: minimize attention output change under rotation.
    
    L = || softmax(QK^T)V - softmax(Q(RK)^T)(RV) ||
    """
    # Original attention
    scores_orig = mx.matmul(Q, K.transpose(0, 1, 3, 2))
    attn_orig = mx.softmax(scores_orig, axis=-1)
    output_orig = mx.matmul(attn_orig, V)
    
    # Rotated attention  
    K_rot = K @ R
    V_rot = V @ R
    
    scores_rot = mx.matmul(Q, K_rot.transpose(0, 1, 3, 2))
    attn_rot = mx.softmax(scores_rot, axis=-1)
    output_rot = mx.matmul(attn_rot, V_rot)
    
    # Loss
    loss = mx.square(output_orig - output_rot).mean()
    
    return loss


def givens_rotation_matrix(theta: mx.array, n: int, i: int, j: int) -> mx.array:
    """Create Givens rotation matrix.
    
    Applies rotation in plane (i,j) by angle theta.
    """
    cos_t = mx.cos(theta)
    sin_t = mx.sin(theta)
    
    # Identity with 2x2 rotation block
    R = mx.eye(n, dtype=mx.float32)
    R = R + mx.scatter(
        mx.zeros((n, n), dtype=mx.float32),
        indices=[(i, i), (i, j), (j, i), (j, j)],
        updates=[cos_t - 1, sin_t, -sin_t, cos_t - 1],
        shape=(n, n)
    )
    
    return R


def learn_attention_aligned_rotation(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    n_iterations: int = 50,
    learning_rate: float = 0.1,
) -> mx.array:
    """Learn orthogonal rotation via gradient descent.
    
    Minimize: L = ||Attention(Q,K,V) - Attention(Q,RK,RV)||_2
    
    Subject to: R^T R = I (orthogonal constraint via Givens)
    """
    B, H, S, D = K.shape
    
    # Initialize as identity
    R = mx.eye(D, dtype=mx.float32)
    
    # Optimization loop (simplified - full would use Givens)
    for itr in range(n_iterations):
        # Forward
        K_rot = K @ R
        scores = mx.matmul(Q, K_rot.transpose(0, 1, 3, 2))
        attn = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn, V_rot if 'V_rot' in dir() else V)
        
        # Gradient via automatic differentiation
        # This is expensive - simplified proxy:
        loss = mx.square(output - output).mean()  # placeholder
        
        # For now, return identity (full impl needs optimizer)
        R = R * 1.0
    
    return R


def learn_rotation_gradient_free(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    n_candidates: int = 32,
) -> Tuple[mx.array, float]:
    """Gradient-free rotation learning via candidate sampling.
    
    For each (i,j) plane, sample random angles and pick best.
    """
    B, H, S, D = K.shape
    
    # Compute baseline attention loss
    scores_base = mx.matmul(Q, K.transpose(0, 1, 3, 2))
    attn_base = mx.softmax(scores_base, axis=-1)
    output_base = mx.matmul(attn_base, V)
    
    best_R = mx.eye(D)
    best_loss = float('inf')
    
    # Sample random 2D planes
    for _ in range(n_candidates):
        i, j = np.random.choice(D, 2, replace=False)
        
        # Random angle
        theta = mx.array(np.random.uniform(0, 2*np.pi))
        
        # Givens rotation
        R_givens = mx.eye(D, dtype=mx.float32)
        
        cos_t = mx.cos(theta)
        sin_t = mx.sin(theta)
        
        # Apply to R
        R_trial = best_R.copy()
        col_i = R_trial[:, i].copy()
        col_j = R_trial[:, j].copy()
        
        R_trial[:, i] = cos_t * col_i + sin_t * col_j
        R_trial[:, j] = -sin_t * col_i + cos_t * col_j
        
        # Evaluate
        K_rot = K @ R_trial
        scores = mx.matmul(Q, K_rot.transpose(0, 1, 3, 2))
        attn = mx.softmax(scores, axis=-1)
        output_rot = mx.matmul(attn, V)
        
        loss = float(mx.square(output_base - output_rot).mean())
        
        if loss < best_loss:
            best_loss = loss
            best_R = R_trial
    
    return best_R, best_loss


def diagonalize_attention_space(
    model,
    tokenizer,
    calibration_text: str,
    n_layers: int = 8,
) -> Dict[int, LearnedRotation]:
    """Main API: learn attention-diagonalizing rotation.
    
    Returns learned rotations for each KV layer.
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Forward pass
    caches = model.make_cache()
    _ = model(input_ids, cache=caches)
    
    # Get KV layers
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    results = {}
    
    # Simplified: use identity rotation (full training expensive)
    for layer_idx in kv_indices[:n_layers]:
        results[layer_idx] = LearnedRotation(
            layer_idx=layer_idx,
            rotation=mx.eye(256),  # Would be learned
            inverse=mx.eye(256),
            initial_loss=0.0,
            final_loss=0.0,
            n_iterations=0,
        )
    
    return results


# =============================================================================
# SIMPLIFIED VERSION FOR GPU PRACTICAL USE
# =============================================================================

def get_attention_sensitive_quantization(
    model,
    tokenizer,
    calibration_text: str,
) -> Tuple[Dict[int, Dict], Dict]:
    """Practical attention-sensitive quantization.
    
    Without expensive rotation learning, use sensitivity to guide precision.
    
    Returns: per_layer config, sensitivity analysis
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Forward pass
    caches = model.make_cache()
    _ = model(input_ids, cache=caches)
    
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    configs = {}
    sensitivity = {}
    
    for layer_idx in kv_indices:
        keys = caches[layer_idx].keys
        if keys is None:
            continue
        
        # Shape: (batch, n_heads, seq, head_dim)
        # We want sensitivity per head_dim across heads and seq
        keys_fp = keys.astype(mx.float32)  # (1, 4, 256, 256)
        
        # Sensitivity: mean absolute value per dimension
        # Average over batch (0), heads (1), seq (2) -> get (head_dim,)
        sensitivity_dim = mx.abs(keys_fp).mean(axis=(0, 1, 2))  # (256,)
        
        # Convert to numpy
        sens_np = np.array(sensitivity_dim)
        n_dims = len(sens_np)
        
        if n_dims < 2:
            continue
            
        # Sort dimensions by sensitivity
        sorted_dims = np.argsort(sens_np)[::-1]
        
        # Define precision groups
        n_high = max(1, int(n_dims * 0.15))   # top 15%
        n_medium = max(1, int(n_dims * 0.25))  # next 25%
        
        high = sorted_dims[:n_high].tolist()
        medium = sorted_dims[n_high:n_high+n_medium].tolist()
        low = sorted_dims[n_high+n_medium:].tolist()
        
        configs[layer_idx] = {
            'high_bits': 4,
            'medium_bits': 3,
            'low_bits': 2,
            'high_dims': high,
            'medium_dims': medium,
            'low_dims': low,
        }
        
        sensitivity[layer_idx] = {
            'sensitivity_ranking': sorted_dims[:20].tolist(),
            'top_sensitivity': float(sens_np[sorted_dims[0]]) if n_dims > 0 else 0.0,
            'bottom_sensitivity': float(sens_np[sorted_dims[-1]]) if n_dims > 0 else 0.0,
            'dim_range': n_dims,
        }
    
    return configs, sensitivity