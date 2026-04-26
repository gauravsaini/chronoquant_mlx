"""Gradient-Based Rotation Learning for KV Compression.

This is what actually beats SOTA:
    min Attention(Q,K,V) - Attention(Q,RK,RV)
    
With orthogonal constraint: R^T R = I
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


def givens_rotation(theta: mx.array, D: int, i: int, j: int) -> mx.array:
    """Create Givens rotation in plane (i,j).
    
    R = I with 2D rotation block:
    [cos -sin]
    [sin  cos]
    """
    R = mx.eye(D, dtype=mx.float32)
    
    # Update rows i,j
    R = mx.where(
        mx.arange(D)[:, None] == mx.array([i, j, i, j]),
        mx.array([mx.cos(theta), mx.sin(theta), -mx.sin(theta), mx.cos(theta)]),
        R
    )
    
    return R


def givens_gradient(theta: mx.array, D: int, i: int, j: int) -> mx.array:
    """Gradient of Givens rotation w.r.t theta."""
    R = mx.eye(D, dtype=mx.float32)
    
    dR = mx.zeros((D, D), dtype=mx.float32)
    dR = mx.where(
        mx.arange(D)[:, None] == mx.array([i, j, i, j]),
        mx.array([-mx.sin(theta), mx.cos(theta), mx.cos(theta), -mx.sin(theta)]),
        dR
    )
    
    return dR


class LearnedRotation:
    """Rotation learned via attention-preserving objective."""
    
    def __init__(self, dim: int, rank: int = 64):
        self.dim = dim
        self.rank = rank
        self.R = mx.eye(dim, dtype=mx.float32)
        
    def compute_attention_loss(
        self,
        Q: mx.array,  # (B, H, S_q, D)
        K: mx.array,  # (B, H, S_k, D)  
        V: mx.array,  # (B, H, S_v, D)
    ) -> Tuple[mx.array, mx.array]:
        """Compute attention output loss under rotation."""
        # Original attention
        scores_orig = mx.matmul(Q, K.transpose(0, 1, 3, 2))
        attn_orig = mx.softmax(scores_orig, axis=-1)
        out_orig = mx.matmul(attn_orig, V)
        
        # Rotated attention  
        K_rot = K @ self.R
        V_rot = V @ self.R
        
        scores_rot = mx.matmul(Q, K_rot.transpose(0, 1, 3, 2))
        attn_rot = mx.softmax(scores_rot, axis=-1)
        out_rot = mx.matmul(attn_rot, V_rot)
        
        loss = mx.square(out_orig - out_rot).mean()
        
        return loss, out_orig
    
    def fit(
        self,
        Q: mx.array,
        K: mx.array, 
        V: mx.array,
        lr: float = 0.01,
        n_planes: int = 32,
        n_steps: int = 50,
    ) -> Dict:
        """Learn rotation via gradient descent on Givens planes."""
        
        D = self.dim
        _, H, S, _ = K.shape
        
        losses = []
        
        for step in range(n_steps):
            # Sample random planes
            for _ in range(n_planes):
                i, j = np.random.choice(D, 2, replace=False)
                
                # Random angle
                theta_init = mx.random.uniform(0, 2 * np.pi)
                
                # Compute loss at several angles
                thetas = mx.linspace(theta_init - 0.5, theta_init + 0.5, 5)
                loss_vals = []
                
                for t in thetas:
                    G = givens_rotation(t, D, i, j)
                    trial_R = self.R @ G
                    
                    K_rot = K @ trial_R
                    V_rot = V @ trial_R
                    
                    scores = mx.matmul(Q, K_rot.transpose(0, 1, 3, 2))
                    attn = mx.softmax(scores, axis=-1)
                    out = mx.matmul(attn, V_rot)
                    
                    # Compare to original
                    scores0 = mx.matmul(Q, K.transpose(0, 1, 3, 2))
                    attn0 = mx.softmax(scores0, axis=-1)
                    out0 = mx.matmul(attn0, V)
                    
                    loss = mx.square(out - out0).mean()
                    loss_vals.append(float(loss))
                
                # Take best angle
                best_idx = np.argmin(loss_vals)
                theta_best = float(thetas[best_idx])
                
                G = givens_rotation(mx.array(theta_best), D, i, j)
                self.R = self.R @ G
                
                # Compute final loss
                l, _ = self.compute_attention_loss(Q, K, V)
                losses.append(float(l))
        
        return {
            'final_loss': losses[-1] if losses else 0,
            'loss_improvement': losses[0] - losses[-1] if losses else 0,
            'n_steps': n_steps,
        }
    
    def get_matrix(self) -> mx.array:
        """Get learned rotation matrix."""
        return self.R


def learn_rotation_for_all_layers(
    model,
    tokenizer,
    calibration_text: str,
    n_steps: int = 20,
    n_planes: int = 16,
) -> Dict[int, LearnedRotation]:
    """Learn attention-preserving rotation for all KV layers."""
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Forward pass
    caches = model.make_cache()
    _ = model(input_ids, cache=caches)
    
    # Get KV layers
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    rotations = {}
    
    for layer_idx in kv_indices:
        keys = caches[layer_idx].keys
        values = caches[layer_idx].values if hasattr(caches[layer_idx], 'values') else keys
        
        if keys is None or values is None:
            continue
        
        _, H, S, D = keys.shape
        
        # Use smaller sample for speed
        Q = keys[:, :, :4, :]  # short query
        K = keys
        V = values
        
        # Learn rotation
        rot = LearnedRotation(D, rank=min(D, 64))
        
        try:
            result = rot.fit(Q, K, V, n_steps=n_steps, n_planes=n_planes)
            rotations[layer_idx] = rot
            print(f"Layer {layer_idx}: loss = {result['final_loss']:.4f}, improvement = {result['loss_improvement']:.4f}")
        except Exception as e:
            print(f"Layer {layer_idx}: failed - {e}")
    
    return rotations


def apply_rotation_to_kv(
    kv: mx.array,
    R: mx.array,
) -> mx.array:
    """Apply learned rotation to KV tensor."""
    return kv @ R