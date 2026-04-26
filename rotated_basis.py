"""Rotation-Aware KV Codec.

Finds the CORRECT coordinate system for KV space.

Key insight:
    KV space is axis-aligned in current basis
    But attention sensitivity is ROTATED
    
Solution:
    Learn transform T such that: KV' = T(KV)
    Then quantize in rotated space where:
        - Redundancy concentrates
        - Compression becomes selective
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SubspaceAnalysis:
    """Analysis of KV subspace structure."""
    layer_idx: int
    
    # Principal components (sorted by importance)
    components: np.ndarray = None
    
    # Variance per component
    explained_variance: np.ndarray = None
    
    # Cumulative variance (top n components)
    cumvar_topk: np.ndarray = None
    
    # Sensitive directions (for attention)
    sensitive_dims: List[int] = field(default_factory=list)


@dataclass 
class LearnedBasis:
    """Learned basis transform for one head."""
    layer_idx: int
    head_idx: int
    
    # Rotation matrix (head_dim, head_dim)
    rotation: mx.array = None
    
    # Inverse rotation
    inverse: mx.array = None
    
    # Which dimensions are redundant (low variance + low sensitivity)
    redundant_dims: List[int] = field(default_factory=list)
    
    # Which dimensions are sensitive (high attention relevance)
    sensitive_dims: List[int] = field(default_factory=list)


def compute_svd_analysis(kv: mx.array, n_components: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SVD of KV to find principal directions.
    
    Returns:
        singular_values, right_singular_vectors
    """
    # kv: (batch, n_heads, seq, head_dim)
    B, H, S, D = kv.shape
    
    # Reshape for SVD
    kv_reshaped = kv.reshape(B * H * S, D)
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(kv_reshaped.asnumpy(), full_matrices=False)
    
    # Top components
    return s[:n_components], Vt[:n_components]


def analyze_subspace_structure(
    model,
    tokenizer,
    calibration_text: str,
    layer_indices: List[int],
) -> Dict[int, SubspaceAnalysis]:
    """Analyze KV subspace to find redundancy.
    
    Key question:
        Is information concentrated in specific directions?
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Forward pass to collect KV
    caches = model.make_cache()
    logits = model(input_ids, cache=caches)
    loss = nn.losses.cross_entropy(logits[:, :-1, :], input_ids[:, 1:], reduction="mean")
    mx.eval(loss)
    
    # Extract KV from caches at specific layers
    analyses = {}
    
    for layer_idx in layer_indices:
        if isinstance(caches[layer_idx], KVCache):
            # Get KV stored in cache
            kv = caches[layer_idx]
            
            # For KVCache that stores actual KV tensors
            if hasattr(kv, 'keys') and kv.keys is not None:
                keys = kv.keys  # (batch, n_kv_heads, seq, head_dim)
                
                # SVD analysis
                if keys is not None:
                    s, V = compute_svd_analysis(keys)
                    
                    total_var = np.sum(s)
                    explained = s / (total_var + 1e-8)
                    cumvar = np.cumsum(explained)
                    
                    analyses[layer_idx] = SubspaceAnalysis(
                        layer_idx=layer_idx,
                        components=s,
                        explained_variance=explained,
                        cumvar_topk=cumvar,
                        sensitive_dims=list(np.argsort(s)[-8:].astype(int)),
                    )
    
    return analyses


def learn_basis_per_head(
    kv_samples: mx.array,
    n_components: int = 8,
) -> Tuple[mx.array, mx.array]:
    """Learn basis transform T for one head.
    
    T = rotation matrix that aligns with data variance
    
    Returns:
        T: rotation matrix (head_dim, head_dim)
        T_inv: inverse
    """
    # kv_samples: (n_samples, head_dim)
    B, D = kv_samples.shape
    
    # Center the data
    mean = mx.mean(kv_samples, axis=0)
    kv_centered = kv_samples - mean
    
    # SVD for principal directions
    _, s, Vt = mx.linalg.svd(kv_centered, full_matrices=False)
    
    # Top n_components as rotation basis
    # This is a Frozen linear transform
    T = Vt[:n_components].T  # (head_dim, n_components)
    T_inv = Vt[:n_components]  # (n_components, head_dim)
    
    return T, T_inv


def quantize_in_rotated_space(
    kv: mx.array,
    basis: LearnedBasis,
    bits_high: int = 4,
    bits_low: int = 2,
) -> Tuple[mx.array, mx.array]:
    """Quantize KV after rotating to learned basis.
    
    High precision on sensitive dims
    Low precision on redundant dims
    """
    # Rotate to basis
    kv_rotated = kv @ basis.rotation  # (..., head_dim) -> (..., n_components)
    
    # Split into sensitive vs redundant
    sensitive = kv_rotated[..., basis.sensitive_dims]
    redundant = mx.concatenate([
        kv_rotated[..., i] for i in basis.redundant_dims
    ], axis=-1)
    
    # Quantize with different precision
    amax_sens = mx.abs(sensitive).max(axis=-1, keepdims=True)
    scale_sens = amax_sens / ((2 ** bits_high) - 1)
    codes_sens = mx.round(sensitive / (scale_sens + 1e-8))
    
    amax_red = mx.abs(redundant).max(axis=-1, keepdims=True)
    scale_red = amax_red / ((2 ** bits_low) - 1)
    codes_red = mx.round(redundant / (scale_red + 1e-8))
    
    return codes_sens, codes_red


def create_rotation_aware_codec(
    model,
    tokenizer,
    calibration_text: str,
) -> Dict[int, LearnedBasis]:
    """Create rotation-aware compression for all heads.
    
    This is the research-grade approach:
    1. Analyze subspace structure
    2. Learn basis T per head
    3. Quantize in rotated space
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Forward pass
    caches = model.make_cache()
    _ = model(input_ids, cache=caches)
    
    # Find KV layers
    caches = model.make_cache()
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    # Learn basis per layer
    bases = {}
    
    for layer_idx in kv_indices:
        if hasattr(caches[layer_idx], 'keys'):
            keys = caches[layer_idx].keys
            if keys is not None:
                B, H, S, D = keys.shape
                
                # Flatten for PCA
                kv_flat = keys.reshape(B * H * S, D)
                
                # Learn transform
                for h in range(H):
                    head_kv = kv_flat[:, :]  # Use all dims
                    
                    T, T_inv = learn_basis_per_head(head_kv, n_components=16)
                    
                    bases[(layer_idx, h)] = LearnedBasis(
                        layer_idx=layer_idx,
                        head_idx=h,
                        rotation=T,
                        inverse=T_inv,
                    )
    
    return bases


def test_subspace_redundancy(
    model,
    tokenizer,
    calibration_text: str,
) -> Dict[int, float]:
    """Test if information is redundant in some KV dimensions.
    
    If top 50% dimensions capture 80%+ variance,
    then KV is low-rank and rotation could help.
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Forward pass
    caches = model.make_cache()
    _ = model(input_ids, cache=caches)
    
    # Find KV indices
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    redundancy = {}
    
    for layer_idx in kv_indices:
        if hasattr(caches[layer_idx], 'keys'):
            keys = caches[layer_idx].keys
            if keys is not None and keys.size > 0:
                # Cast to float32 first
                keys_fp32 = keys.astype(mx.float32)
                kv_np = np.array(keys_fp32, dtype=np.float32)
                kv_reshaped = kv_np.reshape(-1, kv_np.shape[-1])
                
                # SVD on CPU
                s = np.linalg.svd(kv_reshaped, compute_uv=False)
                
                # Variance in top 50% of dimensions
                total_val = np.sum(s)
                half_idx = len(s) // 2
                top_half_val = np.sum(s[:half_idx]) if half_idx > 0 else total_val
                
                redundancy[layer_idx] = top_half_val / total_val if total_val > 0 else 0
    
    return redundancy


# =============================================================================
# MAIN API FOR ROTATION-AWARE COMPRESSION
# =============================================================================

def analyze_kv_geometry(
    model,
    tokenizer,
    calibration_text: str,
) -> Tuple[Dict, Dict]:
    """Main API for KV subspace analysis.
    
    Returns:
        subspace_analysis: dimension importance
        basis_transforms: learned rotations (if applicable)
    """
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    
    input_ids = mx.array(tokenizer.encode(calibration_text))[None]
    
    # Analyze redundancy
    redundancy = test_subspace_redundancy(model, tokenizer, calibration_text)
    
    # Analyze subspace
    subspace = analyze_subspace_structure(
        model, tokenizer, calibration_text,
        list(redundance.keys())
    )
    
    return redundancy, subspace