"""ChronoQuant attention dispatch."""

import mlx.core as mx

from .kernels import MAX_FUSED_HEAD_DIM, chronoquant_sdpa_kernel


def _apply_mask(scores: mx.array, mask):
    if mask is None:
        return scores

    if isinstance(mask, str):
        q_len, k_len = scores.shape[-2:]
        q_indices = mx.arange(k_len - q_len, k_len)
        k_indices = mx.arange(k_len)
        mask = q_indices[:, None] >= k_indices[None]

    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask


def chronoquant_sdpa(
    queries: mx.array,
    keys,
    values,
    cache,
    scale: float,
    mask=None,
):
    """ChronoQuant SDPA with fused generation path and MLX fallback."""
    batch, n_q_heads, q_len, head_dim = queries.shape

    use_fused = (
        cache.use_fused
        and
        batch == 1
        and q_len == 1
        and cache.offset > 0
        and head_dim <= MAX_FUSED_HEAD_DIM
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
        and hasattr(mx.fast, "metal_kernel")
    )

    if use_fused:
        q = (queries * scale).reshape(n_q_heads, head_dim).astype(mx.float32)
        out = chronoquant_sdpa_kernel(
            q=q,
            keyframes_k=cache.kernel_keyframes_k(),
            packed_k=cache.kernel_packed_k(),
            scales_k=cache.kernel_scales_k(),
            keyframes_v=cache.kernel_keyframes_v(),
            packed_v=cache.kernel_packed_v(),
            scales_v=cache.kernel_scales_v(),
            seq_len=cache.offset,
            stride_k=cache.stride_k,
            stride_v=cache.stride_v,
        )
        return out.reshape(batch, n_q_heads, q_len, head_dim).astype(queries.dtype)

    full_k, full_v = cache.reconstruct_history()
    n_kv_heads = full_k.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    q_scaled = queries * scale
    q_grouped = q_scaled.reshape(batch, n_kv_heads, n_repeats, q_len, head_dim)
    k_grouped = full_k[:, :, None, :, :]
    v_grouped = full_v[:, :, None, :, :]

    scores = q_grouped @ k_grouped.transpose(0, 1, 2, 4, 3)
    scores = _apply_mask(scores, mask)
    weights = mx.softmax(scores, axis=-1, precise=True)
    output = weights @ v_grouped
    return output.reshape(batch, n_q_heads, q_len, head_dim).astype(queries.dtype)
