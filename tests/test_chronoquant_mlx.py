import math

import mlx.core as mx
import pytest

from chronoquant_mlx.attention import chronoquant_sdpa
from chronoquant_mlx.cache import ChronoQuantCache
from chronoquant_mlx.kernels import MAX_FUSED_HEAD_DIM, pack_int4_codes, unpack_int4_codes


def _has_metal_kernel():
    return hasattr(mx.fast, "metal_kernel")


def _manual_reconstruct(x: mx.array, stride: int, delta_bits: int = 4):
    half_levels = (2**delta_bits) // 2
    seq_len = x.shape[2]
    out = []
    keyframes = []
    for t in range(seq_len):
        token = x[:, :, t : t + 1, :]
        if t % stride == 0:
            recon = token.astype(mx.float16)
            keyframes.append(recon)
        else:
            anchor = keyframes[t // stride]
            delta = token - anchor.astype(token.dtype)
            amax = mx.abs(delta).max(axis=-1, keepdims=True)
            scale = amax / (half_levels - 1)
            scale_safe = mx.where(scale < 1e-10, 1.0, scale)
            codes = mx.round(delta / scale_safe)
            codes = mx.clip(codes, -half_levels, half_levels - 1)
            codes = mx.where(scale < 1e-10, 0, codes).astype(mx.int8)
            recon = anchor + codes.astype(mx.float16) * scale.astype(mx.float16)
        out.append(recon.astype(mx.float16))
    return mx.concatenate(out, axis=2)


def _reference_sdpa(queries, keys, values, scale):
    batch, n_q_heads, q_len, head_dim = queries.shape
    n_kv_heads = keys.shape[1]
    repeats = n_q_heads // n_kv_heads
    q_grouped = (queries * scale).reshape(batch, n_kv_heads, repeats, q_len, head_dim)
    k_grouped = keys[:, :, None, :, :]
    v_grouped = values[:, :, None, :, :]
    scores = q_grouped @ k_grouped.transpose(0, 1, 2, 4, 3)
    weights = mx.softmax(scores, axis=-1, precise=True)
    return (weights @ v_grouped).reshape(batch, n_q_heads, q_len, head_dim)


class TestPacking:
    def test_pack_unpack_roundtrip(self):
        codes = mx.random.randint(-8, 8, (1, 2, 5, 128)).astype(mx.int8)
        mx.eval(codes)
        packed = pack_int4_codes(codes)
        unpacked = unpack_int4_codes(packed, 128)
        diff = mx.abs(codes.astype(mx.int32) - unpacked.astype(mx.int32)).max().item()
        assert diff == 0


class TestCache:
    def test_reconstruct_history_matches_manual_codec(self):
        cache = ChronoQuantCache(stride_k=4, stride_v=2, delta_bits=4)
        keys = mx.random.normal((1, 2, 9, 128)).astype(mx.float16)
        values = mx.random.normal((1, 2, 9, 128)).astype(mx.float16)
        mx.eval(keys, values)

        cache.update_and_fetch(keys, values)
        recon_k, recon_v = cache.reconstruct_history()
        ref_k = _manual_reconstruct(keys, stride=4)
        ref_v = _manual_reconstruct(values, stride=2)
        mx.eval(recon_k, recon_v, ref_k, ref_v)

        assert mx.abs(recon_k - ref_k).max().item() < 1e-3
        assert mx.abs(recon_v - ref_v).max().item() < 1e-3


@pytest.mark.skipif(not _has_metal_kernel(), reason="mx.fast.metal_kernel not available")
class TestFusedAttention:
    def test_fused_generation_matches_reference(self):
        head_dim = 128
        assert head_dim <= MAX_FUSED_HEAD_DIM

        cache = ChronoQuantCache(stride_k=4, stride_v=2, delta_bits=4)
        keys = mx.random.normal((1, 2, 11, head_dim)).astype(mx.float16)
        values = mx.random.normal((1, 2, 11, head_dim)).astype(mx.float16)
        queries = mx.random.normal((1, 4, 1, head_dim)).astype(mx.float16)
        mx.eval(keys, values, queries)

        cache.update_and_fetch(keys, values)
        full_k, full_v = cache.reconstruct_history()
        scale = 1.0 / math.sqrt(head_dim)

        fused = chronoquant_sdpa(queries, keys, values, cache, scale, mask=None)
        ref = _reference_sdpa(queries, full_k, full_v, scale)
        mx.eval(fused, ref)

        diff = mx.abs(fused.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        assert diff < 2e-2, f"fused chronoquant diff too high: {diff}"
