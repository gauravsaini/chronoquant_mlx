"""ChronoQuant Metal kernels and packing helpers."""

import mlx.core as mx

MAX_FUSED_HEAD_DIM = 256
FUSED_SIMDGROUPS = 31
FUSED_THREADGROUP_SIZE = FUSED_SIMDGROUPS * 32

_PACK_INT4_SOURCE = """
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 8;
    uint total = codes_shape[0];

    uint32_t packed = 0;
    for (uint i = 0; i < 8; i++) {
        uint idx = base + i;
        if (idx < total) {
            packed |= (static_cast<uint32_t>(codes[idx]) & 0xFu) << (i * 4);
        }
    }
    packed_codes[word_idx] = packed;
"""

_pack_int4_kernel = mx.fast.metal_kernel(
    name="chronoquant_pack_int4",
    input_names=["codes"],
    output_names=["packed_codes"],
    source=_PACK_INT4_SOURCE,
)

_INT4_SHIFTS = mx.array([i * 4 for i in range(8)], dtype=mx.uint32)


def pack_int4_codes(codes: mx.array) -> mx.array:
    """Pack signed INT4 codes [-8, 7] into uint32 words."""
    original_shape = codes.shape
    head_dim = original_shape[-1]
    num_words = (head_dim + 7) // 8

    unsigned = mx.clip(codes.astype(mx.int32) + 8, 0, 15).astype(mx.uint32)
    if head_dim % 8 != 0:
        pad = num_words * 8 - head_dim
        unsigned = mx.pad(unsigned, [(0, 0)] * (unsigned.ndim - 1) + [(0, pad)])

    flat = unsigned.reshape(-1)
    total_words = flat.size // 8
    outputs = _pack_int4_kernel(
        inputs=[flat],
        grid=(total_words, 1, 1),
        threadgroup=(min(256, max(total_words, 1)), 1, 1),
        output_shapes=[(total_words,)],
        output_dtypes=[mx.uint32],
    )
    return outputs[0].reshape(*original_shape[:-1], num_words)


def unpack_int4_codes(packed: mx.array, head_dim: int) -> mx.array:
    """Unpack uint32 words back to signed INT4 codes [-8, 7]."""
    expanded = ((packed[..., None] >> _INT4_SHIFTS) & 0xF).astype(mx.int32)
    expanded = expanded.reshape(*packed.shape[:-1], packed.shape[-1] * 8)
    return (expanded[..., :head_dim] - 8).astype(mx.int8)


_CHRONOQUANT_SDPA_HEADER = """
#include <metal_simdgroup>
using namespace metal;

constant uint MAX_HEAD_DIM = 256;
constant uint MAX_SLOTS = MAX_HEAD_DIM / 32;
constant uint NUM_SIMDGROUPS = 31;

inline int unpack_signed_int4(const device uint32_t* packed, uint dim) {
    uint word = dim >> 3;
    uint shift = (dim & 7) << 2;
    return int((packed[word] >> shift) & 0xFu) - 8;
}
"""

_CHRONOQUANT_SDPA_SOURCE = """
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_id = tid >> 5;
    uint lane_id = tid & 31;

    uint n_q_heads = q_shape[0];
    uint D = q_shape[1];
    if (head >= n_q_heads || D > MAX_HEAD_DIM) {
        return;
    }

    uint T_kv = static_cast<uint>(params[0]);
    uint stride_k = static_cast<uint>(params[1]);
    uint stride_v = static_cast<uint>(params[2]);

    uint n_kv_heads = keyframes_k_shape[0];
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;

    uint kf_count_k = keyframes_k_shape[1];
    uint pf_words_k = packed_k_shape[2];
    uint pf_count_k = packed_k_shape[1];
    uint kf_count_v = keyframes_v_shape[1];
    uint pf_words_v = packed_v_shape[2];
    uint pf_count_v = packed_v_shape[1];

    uint slots = (D + 31) >> 5;
    float q_local[MAX_SLOTS];
    float local_acc[MAX_SLOTS];
    for (uint s = 0; s < MAX_SLOTS; s++) {
        q_local[s] = 0.0f;
        local_acc[s] = 0.0f;
    }

    uint q_base = head * D;
    for (uint s = 0; s < slots; s++) {
        uint dim = lane_id + s * 32;
        if (dim < D) {
            q_local[s] = q[q_base + dim];
        }
    }

    float local_max = -1e10f;
    float local_sum = 0.0f;

    for (uint t = simd_id; t < T_kv; t += NUM_SIMDGROUPS) {
        uint k_anchor = t / stride_k;
        bool k_is_keyframe = (t % stride_k) == 0;
        uint k_pf = k_is_keyframe ? 0 : (t - k_anchor - 1);

        uint v_anchor = t / stride_v;
        bool v_is_keyframe = (t % stride_v) == 0;
        uint v_pf = v_is_keyframe ? 0 : (t - v_anchor - 1);

        uint kf_base_k = (kv_head * kf_count_k + k_anchor) * D;
        uint kf_base_v = (kv_head * kf_count_v + v_anchor) * D;
        uint pf_base_k = (kv_head * pf_count_k + k_pf) * pf_words_k;
        uint pf_base_v = (kv_head * pf_count_v + v_pf) * pf_words_v;

        const device uint32_t* token_packed_k = packed_k + pf_base_k;
        const device uint32_t* token_packed_v = packed_v + pf_base_v;

        float k_scale = k_is_keyframe ? 0.0f : static_cast<float>(scales_k[kv_head * pf_count_k + k_pf]);
        float v_scale = v_is_keyframe ? 0.0f : static_cast<float>(scales_v[kv_head * pf_count_v + v_pf]);

        float partial = 0.0f;
        for (uint s = 0; s < slots; s++) {
            uint dim = lane_id + s * 32;
            if (dim >= D) {
                continue;
            }

            float k_val = static_cast<float>(keyframes_k[kf_base_k + dim]);
            if (!k_is_keyframe) {
                k_val += static_cast<float>(unpack_signed_int4(token_packed_k, dim)) * k_scale;
            }
            partial += q_local[s] * k_val;
        }

        float score = simd_sum(partial);

        float new_max = max(local_max, score);
        float factor = metal::fast::exp(local_max - new_max);
        float exp_s = metal::fast::exp(score - new_max);
        local_max = new_max;
        local_sum = local_sum * factor + exp_s;

        for (uint s = 0; s < slots; s++) {
            uint dim = lane_id + s * 32;
            if (dim >= D) {
                continue;
            }

            float v_val = static_cast<float>(keyframes_v[kf_base_v + dim]);
            if (!v_is_keyframe) {
                v_val += static_cast<float>(unpack_signed_int4(token_packed_v, dim)) * v_scale;
            }
            local_acc[s] = local_acc[s] * factor + exp_s * v_val;
        }
    }

    threadgroup float tg_max[NUM_SIMDGROUPS];
    threadgroup float tg_sum[NUM_SIMDGROUPS];
    threadgroup float tg_acc[NUM_SIMDGROUPS * MAX_HEAD_DIM];

    if (lane_id == 0) {
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }
    for (uint s = 0; s < slots; s++) {
        uint dim = lane_id + s * 32;
        if (dim < D) {
            tg_acc[simd_id * MAX_HEAD_DIM + dim] = local_acc[s];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e10f;
    for (uint s = 0; s < NUM_SIMDGROUPS; s++) {
        global_max = max(global_max, tg_max[s]);
    }

    float global_sum = 0.0f;
    float result[MAX_SLOTS];
    for (uint s = 0; s < MAX_SLOTS; s++) {
        result[s] = 0.0f;
    }

    for (uint s = 0; s < NUM_SIMDGROUPS; s++) {
        float factor = metal::fast::exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * factor;
        for (uint slot = 0; slot < slots; slot++) {
            uint dim = lane_id + slot * 32;
            if (dim < D) {
                result[slot] += tg_acc[s * MAX_HEAD_DIM + dim] * factor;
            }
        }
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    uint out_base = head * D;
    for (uint s = 0; s < slots; s++) {
        uint dim = lane_id + s * 32;
        if (dim < D) {
            output[out_base + dim] = result[s] * inv;
        }
    }
"""

_chronoquant_sdpa_kernel = mx.fast.metal_kernel(
    name="chronoquant_sdpa",
    input_names=[
        "q",
        "keyframes_k",
        "packed_k",
        "scales_k",
        "keyframes_v",
        "packed_v",
        "scales_v",
        "params",
    ],
    output_names=["output"],
    source=_CHRONOQUANT_SDPA_SOURCE,
    header=_CHRONOQUANT_SDPA_HEADER,
)


def chronoquant_sdpa_kernel(
    q: mx.array,
    keyframes_k: mx.array,
    packed_k: mx.array,
    scales_k: mx.array,
    keyframes_v: mx.array,
    packed_v: mx.array,
    scales_v: mx.array,
    seq_len: int,
    stride_k: int,
    stride_v: int,
) -> mx.array:
    """Fused ChronoQuant SDPA for B=1, T_q=1 generation."""
    params = mx.array([seq_len, stride_k, stride_v], dtype=mx.float32)
    n_q_heads, head_dim = q.shape
    outputs = _chronoquant_sdpa_kernel(
        inputs=[
            q,
            keyframes_k,
            packed_k,
            scales_k,
            keyframes_v,
            packed_v,
            scales_v,
            params,
        ],
        grid=(n_q_heads * FUSED_THREADGROUP_SIZE, 1, 1),
        threadgroup=(FUSED_THREADGROUP_SIZE, 1, 1),
        output_shapes=[(n_q_heads * head_dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, head_dim)
