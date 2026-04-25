import mlx.core as mx

from .codec import ChronoQuantCodecMLX
from .kernels import pack_int4_codes, unpack_int4_codes


def make_causal_mask(offset: int, num_queries: int, return_array: bool = False, window_size=None):
    """Create attention mask compatible with mlx-lm."""
    if num_queries == 1:
        return None
    if return_array or (window_size and num_queries > window_size):
        from mlx_lm.models.base import create_causal_mask

        return create_causal_mask(num_queries, offset=offset - num_queries, window_size=window_size)
    return "causal"


class ChronoQuantCache:
    """ChronoQuant KV cache with packed INT4 deltas and fused attention path."""

    def __init__(
        self,
        stride_k: int = 32,
        stride_v: int = 8,
        delta_bits: int = 4,
        use_fused: bool = True,
        recent_window: int = 128,
    ):
        self.stride_k = stride_k
        self.stride_v = stride_v
        self.delta_bits = delta_bits
        self.use_fused = use_fused
        self.recent_window = recent_window
        self.codec_k = ChronoQuantCodecMLX(stride=stride_k, delta_bits=delta_bits)
        self.codec_v = ChronoQuantCodecMLX(stride=stride_v, delta_bits=delta_bits)

        self.offset = 0
        self.compressed_offset = 0
        self.head_dim = None

        self.recent_k = None
        self.recent_v = None

        self.keyframes_k = None
        self.keyframes_v = None
        self.packed_codes_k = None
        self.packed_codes_v = None
        self.pframe_scales_k = None
        self.pframe_scales_v = None
        self.velocities_k = None
        self.velocities_v = None

    @staticmethod
    def _num_keyframes(num_tokens: int, stride: int) -> int:
        if num_tokens <= 0:
            return 0
        return ((num_tokens - 1) // stride) + 1

    @classmethod
    def _num_pframes(cls, num_tokens: int, stride: int) -> int:
        return num_tokens - cls._num_keyframes(num_tokens, stride)

    def _active_keyframes(self, component: str):
        tensor = self.keyframes_k if component == "k" else self.keyframes_v
        stride = self.stride_k if component == "k" else self.stride_v
        count = self._num_keyframes(self.compressed_offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count, :]

    def _active_packed(self, component: str):
        tensor = self.packed_codes_k if component == "k" else self.packed_codes_v
        stride = self.stride_k if component == "k" else self.stride_v
        count = self._num_pframes(self.compressed_offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count, :]

    def _active_scales(self, component: str):
        tensor = self.pframe_scales_k if component == "k" else self.pframe_scales_v
        stride = self.stride_k if component == "k" else self.stride_v
        count = self._num_pframes(self.compressed_offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count]

    def _active_velocities(self, component: str):
        tensor = self.velocities_k if component == "k" else self.velocities_v
        stride = self.stride_k if component == "k" else self.stride_v
        count = self._num_keyframes(self.compressed_offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count, :]

    def _append_component(self, component: str, tensor: mx.array, stride: int, codec: ChronoQuantCodecMLX):
        keyframes_attr = "keyframes_k" if component == "k" else "keyframes_v"
        velocities_attr = "velocities_k" if component == "k" else "velocities_v"
        packed_attr = "packed_codes_k" if component == "k" else "packed_codes_v"
        scales_attr = "pframe_scales_k" if component == "k" else "pframe_scales_v"

        existing_keyframes = getattr(self, keyframes_attr)
        existing_velocities = getattr(self, velocities_attr)
        existing_n_kf = self._num_keyframes(self.compressed_offset, stride)

        new_keyframes = []
        new_velocities = []
        new_packed = []
        new_scales = []

        for local_t in range(tensor.shape[2]):
            token_idx = self.compressed_offset + local_t
            token = tensor[:, :, local_t : local_t + 1, :]

            if token_idx % stride == 0:
                new_keyframes.append(token.astype(mx.float16))
                local_end = local_t + stride - 1
                if local_end < tensor.shape[2]:
                    token_end = tensor[:, :, local_end : local_end + 1, :]
                    vel = (token_end - token.astype(token.dtype)) / float(stride)
                else:
                    vel = mx.zeros_like(token)
                new_velocities.append(vel.astype(mx.float16))
                continue

            anchor_idx = token_idx // stride
            if anchor_idx < existing_n_kf:
                anchor = existing_keyframes[:, :, anchor_idx : anchor_idx + 1, :]
                vel = existing_velocities[:, :, anchor_idx : anchor_idx + 1, :]
            else:
                anchor = new_keyframes[anchor_idx - existing_n_kf]
                vel = new_velocities[anchor_idx - existing_n_kf]

            prediction = anchor.astype(mx.float32) + vel.astype(mx.float32) * float(token_idx % stride)
            prediction = prediction.astype(token.dtype)
            delta = token - prediction
            codes, scale = codec.quantize_delta(delta)
            new_packed.append(pack_int4_codes(codes))
            new_scales.append(scale.squeeze(-1).astype(mx.float16))

        if new_keyframes:
            stacked_kf = mx.concatenate(new_keyframes, axis=2)
            stacked_vel = mx.concatenate(new_velocities, axis=2)
            current_kf = getattr(self, keyframes_attr)
            current_vel = getattr(self, velocities_attr)
            setattr(self, keyframes_attr, stacked_kf if current_kf is None else mx.concatenate([current_kf, stacked_kf], axis=2))
            setattr(self, velocities_attr, stacked_vel if current_vel is None else mx.concatenate([current_vel, stacked_vel], axis=2))

        if new_packed:
            packed = mx.concatenate(new_packed, axis=2)
            scales = mx.concatenate(new_scales, axis=2)
            current_packed = getattr(self, packed_attr)
            current_scales = getattr(self, scales_attr)
            setattr(self, packed_attr, packed if current_packed is None else mx.concatenate([current_packed, packed], axis=2))
            setattr(self, scales_attr, scales if current_scales is None else mx.concatenate([current_scales, scales], axis=2))

    def _reconstruct_component(self, component: str):
        if self.compressed_offset == 0:
            return None

        stride = self.stride_k if component == "k" else self.stride_v
        codec = self.codec_k if component == "k" else self.codec_v
        keyframes = self._active_keyframes(component)
        packed = self._active_packed(component)
        scales = self._active_scales(component)

        positions = mx.arange(self.compressed_offset, dtype=mx.int32)
        anchor_idx = positions // stride
        full = mx.take(keyframes, anchor_idx, axis=2).astype(mx.float16)
        
        velocities = self._active_velocities(component)
        if velocities is not None:
            vel = mx.take(velocities, anchor_idx, axis=2).astype(mx.float16)
            alpha = (positions % stride).astype(mx.float16).reshape(1, 1, -1, 1)
            prediction = full + vel * alpha
        else:
            prediction = full

        if packed is None or scales is None or packed.shape[2] == 0:
            return prediction

        pf_idx = positions - anchor_idx - 1
        safe_pf_idx = mx.maximum(pf_idx, 0)
        is_pframe = (positions % stride != 0).reshape(1, 1, -1, 1)

        codes = unpack_int4_codes(packed, self.head_dim)
        gathered_codes = mx.take(codes, safe_pf_idx, axis=2)
        gathered_scales = mx.take(scales, safe_pf_idx, axis=2)[..., None]
        delta = codec.dequantize_delta(gathered_codes, gathered_scales)
        return prediction + mx.where(is_pframe, delta, mx.zeros_like(delta))

    def reconstruct_history(self):
        """Reconstruct full K/V history for fallback attention paths."""
        k_comp = self._reconstruct_component("k")
        v_comp = self._reconstruct_component("v")
        
        k_full = k_comp if k_comp is not None else mx.zeros((self.recent_k.shape[0], self.recent_k.shape[1], 0, self.recent_k.shape[3]))
        v_full = v_comp if v_comp is not None else mx.zeros((self.recent_v.shape[0], self.recent_v.shape[1], 0, self.recent_v.shape[3]))
        
        if self.recent_k is not None and self.recent_k.shape[2] > 0:
            k_full = mx.concatenate([k_full, self.recent_k.astype(mx.float16)], axis=2) if k_full.shape[2] > 0 else self.recent_k.astype(mx.float16)
        if self.recent_v is not None and self.recent_v.shape[2] > 0:
            v_full = mx.concatenate([v_full, self.recent_v.astype(mx.float16)], axis=2) if v_full.shape[2] > 0 else self.recent_v.astype(mx.float16)
            
        return k_full, v_full

    def kernel_keyframes_k(self):
        kf = self._active_keyframes("k")
        if kf is not None:
            return kf.squeeze(0)
        return mx.zeros((self.recent_k.shape[1], 1, self.recent_k.shape[3]), dtype=mx.float16)

    def kernel_keyframes_v(self):
        kf = self._active_keyframes("v")
        if kf is not None:
            return kf.squeeze(0)
        return mx.zeros((self.recent_v.shape[1], 1, self.recent_v.shape[3]), dtype=mx.float16)

    def kernel_packed_k(self):
        packed = self._active_packed("k")
        if packed is not None:
            return packed.squeeze(0)
        return mx.zeros((self.recent_k.shape[1], 1, 1), dtype=mx.uint32)

    def kernel_packed_v(self):
        packed = self._active_packed("v")
        if packed is not None:
            return packed.squeeze(0)
        return mx.zeros((self.recent_v.shape[1], 1, 1), dtype=mx.uint32)

    def kernel_scales_k(self):
        scales = self._active_scales("k")
        if scales is not None:
            return scales.squeeze(0)
        return mx.zeros((self.recent_k.shape[1], 1), dtype=mx.float16)

    def kernel_scales_v(self):
        scales = self._active_scales("v")
        if scales is not None:
            return scales.squeeze(0)
        return mx.zeros((self.recent_v.shape[1], 1), dtype=mx.float16)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Compress new KV states and return lightweight placeholders."""
        _, _, _, head_dim = keys.shape
        if self.head_dim is None:
            self.head_dim = head_dim
        elif head_dim != self.head_dim:
            raise ValueError(f"ChronoQuantCache head_dim changed: {self.head_dim} -> {head_dim}")

        if self.recent_k is None:
            self.recent_k = keys
            self.recent_v = values
        else:
            self.recent_k = mx.concatenate([self.recent_k, keys], axis=2)
            self.recent_v = mx.concatenate([self.recent_v, values], axis=2)
            
        self.offset += keys.shape[2]

        while self.recent_k.shape[2] > self.recent_window:
            chunk_k = self.recent_k[:, :, :self.stride_k, :]
            chunk_v = self.recent_v[:, :, :self.stride_k, :]
            
            self._append_component("k", chunk_k, self.stride_k, self.codec_k)
            self._append_component("v", chunk_v, self.stride_v, self.codec_v)
            self.compressed_offset += self.stride_k
            
            self.recent_k = self.recent_k[:, :, self.stride_k:, :]
            self.recent_v = self.recent_v[:, :, self.stride_k:, :]

        state = self.state
        if state:
            mx.eval(*state)

        return keys, values

    def make_mask(self, num_queries, return_array=False, window_size=None, **kwargs):
        return make_causal_mask(self.offset, num_queries, return_array, window_size)

    @property
    def state(self):
        parts = []
        if self.keyframes_k is not None:
            parts.extend([
                self._active_keyframes("k"),
                self._active_keyframes("v"),
            ])
            if self.packed_codes_k is not None:
                parts.extend([self._active_packed("k"), self._active_scales("k")])
            if self.packed_codes_v is not None:
                parts.extend([self._active_packed("v"), self._active_scales("v")])
        if self.recent_k is not None:
            parts.extend([self.recent_k, self.recent_v])
        return parts

    @state.setter
    def state(self, values):
        if not values:
            return

        print(f"DEBUG state.setter called with {len(values)} values. values[0] is {type(values[0])}")

        # Distinguish between [recent_k, recent_v] and full state based on shape and length
        if len(values) == 2:
            self.recent_k, self.recent_v = values
            self.head_dim = self.recent_k.shape[-1]
            self.compressed_offset = 0
            self.offset = self.recent_k.shape[2]
            return

        self.keyframes_k = values[0]
        self.keyframes_v = values[1]
        index = 2

        if len(values) >= index + 4 and len(values) % 2 == 0:
            # We have packed_k, scales_k, packed_v, scales_v
            # Wait, if we have recent_k/v, len is 8 (kf*2 + pk*4 + recent*2)
            if len(values) == 8:
                self.packed_codes_k = values[2]
                self.pframe_scales_k = values[3]
                self.packed_codes_v = values[4]
                self.pframe_scales_v = values[5]
                self.recent_k = values[6]
                self.recent_v = values[7]
            elif len(values) == 6:
                # Either [kf, pk, recent] or [kf, pk, pv]?
                # Just safe assignment assuming the standard order:
                self.packed_codes_k = values[2]
                self.pframe_scales_k = values[3]
                self.recent_k = values[4]
                self.recent_v = values[5]
            elif len(values) == 4:
                self.recent_k = values[2]
                self.recent_v = values[3]
        else:
             pass

        self.head_dim = self.recent_k.shape[-1] if self.recent_k is not None else self.keyframes_k.shape[-1]
        self.compressed_offset = self.keyframes_k.shape[2] + (0 if self.packed_codes_k is None else self.packed_codes_k.shape[2])
        self.offset = self.compressed_offset + (0 if self.recent_k is None else self.recent_k.shape[2])

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, value):
        pass

    def is_trimmable(self):
        return False

    def empty(self):
        return self.offset == 0

    def has_previous_state(self) -> bool:
        return self.offset > 0

    @property
    def nbytes(self) -> int:
        total = 0
        for tensor in [
            self._active_keyframes("k"),
            self._active_keyframes("v"),
            self._active_packed("k"),
            self._active_packed("v"),
            self._active_scales("k"),
            self._active_scales("v"),
            self.recent_k,
            self.recent_v,
        ]:
            if tensor is not None:
                total += tensor.nbytes
        return total
