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
    ):
        self.stride_k = stride_k
        self.stride_v = stride_v
        self.delta_bits = delta_bits
        self.use_fused = use_fused
        self.codec_k = ChronoQuantCodecMLX(stride=stride_k, delta_bits=delta_bits)
        self.codec_v = ChronoQuantCodecMLX(stride=stride_v, delta_bits=delta_bits)

        self.offset = 0
        self.head_dim = None

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
        count = self._num_keyframes(self.offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count, :]

    def _active_packed(self, component: str):
        tensor = self.packed_codes_k if component == "k" else self.packed_codes_v
        stride = self.stride_k if component == "k" else self.stride_v
        count = self._num_pframes(self.offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count, :]

    def _active_scales(self, component: str):
        tensor = self.pframe_scales_k if component == "k" else self.pframe_scales_v
        stride = self.stride_k if component == "k" else self.stride_v
        count = self._num_pframes(self.offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count]

    def _active_velocities(self, component: str):
        tensor = self.velocities_k if component == "k" else self.velocities_v
        stride = self.stride_k if component == "k" else self.stride_v
        count = self._num_keyframes(self.offset, stride)
        if tensor is None:
            return None
        return tensor[:, :, :count, :]

    def _append_component(self, component: str, tensor: mx.array, stride: int, codec: ChronoQuantCodecMLX):
        keyframes_attr = "keyframes_k" if component == "k" else "keyframes_v"
        velocities_attr = "velocities_k" if component == "k" else "velocities_v"
        packed_attr = "packed_codes_k" if component == "k" else "packed_codes_v"
        scales_attr = "pframe_scales_k" if component == "k" else "pframe_scales_v"

        n_kv_heads = tensor.shape[1]
        all_head_keyframes = []
        all_head_velocities = []
        all_head_packed = []
        all_head_scales = []

        for h in range(n_kv_heads):
            head_tensor = tensor[:, h : h + 1, :, :]
            offset_h = (h * (stride // n_kv_heads)) % stride
            
            existing_keyframes = getattr(self, keyframes_attr)
            existing_velocities = getattr(self, velocities_attr)
            if offset_h == 0:
                existing_n_kf = 0 if self.offset == 0 else ((self.offset - 1) // stride) + 1
            else:
                existing_n_kf = 0 if self.offset == 0 else (1 if self.offset <= offset_h else ((self.offset - offset_h - 1) // stride) + 2)
            
            head_keyframes = []
            head_velocities = []
            head_packed = []
            head_scales = []
            
            for local_t in range(head_tensor.shape[2]):
                token_idx = self.offset + local_t
                token = head_tensor[:, :, local_t : local_t + 1, :]
                
                is_kf = (token_idx == 0) or (token_idx >= offset_h and (token_idx - offset_h) % stride == 0)
                if is_kf:
                    head_keyframes.append(token.astype(mx.float16))
                    local_end = local_t + stride - 1
                    if local_end < head_tensor.shape[2]:
                        token_end = head_tensor[:, :, local_end : local_end + 1, :]
                        vel = (token_end - token.astype(token.dtype)) / float(stride)
                    else:
                        vel = mx.zeros_like(token)
                    head_velocities.append(vel.astype(mx.float16))
                    continue
                
                anchor_idx = 0 if token_idx < offset_h else (token_idx - offset_h) // stride + (1 if offset_h > 0 else 0)
                if anchor_idx < existing_n_kf:
                    anchor = existing_keyframes[:, h : h + 1, anchor_idx : anchor_idx + 1, :]
                    vel = existing_velocities[:, h : h + 1, anchor_idx : anchor_idx + 1, :]
                else:
                    anchor = head_keyframes[anchor_idx - existing_n_kf]
                    vel = head_velocities[anchor_idx - existing_n_kf]
                
                prediction = anchor.astype(mx.float32) + vel.astype(mx.float32) * float(token_idx - (0 if anchor_idx == 0 else offset_h + (anchor_idx - (1 if offset_h > 0 else 0)) * stride))
                prediction = prediction.astype(token.dtype)
                delta = token - prediction
                codes, scale = codec.quantize_delta(delta)
                head_packed.append(pack_int4_codes(codes))
                head_scales.append(scale.squeeze(-1).astype(mx.float16))
            
            packed_words = (tensor.shape[3] + 7) // 8
            all_head_keyframes.append(mx.concatenate(head_keyframes, axis=2) if head_keyframes else mx.zeros((1, 1, 0, tensor.shape[3]), dtype=mx.float16))
            all_head_velocities.append(mx.concatenate(head_velocities, axis=2) if head_velocities else mx.zeros((1, 1, 0, tensor.shape[3]), dtype=mx.float16))
            all_head_packed.append(mx.concatenate(head_packed, axis=2) if head_packed else mx.zeros((1, 1, 0, packed_words), dtype=mx.uint32))
            all_head_scales.append(mx.concatenate(head_scales, axis=2) if head_scales else mx.zeros((1, 1, 0), dtype=mx.float16))

        # Pad across heads to maximum length
        max_kf = max(k.shape[2] for k in all_head_keyframes) if all_head_keyframes else 0
        if max_kf > 0:
            padded_kf = []
            padded_vel = []
            for k, v in zip(all_head_keyframes, all_head_velocities):
                pad_len = max_kf - k.shape[2]
                if pad_len > 0:
                    k = mx.concatenate([k, mx.zeros((1, 1, pad_len, k.shape[3]), dtype=k.dtype)], axis=2)
                    v = mx.concatenate([v, mx.zeros((1, 1, pad_len, v.shape[3]), dtype=v.dtype)], axis=2)
                padded_kf.append(k)
                padded_vel.append(v)
            stacked_kf = mx.concatenate(padded_kf, axis=1)
            stacked_vel = mx.concatenate(padded_vel, axis=1)
            
            current_kf = getattr(self, keyframes_attr)
            current_vel = getattr(self, velocities_attr)
            setattr(self, keyframes_attr, stacked_kf if current_kf is None else mx.concatenate([current_kf, stacked_kf], axis=2))
            setattr(self, velocities_attr, stacked_vel if current_vel is None else mx.concatenate([current_vel, stacked_vel], axis=2))

        max_pf = max(p.shape[2] for p in all_head_packed) if all_head_packed else 0
        if max_pf > 0:
            padded_packed = []
            padded_scales = []
            for p, s in zip(all_head_packed, all_head_scales):
                pad_len = max_pf - p.shape[2]
                if pad_len > 0:
                    p = mx.concatenate([p, mx.zeros((1, 1, pad_len, p.shape[3]), dtype=p.dtype)], axis=2)
                    s = mx.concatenate([s, mx.zeros((1, 1, pad_len), dtype=s.dtype)], axis=2)
                padded_packed.append(p)
                padded_scales.append(s)
            stacked_packed = mx.concatenate(padded_packed, axis=1)
            stacked_scales = mx.concatenate(padded_scales, axis=1)
            
            current_packed = getattr(self, packed_attr)
            current_scales = getattr(self, scales_attr)
            setattr(self, packed_attr, stacked_packed if current_packed is None else mx.concatenate([current_packed, stacked_packed], axis=2))
            setattr(self, scales_attr, stacked_scales if current_scales is None else mx.concatenate([current_scales, stacked_scales], axis=2))

    def _reconstruct_component(self, component: str):
        if self.offset == 0:
            return None

        stride = self.stride_k if component == "k" else self.stride_v
        codec = self.codec_k if component == "k" else self.codec_v
        keyframes = self._active_keyframes(component)
        packed = self._active_packed(component)
        scales = self._active_scales(component)
        velocities = self._active_velocities(component)

        n_kv_heads = keyframes.shape[1] if keyframes is not None else 1
        all_head_reconstructed = []

        for h in range(n_kv_heads):
            offset_h = (h * (stride // n_kv_heads)) % stride
            
            # Reconstruct for head h
            head_kf = keyframes[:, h : h + 1, :, :] if keyframes is not None else None
            head_vel = velocities[:, h : h + 1, :, :] if velocities is not None else None
            head_packed = packed[:, h : h + 1, :, :] if packed is not None else None
            head_scales = scales[:, h : h + 1, :] if scales is not None else None
            
            positions = mx.arange(self.offset, dtype=mx.int32)
            
            # Head-wise anchor and time logic
            is_kf = (positions == 0) | ((positions >= offset_h) & ((positions - offset_h) % stride == 0))
            anchor_idx = mx.where(positions < offset_h, 0, (positions - offset_h) // stride + (1 if offset_h > 0 else 0))
            kf_time = mx.where(anchor_idx == 0, 0, offset_h + (anchor_idx - (1 if offset_h > 0 else 0)) * stride)
            
            full = mx.take(head_kf, anchor_idx, axis=2).astype(mx.float16) if head_kf is not None else mx.zeros((1, 1, self.offset, self.head_dim), dtype=mx.float16)
            
            if head_vel is not None:
                vel = mx.take(head_vel, anchor_idx, axis=2).astype(mx.float16)
                alpha = (positions - kf_time).astype(mx.float16).reshape(1, 1, -1, 1)
                prediction = full + vel * alpha
            else:
                prediction = full
                
            if head_packed is None or head_scales is None or head_packed.shape[2] == 0:
                all_head_reconstructed.append(prediction)
                continue
                
            pf_idx = positions - kf_time - 1
            # For keyframes, pf_idx is negative, we use 0 safely since we mask it out anyway
            safe_pf_idx = mx.maximum(pf_idx, 0)
            
            # Unpack codes
            codes = unpack_int4_codes(head_packed, self.head_dim)
            gathered_codes = mx.take(codes, safe_pf_idx, axis=2)
            gathered_scales = mx.take(head_scales, safe_pf_idx, axis=2)[..., None]
            delta = codec.dequantize_delta(gathered_codes, gathered_scales)
            
            # Apply delta only where it's a P-frame
            is_pframe = (~is_kf).reshape(1, 1, -1, 1)
            reconstructed = prediction + mx.where(is_pframe, delta, mx.zeros_like(delta))
            all_head_reconstructed.append(reconstructed)

        return mx.concatenate(all_head_reconstructed, axis=1)

    def reconstruct_history(self):
        """Reconstruct full K/V history for fallback attention paths."""
        return self._reconstruct_component("k"), self._reconstruct_component("v")

    def kernel_keyframes_k(self):
        return self._active_keyframes("k").squeeze(0)

    def kernel_keyframes_v(self):
        return self._active_keyframes("v").squeeze(0)

    def kernel_packed_k(self):
        packed = self._active_packed("k")
        if packed is not None:
            return packed.squeeze(0)
        return mx.zeros((self.keyframes_k.shape[1], 1, 1), dtype=mx.uint32)

    def kernel_packed_v(self):
        packed = self._active_packed("v")
        if packed is not None:
            return packed.squeeze(0)
        return mx.zeros((self.keyframes_v.shape[1], 1, 1), dtype=mx.uint32)

    def kernel_scales_k(self):
        scales = self._active_scales("k")
        if scales is not None:
            return scales.squeeze(0)
        return mx.zeros((self.keyframes_k.shape[1], 1), dtype=mx.float16)

    def kernel_scales_v(self):
        scales = self._active_scales("v")
        if scales is not None:
            return scales.squeeze(0)
        return mx.zeros((self.keyframes_v.shape[1], 1), dtype=mx.float16)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Compress new KV states and return lightweight placeholders."""
        _, _, _, head_dim = keys.shape
        if self.head_dim is None:
            self.head_dim = head_dim
        elif head_dim != self.head_dim:
            raise ValueError(f"ChronoQuantCache head_dim changed: {self.head_dim} -> {head_dim}")

        self._append_component("k", keys, self.stride_k, self.codec_k)
        self._append_component("v", values, self.stride_v, self.codec_v)
        self.offset += keys.shape[2]

        state = self.state
        if state:
            mx.eval(*state)

        return keys, values

    def make_mask(self, num_queries, return_array=False, window_size=None, **kwargs):
        return make_causal_mask(self.offset, num_queries, return_array, window_size)

    @property
    def state(self):
        if self.offset == 0 or self.keyframes_k is None or self.keyframes_v is None:
            return []

        parts = [self._active_keyframes("k"), self._active_keyframes("v")]
        packed_k = self._active_packed("k")
        scales_k = self._active_scales("k")
        packed_v = self._active_packed("v")
        scales_v = self._active_scales("v")
        if packed_k is not None:
            parts.extend([packed_k, scales_k])
        if packed_v is not None:
            parts.extend([packed_v, scales_v])
        return parts

    @state.setter
    def state(self, values):
        if not values:
            return

        self.keyframes_k = values[0]
        self.keyframes_v = values[1]
        index = 2

        if len(values) >= 4:
            self.packed_codes_k = values[index]
            self.pframe_scales_k = values[index + 1]
            index += 2
        if len(values) >= 6:
            self.packed_codes_v = values[index]
            self.pframe_scales_v = values[index + 1]

        self.head_dim = self.keyframes_k.shape[-1]
        key_tokens = self.keyframes_k.shape[2] + (0 if self.packed_codes_k is None else self.packed_codes_k.shape[2])
        self.offset = key_tokens

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
        ]:
            if tensor is not None:
                total += tensor.nbytes
        return total
