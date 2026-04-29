# ChronoQuant MLX

Apple Silicon / MLX implementation of ChronoQuant, a temporal predictive KV-cache codec.

ChronoQuant treats the KV cache like a video stream:

1. Periodic FP16 I-frames store exact key/value anchors.
2. Intermediate P-frames store quantized residual deltas from the nearest anchor.
3. A fused Metal attention path decodes the residuals inside attention instead of reconstructing full KV tensors in the Python/MLX graph.

## Main Modes

### Standard Mode

Best default for quality and simplicity.

```python
import chronoquant_mlx

chronoquant_mlx.apply()
caches = chronoquant_mlx.create_chronoquant_caches(
    model,
    stride_k=32,
    stride_v=8,
)
logits = model(input_ids, cache=caches)
```

This uses INT4 residual deltas for both keys and values.

### Full-Throttle V3

Measured aggressive no-learned-state preset for Qwen3.5-9B long-context runs.

```python
import chronoquant_mlx

chronoquant_mlx.apply()
caches = chronoquant_mlx.create_full_throttle_caches(model)
logits = model(input_ids, cache=caches)
```

Equivalent explicit config:

```python
caches = chronoquant_mlx.create_chronoquant_caches(
    model,
    stride_k=64,
    stride_v=32,
    delta_bits_k=4,
    delta_bits_v=3,
    dead_zone_k=0.0,
    dead_zone_v=0.05,
)
```

Important caveat: the current fused MLX storage path still packs residual codes into INT4 lanes. So Full-Throttle V3 is a measured quality/speed/active-byte preset, not proof of true physical variable-bit packing.

## Measured Qwen3.5-9B Results

Long-context validation on Apple Silicon / MLX:

| Context | Method | PPL | TPS | KV Bytes/Token | Compression |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | FP16 Baseline | 1.2027 | 7.58 | 45344.0 | 1.00x |
| 4096 | Standard `k=32,v=8` | 1.2004 | 6.96 | 22806.0 | 1.99x |
| 4096 | Full-Throttle V3 | 1.2004 | 6.83 | 21469.0 | 2.11x |
| 8192 | FP16 Baseline | 1.1080 | 7.48 | 39056.0 | 1.00x |
| 8192 | Standard `k=32,v=8` | 1.1074 | 6.55 | 16518.0 | 2.36x |
| 8192 | Full-Throttle V3 | 1.1053 | 6.56 | 15181.0 | 2.57x |
| 16384 | FP16 Baseline | 1.0593 | 6.99 | 35912.0 | 1.00x |
| 16384 | Standard `k=32,v=8` | 1.0593 | 5.68 | 13374.0 | 2.69x |
| 16384 | Full-Throttle V3 | 1.0560 | 5.67 | 12037.0 | 2.98x |

Full-Throttle V2 was tested as a more aggressive 2-bit value-residual ablation. It achieved the same measured storage as V3 but degraded perplexity, so it is intentionally not exposed as a recommended preset.

## Components

- `codec.py`: per-token residual quantization and dead-zone handling.
- `cache.py`: ChronoQuant KV cache object with active-byte accounting.
- `kernels.py`: INT4 packing helpers and fused Metal SDPA kernel.
- `attention.py`: dispatch logic for fused generation path and fallback reconstruction path.
- `patch.py`: MLX attention monkey-patch and cache factory helpers.

## Validation

The main validation harness lives in:

```bash
scripts/run_chronoquant_full_throttle_eval.py
```

The remote helper for the Qwen3.5-9B benchmark box lives in:

```bash
scripts/run_chronoquant_full_throttle_remote.sh
```
