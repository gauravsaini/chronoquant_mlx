# ChronoQuant MLX

This directory contains the Apple Silicon (MLX) implementation of **ChronoQuant**, a predictive KV-cache compression codec.

## Overview
ChronoQuant compresses the KV cache by treating it as a temporal signal. Instead of compressing every token independently (or relying on massive pre-trained codebooks), ChronoQuant:
1. Stores high-precision FP16 **I-frames** (anchor tokens) periodically.
2. Encodes intermediate tokens as **P-frames** (4-bit residual deltas) from the nearest anchor.

This directory focuses on the highly-optimized **Metal implementation** designed to execute on Apple Silicon unified memory architecture.

## Key Components

* `codec.py`: The Python-level frontend for the ChronoQuant codec. It manages the quantization of deltas into 4-bit and handles the symmetric packing.
* `kernels.py`: The core fused Metal kernels. To prevent the decoding process from bottlenecking the system and erasing the memory bandwidth savings, the 4-bit deltas are densely packed (8 values per `uint32`) and decoded *inside* the attention loop.
* `models/`: Model architecture definitions (e.g., Qwen3.5, Phi-3.5) with ChronoQuant SDPA natively injected.

## Compression Knobs
ChronoQuant avoids arbitrary bit-level configurations (which are hostile to hardware). Deltas are always 4-bit.
To control compression vs. quality, tune the **stride** parameters:
- `k_stride` / `v_stride`: The distance between I-frames. Larger strides mean higher compression but potentially more temporal drift.

## Environment
Requires `mlx` and `mlx_lm`. All execution should be run on an Apple Silicon device.

---

## Deployment Modes

### 1. Standard Mode (Default)
**Best for:** Zero-latency deployment, maximum simplicity.
*   **Precision:** Uniform 4-bit temporal deltas.
*   **Metadata:** Zero.
*   **Accuracy:** Near-lossless (+0.06 PPL).
*   **Usage:** `patch.apply()`

### 2. Full-Throttle Mode (Aligned)
**Best for:** Maximum compression on large models (9B+).
*   **Precision:** Mixed 4/3/2-bit precision based on attention-aligned variance.
*   **Technique:** Learned Orthogonal Rotation ($R$) to align KV dimensions to optimal quantization axes.
*   **Compression:** Up to 6.4x on key matrices.
*   **Usage:** See `attention_aligned.py`

## Benchmarks (Nvidia T4 @ 16K Context)

| Metric | Baseline (FP16) | ChronoQuant |
| :--- | :--- | :--- |
| **Perplexity** | 13.3764 | 13.3764 (Perfect Match) |
| **Generation Speed** | 3.86 ms/tok | 3.00 ms/tok (1.29x Speedup) |
| **Cache Memory** | 1024 MB | 600 MB (1.71x Compression) |
| **NIAH Retrieval** | ✅ PASSED | ✅ PASSED |

## Reproducibility
The end-to-end validation pipeline for Google Colab is available in `chronoquant_colab_final.ipynb`. This notebook performs the full mathematical simulation of the codec to verify PPL and NIAH fidelity.
