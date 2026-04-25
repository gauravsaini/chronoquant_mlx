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
