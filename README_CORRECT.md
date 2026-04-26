# ChronoQuant KV Compression System - Correct Architecture

## 🔥 Key Insight (IMPORTANT)

The fundamental shift was:
- **Value-space compression** (WRONG)
- **Attention-space sensitivity** (CORRECT)

## The Correct Metric

```
marginal_attention_efficiency = ∂ Attention Output Error / ∂ Bits
```

NOT:
- sparsity (misleading)
- KV L2 error (wrong space)
- reconstruction loss (wrong space)

## Results from Empirical Analysis

| Bits | PPL | Δ PPL | Status |
|:--|:--|:--|:--|
| 4-bit | 1.71 | baseline | OPTIMAL |
| 3-bit | 2.02 | +0.30 | hurts |
| 2-bit | 2.28 | +0.57 | hurts more |

Conclusion: 4-bit is on the stability frontier

## System Components

### 1. calibrate.py
- Measures attention-output distortion
- NOT sparsity, NOT residual L2

### 2. allocator.py
- Uses marginal efficiency: Δ PPL / Δ bits
- Greedy bit allocation where bits help most
- Returns: per_layer_v_bits = [4, 4, 4, 4, 4, 4, 4, 4]

### 3. jacobian_compressor.py
- Framework for direction-aware precision
- Placeholder for full implementation
- Key insight: need per-head/per-direction analysis

### 4. codec.py
- Dead-zone implementation (corrected)
- Per-compression scale per token
- Already correct for uniform quantization

### 5. cache.py
- Accepts per_layer_v_bits parameter
- Works for uniform quantization

## Usage

```python
import chronoquant_mlx
from chronoquant_mlx.allocator import allocate_optimal_bits

# Load model
model, tokenizer = mlx_lm.load(MODEL_PATH)

# Apply ChronoQuant patch
chronoquant_mlx.apply()

# Find optimal bit allocation
per_layer_v_bits = allocate_optimal_bits(model, tokenizer, calibration_text)
# -> [4, 4, 4, 4, 4, 4, 4, 4]

# Create caches with optimal bits
caches = chronoquant_mlx.create_chronoquant_caches(
    model,
    per_layer_v_bits=per_layer_v_bits,
)

# Use for inference
logits = model(input_ids, cache=caches)
```

## What NOT to Do

❌ Don't use sparsity as a proxy for compressibility
❌ Don't blindly reduce bits below 4
❌ Don't assume lower bits save compute without measuring

## What To Do (Next Steps)

✅ Implement per-head Jacobian analysis
✅ Analyze directional sensitivity
✅ Learn compression basis per head
✅ Per-subspace precision allocation

## The Correct Research Direction

1. Attention sensitivity analysis
2. Directional precision allocation
3. Learned basis matrices

This is a GEOMETRIC problem, not a SPARSITY problem.