import mlx.core as mx
import mlx_lm
import mlx_lm.models.base as _base
import e2e_learnable
import math

_original_sdpa = _base.scaled_dot_product_attention
layer_codecs = {}

def apply_calibration_patch():
    def _patched_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
        # Find which layer we are in based on cache (a hack for benchmarking)
        # MLX cache object doesn't trivially store layer idx, but we assume
        # if it's the first time we see a cache object, it's layer 0, etc.
        # For simplicity, we just apply the same codec if only testing one layer,
        # but since we have multiple, we'll try to guess based on keys shape or just
        # skip patching if we can't identify.
        # Actually, let's just rotate K, V and unrotate for perplexity check.
        layer_idx = getattr(cache, 'layer_idx', None)
        
        if layer_idx is not None and layer_idx in layer_codecs:
            codec = layer_codecs[layer_idx]
            
            # Encode (rotates and quantizes)
            K_rot, k_info = codec.encode(keys)
            V_rot, v_info = codec.encode(values) if values is not None else (None, None)
            
            # Decode (dequantizes and inverse rotates)
            k_recon = codec.decode(k_info['codes'], k_info['scales'])
            v_recon = codec.decode(v_info['codes'], v_info['scales']) if values is not None else None
            
            return _original_sdpa(queries, k_recon, v_recon, cache, scale, mask, **kwargs)
        
        return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)
    
    import sys
    _base.scaled_dot_product_attention = _patched_sdpa
    for name, module in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and hasattr(module, "scaled_dot_product_attention"):
            module.scaled_dot_product_attention = _patched_sdpa

def calculate_perplexity(model, tokenizer, text):
    input_ids = mx.array(tokenizer.encode(text))[None]
    
    # Label shift
    target_ids = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    
    # We assign layer_idx to caches
    caches = model.make_cache()
    from mlx_lm.models.cache import KVCache
    idx = 0
    for c in caches:
        if isinstance(c, KVCache):
            c.layer_idx = idx
            idx += 1

    import mlx.nn as nn
    
    logits = model(input_ids, cache=caches)
    
    # Calculate cross entropy
    # Flatten for loss calculation
    logits = logits.reshape(-1, logits.shape[-1])
    target_ids = target_ids.reshape(-1)
    
    loss = mx.mean(nn.losses.cross_entropy(logits, target_ids))
    ppl = math.exp(float(loss))
    
    return ppl

def main():
    print("=== ChronoQuant Full Calibration & Benchmark ===")
    model_name = "Qwen/Qwen3.5-0.8B"
    print(f"Loading {model_name}...")
    model, tokenizer = mlx_lm.load(model_name)
    
    # 1. Baseline PPL
    test_text = "Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at and below the scale of atoms."
    baseline_ppl = calculate_perplexity(model, tokenizer, test_text)
    print(f"Baseline FP16 PPL: {baseline_ppl:.4f}")
    
    # 2. Calibration
    calib_text = "The principles of quantum mechanics dictate that observing a particle alters its state. This phenomenon, known as the observer effect, highlights a fundamental limitation in precision measurement. In 1927, Werner Heisenberg formulated his famous uncertainty principle." * 5
    
    print("\nStarting Calibration (Gradient Rotation Learning & Quantization)...")
    
    from mlx_lm.models.cache import KVCache
    caches = model.make_cache()
    _ = model(mx.array(tokenizer.encode(calib_text))[None], cache=caches)
    
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    # Train layers 0 to 3
    for layer_idx in kv_indices[:4]:
        keys = caches[layer_idx].keys
        values = caches[layer_idx].values if hasattr(caches[layer_idx], 'values') else keys
        
        seq_len = min(256, keys.shape[2])
        Q = keys[:, :, :seq_len, :]
        K = keys[:, :, :seq_len, :]
        V = values[:, :, :seq_len, :] if values is not None else K
        
        print(f"Training Layer {layer_idx}...")
        codec = e2e_learnable.EndToEndKVCodec(
            keys.shape[-1], 
            {4: list(range(64)), 3: list(range(64, 128)), 2: list(range(128, 256))}
        )
        codec.train(Q, K, V, n_steps=50)
        layer_codecs[layer_idx] = codec
    
    # 3. Apply Patch and Test
    print("\nApplying Attention-Aligned Patch...")
    apply_calibration_patch()
    
    patched_ppl = calculate_perplexity(model, tokenizer, test_text)
    print(f"Post-Calibration PPL (first 4 layers compressed): {patched_ppl:.4f}")
    print(f"PPL Gap: {patched_ppl - baseline_ppl:.4f}")
    
    # Calculate approx bits per parameter
    print("\nCompression Achieved (in first 4 layers):")
    dim = keys.shape[-1]
    avg_bits = (64*4 + 64*3 + 128*2) / dim
    print(f"Average Bits/Param: {avg_bits:.2f}")

if __name__ == "__main__":
    main()
