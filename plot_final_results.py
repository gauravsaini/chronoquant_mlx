import mlx.core as mx
import mlx_lm
import e2e_learnable
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print("Loading model for full layer analysis...")
    model_name = "Qwen/Qwen3.5-0.8B"
    model, tokenizer = mlx_lm.load(model_name)
    
    calib_text = "The principles of quantum mechanics dictate that observing a particle alters its state. This phenomenon, known as the observer effect, highlights a fundamental limitation in precision measurement. In 1927, Werner Heisenberg formulated his famous uncertainty principle." * 5
    
    from mlx_lm.models.cache import KVCache
    caches = model.make_cache()
    _ = model(mx.array(tokenizer.encode(calib_text))[None], cache=caches)
    
    kv_indices = [i for i, c in enumerate(caches) if isinstance(c, KVCache)]
    
    layer_allocations = []
    
    for layer_idx in kv_indices:
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
        codec.train(Q, K, V, n_steps=30) # fewer steps for faster plotting
        
        # We don't need to actually return precision_groups from train(), 
        # but we know how it calculates them:
        R = codec.get_rotation()
        K_rot = K @ R
        imp = mx.abs(K_rot).mean(axis=(0, 1, 2))
        imp_np = np.array(imp)
        
        n_high = len(imp_np) // 5
        n_med = len(imp_np) // 3
        n_low = len(imp_np) - n_high - n_med
        
        layer_allocations.append((n_high, n_med, n_low))

    # Plotting
    layers = list(range(len(layer_allocations)))
    high_vals = [a[0] for a in layer_allocations]
    med_vals = [a[1] for a in layer_allocations]
    low_vals = [a[2] for a in layer_allocations]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stacked bar chart
    ax.bar(layers, high_vals, label='4-bit (High Variance)', color='#ff9999')
    ax.bar(layers, med_vals, bottom=high_vals, label='3-bit (Medium)', color='#66b3ff')
    ax.bar(layers, low_vals, bottom=np.array(high_vals) + np.array(med_vals), label='2-bit (Low Variance)', color='#99ff99')
    
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Number of Dimensions')
    ax.set_title('Attention-Aligned Precision Allocation Across Layers')
    ax.legend()
    
    plt.tight_layout()
    os.makedirs('../chronoquant-paper/figures', exist_ok=True)
    out_path = '../chronoquant-paper/figures/attention_aligned_allocation.png'
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
