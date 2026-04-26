import mlx.core as mx
import mlx_lm
import e2e_learnable

def main():
    print("Loading model for rotation training test...")
    model, tokenizer = mlx_lm.load("Qwen/Qwen3.5-0.8B")
    
    calibration_text = "The principles of quantum mechanics dictate that observing a particle alters its state, a fundamental limitation in precision measurement."
    
    print("Training end-to-end codec...")
    result = e2e_learnable.train_and_export(model, tokenizer, calibration_text, n_steps=50)
    print("Training finished.")

if __name__ == "__main__":
    main()
