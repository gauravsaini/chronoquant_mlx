"""
Needle-In-A-Haystack worker for ChronoQuant evaluation.

Follows the same architecture as benchmark_honest_worker.py:
  - Runs inside a subprocess
  - Prints a single JSON line to stdout with results
  - All diagnostics go to stderr

Usage:
  python scripts/niah_worker.py \
      --type chronoquant --model /path/to/model \
      --stride-k 32 --stride-v 8 \
      --context-tokens 4096 --depth 50
"""
import os
import sys
import json
import time
import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "turboquant_mlx"))
sys.path.insert(0, "/Users/Shared/AI/models")

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache, KVCache


# ── Haystack filler ──
FILLER_PARA = (
    "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
    "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
    "modern AI were planted by philosophers who attempted to describe the process of human thinking "
    "as the mechanical manipulation of symbols. This work culminated in the invention of the "
    "programmable digital computer in the 1940s, a machine based on the abstract essence of "
    "mathematical reasoning. This device and the ideas behind it inspired a handful of scientists "
    "to begin seriously discussing the possibility of building an electronic brain. The field of AI "
    "research was founded at a workshop held on the campus of Dartmouth College during the summer "
    "of 1956. Those who attended would become the leaders of AI research for decades. Many of them "
    "predicted that a machine as intelligent as a human being would exist in no more than a "
    "generation and they were given millions of dollars to make this vision come true."
)

NEEDLE = "The special secret code is: ALBATROSS-7492-KILO."
QUESTION = "\n\nBased on the text above, what is the special secret code? The special secret code is:"
EXPECTED = "ALBATROSS-7492-KILO"


def build_niah_prompt(tokenizer, context_tokens: int, depth_pct: int) -> mx.array:
    """Build a haystack prompt with a needle at a given depth percentage.
    
    Returns input_ids as (1, seq_len) mx.array.
    """
    needle_ids = tokenizer.encode(NEEDLE, add_special_tokens=False)
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER_PARA, add_special_tokens=False)

    # How many filler tokens do we need?
    filler_budget = context_tokens - len(needle_ids) - len(question_ids) - 2  # small buffer
    if filler_budget < 0:
        raise ValueError(f"context_tokens={context_tokens} too small for NIAH test")

    # Build filler by repeating the paragraph
    haystack = []
    while len(haystack) < filler_budget:
        haystack.extend(filler_ids)
    haystack = haystack[:filler_budget]

    # Insert needle at depth
    insert_pos = int(len(haystack) * (depth_pct / 100.0))
    prompt = haystack[:insert_pos] + needle_ids + haystack[insert_pos:] + question_ids

    return mx.array(prompt[:context_tokens])[None]


def run_niah(model, tokenizer, cache_fn, context_tokens: int, depth_pct: int, max_gen: int = 30):
    """Run a single NIAH test and return result dict."""
    prompt_ids = build_niah_prompt(tokenizer, context_tokens, depth_pct)
    actual_len = prompt_ids.shape[1]

    cache = cache_fn()

    # Prefill
    logits = model(prompt_ids, cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)

    # Generate up to max_gen tokens
    generated_tokens = []
    for _ in range(max_gen):
        generated_tokens.append(token.item())
        if token.item() == tokenizer.eos_token_id:
            break
        logits = model(token[None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)

    generated_text = tokenizer.decode(generated_tokens).strip()

    # Check success: does the generated text contain the expected answer?
    success = EXPECTED.lower() in generated_text.lower()

    return {
        "context_tokens": actual_len,
        "depth_pct": depth_pct,
        "generated": generated_text[:200],  # truncate for JSON
        "expected": EXPECTED,
        "success": success,
    }


# ── Cache factories (reused from benchmark_honest_worker.py) ──

def setup_baseline(model):
    def make_cache():
        return make_prompt_cache(model)
    return make_cache


def setup_chronoquant(model, stride_k, stride_v, use_fused=True, heterogeneous=False):
    from chronoquant_mlx.patch import apply_patch as cq_apply_patch
    from chronoquant_mlx.patch import create_chronoquant_caches
    cq_apply_patch()
    def make_cache():
        return create_chronoquant_caches(
            model, stride_k=stride_k, stride_v=stride_v, use_fused=use_fused,
            heterogeneous_layers=heterogeneous,
        )
    return make_cache


def setup_turboquant(model, head_dim, bits):
    import turboquant.patch as tq_patch
    from turboquant.cache_v3 import TurboQuantKVCacheV3
    tq_patch.apply()
    def make_cache():
        caches = make_prompt_cache(model)
        for i, c in enumerate(caches):
            if isinstance(c, KVCache):
                caches[i] = TurboQuantKVCacheV3(head_dim=head_dim, bits=bits, seed=42)
        return caches
    return make_cache


def infer_head_dim(model):
    try:
        return int(model.layers[0].self_attn.head_dim)
    except Exception:
        pass
    try:
        cfg = model.config
        return int(cfg.hidden_size // cfg.num_attention_heads)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="NIAH worker for ChronoQuant evaluation")
    parser.add_argument("--type", choices=["baseline", "chronoquant", "turboquant"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stride-k", type=int, default=32)
    parser.add_argument("--stride-v", type=int, default=8)
    parser.add_argument("--disable-fused", action="store_true")
    parser.add_argument("--heterogeneous", action="store_true", help="Use heterogeneous layer-wise compression")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=50, help="Needle depth as percentage (0-100)")
    parser.add_argument("--max-gen", type=int, default=30, help="Max tokens to generate for answer")
    args = parser.parse_args()

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = load(args.model)

    head_dim = args.head_dim if args.head_dim > 0 else infer_head_dim(model)
    if head_dim is None:
        print("ERROR: cannot infer head_dim; pass --head-dim", file=sys.stderr)
        sys.exit(1)
    print(f"head_dim={head_dim}", file=sys.stderr)

    if args.type == "baseline":
        make_cache = setup_baseline(model)
    elif args.type == "chronoquant":
        make_cache = setup_chronoquant(model, args.stride_k, args.stride_v, not args.disable_fused, args.heterogeneous)
    elif args.type == "turboquant":
        make_cache = setup_turboquant(model, head_dim, args.bits)

    print(f"Running NIAH: type={args.type}, ctx={args.context_tokens}, depth={args.depth}%", file=sys.stderr)
    result = run_niah(model, tokenizer, make_cache, args.context_tokens, args.depth, args.max_gen)

    result["type"] = args.type
    result["model"] = args.model
    if args.type == "chronoquant":
        result["stride_k"] = args.stride_k
        result["stride_v"] = args.stride_v
        result["use_fused"] = not args.disable_fused
    elif args.type == "turboquant":
        result["bits"] = args.bits

    print(f"  Success: {result['success']}", file=sys.stderr)
    print(f"  Generated: {result['generated'][:80]}...", file=sys.stderr)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
