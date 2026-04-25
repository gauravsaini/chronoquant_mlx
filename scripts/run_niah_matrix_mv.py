"""Run a full NIAH matrix for the motion-vectors ChronoQuant branch (pure 4-bit, no heterogeneous)."""
import subprocess
import sys
import json
import time

PYTHON = "/Users/ektasaini/mlx-env/bin/python3"
WORKER = "/Users/Shared/AI/models/niah_worker.py"
MODEL = "/Users/Shared/AI/models/Qwen3.5-4B"
HEAD_DIM = "128"

CONTEXTS = [512, 1024, 2048, 4096]
DEPTHS = [10, 50, 90]

results = []

total = len(CONTEXTS) * len(DEPTHS)
i = 0
for ctx in CONTEXTS:
    for depth in DEPTHS:
        i += 1
        label = f"ctx={ctx} depth={depth}%"
        print(f"[{i}/{total}] Motion Vectors 4-bit @ {label}", flush=True)
        
        cmd = [
            PYTHON, WORKER,
            "--type", "chronoquant",
            "--model", MODEL,
            "--head-dim", HEAD_DIM,
            "--context-tokens", str(ctx),
            "--depth", str(depth),
            "--max-gen", "200",
        ]
        
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            for line in reversed(proc.stdout.strip().splitlines()):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                result = {"error": "no JSON", "label": label}
            
            result["label"] = label
            success = result.get("success", "ERROR")
            icon = "✅" if success else "❌"
            print(f"  -> {icon} {success}", flush=True)
            
        except subprocess.TimeoutExpired:
            result = {"label": label, "error": "timeout"}
            print(f"  -> ⏱️ TIMEOUT", flush=True)
        except Exception as e:
            result = {"label": label, "error": str(e)}
            print(f"  -> 💥 {e}", flush=True)
        
        results.append(result)

# Print summary table
print("\n" + "="*80)
print("NIAH RESULTS: Motion Vectors ChronoQuant (Pure 4-bit)")
print(f"Model: {MODEL}")
print("="*80)
print(f"{'Context':>8} | {'Depth':>6} | {'Result':>6} | Generated (truncated)")
print("-"*80)
for r in results:
    ctx = r.get("context_tokens", "?")
    depth = r.get("depth_pct", "?")
    if "error" in r:
        print(f"{str(ctx):>8} | {str(depth):>6} | {'ERROR':>6} | {r['error']}")
    else:
        icon = "✅" if r["success"] else "❌"
        gen = r.get("generated", "")[:60].replace("\n", " ")
        print(f"{ctx:>8} | {depth:>6}% | {icon:>6} | {gen}")

n_pass = sum(1 for r in results if r.get("success") is True)
n_fail = sum(1 for r in results if r.get("success") is False)
n_error = sum(1 for r in results if "error" in r)
print("-"*80)
print(f"Total: {n_pass} pass, {n_fail} fail, {n_error} error out of {len(results)} tests")

# Save JSON
with open("/Users/Shared/AI/models/niah_motionvectors_results.json", "w") as f:
    json.dump({"model": MODEL, "results": results, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
print(f"\nSaved to /Users/Shared/AI/models/niah_motionvectors_results.json")
