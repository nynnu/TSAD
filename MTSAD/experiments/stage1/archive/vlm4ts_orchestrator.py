"""
Orchestrator: runs vlm4ts_worker.py in a subprocess per signal.
Segfaults don't crash the whole run — the signal is skipped and noted.
Saves partial results incrementally.
"""
import json, os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    "MSL":  ["C-1","D-14","D-15","D-16","F-7","F-8","P-11","P-14","T-12","T-13","T-8"],
    "NAB":  ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38",
             "ec2_cpu_utilization_5f5533","ec2_cpu_utilization_77c1ca",
             "ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
             "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de",
             "ec2_disk_write_bytes_c0d644","ec2_network_in_257a54",
             "ec2_network_in_5abac7","elb_request_count_8c0756",
             "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
             "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"],
    "SMAP": ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"],
}

BASE        = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
RESULTS_DIR = BASE / "experiments/results_vlm4ts_official"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL     = RESULTS_DIR / "partial_results.jsonl"

WORKER = str(BASE / "experiments/vlm4ts_worker.py")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Load already-done signals
done = {}
if PARTIAL.exists():
    for line in PARTIAL.read_text().splitlines():
        try:
            r = json.loads(line)
            done[f"{r['ds']}__{r['sig']}"] = r
        except Exception:
            pass
print(f"Already done: {len(done)} signals", flush=True)

all_results = list(done.values())

for ds, signals in DATASETS.items():
    for sig in signals:
        key = f"{ds}__{sig}"
        if key in done:
            print(f"  [SKIP] {ds}/{sig} (cached)", flush=True)
            continue

        print(f"\n  [{ds}] {sig}", flush=True)
        env = {**os.environ, "OPENAI_API_KEY": API_KEY}
        try:
            timeout_s = 1200 if ds == "SMAP" else 300
            proc = subprocess.run(
                [sys.executable, WORKER, ds, sig],
                capture_output=True, text=True, timeout=timeout_s, env=env,
            )
            stdout = proc.stdout
            stderr = proc.stderr

            # Print stderr (progress lines) for visibility
            for line in stderr.splitlines():
                if line.strip():
                    print(f"    | {line}", flush=True)

            # Extract JSON result
            result_line = None
            for line in stdout.splitlines():
                if line.startswith("RESULT:"):
                    result_line = line[len("RESULT:"):]
            if result_line:
                r = json.loads(result_line)
                print(f"    -> S1_fix={r['stage1_f1_fixed']:.4f}  "
                      f"S2_fix={r['stage2_f1_fixed']:.4f} "
                      f"(P={r['stage2_p_fixed']:.2f} R={r['stage2_r_fixed']:.2f})"
                      f"  s1={r['n_stage1']}→s2={r['n_stage2']}", flush=True)
                print(f"    S1:{r['s1_ivs']}  S2:{r['s2_ivs']}", flush=True)
                all_results.append(r)
                with open(PARTIAL, "a") as f:
                    f.write(json.dumps(r) + "\n")
            else:
                print(f"    [NO RESULT] exit={proc.returncode}", flush=True)
                if proc.returncode == -11 or proc.returncode == 139:
                    print(f"    [SEGFAULT — skipping]", flush=True)

        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT - skipping]", flush=True)
        except Exception as exc:
            print(f"    [ERROR] {exc}", flush=True)

if not all_results:
    print("No results.", flush=True)
    raise SystemExit

W = 95
print(f"\n{'='*W}", flush=True)
print("VLM4TS Official — ViT-B-16 Stage1 + GPT-4o full-series Stage2", flush=True)
print(f"{'='*W}", flush=True)
print(f"  {'Signal':<45} {'S1-fix':>7} {'S2-fix':>7} {'S2-P':>6} {'S2-R':>6}", flush=True)
print(f"  {'-'*72}", flush=True)

for ds in ["NAB", "SMAP", "MSL"]:
    rows = [r for r in all_results if r["ds"] == ds]
    if not rows:
        continue
    print(f"\n  {ds} ({len(rows)} signals):", flush=True)
    for r in rows:
        print(f"  {r['sig']:<45} {r['stage1_f1_fixed']:>7.4f} {r['stage2_f1_fixed']:>7.4f} "
              f"{r['stage2_p_fixed']:>6.2f} {r['stage2_r_fixed']:>6.2f}", flush=True)
    print(f"  {'AVG':<45} "
          f"{np.mean([r['stage1_f1_fixed'] for r in rows]):>7.4f} "
          f"{np.mean([r['stage2_f1_fixed'] for r in rows]):>7.4f}", flush=True)

all_s1 = np.mean([r["stage1_f1_fixed"] for r in all_results])
all_s2 = np.mean([r["stage2_f1_fixed"] for r in all_results])
print(f"\n  {'ALL ({} signals)'.format(len(all_results)):<45} {all_s1:>7.4f} {all_s2:>7.4f}", flush=True)
print(f"{'='*W}", flush=True)

print(f"\n  ┌────────────────────────────────────────────────────────────┐")
print(f"  │  Comparison (fixed F1 formula, our {len(all_results)} signals)           │")
print(f"  ├────────────────────────────────────────────────────────────┤")
print(f"  │  VLM4TS official Stage1 (ViT-B-16)      : {all_s1:.4f}           │")
print(f"  │  VLM4TS official Stage2 (full-series)   : {all_s2:.4f}           │")
print(f"  │                                                            │")
print(f"  │  Our Stage1 (DINOv2 dino_k5)            : 0.6174           │")
print(f"  │  Our Stage2 v4 (per-cand. z_max-aware)  : 0.6526           │")
print(f"  └────────────────────────────────────────────────────────────┘")

pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
print(f"\nSaved -> {RESULTS_DIR}", flush=True)
