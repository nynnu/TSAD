"""Smoke test: single signal across all three modes. Run with OPENAI_API_KEY set."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("Set OPENAI_API_KEY in environment.")

from experiment_stage2_univar_v5_1 import run_signal

for mode in ["A", "B", "C"]:
    print(f"\n{'='*60}")
    print(f"MODE {mode}")
    r = run_signal("NAB", "ec2_cpu_utilization_fe7f93", mode)
    print(f"  S1={r['f1_s1']:.4f} OUT={r['f1_out']:.4f} Oracle={r['f1_oracle']:.4f}")
    print(f"  tp_retention={r.get('tp_retention',float('nan')):.0%}  "
          f"fp_rejection={r.get('fp_rejection',float('nan')):.0%}  "
          f"false_discard={r.get('false_discard',float('nan')):.0%}")
