"""Quick smoke test — one NAB signal only. Run with OPENAI_API_KEY set in env."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("Set OPENAI_API_KEY in environment before running.")

from experiment_stage2_univar_v5 import run_signal

r = run_signal("NAB", "ec2_cpu_utilization_fe7f93")
print("\n=== RESULT ===")
print(f"Stage1 F1  : {r['f1_s1']:.4f}")
print(f"v5 F1      : {r['f1_v5']:.4f}")
print(f"Oracle F1  : {r['f1_oracle']:.4f}")
print(f"Gap recov  : {r['gap_recovery']:.1%}" if isinstance(r['gap_recovery'], float) and r['gap_recovery']==r['gap_recovery'] else "Gap recov  : N/A")
print(f"L-acc      : {r['l_option_acc']:.1%}")
print(f"R-acc      : {r['r_option_acc']:.1%}")
print(f"Kept       : {r['n_kept']}/{r['n_s1']}")
print(f"\nPer-candidate:")
for pc in r["per_candidate"]:
    print(f"  C{pc['cid']} orig={pc['orig']} -> decision={pc['vlm_decision']} "
          f"L{pc['vlm_li']}/R{pc['vlm_ri']} "
          f"(oracle L{pc['oracle_li']}/R{pc['oracle_ri']}) "
          f"final={pc['final_iv']}")
