"""
실행 진입점: experiment_stage2_perchannel.py
.env 파일에서 OPENAI_API_KEY를 읽어 실행합니다.

사용법:
  python run_stage2_perchannel.py
  (VLM4TS 루트 디렉토리에서 실행)
"""
import os, sys
from pathlib import Path

# .env 로드 (실험 파일과 동일 로직)
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in .env or environment.")
    print("Create .env file with: OPENAI_API_KEY=sk-...")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent / "experiments"))
from experiment_stage2_perchannel import run_entity, SMD_ENTITIES, RESULTS_DIR
import numpy as np
import pandas as pd

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
all_results, all_logs = [], []

for entity in SMD_ENTITIES:
    try:
        r = run_entity(entity)
    except Exception as exc:
        print(f"\n[ERROR] {entity}: {exc}")
        import traceback; traceback.print_exc()
        r = None

    if r is not None:
        all_logs.extend(r.pop("candidate_log"))
        all_results.append(r)

if all_results:
    print(f"\n{'='*75}")
    print("FINAL: Per-Channel Consensus Before/After Stage2")
    print(f"{'='*75}")
    print(f"{'Entity':<15} {'Oracle':>8} {'Loose':>8} {'Stage2':>8} "
          f"{'dOracle':>8} {'dLoose':>7}  n_conf/n_cand")
    print("-" * 75)
    for r in all_results:
        print(f"{r['entity']:<15} {r['oracle_f1']:>8.4f} {r['loose_f1']:>8.4f} "
              f"{r['stage2_f1']:>8.4f} {r['change_vs_oracle']:>+8.4f} "
              f"{r['change_vs_loose']:>+7.4f}  "
              f"{r['stage2_n']}/{r['loose_n']}")
    print("-" * 75)
    o_avg  = np.mean([r["oracle_f1"]  for r in all_results])
    l_avg  = np.mean([r["loose_f1"]   for r in all_results])
    s2_avg = np.mean([r["stage2_f1"]  for r in all_results])
    print(f"{'AVG':<15} {o_avg:>8.4f} {l_avg:>8.4f} {s2_avg:>8.4f} "
          f"{s2_avg - o_avg:>+8.4f} {s2_avg - l_avg:>+7.4f}")

    pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
    pd.DataFrame(all_logs).to_csv(RESULTS_DIR / "candidate_verdicts.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'summary.csv'}")
