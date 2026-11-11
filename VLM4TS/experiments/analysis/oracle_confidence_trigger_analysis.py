"""
Self-reported confidence as a re-verification trigger (design doc §13's
unexecuted item, audited 2026-08-05): "Self-reported confidence를 재검증
트리거로 활용하는 아이디어를 Step5 설계에 추가 검토한다."

This idea originates from our OWN Oracle Reasoning Pilot's prompt design
(run_sanity_oracle.py's build_prompt asks the VLM to self-report confidence:
low/medium/high alongside its onset/affected-channel answers) -- not a
specific external paper's core method, so no separate paper+GitHub read is
required under SCHEMA.md's rule (that rule applies to importing an external
paper's mechanism; this is analyzing our own already-collected field).
[[tiny but trusted]] was checked as the nearest related paper (reward
components: anomaly accuracy/visual groundedness/axis awareness/clarity) but
does not include a self-reported-confidence-as-trigger mechanism -- confirmed
not directly applicable here.

Prerequisite question before building any actual "trigger" (a second,
more expensive re-verification pass for low-confidence cases): IS
self-reported confidence even predictive of correctness in our data? If
low-confidence cases aren't meaningfully worse than high-confidence ones,
a confidence-gated trigger has no signal to act on. This uses EXISTING data
(no new API calls) -- both completed Oracle pilot runs (original subplot,
2026-08-04, and the overlay rerun, §28, 2026-08-05), which already collected
self_reported_confidence alongside ground-truth-scored onset_abs_error and
affected_recall/precision for every case.
"""

import csv
from pathlib import Path

PATHS = [
    Path(r"C:\Users\김나영\Desktop\ny\TSAD-code\VLM4TS\sanity\results\sanity_oracle\runs\20260804_234430\raw\oracle_samples.csv"),
    Path(r"C:\Users\김나영\Desktop\ny\TSAD-code\VLM4TS\sanity\results\sanity_oracle_overlay\runs\20260805_184427\raw\oracle_samples.csv"),
]


def load_rows():
    rows = []
    for p in PATHS:
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["parse_status"] != "OK":
                    continue
                row["_source"] = p.parts[-4]  # "sanity_oracle" or "sanity_oracle_overlay"
                rows.append(row)
    return rows


def run():
    rows = load_rows()
    print(f"n={len(rows)} OK-parsed cases across {len(PATHS)} runs")

    buckets = {}
    for r in rows:
        conf = r["self_reported_confidence"]
        buckets.setdefault(conf, []).append(r)

    print(f"\n{'='*70}\nCONFIDENCE vs CORRECTNESS (existing Oracle pilot data, no new API calls)\n{'='*70}")
    print(f"{'confidence':<10} {'n':>4} {'mean_onset_err':>15} {'exact_onset_rate':>17} {'mean_recall':>12} {'mean_precision':>15}")
    for conf in sorted(buckets, key=lambda c: {"low": 0, "medium": 1, "high": 2}.get(c, 9)):
        sub = buckets[conf]
        n = len(sub)
        errs = [float(r["onset_abs_error"]) for r in sub if r["onset_abs_error"] not in ("", "None")]
        exact_rate = sum(1 for e in errs if e == 0) / len(errs) if errs else float("nan")
        mean_err = sum(errs) / len(errs) if errs else float("nan")
        recalls = [float(r["affected_recall"]) for r in sub if r["affected_recall"] not in ("", "None")]
        precisions = [float(r["affected_precision"]) for r in sub if r["affected_precision"] not in ("", "None")]
        mean_recall = sum(recalls) / len(recalls) if recalls else float("nan")
        mean_prec = sum(precisions) / len(precisions) if precisions else float("nan")
        print(f"{conf:<10} {n:>4} {mean_err:>15.3f} {exact_rate:>17.3f} {mean_recall:>12.3f} {mean_prec:>15.3f}")

    # "imperfect" case = onset_abs_error>0 OR recall<1.0 OR precision<1.0
    print(f"\n{'imperfect (any metric<perfect) rate by confidence bucket'}")
    for conf in sorted(buckets, key=lambda c: {"low": 0, "medium": 1, "high": 2}.get(c, 9)):
        sub = buckets[conf]
        imperfect = 0
        for r in sub:
            err = float(r["onset_abs_error"]) if r["onset_abs_error"] not in ("", "None") else 0
            rec = float(r["affected_recall"]) if r["affected_recall"] not in ("", "None") else 1.0
            prec = float(r["affected_precision"]) if r["affected_precision"] not in ("", "None") else 1.0
            if err > 0 or rec < 1.0 or prec < 1.0:
                imperfect += 1
        rate = imperfect / len(sub) if sub else float("nan")
        print(f"  {conf:<10} n={len(sub):>4}  imperfect_rate={rate:.3f}")


if __name__ == "__main__":
    run()
