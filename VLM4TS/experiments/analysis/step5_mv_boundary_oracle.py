"""
Step5 (multivariate boundary separation) -- first, cheapest question: is
there meaningful oracle-boundary headroom in the MULTIVARIATE (SMD) setting,
the way there was in the univariate one (oracle_boundary_analysis.py, run
2026-08-05, ALL gap=+0.0643)?

Reuses v16's already-computed stage1() (loose thresholding + oracle F1 sweep
over 12 alpha values on cached DINOv2 overlay-group scores) -- no new
computation, no API calls, no DINOv2 needed. Just answers: does the gap
between "loose" (Stage1 heuristic threshold) and "oracle" (best possible
threshold sweep) F1 justify building a VLM-based constrained-boundary-
selection call for multivariate, the way v5.2c did for univariate?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2" / "active"))
import experiment_stage2_v16 as v16  # noqa: E402

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = []
    for entity in SMD_ENTITIES:
        train, test, labels = v16.load_smd(entity)
        T = len(labels)
        ch_scores, ov_scores = v16.load_scores(entity)
        inter, loose_ivs, gt_ivs, oracle_f1, oracle_ivs, all_ws, mu, sig = v16.stage1(ov_scores, T, labels)
        lf1, lp, lr = v16.f1(gt_ivs, loose_ivs)
        gap = oracle_f1 - lf1
        results.append((entity, lf1, oracle_f1, gap, len(gt_ivs), len(loose_ivs)))
        print(f"{entity}: GT={len(gt_ivs)} loose_cand={len(loose_ivs)} "
              f"loose_F1={lf1:.4f} oracle_F1={oracle_f1:.4f} gap={gap:+.4f}")

    import numpy as np
    avg_loose = np.mean([r[1] for r in results])
    avg_oracle = np.mean([r[2] for r in results])
    avg_gap = np.mean([r[3] for r in results])
    print(f"\n{'='*60}\nMULTIVARIATE (SMD) BOUNDARY ORACLE SUMMARY\n{'='*60}")
    print(f"Loose (Stage1 heuristic) avg F1 = {avg_loose:.4f}")
    print(f"Oracle (best threshold)  avg F1 = {avg_oracle:.4f}")
    print(f"Gap                             = {avg_gap:+.4f}")
    print(f"\n(for reference, univariate MSL/SMAP/NAB gap 2026-08-05 = +0.0643)")
