"""Conformal alarm budget on the 28 SMD machines with the model-free statistic (no VLM; GT only evaluates).
fit = first 50% of train blocks, calibration = next 25%, audit = last 25% (see conformal_budget.run_series)."""
import sys
import numpy as np
from paired_bank_vlm_pilot import ROOT, save
from conformal_budget import block_stats, run_series, summarize, show

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/conformal_budget_20261111"
out.mkdir(parents=True, exist_ok=True)
ALPHAS = (0.003, 0.01, 0.03, 0.1)
AGG = "fisher" if "--fisher" in sys.argv else "min"
results, ys = [], []
for path in sorted((data/"test").glob("machine-*.txt")):
    machine = path.stem
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
    test = np.loadtxt(path, delimiter=",")
    ys.append(np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel())
    results.append(run_series(block_stats(train), block_stats(test), ALPHAS, agg=AGG))
    print(machine, "done", flush=True)
summary = summarize(results, ys, ALPHAS)
save(out/("smd_statistic_fisher.json" if AGG == "fisher" else "smd_statistic.json"), dict(summary={str(k): v for k, v in summary.items()},
     limitation="Pooled over 28 machines; far_on_normal = alarm share among test blocks with no GT tick. Statistic = largest one-tick change "
                "per channel in a 14-tick block, aggregated by the smallest channel tail probability."))
print("DINO Stage1 for reference: alarm share ~4%, segment precision 0.396, event recall 0.875")
show(summary, "model-free statistic, 28 SMD machines")
