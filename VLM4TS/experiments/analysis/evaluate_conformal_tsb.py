"""Universality check: the same conformal budget on univariate TSB-AD-U series from other domains (no SMD, no retuning).
Series with a train prefix of at least 5,600 ticks (so 25% calibration still gives >= 100 blocks). GT only evaluates."""
import re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from paired_bank_vlm_pilot import ROOT, save
from conformal_budget import block_stats, run_series, summarize, show

src = ROOT.parents[1]/"reference/TSB-AD-U"
out = ROOT/"analysis/results/conformal_budget_20261111"
ALPHAS = (0.003, 0.01, 0.03, 0.1)
MIN_TRAIN = 5600
groups = defaultdict(lambda: ([], []))
for f in sorted(src.glob("*.csv")):
    m = re.match(r"\d+_(.+?)_id_\d+_.*_tr_(\d+)_1st_\d+\.csv", f.name)
    if not m or m.group(1) == "SMD" or int(m.group(2)) < MIN_TRAIN:
        continue
    ntr = int(m.group(2))
    arr = np.loadtxt(f, delimiter=",", skiprows=1)
    x, y = arr[:, 0], arr[:, 1].astype(bool)
    if len(x)-ntr < 14*50 or not y[ntr:].any() or not np.isfinite(x).all():
        continue
    res, ys = groups[m.group(1)]
    print(len(res), f.name, flush=True)
    res.append(run_series(block_stats(x[:ntr]), block_stats(x[ntr:]), ALPHAS, agg="fisher")); ys.append(y[ntr:])
print({k: len(v[0]) for k, v in groups.items()}, flush=True)
per = {k: summarize(v[0], v[1], ALPHAS) for k, v in groups.items()}
allr = [r for v in groups.values() for r in v[0]]; ally = [y for v in groups.values() for y in v[1]]
overall = summarize(allr, ally, ALPHAS)
save(out/"tsb_univariate.json", dict(series={k: len(v[0]) for k, v in groups.items()}, overall={str(a): s for a, s in overall.items()},
     per_dataset={k: {str(a): s for a, s in v.items()} for k, v in per.items()},
     limitation="Univariate, statistic = largest one-tick change per 14-tick block, Fisher aggregation (one channel), split fit/calibration/audit 50/25/25."))
show(overall, "TSB-AD-U overall (%d series)" % len(allr))
print("false-alarm rate on label-normal test blocks / alpha, per dataset (static | sliding_all):")
for k, v in per.items():
    print("%-12s n=%-3d " % (k, len(groups[k][0])) + "  ".join("a=%.3f: %.1fx|%.1fx" % (a, v[a]["static"]["far_on_normal"]/a, v[a]["sliding_all"]["far_on_normal"]/a) for a in ALPHAS[1:]))
