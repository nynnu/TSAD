"""Which period detector finds real periodicity? Out-of-sample test on train (28 machines, no GT, no labels).
Period is estimated on the first half of a channel's train only; the seasonal profile built from that half must predict
the second half better than a constant (skill = 1 - SSE_season / SSE_constant). Null: 30 random periods in [5, 1000]."""
import sys
import numpy as np
from paired_bank_vlm_pilot import ROOT, estimate_reference_period, save
sys.path.insert(0, str(ROOT/"stage1_final"))
from stage1 import period as stage1_period

data = ROOT.parent/"mv_data/SMD"
rng = np.random.default_rng(0)


def skill(a, b, p):
    prof = np.array([a[k::p].mean() for k in range(p)])
    pred = prof[(len(a)+np.arange(len(b))) % p]
    return 1-((b-pred)**2).sum()/max(((b-a.mean())**2).sum(), 1e-12)


rows = []
for path in sorted((data/"train").glob("machine-*.txt")):
    tr = np.loadtxt(path, delimiter=",")
    h = len(tr)//2
    for c in range(tr.shape[1]):
        a, b = tr[:h, c], tr[h:, c]
        if a.std() < 1e-6:
            continue
        lo, hi = np.quantile(a, [.01, .99]); hi = lo+1. if hi-lo < 1e-8 else hi
        p1 = stage1_period(a)
        p2, _ = estimate_reference_period((a-lo)/(hi-lo))
        null = [skill(a, b, int(q)) for q in rng.integers(5, 1001, 30)]
        rows.append(dict(s1=skill(a, b, p1) if p1 else None, s2=skill(a, b, p2) if p2 else None, null95=float(np.quantile(null, .95)), null_med=float(np.median(null))))
    print(path.stem, flush=True)
n = len(rows)
res = {}
for name in ("s1", "s2"):
    got = [r for r in rows if r[name] is not None]
    res[name] = dict(channels_with_period=len(got), share=len(got)/n, skill_median=float(np.median([r[name] for r in got])),
                     beats_constant=float(np.mean([r[name] > 0 for r in got])), beats_random_period_95=float(np.mean([r[name] > r["null95"] for r in got])))
res["null"] = dict(skill_median=float(np.median([r["null_med"] for r in rows])), skill_p95_median=float(np.median([r["null95"] for r in rows])))
res["channels"] = n
save(ROOT/"analysis/results/period_validity_20261111/summary.json", res)
print(res)
