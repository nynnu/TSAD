"""Do train-scaled features separate GT-overlap cores from non-GT cores? (28 machines, no VLM)

spike  = largest in-core step of a shown channel / train 99.9% step (what Stage1 + the core question focus on)
regime = share of the 224-tick tile where a shown channel sits outside its train 0.1-99.9% range
Evaluation only. AUROC is reported per machine (pooled AUROC would mix machine base rates).
"""
import json
import numpy as np
from scipy.stats import rankdata
from paired_bank_vlm_pilot import ROOT, save

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/miss_funnel_20261110"
cores = json.loads((out/"summary.json").read_text(encoding="utf8"))["cores"]


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return None
    r = rankdata(score)
    return float((r[label].sum()-label.sum()*(label.sum()+1)/2)/(label.sum()*(~label).sum()))


rows = []
for machine in sorted({c["machine"] for c in cores}):
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    step_scale = np.quantile(np.abs(np.diff(train, axis=0)), 0.999, axis=0)+1e-3
    lo, hi = np.quantile(train, [0.001, 0.999], axis=0)
    mine = [c for c in cores if c["machine"] == machine]
    spike, regime = [], []
    for c in mine:
        a, b = c["core"]; ch = c["shown"]
        left = int(np.clip((a+b-224)//2, 0, len(test)-224))
        tile = test[left:left+224][:, ch]
        spike.append((np.abs(np.diff(test[max(a-1, 0):b][:, ch], axis=0)).max(axis=0)/step_scale[ch]).max())
        regime.append(((tile > hi[ch]+1e-3) | (tile < lo[ch]-1e-3)).mean(axis=0).max())
    y = [c["gt_overlap"] for c in mine]
    rows.append(dict(machine=machine, cores=len(mine), gt_cores=int(sum(y)), auroc_spike=auroc(spike, y), auroc_regime=auroc(regime, y)))
    print(rows[-1], flush=True)
valid = [r for r in rows if r["auroc_spike"] is not None]
summary = dict(machines_evaluated=len(valid),
               mean_auroc_spike=float(np.mean([r["auroc_spike"] for r in valid])),
               mean_auroc_regime=float(np.mean([r["auroc_regime"] for r in valid])),
               regime_better_on=sum(r["auroc_regime"] > r["auroc_spike"] for r in valid))
save(out/"regime_vs_spike_auroc.json", dict(summary=summary, machines=rows,
     limitation="Overlap-based core labels; train quantile range is not a calibrated normality bound; exploratory, thresholds not tuned."))
print(json.dumps(summary, indent=2))
