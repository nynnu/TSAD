"""Which signal property separates GT from non-GT Stage1 cores? Per-machine AUROC (no VLM, evaluation only)."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from paired_bank_vlm_pilot import ROOT, save

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/normal_spike_root_20261110"
rows = json.loads((out/"summary.json").read_text(encoding="utf8"))["cores"]
shown = {(c["machine"], tuple(c["core"])): c["shown"] for c in json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]}


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return None
    r = rankdata(score)
    return float((r[label].sum()-label.sum()*(label.sum()+1)/2)/(label.sum()*(~label).sum()))


names = ["step_vs_train", "rarity_in_train", "rarity_in_test", "channels_stepping_together", "burst_density_vs_train", "level_shift_after", "core_length"]
table = {n: [] for n in names}
for machine in sorted({r["machine"] for r in rows}):
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    dtr, dte = np.abs(np.diff(train, axis=0)), np.abs(np.diff(test, axis=0))
    scale = np.quantile(dtr, 0.999, axis=0)+1e-3
    big_tr, big_te = dtr > scale, dte > scale                       # "large step" events per channel
    # ponytail: 224-tick non-overlapping train windows as the burst reference; overlapping windows if this is kept
    tr_density = big_tr[:len(big_tr)//224*224].reshape(-1, 224, train.shape[1]).sum(axis=1)
    mine = [r for r in rows if r["machine"] == machine]
    feats = {n: [] for n in names}
    for r in mine:
        a, b = r["core"]; ch = r["lead"]; mid = (a+b)//2
        feats["step_vs_train"].append(r["step"]/scale[ch])
        feats["rarity_in_train"].append(-r["train_count"])
        feats["rarity_in_test"].append(-r["test_count_elsewhere"])
        feats["channels_stepping_together"].append(int(big_te[max(a-1, 0):b-1].any(axis=0).sum()))
        density = big_te[max(mid-112, 0):mid+112].sum(axis=0)
        feats["burst_density_vs_train"].append(float(((density[None] > tr_density).mean(axis=0))[shown[(machine, (a, b))]].max()))
        pre, post = test[max(a-60, 0):a], test[b:b+60]
        noise = 1.4826*np.median(np.abs(pre-np.median(pre, axis=0)), axis=0)+0.01
        feats["level_shift_after"].append(float((np.abs(np.median(post, axis=0)-np.median(pre, axis=0))/noise)[shown[(machine, (a, b))]].max()) if len(pre) and len(post) else 0.0)
        feats["core_length"].append(b-a)
    y = [r["gt"] for r in mine]
    for n in names:
        table[n].append(auroc(feats[n], y))
result = {n: dict(mean_auroc=float(np.mean(v)), machines_above_0_7=int(sum(x > .7 for x in v)), machines_below_0_5=int(sum(x < .5 for x in v))) for n, v in table.items()}
save(out/"core_feature_auroc.json", dict(features=result, per_machine=table,
     limitation="Exploratory single features on Stage1-selected cores; overlap labels; nothing tuned or combined."))
print(json.dumps(result, indent=1))

# one picture: a GT core and a non-GT core whose step size is routine in the same channel
fig, axes = plt.subplots(2, 1, figsize=(16, 6))
for ax, gt in zip(axes, (False, True)):
    r = next(x for x in rows if x["gt"] == gt and x["train_count"] >= 5 and x["test_count_elsewhere"] >= 20 and x["source"] != "deseason" and x["core"][0] > 2000)
    test = np.loadtxt(data/"test"/f"{r['machine']}.txt", delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{r['machine']}.txt", delimiter=",").astype(bool).ravel()
    lo, hi = r["core"][0]-1500, min(len(test), r["core"][1]+1500)
    ax.plot(range(lo, hi), test[lo:hi, r["lead"]], "k", lw=.6)
    ax.fill_between(range(lo, hi), 0, 1, where=y[lo:hi], color="red", alpha=.12, transform=ax.get_xaxis_transform())
    ax.axvspan(*r["core"], color="tab:orange", alpha=.6)
    ax.set_title(f"{'GT' if gt else 'non-GT'} core {r['machine']} {r['core']} ch{r['lead']} · same-size steps: train {r['train_count']}, elsewhere in test {r['test_count_elsewhere']} · orange=core, red=GT", fontsize=9)
fig.tight_layout(); fig.savefig(out/"routine_spike_gt_vs_nongt.png", dpi=70)
