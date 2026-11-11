"""Why does the pipeline raise normal spikes? Decompose the 630 non-GT Stage1 cores (28 machines, no VLM).

For each core, the lead channel is the shown channel whose in-core step is largest relative to train.
train_count = how many train steps of that channel are at least as large (0 = never seen in train).
GT is used only to label cores for this audit.
"""
import json
from collections import Counter
import numpy as np
from paired_bank_vlm_pilot import ROOT, save

data = ROOT.parent/"mv_data/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out = ROOT/"analysis/results/normal_spike_root_20261110"
out.mkdir(parents=True, exist_ok=True)
cores = json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]


def bucket(n):
    return "never_in_train" if n == 0 else "rare_in_train(1-4)" if n < 5 else "routine_in_train(5+)"


machines, rows = [], []
for machine in sorted({c["machine"] for c in cores}):
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    doc = json.loads((snap/f"{machine}.json").read_text(encoding="utf8"))
    dtr, dte = np.abs(np.diff(train, axis=0)), np.abs(np.diff(test, axis=0))
    scale = np.quantile(dtr, 0.999, axis=0)+1e-3
    gt_idx = np.flatnonzero(y)
    mine = [c for c in cores if c["machine"] == machine]
    mask = np.zeros(len(y), bool)
    for c in mine:
        a, b = c["core"]; mask[a:b] = True
        step = dte[max(a-1, 0):b-1].max(axis=0)
        lead = max(c["shown"], key=lambda ch: step[ch]/scale[ch])
        parent = next(e for e in doc["intervals"] if e["start"] <= a and b <= e["end"])
        source = next(r["source"] for r in parent["channels"] if r["channel"] == lead)
        rows.append(dict(machine=machine, core=c["core"], gt=c["gt_overlap"], lead=int(lead), step=float(step[lead]),
                         train_count=int((dtr[:, lead] >= step[lead]-1e-9).sum()),
                         test_count_elsewhere=int((dte[:, lead] >= step[lead]-1e-9).sum())-int((dte[max(a-1, 0):b-1, lead] >= step[lead]-1e-9).sum()),
                         source=source, length=b-a,
                         gap_to_gt=int(np.abs(gt_idx-a).min()) if len(gt_idx) and not c["gt_overlap"] else 0))
    machines.append(dict(machine=machine, gt_fraction=float(y.mean()), alarm_fraction=float(mask.mean()),
                         cores=len(mine), non_gt_cores=sum(not c["gt_overlap"] for c in mine), fp_ticks=int((mask & ~y).sum())))

non, pos = [r for r in rows if not r["gt"]], [r for r in rows if r["gt"]]
alarm = np.array([m["alarm_fraction"] for m in machines]); gtf = np.array([m["gt_fraction"] for m in machines])
summary = dict(
    alarm_budget=dict(alarm_fraction_min=float(alarm.min()), alarm_fraction_median=float(np.median(alarm)), alarm_fraction_max=float(alarm.max()),
                      gt_fraction_min=float(gtf.min()), gt_fraction_median=float(np.median(gtf)), gt_fraction_max=float(gtf.max()),
                      corr_alarm_vs_gt_fraction=float(np.corrcoef(alarm, gtf)[0, 1]),
                      corr_non_gt_cores_vs_gt_fraction=float(np.corrcoef([m["non_gt_cores"] for m in machines], gtf)[0, 1])),
    non_gt_by_train=dict(Counter(bucket(r["train_count"]) for r in non)),
    gt_by_train=dict(Counter(bucket(r["train_count"]) for r in pos)),
    non_gt_by_source=dict(Counter(r["source"] for r in non)), gt_by_source=dict(Counter(r["source"] for r in pos)),
    non_gt_source_x_train={s: dict(Counter(bucket(r["train_count"]) for r in non if r["source"] == s)) for s in sorted({r["source"] for r in non})},
    gt_source_x_train={s: dict(Counter(bucket(r["train_count"]) for r in pos if r["source"] == s)) for s in sorted({r["source"] for r in pos})},
    non_gt_within_100_ticks_of_gt=sum(r["gap_to_gt"] <= 100 for r in non),
    non_gt_never_in_train_but_recurs_in_test=sum(r["train_count"] == 0 and r["test_count_elsewhere"] >= 5 for r in non),
    non_gt_never_in_train_and_unique_in_test=sum(r["train_count"] == 0 and r["test_count_elsewhere"] == 0 for r in non))
save(out/"summary.json", dict(summary=summary, machines=machines, cores=rows,
     limitation="Non-GT is not proof of normal. One-tick step on one lead channel; slow changes are not captured by this feature."))
print(json.dumps(summary, indent=1))
for m in machines:
    print(m["machine"], "gt %.3f alarm %.4f cores %d nonGT %d" % (m["gt_fraction"], m["alarm_fraction"], m["cores"], m["non_gt_cores"]))
