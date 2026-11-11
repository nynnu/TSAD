"""Stage1 candidate rule by machine-level rarity instead of a test percentile.

Hypothesis: an anomaly is a moment when MORE channels make a large step together than this
machine ever does in normal train. Per tick, S = number of channels whose one-tick change
(within +-1 tick) exceeds that channel's train 99.9% step. A tick is a candidate when S is
above the train 99.9% quantile of S. Every threshold comes from train; the number of alarms
is free to follow the data. Candidates are padded to the 14-tick unit of the DINO cores.

Protocol: design on machine-1-x only (default). `--heldout` evaluates machine-2-x/3-x once.
GT is used only for evaluation.
"""
import json, sys
import numpy as np
from paired_bank_vlm_pilot import ROOT, save

data = ROOT.parent/"mv_data/SMD"
out = ROOT/"analysis/results/stage1_costep_20261110"
out.mkdir(parents=True, exist_ok=True)
dino = json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]
HELDOUT = "--heldout" in sys.argv
PAD = 7


def costep(series, scale):
    big = np.abs(np.diff(series, axis=0, prepend=series[:1])) > scale
    near = big.copy(); near[1:] |= big[:-1]; near[:-1] |= big[1:]   # +-1 tick tolerance
    return near.sum(axis=1)


def segments(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def measure(y, mask):
    segs, events = segments(mask), segments(y)
    return dict(tp=int((mask & y).sum()), fp=int((mask & ~y).sum()), fn=int((~mask & y).sum()),
                hit=int(sum(mask[a:b].any() for a, b in events)), events=len(events),
                segments=len(segs), segments_on_gt=int(sum(y[a:b].any() for a, b in segs)))


def total(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0] if k != "machine"}
    p, r = s["tp"]/max(s["tp"]+s["fp"], 1), s["tp"]/(s["tp"]+s["fn"])
    ep, er = s["segments_on_gt"]/max(s["segments"], 1), s["hit"]/s["events"]
    return dict(point_precision=p, point_recall=r, point_f1=2*p*r/max(p+r, 1e-12), fp_ticks=s["fp"],
                segment_precision=ep, event_recall=er, event_f1=2*ep*er/max(ep+er, 1e-12),
                segments=s["segments"], segments_on_gt=s["segments_on_gt"], events_hit=s["hit"], events=s["events"])


names = ("dino", "costep", "dino_with_costep", "dino_or_costep")
rows = {n: [] for n in names}
info = []
for path in sorted((data/"test").glob("machine-*.txt")):
    machine = path.stem
    if machine.startswith("machine-1-") == HELDOUT:
        continue
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
    test = np.loadtxt(path, delimiter=",")
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    scale = np.quantile(np.abs(np.diff(train, axis=0)), 0.999, axis=0)+1e-3
    limit = np.quantile(costep(train, scale), 0.999)
    hot = costep(test, scale) > limit
    new = np.zeros(len(y), bool)
    for t in np.flatnonzero(hot):
        new[max(t-PAD, 0):t+PAD] = True
    old = np.zeros(len(y), bool); gated = old.copy()
    for c in dino:
        if c["machine"] == machine:
            a, b = c["core"]; old[a:b] = True
            if hot[a:b].any():
                gated[a:b] = True
    for n, mask in zip(names, (old, new, gated, old | new)):
        rows[n].append(dict(machine=machine, **measure(y, mask)))
    info.append(dict(machine=machine, train_costep_limit=float(limit), hot_ticks=int(hot.sum()), gt_fraction=float(y.mean()), alarm_fraction=float(new.mean())))
    print(machine, "limit", limit, "hot", int(hot.sum()), flush=True)

summary = {n: total(rows[n]) for n in names}
save(out/("heldout.json" if HELDOUT else "dev.json"), dict(summary=summary, machines=info, per_machine=rows,
     limitation="One-tick step statistic: slow drifts without steps cannot be found. Offline train quantiles; "
                "segments counted after merging; overlap matching, not one-to-one."))
for n in names:
    s = summary[n]
    print("%-17s point P %.3f R %.3f F1 %.3f | segments %4d on GT %3d (%.2f) | events %d/%d | event F1 %.3f" % (
        n, s["point_precision"], s["point_recall"], s["point_f1"], s["segments"], s["segments_on_gt"], s["segment_precision"], s["events_hit"], s["events"], s["event_f1"]))
