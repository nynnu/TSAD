"""Within-machine ranking of Stage1 cores as a false-alarm budget (no VLM).

Why: no fixed cross-machine threshold worked, but within a machine several signal properties rank GT
cores above non-GT cores (AUROC 0.7-0.8). Rank-normalising per machine removes the scale problem, and
the kept fraction is the explicit alarm budget. Features: Stage1 score of the parent interval, train-scaled
largest step, number of channels stepping together. Selection of the score combination uses machine-1-x
only; machine-2-x/3-x are evaluated once with the chosen one (`--heldout`). GT only evaluates.
"""
import json, sys
from itertools import combinations
import numpy as np
from scipy.stats import rankdata
from paired_bank_vlm_pilot import ROOT, save

data = ROOT.parent/"mv_data/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
out = ROOT/"analysis/results/rank_filter_20261111"
out.mkdir(parents=True, exist_ok=True)
HELDOUT = "--heldout" in sys.argv
FEATURES = ("s1", "ratio", "costep")
FRACTIONS = (1.0, 0.75, 0.5, 0.35)
cores = json.loads((ROOT/"analysis/results/miss_funnel_20261110/summary.json").read_text(encoding="utf8"))["cores"]


def features(machine, mine):
    train = np.loadtxt(data/"train"/f"{machine}.txt", delimiter=",")
    test = np.loadtxt(data/"test"/f"{machine}.txt", delimiter=",")
    dtr, dte = np.abs(np.diff(train, axis=0)), np.abs(np.diff(test, axis=0))
    scale = np.quantile(dtr, 0.999, axis=0)+1e-3
    big = dte > scale
    doc = json.loads((snap/f"{machine}.json").read_text(encoding="utf8"))
    rows = []
    for c in mine:
        a, b = c["core"]
        parent = next(e for e in doc["intervals"] if e["start"] <= a and b <= e["end"])
        s1 = max([r["spike_score"] for r in parent["channels"] if r["spike_score"] is not None] or [0.0])
        ratio = float((dte[max(a-1, 0):b-1][:, c["shown"]].max(axis=0)/scale[c["shown"]]).max())
        costep = int(big[max(a-1, 0):b-1].any(axis=0).sum())
        rows.append(dict(s1=s1, ratio=ratio, costep=costep))
    return rows


def segments(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def evaluate(per_machine, scores, fraction, rng=None):
    """Keep the top `fraction` of cores per machine. scores: {machine: array}; rng: random order instead."""
    tp = fp = fn = kept_gt = kept = hit = events = 0
    for machine, (mine, y) in per_machine.items():
        n = len(mine)
        k = max(1, int(np.ceil(fraction*n)))
        order = rng.permutation(n) if rng is not None else np.argsort(-scores[machine], kind="stable")
        keep = order[:k]
        mask = np.zeros(len(y), bool)
        for i in keep:
            a, b = mine[i]["core"]
            mask[a:b] = True
        kept += k; kept_gt += sum(mine[i]["gt_overlap"] for i in keep)
        tp += int((mask & y).sum()); fp += int((mask & ~y).sum()); fn += int((~mask & y).sum())
        gt = segments(y); events += len(gt); hit += sum(mask[a:b].any() for a, b in gt)
    p, r = kept_gt/kept, hit/events
    pp, pr = tp/max(tp+fp, 1), tp/(tp+fn)
    return dict(core_precision=p, event_recall=r, event_f1=2*p*r/(p+r), point_f1=2*pp*pr/(pp+pr), kept=kept)


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return None
    r = rankdata(score)
    return float((r[label].sum()-label.sum()*(label.sum()+1)/2)/(label.sum()*(~label).sum()))


machines = sorted({c["machine"] for c in cores})
machines = [m for m in machines if m.startswith("machine-1-") != HELDOUT]
per_machine, ranks, aurocs = {}, {}, {f: [] for f in FEATURES}
for machine in machines:
    mine = [c for c in cores if c["machine"] == machine]
    feats = features(machine, mine)
    y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()
    per_machine[machine] = (mine, y)
    ranks[machine] = {f: rankdata([r[f] for r in feats])/len(feats) for f in FEATURES}
    for f in FEATURES:
        aurocs[f].append(auroc(ranks[machine][f], [c["gt_overlap"] for c in mine]))
    print(machine, "features done", flush=True)

combos = [list(s) for n in (1, 2, 3) for s in combinations(FEATURES, n)]
rng = np.random.default_rng(0)
random_baseline = {}
for f in FRACTIONS:
    trials = [evaluate(per_machine, None, f, rng) for _ in range(200)]
    random_baseline[f] = {k: float(np.mean([t[k] for t in trials])) for k in ("core_precision", "event_recall", "event_f1", "point_f1")}
table = {}
for combo in combos:
    scores = {m: np.mean([ranks[m][f] for f in combo], axis=0) for m in machines}
    table["+".join(combo)] = {str(f): evaluate(per_machine, scores, f) for f in FRACTIONS}
summary = dict(heldout=HELDOUT, machines=len(machines), auroc_per_machine_mean={f: float(np.mean([a for a in v if a is not None])) for f, v in aurocs.items()},
               random=random_baseline, combos=table)
if not HELDOUT:
    best = max(table, key=lambda k: table[k]["0.5"]["event_f1"])
    summary["chosen_on_dev"] = best
save(out/("heldout.json" if HELDOUT else "dev.json"), dict(summary=summary,
     limitation="Core level; precision = kept cores overlapping GT / kept cores, event recall = GT events overlapped by kept cores. "
                "Selection on machine-1-x at keep fraction 0.5 by event F1; heldout evaluates every combination only for the record, the headline is the chosen one."))
print(json.dumps(summary["auroc_per_machine_mean"]))
for f in FRACTIONS:
    r = random_baseline[f]
    print("keep %.2f random: precision %.3f event recall %.3f event F1 %.3f" % (f, r["core_precision"], r["event_recall"], r["event_f1"]))
for name, rows in table.items():
    print("%-22s" % name, " | ".join("%.2f: P %.3f R %.3f F1 %.3f" % (f, rows[str(f)]["core_precision"], rows[str(f)]["event_recall"], rows[str(f)]["event_f1"]) for f in FRACTIONS))
if not HELDOUT:
    print("chosen on dev:", summary["chosen_on_dev"])
