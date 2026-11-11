"""Does Stage1's own confidence (spike score, source) separate GT from non-GT? 28 machines, no VLM, GT only evaluates.
Dev = machine-1-x, held = machine-2-x/3-x. (a) channel order inside an interval, (b) interval-level score within a machine."""
import json
import numpy as np
from scipy.stats import rankdata
from paired_bank_vlm_pilot import ROOT, save
from annotation_contract import read_annotation_contract

data = ROOT.parent/"mv_data/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
TIER = {"both": 2, "spike": 1, "deseason": 0}


def auroc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return None
    r = rankdata(score)
    return float((r[label].sum()-label.sum()*(label.sum()+1)/2)/(label.sum()*(~label).sum()))


def orderings(chans):
    sp = np.array([c["spike_score"] if c["spike_score"] is not None else -1e9 for c in chans])
    ds = np.array([c["deseason_score"] or 0. for c in chans])
    tier = np.array([TIER[c["source"]] for c in chans])
    rk = lambda v: rankdata(v)/len(v)
    return {"spike_score (current)": sp, "source tier, then spike_score": tier*1e6+np.clip(sp, -1e5, 1e5),
            "rank(spike)+rank(seasonal)": rk(sp)+rk(ds), "seasonal_score": ds}


acc = {s: {} for s in ("dev", "held")}; cover = {s: {} for s in ("dev", "held")}
interval = {s: {"max_spike": [], "n_both": [], "frac_both": [], "n_channels": [], "length": []} for s in ("dev", "held")}
for path in sorted(snap.glob("machine-*.json")):
    split = "dev" if path.stem.startswith("machine-1-") else "held"
    doc = json.loads(path.read_text(encoding="utf8")); truth = read_annotation_contract(data, path.stem)
    y = np.loadtxt(data/"test_label"/f"{path.stem}.txt", delimiter=",").astype(bool).ravel()
    feats = {k: [] for k in interval[split]}; labels = []
    for e in doc["intervals"]:
        chans = e["channels"]
        sp = [c["spike_score"] for c in chans if c["spike_score"] is not None]
        feats["max_spike"].append(max(sp) if sp else -1e9); feats["n_both"].append(sum(c["source"] == "both" for c in chans))
        feats["frac_both"].append(feats["n_both"][-1]/max(len(chans), 1)); feats["n_channels"].append(len(chans)); feats["length"].append(e["end"]-e["start"])
        labels.append(bool(y[e["start"]:e["end"]].any()))
        hit = [t for t in truth if max(e["start"], t["start"]) < min(e["end"], t["end"])]
        if len(hit) != 1 or hit[0]["annotation_status"] != "exact" or len(chans) < 2:
            continue
        gt = [c["channel"] in set(hit[0]["channels"]) for c in chans]
        for name, score in orderings(chans).items():
            a = auroc(score, gt)
            if a is not None:
                acc[split].setdefault(name, []).append(a)
            order = np.argsort(-score, kind="stable")[:8]
            cover[split].setdefault(name, []).append((sum(gt[i] for i in order), sum(gt)))
    for k in feats:
        a = auroc(rankdata(feats[k])/len(labels), labels)
        if a is not None:
            interval[split][k].append(a)
summary = {}
for s in ("dev", "held"):
    summary[s] = dict(channel_order={n: dict(auroc=float(np.mean(v)), gt_channels_in_first_page=float(sum(a for a, _ in cover[s][n])/sum(b for _, b in cover[s][n]))) for n, v in acc[s].items()},
                      interval_score_auroc_within_machine={k: float(np.mean(v)) for k, v in interval[s].items()})
save(ROOT/"analysis/results/stage1_confidence_20261111/summary.json", summary)
for s in summary:
    print(s)
    for n, v in summary[s]["channel_order"].items():
        print("   channel order %-34s AUROC %.3f | GT channels on page 1: %.3f" % (n, v["auroc"], v["gt_channels_in_first_page"]))
    print("   interval score AUROC (within machine):", {k: round(v, 3) for k, v in summary[s]["interval_score_auroc_within_machine"].items()})
