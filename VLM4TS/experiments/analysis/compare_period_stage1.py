"""Stage1 with the old vs the validated period test on the 3 machines whose DINO scores are cached (DINO is unchanged;
only the seasonal component differs). GT only evaluates."""
import json, shutil, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from paired_bank_vlm_pilot import ROOT, save
from annotation_contract import read_annotation_contract
sys.path.insert(0, str(ROOT/"stage1_final"))
import stage1

data = ROOT.parent/"mv_data/SMD"
cache_src = ROOT/"stage1_final/cache_statistical/SMD"
snap = ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"
MACHINES = ("machine-1-1", "machine-1-2", "machine-1-6")
new_period = stage1.period


def old_period(x, lo=5, hi=1000):
    xc = x - x.mean()
    if xc.std() < 1e-9:
        return None
    f = np.abs(np.fft.rfft(xc)); fr = np.fft.rfftfreq(len(xc)); m = (fr > 1/hi) & (fr < 1/lo)
    if not m.any() or f[m].max() < 3*np.median(f[m]):
        return None
    return int(np.clip(round(1/fr[m][np.argmax(f[m])]), lo, hi))


def metrics(results):
    seg = seg_gt = ev = hit = 0; rec = []; prec = []; srcs = {"gt": Counter(), "non": Counter()}; ch_total = 0
    for machine, r in results.items():
        y = np.loadtxt(data/"test_label"/f"{machine}.txt", delimiter=",").astype(bool).ravel()[:r["length"]]
        truth = read_annotation_contract(data, machine)
        mask = np.zeros(len(y), bool)
        for e in r["intervals"]:
            mask[e["start"]:e["end"]] = True; seg += 1; seg_gt += y[e["start"]:e["end"]].any()
        d = np.diff(np.r_[0, y.astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ev += 1; hit += mask[a:b].any()
        for t in truth:
            if t["annotation_status"] != "exact":
                continue
            over = [e for e in r["intervals"] if max(t["start"], e["start"]) < min(t["end"], e["end"])]
            if not over:
                continue
            got = {c["channel"] for e in over for c in e["channels"]}; gt = set(t["channels"])
            rec.append(len(got & gt)/len(gt)); prec.append(len(got & gt)/max(len(got), 1))
        for e in r["intervals"]:
            hit_gt = y[e["start"]:e["end"]].any()
            for c in e["channels"]:
                ch_total += 1
    return dict(intervals=seg, interval_precision=seg_gt/seg, event_recall=hit/ev, channel_recall=float(np.mean(rec)), channel_precision=float(np.mean(prec)),
                channels_per_interval=float(np.mean([len(e["channels"]) for r in results.values() for e in r["intervals"]])))


out = {}
with tempfile.TemporaryDirectory() as tmp:
    for m in MACHINES:
        shutil.copy(cache_src/f"{m}_dino_test.npz", Path(tmp)/f"{m}_dino.npz")
    for name, fn in (("old_period", old_period), ("validated_period", new_period)):
        stage1.period = fn
        res = {m: stage1.run_machine(m, data, Path(tmp), lambda: None) for m in MACHINES}
        out[name] = metrics(res)
        out[name]["share_of_channels_with_a_period"] = None
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[name].items() if v is not None}, flush=True)
        if name == "old_period":
            same = all(len(res[m]["intervals"]) == len(json.loads((snap/f"{m}.json").read_text(encoding="utf8"))["intervals"]) for m in MACHINES)
            print("   old rule reproduces the saved snapshots' interval counts:", same)
    # how many channels carry a seasonal term
    for name, fn in (("old_period", old_period), ("validated_period", new_period)):
        n = k = 0
        for m in MACHINES:
            tr = np.loadtxt(data/"train"/f"{m}.txt", delimiter=",")
            for c in range(tr.shape[1]):
                if tr[:, c].std() > 1e-6:
                    n += 1; k += fn(tr[:, c]) is not None
        out[name]["share_of_channels_with_a_period"] = k/n
        print(name, "channels with a seasonal term: %.2f" % (k/n))
save(ROOT/"analysis/results/period_validity_20261111/stage1_compare.json", out)
