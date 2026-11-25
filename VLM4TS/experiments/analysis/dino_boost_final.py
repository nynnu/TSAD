"""Final check of the boosted zero-shot DINOv2 Stage1 score on all 20 TSB-AD-M-Eva SMD files with TSB-AD's own metric code (get_metrics, oracle where the
benchmark uses it). Fills the missing patch scores (machine-1-7 after its features arrive) into boost_scores.npz.
Variants: UZ (current); UZ+MuSc50 rank average with moving mean 101 (chosen in advance on the 5 non-TSB-AD SMD machines); the same with 51 (best on Eva, post hoc);
and a LABEL-FREE window declared now, before its numbers: w = median length of the machine's own Stage1 incidents (decision layer on UZ), clipped to [15, 201],
rounded to odd - the smoothing follows the episode scale the detector itself sees on that machine.
Usage: python dino_boost_final.py"""
import torch  # noqa: F401
import sys, json, glob, numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent/"pipeline_final")); sys.argv = sys.argv[:1]
import dino_score_boost as B
from TSB_AD.evaluation.metrics import get_metrics
from paper_stage2 import decide_incidents
E = B.E; D = B.D; changed = False
for m in D:
    if "M50" not in D[m] and (B.MD.OUT/"feats"/f"{m}.npz").exists():
        T = len(D[m]["y"]); D[m]["M30"] = B.musc_jb(m, T, .3); D[m]["M50"] = B.musc_jb(m, T, .5); changed = True; print("patch scores added", m, flush=True)
if changed: np.savez(B.CACHE, **{f"{m}|{k}": v for m, dd in D.items() for k, v in dd.items()})


def adaptive(d):
    L = [e["end"]-e["start"] for e in decide_incidents(d["UZ"])]; w = int(np.clip(np.median(L) if L else 51, 15, 201)) | 1
    return uniform_filter1d(B.rk(d["UZ"])+B.rk(d["M50"]), w), w


if __name__ == "__main__":
    V = {"UZ (current)": lambda d: d["UZ"], "fusion + mean101 (chosen in advance)": lambda d: uniform_filter1d(B.rk(d["UZ"])+B.rk(d["M50"]), 101),
         "fusion + mean51 (post hoc)": lambda d: uniform_filter1d(B.rk(d["UZ"])+B.rk(d["M50"]), 51), "fusion + adaptive window (label-free)": lambda d: adaptive(d)[0]}
    res = {}
    for part, items in (("tune (5 non-TSB-AD SMD)", [(m, 0, None) for m in B.TUNE]), ("TSB-AD Eva-20", [(E.MAP[i][0], E.MAP[i][1], i) for i in E.MAP if i not in (6, 16)])):
        res[part] = {}
        for n, fn in V.items():
            rows = []
            for m, off, i in items:
                s, y = np.asarray(fn(D[m]), float), D[m]["y"]
                if i is not None: n_ = len(pd.read_csv(glob.glob(str(E.TS/f"*_SMD_id_{i}_*.csv"))[0]).iloc[:, -1]); s, y = s[off:off+n_], y[off:off+n_]
                rows.append(get_metrics(s, y.astype(int), slidingWindow=100))
            res[part][n] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("Standard-F1", "VUS-PR", "AUC-PR", "Affiliation-F", "Event-based-F1", "R-based-F1", "PA-F1")}
            print(part, "|", n, res[part][n], flush=True)
    res["adaptive windows"] = {m: adaptive(D[m])[1] for m in D if "M50" in D[m]}
    json.dump(res, open(HERE/"results/resid_dino_20261125/boost_final.json", "w"), indent=1)
