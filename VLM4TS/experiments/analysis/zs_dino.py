"""Zero-shot DINO score (no train segment): the Stage1 DINOv2 spike / height scores (stage1_final/stage1.py dino_scores; heads fitted on SYNTHETIC spikes, not on
SMD / ASD labels) with the only train use - the plot y range and the flat-window test - replaced by the test channel's p0.5..p99.5 range. Per tick:
D = max over channels of calib(spike) + calib(height) (calib = per-channel test median / MAD), as multi_metric_scan.dino_tick. Cached per machine.
Usage: python zs_dino.py [--machines SMD/machine-1-1 ...]   -> results/testonly_20261124/cache/{ds}_{m}_dino_zs.npz"""
import argparse, sys, time
from pathlib import Path
import torch   # torch DLLs before scipy / sklearn (Windows)
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"stage1_final"), str(ROOT/"analysis")]
from stage1 import load_dino, dino_scores, drop_flat, calib
OUT = ROOT/"analysis/results/testonly_20261124/cache"; DATA = ROOT.parent/"mv_data"


def zs_range(te):
    """Two-row stand-in for the train split: the test p0.5 / p99.5 per channel (dino_scores and drop_flat only read its min / max / ptp)."""
    return np.vstack([np.quantile(te, .005, 0), np.quantile(te, .995, 0)])


def dino_zs(ds, m, dino=None):
    c = OUT/f"{ds}_{m}_dino_zs.npz"
    if c.exists(): return np.load(c)["D"]
    te = np.loadtxt(DATA/ds/"test"/f"{m}.txt", delimiter=",", ndmin=2); tr = zs_range(te); sp, ht = dino_scores(dino or load_dino(), tr, te)
    X = calib(drop_flat(sp, te, tr))+calib(drop_flat(ht, te, tr)); d = np.where(np.isfinite(X), X, -np.inf).max(0); fin = np.isfinite(d)
    D = np.where(fin, d, d[fin].min() if fin.any() else 0.); np.savez(c, D=D, spike=sp, height=ht); return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--machines", nargs="*"); a = ap.parse_args(); dino = load_dino(); torch.set_num_threads(max(1, torch.get_num_threads()))
    keys = [tuple(k.split("/")) for k in a.machines] if a.machines else [(ds, f.stem) for ds in ("SMD", "ASD") for f in sorted((DATA/ds/"test").glob("*.txt"))]
    for ds, m in keys:
        t0 = time.time(); dino_zs(ds, m, dino); print(ds, m, f"{time.time()-t0:.0f}s", flush=True)
