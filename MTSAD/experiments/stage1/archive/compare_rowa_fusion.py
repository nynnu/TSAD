"""
ROWA Fusion: Reconstruction One-Way Additive
=============================================

Problem with standard additive fusion
--------------------------------------
  add(k5,k30) + 0.3*Recon  degrades NAB (0.7687 → 0.5994)
  Root cause: Recon modifies the score distribution globally,
  shifting EVT threshold away from true anomaly peaks.

ROWA fix
--------
  Add Recon ONLY where it exceeds the multi-scale LTR score:

    base      = normalize_01(add(s_k5, s_k30))
    increment = max(0,  normalize_01(s_recon) - base)
    s_final   = base + w * increment

  When LTR is already high (true anomaly):
    norm(Recon) - base ≤ 0  → increment = 0  → LTR peak preserved ✓
  When LTR is low but Recon fires (e.g. SMAP F-2):
    norm(Recon) - base > 0  → Recon detection surfaced ✓

  No FP amplification. No peak suppression.

Ablations
---------
  add_ms     : add(LTR k5, LTR k30) + EVT            [current best = 0.6256]
  rowa_w01   : ROWA w=0.1
  rowa_w02   : ROWA w=0.2
  rowa_w03   : ROWA w=0.3
  rowa_w05   : ROWA w=0.5
  std_add_w03: add_ms + 0.3*Recon  (standard additive, for comparison)

Checkpoint reuse (no inference):
  k=5  : {MGMR_CKPT}/{DS}__{sig}__ltr.pkl       keys: scores, timestamps
  k=30 : {K30_CKPT}/{DS}__{sig}__ltr_k30.pkl    key:  scores
  Recon: {RECON_CKPT}/{DS}__{sig}__recon.pkl     key:  scores

Usage:
  python experiments/compare_rowa_fusion.py
"""

from __future__ import annotations

import ast
import os
import pickle
import sys
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import genpareto

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

print(f"Project root : {PROJECT_ROOT}")

subprocess.run(
    ["pip", "install", "scipy", "--quiet"], check=True,
)

from evaluation.evaluate import evaluate_intervals

# ── Checkpoint paths ──────────────────────────────────────────────────────────
def _find_dir(candidates: list[str]) -> str:
    found = next((p for p in candidates if os.path.isdir(p)), candidates[-1])
    os.makedirs(found, exist_ok=True)
    return found


K5_CKPT_DIR = _find_dir([
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
])
K30_CKPT_DIR = _find_dir([
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
])
RECON_CKPT_DIR = _find_dir([
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_recon_additive", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_recon_additive", "checkpoints"),
])

print(f"k=5  ckpt : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR))} files)")
print(f"k=30 ckpt : {K30_CKPT_DIR}  ({len(os.listdir(K30_CKPT_DIR))} files)")
print(f"Recon ckpt: {RECON_CKPT_DIR}  ({len(os.listdir(RECON_CKPT_DIR))} files)")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_rowa_fusion")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(PROJECT_ROOT, "data")
NAB_DIR   = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR  = os.path.join(DATA_DIR, "SMAP")
MSL_DIR   = os.path.join(DATA_DIR, "MSL")
ANOM_CSV  = os.path.join(DATA_DIR, "anomalies.csv")

EVT_Q_INIT = 0.90
EVT_FPR    = 0.01
ROWA_WEIGHTS = [0.1, 0.2, 0.3, 0.5]


# ── EVT ───────────────────────────────────────────────────────────────────────

def evt_threshold(scores: np.ndarray, q_init: float = EVT_Q_INIT,
                  target_fpr: float = EVT_FPR) -> float:
    u           = float(np.percentile(scores, q_init * 100.0))
    exceedances = scores[scores > u] - u
    fallback    = float(np.percentile(scores, (1.0 - target_fpr) * 100.0))
    if len(exceedances) < 10:
        return fallback
    try:
        c, _, scale = genpareto.fit(exceedances, floc=0)
        p_cond = min(target_fpr / max(1.0 - q_init, 1e-9), 1.0 - 1e-9)
        excess = genpareto.ppf(1.0 - p_cond, c, loc=0, scale=scale)
        thr    = u + max(0.0, excess)
        return thr if u <= thr <= scores.max() else fallback
    except Exception:
        return fallback


def evt_detect(scores: np.ndarray, timestamps: np.ndarray) -> list:
    if scores.max() - scores.min() < 1e-8:
        return []
    flags = scores > evt_threshold(scores)
    if not flags.any():
        return []
    intervals, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg:
            in_seg = True; seg_start = i
        elif not f and in_seg:
            in_seg = False
            intervals.append([timestamps[seg_start], timestamps[i - 1]])
    if in_seg:
        intervals.append([timestamps[seg_start], timestamps[len(flags) - 1]])
    return intervals


# ── Score helpers ─────────────────────────────────────────────────────────────

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-8 else (arr - lo) / (hi - lo)


def add_fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    T = min(len(a), len(b))
    return normalize_01(a[:T]) + normalize_01(b[:T])


def rowa_fuse(s_ltr_add: np.ndarray, s_recon: np.ndarray, w: float) -> np.ndarray:
    """ROWA: add Recon only where it exceeds the multi-scale LTR base.

    s_ltr_add : already add-fused LTR (sum of two normalized signals, in [0,2])
    s_recon   : raw Recon scores
    w         : weight on the incremental Recon contribution
    """
    T         = min(len(s_ltr_add), len(s_recon))
    base      = normalize_01(s_ltr_add[:T])          # re-normalize add to [0, 1]
    recon_n   = normalize_01(s_recon[:T])
    increment = np.maximum(0.0, recon_n - base)      # only where Recon > LTR
    return base + w * increment


def std_add_fuse(s_ltr_add: np.ndarray, s_recon: np.ndarray, w: float) -> np.ndarray:
    """Standard additive (benchmark): normalize(add(k5,k30)) + w*normalize(Recon)."""
    T     = min(len(s_ltr_add), len(s_recon))
    base  = normalize_01(s_ltr_add[:T])
    return base + w * normalize_01(s_recon[:T])


# ── Evaluation ────────────────────────────────────────────────────────────────

def load_gt(anom_csv: str) -> dict:
    gt = {}
    with open(anom_csv, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt


def _eval(detected: list, gt_ivs: list) -> float:
    return evaluate_intervals(gt_ivs, detected)["F1"]


# ── Dataset config ────────────────────────────────────────────────────────────

gt = load_gt(ANOM_CSV)
nab_files = sorted(f for f in os.listdir(NAB_DIR) if f.endswith(".csv"))
NAB_SIGS  = [f[:-4] for f in nab_files if gt.get(f[:-4])]
SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
              "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
              "T-13","D-16","T-8","P-14","D-14"]

DATASET_CONFIGS = {
    "NAB":  {"dir": NAB_DIR,  "sigs": NAB_SIGS},
    "SMAP": {"dir": SMAP_DIR, "sigs": SMAP_SIGS},
    "MSL":  {"dir": MSL_DIR,  "sigs": MSL_SIGS},
}

COLS = ["add_ms"] + [f"rowa_w0{int(w*10)}" for w in ROWA_WEIGHTS] + ["std_add_w03"]


# ── Main loop ─────────────────────────────────────────────────────────────────

rows: list = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*70}\nDataset: {ds_name}  ({len(cfg['sigs'])} signals)\n{'='*70}")

    for sig in cfg["sigs"]:
        gt_ivs = gt.get(sig, [])
        if not os.path.exists(os.path.join(cfg["dir"], f"{sig}.csv")) or not gt_ivs:
            print(f"  SKIP {sig}")
            continue

        print(f"\n  [{sig}]")
        sig_row = {"dataset": ds_name, "signal": sig}

        k5_ckpt    = os.path.join(K5_CKPT_DIR,    f"{ds_name}__{sig}__ltr.pkl")
        k30_ckpt   = os.path.join(K30_CKPT_DIR,   f"{ds_name}__{sig}__ltr_k30.pkl")
        recon_ckpt = os.path.join(RECON_CKPT_DIR,  f"{ds_name}__{sig}__recon.pkl")

        # All three must exist — this script is pure fusion, no compute fallback
        missing = [p for p in [k5_ckpt, k30_ckpt, recon_ckpt] if not os.path.exists(p)]
        if missing:
            print(f"    SKIP — missing checkpoints: {[os.path.basename(p) for p in missing]}")
            continue

        try:
            c_k5    = pickle.load(open(k5_ckpt,    "rb"))
            c_k30   = pickle.load(open(k30_ckpt,   "rb"))
            c_recon = pickle.load(open(recon_ckpt, "rb"))

            s_k5       = c_k5["scores"]
            timestamps = c_k5["timestamps"]
            s_k30      = c_k30["scores"]
            s_recon    = c_recon.get("scores", c_recon.get("mean"))
        except Exception as e:
            print(f"    SKIP — load error: {e}")
            continue

        if s_recon is None:
            print("    SKIP — Recon checkpoint has no 'scores' or 'mean' key")
            continue

        T       = min(len(s_k5), len(s_k30), len(s_recon))
        ts_trim = timestamps[:T]
        s_add   = add_fuse(s_k5, s_k30)   # shape [T], in [0, 2]

        # 1. add(k5, k30) — current best reference
        ivs = evt_detect(normalize_01(s_add[:T]), ts_trim)
        f1  = _eval(ivs, gt_ivs)
        sig_row["add_ms_f1"] = f1
        print(f"    add_ms      F1={f1:.4f}  ivs={len(ivs)}")

        # 2. ROWA at multiple weights
        for w in ROWA_WEIGHTS:
            key = f"rowa_w0{int(w*10)}"
            s_f = rowa_fuse(s_add[:T], s_recon[:T], w)
            ivs = evt_detect(s_f, ts_trim)
            f1  = _eval(ivs, gt_ivs)
            sig_row[f"{key}_f1"] = f1
            delta = f1 - sig_row["add_ms_f1"]
            print(f"    {key}     F1={f1:.4f}  ivs={len(ivs)}  ({delta:+.4f})")

        # 3. Standard additive w=0.3 (comparison)
        s_std = std_add_fuse(s_add[:T], s_recon[:T], 0.3)
        ivs   = evt_detect(s_std, ts_trim)
        f1    = _eval(ivs, gt_ivs)
        sig_row["std_add_w03_f1"] = f1
        delta = f1 - sig_row["add_ms_f1"]
        print(f"    std_add_w03 F1={f1:.4f}  ivs={len(ivs)}  ({delta:+.4f})")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────

results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

COL_W = 13
SEP   = 36 + COL_W * len(COLS)


def _fmt(v) -> str:
    try:
        f = float(v)
        return "  NaN" if (f != f) else f"{f:.4f}"
    except Exception:
        return "  NaN"


def _print_table(ds: str, df: pd.DataFrame):
    sub = df[df["dataset"] == ds]
    if sub.empty:
        return
    print(f"\n{'='*SEP}\n=== {ds} — ROWA Fusion ===\n{'='*SEP}")
    print(f"{'Signal':<36}" + "".join(f"{c:>{COL_W}}" for c in COLS))
    print("-" * SEP)
    for _, r in sub.iterrows():
        fs   = [r.get(f"{c}_f1", float("nan")) for c in COLS]
        best = max((f for f in fs if f == f), default=float("nan"))
        line = f"{r['signal']:<36}"
        for f in fs:
            mark = "→" if (f == f and abs(f - best) < 1e-9) else " "
            line += f"{mark + _fmt(f):>{COL_W}}"
        print(line)
    print("-" * SEP)
    print(f"{'AVG':<36}" + "".join(f" {sub[f'{c}_f1'].mean():>{COL_W-1}.4f}" for c in COLS))


for ds in ["NAB", "SMAP", "MSL"]:
    _print_table(ds, results_df)

# ── Cross-dataset summary ─────────────────────────────────────────────────────
print(f"\n{'='*SEP}\nOVERALL SUMMARY  (ROWA: base=add(k5,k30), increment=max(0,Recon-base))\n{'='*SEP}")
print(f"{'Dataset':<14}" + "".join(f"{c:>{COL_W}}" for c in COLS))
print("-" * SEP)

all_f1: dict[str, list] = {c: [] for c in COLS}
for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    row = f"{ds:<14}"
    for c in COLS:
        v = sub[f"{c}_f1"].mean() if f"{c}_f1" in sub.columns else float("nan")
        row += f" {v:>{COL_W-1}.4f}"
        if not np.isnan(v):
            all_f1[c].append((ds, len(sub), v))
    print(row)

print("-" * SEP)
# Signal-weighted ALL (matches KEY FINDINGS in multiscale experiment)
ds_counts = {"NAB": 16, "SMAP": 13, "MSL": 11}
overall = f"{'ALL (weighted)':<14}"
for c in COLS:
    vals = all_f1[c]
    if not vals:
        overall += f"{'NaN':>{COL_W}}"
        continue
    total_n = sum(ds_counts.get(ds, 0) for ds, _, _ in vals)
    weighted = sum(ds_counts.get(ds, 0) * v for ds, _, v in vals) / max(total_n, 1)
    overall += f" {weighted:>{COL_W-1}.4f}"
print(overall)

print(f"\nResults saved → {out_csv}")
