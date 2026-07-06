"""
Multi-Spectral SMA Score Ensemble
====================================
Hypothesis: SMA at different β values captures different frequency regimes.
  β=0.25 → mild alignment, preserves high-freq (spikes, discontinuities)
  β=0.50 → moderate
  β=0.75 → strong alignment (stuck sensors, low-freq anomalies)
  β=1.00 → full alignment to 1/f²

Ensembling across all β values builds robustness to unknown anomaly
frequency profile — no signal type assumed in advance.

Zero new inference: all 4 SMA-k5 checkpoints already exist from
compare_sma_ltr.py.  k30 checkpoints from compare_ltr_multiscale.py.

Conditions
----------
  sma075_k30      : add(sma075, k30) + EVT          ← current best (0.6391)
  add_sma_all     : Σ normalize(sma_β) all 4 betas + EVT
  max_sma_all     : max across 4 beta scores + EVT
  add_sma_all_k30 : add_sma_all + normalize(k30) + EVT  ← main hypothesis

Usage
-----
  python experiments/compare_multispectral_sma.py
"""

from __future__ import annotations

import ast
import os
import pickle
import sys

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

from evaluation.evaluate import evaluate_intervals  # noqa: E402

print(f"Project root : {PROJECT_ROOT}")

# ── Checkpoint directories ────────────────────────────────────────────────────
def _first_existing(*candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[-1]

SMA_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_sma_ltr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_sma_ltr", "checkpoints"),
)
K30_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
)
K5_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)

print(f"SMA checkpoints : {SMA_CKPT_DIR}  ({len(os.listdir(SMA_CKPT_DIR))} files)")
print(f"k30 checkpoints : {K30_CKPT_DIR}  ({len(os.listdir(K30_CKPT_DIR))} files)")
print(f"k5  checkpoints : {K5_CKPT_DIR}")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_multispectral_sma")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Signal lists ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
             "T-13","D-16","T-8","P-14","D-14"]

def _load_nab_sigs(gt):
    nab_dir = os.path.join(DATA_DIR, "realAWSCloudwatch")
    return sorted(f[:-4] for f in os.listdir(nab_dir)
                  if f.endswith(".csv") and gt.get(f[:-4]))

def _load_gt():
    gt = {}
    with open(ANOM_CSV, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt

gt = _load_gt()
NAB_SIGS = _load_nab_sigs(gt)

DATASET_CONFIGS = {
    "NAB":  {"dir": os.path.join(DATA_DIR, "realAWSCloudwatch"), "sigs": NAB_SIGS},
    "SMAP": {"dir": os.path.join(DATA_DIR, "SMAP"),             "sigs": SMAP_SIGS},
    "MSL":  {"dir": os.path.join(DATA_DIR, "MSL"),              "sigs": MSL_SIGS},
}

# ── EVT ───────────────────────────────────────────────────────────────────────
EVT_Q    = 0.90
EVT_FPR  = 0.01

def evt_threshold(scores: np.ndarray) -> float:
    u           = float(np.percentile(scores, EVT_Q * 100))
    exceedances = scores[scores > u] - u
    fallback    = float(np.percentile(scores, (1.0 - EVT_FPR) * 100))
    if len(exceedances) < 10:
        return fallback
    try:
        c, _, scale = genpareto.fit(exceedances, floc=0)
        p_cond = min(EVT_FPR / max(1.0 - EVT_Q, 1e-9), 1.0 - 1e-9)
        thr    = u + max(0.0, float(genpareto.ppf(1.0 - p_cond, c, loc=0, scale=scale)))
        return thr if u <= thr <= scores.max() else fallback
    except Exception:
        return fallback

def evt_detect(scores: np.ndarray, timestamps: np.ndarray) -> list:
    if scores.max() - scores.min() < 1e-8:
        return []
    thr   = evt_threshold(scores)
    flags = scores > thr
    if not flags.any():
        return []
    ivs, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg:
            in_seg = True; seg_start = i
        elif not f and in_seg:
            in_seg = False
            ivs.append([timestamps[seg_start], timestamps[i - 1]])
    if in_seg:
        ivs.append([timestamps[seg_start], timestamps[len(flags) - 1]])
    return ivs

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-8 else (arr - lo) / (hi - lo)

# ── Checkpoint loading ────────────────────────────────────────────────────────
SMA_BETAS     = ["025", "050", "075", "100"]
SMA_BETA_VALS = [0.25,  0.50,  0.75,  1.00]

def load_sma(ds, sig, beta_str):
    """Load SMA-k5 scores for one beta. Returns scores array or None."""
    p = os.path.join(SMA_CKPT_DIR, f"{ds}__{sig}__ltr_sma{beta_str}.pkl")
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb"))["scores"]

def load_k30(ds, sig):
    p = os.path.join(K30_CKPT_DIR, f"{ds}__{sig}__ltr_k30.pkl")
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb"))["scores"]

def load_k5_timestamps(ds, sig):
    """Load timestamps from k5 baseline checkpoint (only source that has them)."""
    p = os.path.join(K5_CKPT_DIR, f"{ds}__{sig}__ltr.pkl")
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb"))["timestamps"]

# ── Per-signal evaluation ─────────────────────────────────────────────────────
METHODS = ["sma075_k30", "add_sma_all", "max_sma_all", "add_sma_all_k30"]
rows: list = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}  ({len(cfg['sigs'])} signals)")
    print(f"{'='*60}")

    for sig in cfg["sigs"]:
        gt_ivs = gt.get(sig, [])
        if not gt_ivs:
            continue

        # Load all 4 SMA betas
        sma_scores = {}
        for b in SMA_BETAS:
            s = load_sma(ds_name, sig, b)
            if s is None:
                break
            sma_scores[b] = s
        if len(sma_scores) < 4:
            print(f"  SKIP {sig} (missing SMA ckpts)")
            continue

        ts = load_k5_timestamps(ds_name, sig)
        if ts is None:
            print(f"  SKIP {sig} (missing k5 timestamps)")
            continue

        s_k30 = load_k30(ds_name, sig)

        T   = min(len(s) for s in sma_scores.values())
        if s_k30 is not None:
            T = min(T, len(s_k30))
        ts_t = ts[:T]

        row = {"dataset": ds_name, "signal": sig}

        # norm each beta score
        norm_sma = {b: normalize_01(sma_scores[b][:T]) for b in SMA_BETAS}
        norm_k30 = normalize_01(s_k30[:T]) if s_k30 is not None else None

        # ── Condition 1: sma075 + k30 (current best reference) ──────────────
        if norm_k30 is not None:
            s = norm_sma["075"] + norm_k30
            row["sma075_k30_f1"] = evaluate_intervals(
                gt_ivs, evt_detect(s, ts_t))["F1"]
        else:
            row["sma075_k30_f1"] = float("nan")

        # ── Condition 2: add of all 4 betas ──────────────────────────────────
        s = sum(norm_sma[b] for b in SMA_BETAS)
        row["add_sma_all_f1"] = evaluate_intervals(
            gt_ivs, evt_detect(s, ts_t))["F1"]

        # ── Condition 3: max of all 4 betas ──────────────────────────────────
        s = np.stack([norm_sma[b] for b in SMA_BETAS]).max(axis=0)
        row["max_sma_all_f1"] = evaluate_intervals(
            gt_ivs, evt_detect(s, ts_t))["F1"]

        # ── Condition 4: add_sma_all + k30 ───────────────────────────────────
        if norm_k30 is not None:
            s = sum(norm_sma[b] for b in SMA_BETAS) + norm_k30
            row["add_sma_all_k30_f1"] = evaluate_intervals(
                gt_ivs, evt_detect(s, ts_t))["F1"]
        else:
            row["add_sma_all_k30_f1"] = float("nan")

        rows.append(row)
        print(f"  {sig:<12}  "
              f"sma075+k30={row['sma075_k30_f1']:.4f}  "
              f"add_all={row['add_sma_all_f1']:.4f}  "
              f"max_all={row['max_sma_all_f1']:.4f}  "
              f"add_all+k30={row['add_sma_all_k30_f1']:.4f}")

# ── Summary table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "multispectral_sma.csv"), index=False)

COL_W  = 14
F1_COLS = [f"{m}_f1" for m in METHODS]

print(f"\n{'='*80}")
print("SUMMARY — Multi-Spectral SMA Ensemble")
print(f"{'='*80}")
print(f"{'Dataset':<10}" + "".join(f"{m:>{COL_W}}" for m in METHODS))
print("-" * (10 + COL_W * len(METHODS)))

sig_counts = {"NAB": 16, "SMAP": 13, "MSL": 11}
all_avgs   = {c: [] for c in F1_COLS}

for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    avgs = [sub[c].mean() for c in F1_COLS]
    print(f"{ds:<10}" + "".join(f"{a:>{COL_W}.4f}" for a in avgs))
    for c, a in zip(F1_COLS, avgs):
        all_avgs[c].extend([a] * sig_counts[ds])

print("-" * (10 + COL_W * len(METHODS)))

# Signal-weighted ALL F1
all_f1 = {c: np.mean(all_avgs[c]) / sum(sig_counts.values()) * 40
          for c in F1_COLS}
# Actually: weighted average = sum(ds_avg * ds_count) / total_count
all_f1_weighted = {}
for c in F1_COLS:
    total = 0.0
    for ds in ["NAB", "SMAP", "MSL"]:
        sub = results_df[results_df["dataset"] == ds]
        if not sub.empty:
            total += sub[c].mean() * sig_counts[ds]
    all_f1_weighted[c] = total / sum(sig_counts.values())

print(f"{'ALL (wtd)':<10}" + "".join(f"{all_f1_weighted[c]:>{COL_W}.4f}" for c in F1_COLS))
print(f"\nResults → {RESULTS_DIR}/multispectral_sma.csv")
