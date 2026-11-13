"""
compare_bltr_scoring.py
=======================
Bidirectional LTR (B-LTR): score anomalies in BOTH directions.

Core insight
------------
LTR fails for two opposite reasons:
  1. Spike anomaly: window is very DIFFERENT from neighbors → LTR score HIGH ✓ (works)
  2. Stuck-sensor / regime-shift: window is very SIMILAR to (also-anomalous)
     neighbors → LTR score LOW ✗ (inverted — D-15, T-1)

Both are anomalies. Both deviate from the TYPICAL LTR score of the signal.
They deviate in OPPOSITE DIRECTIONS.

The fix
-------
Score by absolute deviation from the SIGNAL-LEVEL MEDIAN LTR score:

  score_t = | ltr_t − median(ltr_all) |

  - Spike (high LTR)      : far ABOVE median → high bidirectional score ✓
  - Stuck sensor (low LTR): far BELOW median → high bidirectional score ✓
  - Normal (moderate LTR) : near median      → low score ✓

No new inference. Zero cost. No new hyperparameters.

Why this doesn't regress on working signals
-------------------------------------------
For signals LTR already handles (e.g., NAB spikes with F1=1.0):
  anomaly LTR score >> median → |anomaly − median| is just as large as LTR alone
  EVT on the B-LTR distribution still correctly flags the anomaly.

The only risk: occasional very STABLE periods in normal data get mild B-LTR
scores (their LTR ≈ 0 → |0 − median| is moderate). EVT should set the
threshold above this natural variation. In practice, anomalies are always the
MOST EXTREME deviations in either direction.

Variants
--------
  bltr_k5       : |ltr_k5 − median(ltr_k5)| + EVT
  bltr_k5_k30   : |ltr_k5 − median| + |ltr_k30 − median| (multi-scale)
  bltr_add_raw  : raw LTR k5 + B-LTR k5  (preserves upward signal, adds downward)

All checkpoints already exist — NO re-inference.

Usage
-----
  python experiments/compare_bltr_scoring.py
"""

from __future__ import annotations

import ast
import os
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import genpareto

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from evaluation.evaluate import evaluate_intervals

print(f"Project root: {PROJECT_ROOT}")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV    = os.path.join(DATA_DIR, "anomalies.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_bltr_scoring")
os.makedirs(RESULTS_DIR, exist_ok=True)

LTR_CSV = os.path.join(
    PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "comparison.csv"
)

def _first_existing(*candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[-1]

K5_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)
K30_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
)

# ── Signal lists ──────────────────────────────────────────────────────────────
SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
             "T-13","D-16","T-8","P-14","D-14"]

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

def _load_nab_sigs(gt):
    nab_dir = os.path.join(DATA_DIR, "realAWSCloudwatch")
    return sorted(f[:-4] for f in os.listdir(nab_dir)
                  if f.endswith(".csv") and gt.get(f[:-4]))

gt = _load_gt()
NAB_SIGS = _load_nab_sigs(gt)

DATASET_CONFIGS = {
    "NAB":  {"dir": os.path.join(DATA_DIR, "realAWSCloudwatch"), "sigs": NAB_SIGS},
    "SMAP": {"dir": os.path.join(DATA_DIR, "SMAP"),             "sigs": SMAP_SIGS},
    "MSL":  {"dir": os.path.join(DATA_DIR, "MSL"),              "sigs": MSL_SIGS},
}

# ── EVT ───────────────────────────────────────────────────────────────────────
EVT_Q   = 0.90
EVT_FPR = 0.01

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

# ── B-LTR core ────────────────────────────────────────────────────────────────
def bidirectional_ltr(scores: np.ndarray) -> np.ndarray:
    """
    |score_t - median(scores)|

    Captures deviations from the signal's typical LTR level in BOTH directions:
    - Spikes/novelties (LTR >> median) → high B-LTR score
    - Stuck sensors / homogenization (LTR << median) → high B-LTR score

    Using MEDIAN (not mean) as center because:
    - Median is robust to outliers (anomalies don't shift the center)
    - If anomalies have very low LTR (inverted), mean shifts downward →
      normal windows appear to have high B-LTR scores (false positives)
    - Median stays at the typical normal LTR level regardless of anomaly content
    """
    med = np.median(scores)
    return np.abs(scores - med)

# ── Checkpoint loading ────────────────────────────────────────────────────────
def load_k5(ds, sig):
    p = os.path.join(K5_CKPT_DIR, f"{ds}__{sig}__ltr.pkl")
    if not os.path.exists(p):
        return None, None
    c = pickle.load(open(p, "rb"))
    return c["scores"], c["timestamps"]

def load_k30(ds, sig):
    p = os.path.join(K30_CKPT_DIR, f"{ds}__{sig}__ltr_k30.pkl")
    if not os.path.exists(p):
        return None, None
    return pickle.load(open(p, "rb"))["scores"], None

# Load LTR baseline
ltr_baseline = {}
if os.path.exists(LTR_CSV):
    for _, r in pd.read_csv(LTR_CSV).iterrows():
        ltr_baseline[(r["dataset"], r["signal"])] = r.get("add_k5_k30_f1", float("nan"))

# ── Main loop ─────────────────────────────────────────────────────────────────
rows = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}  ({len(cfg['sigs'])} signals)")
    print(f"{'='*60}")

    for sig in cfg["sigs"]:
        gt_ivs = gt.get(sig, [])
        if not gt_ivs:
            continue

        k5_scores, k5_ts = load_k5(ds_name, sig)
        if k5_scores is None:
            print(f"  SKIP {sig} (no k5 checkpoint)")
            continue

        k30_scores, _ = load_k30(ds_name, sig)

        T  = len(k5_scores)
        ts = k5_ts[:T]

        # ── Condition 1: B-LTR k5 alone ──────────────────────────────────────
        bltr5 = bidirectional_ltr(k5_scores)
        f1_bltr5 = evaluate_intervals(gt_ivs, evt_detect(bltr5, ts))["F1"]

        # ── Condition 2: B-LTR k5 + B-LTR k30 (multi-scale bidirectional) ────
        f1_bltr5_30 = float("nan")
        if k30_scores is not None:
            Tk = min(T, len(k30_scores))
            bltr30 = bidirectional_ltr(k30_scores[:Tk])
            s = normalize_01(bltr5[:Tk]) + normalize_01(bltr30)
            f1_bltr5_30 = evaluate_intervals(gt_ivs, evt_detect(s, ts[:Tk]))["F1"]

        # ── Condition 3: raw LTR k5 + B-LTR k5 (preserves upward, adds downward)
        s_raw_bi = normalize_01(k5_scores) + normalize_01(bltr5)
        f1_raw_bi = evaluate_intervals(gt_ivs, evt_detect(s_raw_bi, ts))["F1"]

        # ── Condition 4: B-LTR multi-scale + raw k5 (best of all) ────────────
        f1_bltr_full = float("nan")
        if k30_scores is not None:
            Tk = min(T, len(k30_scores))
            bltr30 = bidirectional_ltr(k30_scores[:Tk])
            s = (normalize_01(bltr5[:Tk]) +
                 normalize_01(bltr30) +
                 normalize_01(k5_scores[:Tk]))
            f1_bltr_full = evaluate_intervals(gt_ivs, evt_detect(s, ts[:Tk]))["F1"]

        f1_ltr = ltr_baseline.get((ds_name, sig), float("nan"))

        row = {
            "dataset": ds_name, "signal": sig,
            "ltr_add_f1":   f1_ltr,
            "bltr5_f1":     f1_bltr5,
            "bltr5_30_f1":  f1_bltr5_30,
            "raw_bi_f1":    f1_raw_bi,
            "bltr_full_f1": f1_bltr_full,
        }
        rows.append(row)

        d1 = f1_bltr5 - f1_ltr if not np.isnan(f1_ltr) else float("nan")
        print(f"  {sig:<40} ltr={f1_ltr:.3f}  bltr5={f1_bltr5:.3f}  "
              f"bltr5+30={f1_bltr5_30:.3f}  raw+bi={f1_raw_bi:.3f}  "
              f"full={f1_bltr_full:.3f}  Δ={d1:+.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

SIG_COUNTS = {"NAB": 16, "SMAP": 13, "MSL": 11}
COLS       = ["ltr_add_f1", "bltr5_f1", "bltr5_30_f1", "raw_bi_f1", "bltr_full_f1"]
COL_W      = 14

print(f"\n{'='*80}")
print("SUMMARY — Bidirectional LTR")
print(f"{'='*80}")
print(f"{'Dataset':<10}" + "".join(f"{c:>{COL_W}}" for c in COLS))
print("-" * (10 + COL_W * len(COLS)))

all_wtd = {c: 0.0 for c in COLS}
total_n = 0

for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    avgs = [sub[c].mean() for c in COLS]
    print(f"{ds:<10}" + "".join(f"{a:>{COL_W}.4f}" for a in avgs))
    n = SIG_COUNTS[ds]
    for c, a in zip(COLS, avgs):
        all_wtd[c] += a * n
    total_n += n

print("-" * (10 + COL_W * len(COLS)))
print(f"{'ALL (wtd)':<10}" +
      "".join(f"{all_wtd[c]/total_n:>{COL_W}.4f}" for c in COLS))
print(f"\nResults → {RESULTS_DIR}/comparison.csv")

# ── Spotlight ─────────────────────────────────────────────────────────────────
SPOTLIGHT = [
    ("MSL",  "D-15",  "inv  sep=2.25"),
    ("SMAP", "T-1",   "inv  sep=1.20"),
    ("NAB",  "iio_us-east-1_i-a2eb1cd9_NetworkIn", "sep=2.60 thr?"),
    ("SMAP", "F-3",   "blind"),
    ("MSL",  "T-13",  "blind sep=0.96"),
]
print(f"\n{'='*70}")
print("Spotlight")
print(f"{'='*70}")
print(f"{'Signal':<44} ltr  bltr5 b5+30 r+bi  full")
print("-" * 70)
for ds, sig, note in SPOTLIGHT:
    sub = results_df[(results_df["dataset"] == ds) & (results_df["signal"] == sig)]
    if sub.empty:
        continue
    r = sub.iloc[0]
    print(f"{ds}/{sig:<40} "
          f"{r['ltr_add_f1']:.3f} {r['bltr5_f1']:.3f} "
          f"{r['bltr5_30_f1']:.3f} {r['raw_bi_f1']:.3f} "
          f"{r['bltr_full_f1']:.3f}  {note}")
