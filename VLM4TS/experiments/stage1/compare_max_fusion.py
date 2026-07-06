"""
compare_max_fusion.py
=====================
MAX fusion of complementary scorers (zero new inference).

Structural finding from GMD + CMR experiments
----------------------------------------------
LTR and GMD/CMR are ORTHOGONAL detectors:
  LTR:     good at CONTEXTUAL anomalies (transient spikes, local deviations)
           fails for SUSTAINED anomalies when local neighbors are also anomalous
  GMD/CMR: good at GEOMETRIC anomalies (global distribution outliers)
           fails for CONTEXTUAL anomalies when normal data is multimodal

Why MAX, not ADD
----------------
ADD blends noise from both signals. When GMD fails on NAB (normal windows
appear as geometric outliers due to multimodal CPU utilization), adding GMD
injects noise into LTR's clean NAB detection → threshold rises → misses.

MAX: score_t = max(norm(ltr_t), norm(gmd_t))
  - NAB anomaly: LTR score near 1.0 → max = LTR ✓ (GMD's noise is lower)
  - D-15 anomaly: GMD score near 1.0, LTR near 0.0 → max = GMD ✓
  - Normal window: both low → max is low ✓
  - GMD false-positive normal window: GMD moderate, LTR low → max = moderate
    → EVT threshold accounts for this in the distribution

Conditions (all from existing checkpoints, NO re-inference)
-----------------------------------------------------------
  ltr_add           : add(k5, k30) — current best baseline
  max_ltr_gmd       : max(norm(ltr_k5+k30), norm(gmd))      ← main hypothesis
  max_ltr_cmr       : max(norm(ltr_k5+k30), norm(cmr))
  add_ltr_gmd       : add(norm(ltr_k5+k30), norm(gmd))      ← control for max vs add
  max_all           : max(norm(ltr_k5+k30), norm(gmd), norm(cmr))

Usage
-----
  python experiments/compare_max_fusion.py
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_max_fusion")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
GMD_CKPT_DIR = os.path.join(PROJECT_ROOT, "results_gmd_scoring", "checkpoints")
CMR_CKPT_DIR = os.path.join(PROJECT_ROOT, "results_cmr_scoring", "checkpoints")

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
    "NAB":  {"sigs": NAB_SIGS},
    "SMAP": {"sigs": SMAP_SIGS},
    "MSL":  {"sigs": MSL_SIGS},
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

def window_scores_to_ts(window_scores, T_full, window_size=224, step_size=56):
    aligned = np.zeros(T_full)
    counts  = np.zeros(T_full)
    for i, s in enumerate(window_scores):
        start = i * step_size
        end   = min(start + window_size, T_full)
        aligned[start:end] += s
        counts[start:end]  += 1.0
    return aligned / np.maximum(counts, 1.0)

# ── Checkpoint loading ────────────────────────────────────────────────────────
def load_k5(ds, sig):
    p = os.path.join(K5_CKPT_DIR, f"{ds}__{sig}__ltr.pkl")
    if not os.path.exists(p): return None, None
    c = pickle.load(open(p, "rb"))
    return c["scores"], c["timestamps"]

def load_k30(ds, sig):
    p = os.path.join(K30_CKPT_DIR, f"{ds}__{sig}__ltr_k30.pkl")
    if not os.path.exists(p): return None
    return pickle.load(open(p, "rb"))["scores"]

def load_gmd(ds, sig):
    p = os.path.join(GMD_CKPT_DIR, f"{ds}__{sig}__gmd.pkl")
    if not os.path.exists(p): return None
    return pickle.load(open(p, "rb"))["scores"]

def load_cmr(ds, sig):
    p = os.path.join(CMR_CKPT_DIR, f"{ds}__{sig}__cmr.pkl")
    if not os.path.exists(p): return None
    return pickle.load(open(p, "rb"))["scores"]

# ── Main loop ─────────────────────────────────────────────────────────────────
rows = []
LTR_BASELINE = {
    "NAB": 0.7687, "SMAP": 0.5274, "MSL": 0.5333
}

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

    for sig in cfg["sigs"]:
        gt_ivs = gt.get(sig, [])
        if not gt_ivs:
            continue

        k5_scores, ts = load_k5(ds_name, sig)
        if k5_scores is None:
            print(f"  SKIP {sig}")
            continue

        T = len(k5_scores)
        ts = ts[:T]

        # ── Compute add(k5, k30) = current best LTR ──────────────────────────
        k30_scores = load_k30(ds_name, sig)
        if k30_scores is not None:
            Tk = min(T, len(k30_scores))
            ltr_add = normalize_01(k5_scores[:Tk]) + normalize_01(k30_scores[:Tk])
            ts_ltr  = ts[:Tk]
        else:
            ltr_add = normalize_01(k5_scores)
            ts_ltr  = ts
            Tk      = T

        f1_ltr_add = evaluate_intervals(gt_ivs, evt_detect(ltr_add, ts_ltr))["F1"]

        # ── Load GMD and CMR (T_full aligned) ─────────────────────────────────
        gmd_scores = load_gmd(ds_name, sig)
        cmr_scores = load_cmr(ds_name, sig)

        # ── Condition: max(ltr_add, gmd) ─────────────────────────────────────
        f1_max_ltr_gmd = float("nan")
        if gmd_scores is not None:
            Tg = min(Tk, len(gmd_scores))
            s  = np.maximum(normalize_01(ltr_add[:Tg]),
                            normalize_01(gmd_scores[:Tg]))
            f1_max_ltr_gmd = evaluate_intervals(gt_ivs, evt_detect(s, ts_ltr[:Tg]))["F1"]

        # ── Condition: max(ltr_add, cmr) ─────────────────────────────────────
        f1_max_ltr_cmr = float("nan")
        if cmr_scores is not None:
            Tc = min(Tk, len(cmr_scores))
            s  = np.maximum(normalize_01(ltr_add[:Tc]),
                            normalize_01(cmr_scores[:Tc]))
            f1_max_ltr_cmr = evaluate_intervals(gt_ivs, evt_detect(s, ts_ltr[:Tc]))["F1"]

        # ── Condition: add(ltr_add, gmd) — control for max vs add ────────────
        f1_add_ltr_gmd = float("nan")
        if gmd_scores is not None:
            s  = normalize_01(ltr_add[:Tg]) + normalize_01(gmd_scores[:Tg])
            f1_add_ltr_gmd = evaluate_intervals(gt_ivs, evt_detect(s, ts_ltr[:Tg]))["F1"]

        # ── Condition: max(ltr_add, gmd, cmr) ────────────────────────────────
        f1_max_all = float("nan")
        if gmd_scores is not None and cmr_scores is not None:
            Tall = min(Tg, Tc)
            s    = np.maximum(np.maximum(normalize_01(ltr_add[:Tall]),
                                         normalize_01(gmd_scores[:Tall])),
                               normalize_01(cmr_scores[:Tall]))
            f1_max_all = evaluate_intervals(gt_ivs, evt_detect(s, ts_ltr[:Tall]))["F1"]

        row = {
            "dataset": ds_name, "signal": sig,
            "ltr_add_f1":      f1_ltr_add,
            "max_ltr_gmd_f1":  f1_max_ltr_gmd,
            "max_ltr_cmr_f1":  f1_max_ltr_cmr,
            "add_ltr_gmd_f1":  f1_add_ltr_gmd,
            "max_all_f1":      f1_max_all,
        }
        rows.append(row)

        delta = f1_max_ltr_gmd - f1_ltr_add if not np.isnan(f1_max_ltr_gmd) else float("nan")
        print(f"  {sig:<40} ltr={f1_ltr_add:.3f}  "
              f"max_gmd={f1_max_ltr_gmd:.3f}  max_cmr={f1_max_ltr_cmr:.3f}  "
              f"add_gmd={f1_add_ltr_gmd:.3f}  max_all={f1_max_all:.3f}  "
              f"Δmax_gmd={delta:+.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

SIG_COUNTS = {"NAB": 16, "SMAP": 13, "MSL": 11}
COLS       = ["ltr_add_f1", "max_ltr_gmd_f1", "max_ltr_cmr_f1",
              "add_ltr_gmd_f1", "max_all_f1"]
COL_W      = 16

print(f"\n{'='*85}")
print("SUMMARY — MAX fusion of LTR + GMD/CMR")
print(f"{'='*85}")
print(f"{'Dataset':<10}" + "".join(f"{c:>{COL_W}}" for c in COLS))
print("-" * (10 + COL_W * len(COLS)))

all_wtd = {c: 0.0 for c in COLS}
total_n = 0

for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty: continue
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

# ── Critical signals spotlight ────────────────────────────────────────────────
SPOTLIGHT = [
    ("MSL",  "D-15",  "inv  sep=2.25"),
    ("SMAP", "T-1",   "inv  sep=1.20"),
    ("NAB",  "iio_us-east-1_i-a2eb1cd9_NetworkIn", "sep=2.60"),
    ("NAB",  "ec2_cpu_utilization_77c1ca",           "ltr-good"),
    ("NAB",  "ec2_disk_write_bytes_1ef3de",           "ltr-good"),
    ("MSL",  "P-11",  "gmd-good"),
    ("MSL",  "T-13",  "blind"),
]
print(f"\n{'='*85}")
print("Signal-level spotlight")
print(f"{'='*85}")
print(f"{'Signal':<46} ltr  max_g max_c add_g max_a  note")
print("-" * 85)
for ds, sig, note in SPOTLIGHT:
    sub = results_df[(results_df["dataset"] == ds) & (results_df["signal"] == sig)]
    if sub.empty: continue
    r = sub.iloc[0]
    print(f"{ds}/{sig:<42} "
          f"{r['ltr_add_f1']:.3f} "
          f"{r['max_ltr_gmd_f1']:.3f} "
          f"{r['max_ltr_cmr_f1']:.3f} "
          f"{r['add_ltr_gmd_f1']:.3f} "
          f"{r['max_all_f1']:.3f}  {note}")
