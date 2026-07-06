"""Spectrogram + PatchCore vs LTR baselines
==========================================

Hypothesis
----------
Line plot leaves 85% of MAE patches empty (background).
Spectrogram fills every patch with power(time, frequency) content.
PatchCore memory-bank scoring uses all 196 patch features per window
instead of collapsing to one vector (LTR).

Together: denser visual encoding + richer patch-level scoring.

Ablation columns
----------------
  baseline_f1     : line + LTR k=5  + EVT          [cached mgmr]
  add_k5k30_f1    : add(k5, k30)    + EVT          [cached multiscale]
  spec_pc_f1      : spectrogram + PatchCore + EVT  [new]

Usage
-----
  python experiments/compare_spectrogram_patchcore.py
"""
from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import genpareto

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy", "transformers", "--quiet"],
    check=True,
)

import torch
from preprocessing.data_utils import orion_to_internal
from preprocessing.spectrogram_encoder import render_spectrogram_windows
from models.mae_patchcore import MAE_PatchCore, scores_to_timeseries
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_SIZE   = int(WINDOW_SIZE / 4.0)   # 56
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

# PatchCore
N_REF_RATIO = 0.20   # first 20% of windows as normal reference
TOP_P       = 0.10   # score = mean of top 10% patch distances

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_spec_patchcore")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# Baseline cache paths
_K5_CANDS = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_DIR = next((p for p in _K5_CANDS if os.path.isdir(p)), _K5_CANDS[-1])

_K30_CANDS = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
]
K30_DIR = next((p for p in _K30_CANDS if os.path.isdir(p)), _K30_CANDS[-1])

print(f"k=5  cache : {K5_DIR}  ({len(os.listdir(K5_DIR)) if os.path.isdir(K5_DIR) else 0} files)")
print(f"k=30 cache : {K30_DIR}  ({len(os.listdir(K30_DIR)) if os.path.isdir(K30_DIR) else 0} files)")
print(f"ckpt dir   : {CKPT_DIR}")


# ── EVT ───────────────────────────────────────────────────────────────────────
def evt_threshold(scores: np.ndarray) -> float:
    u  = float(np.percentile(scores, EVT_Q_INIT * 100))
    ex = scores[scores > u] - u
    fb = float(np.percentile(scores, (1 - EVT_FPR) * 100))
    if len(ex) < 10:
        return fb
    try:
        c, _, sc = genpareto.fit(ex, floc=0)
        p   = min(EVT_FPR / max(1 - EVT_Q_INIT, 1e-9), 1 - 1e-9)
        thr = u + max(0., genpareto.ppf(1 - p, c, loc=0, scale=sc))
        return thr if u <= thr <= scores.max() else fb
    except Exception:
        return fb


def evt_detect(scores: np.ndarray, timestamps: np.ndarray) -> list:
    if scores.max() - scores.min() < 1e-8:
        return []
    flags = scores > evt_threshold(scores)
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


def normalize_01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-8 else (a - lo) / (hi - lo)


def f1_score(detected: list, gt_ivs: list) -> float:
    return evaluate_intervals(gt_ivs, detected)["F1"]


# ── GT loading ────────────────────────────────────────────────────────────────
def load_gt(path: str) -> dict:
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt


def load_cached_scores(ds: str, sig: str, directory: str, suffix: str):
    p = os.path.join(directory, f"{ds}__{sig}__{suffix}.pkl")
    if not os.path.exists(p):
        return None, None
    try:
        c = pickle.load(open(p, "rb"))
        return c.get("scores"), c.get("timestamps")
    except Exception:
        return None, None


# ── Dataset config ─────────────────────────────────────────────────────────────
gt = load_gt(ANOM_CSV)

nab_files = sorted(f for f in os.listdir(NAB_DIR) if f.endswith(".csv"))
NAB_SIGS  = [f[:-4] for f in nab_files if gt.get(f[:-4])]
SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
             "T-13","D-16","T-8","P-14","D-14"]

DATASET_CONFIGS = {
    "NAB":  {"dir": NAB_DIR,  "channels": NAB_SIGS},
    "SMAP": {"dir": SMAP_DIR, "channels": SMAP_SIGS},
    "MSL":  {"dir": MSL_DIR,  "channels": MSL_SIGS},
}

# ── Lazy model ────────────────────────────────────────────────────────────────
_patchcore = None

def get_patchcore() -> MAE_PatchCore:
    global _patchcore
    if _patchcore is None:
        _patchcore = MAE_PatchCore(
            device=DEVICE,
            n_ref_ratio=N_REF_RATIO,
            top_p=TOP_P,
        )
    return _patchcore


# ── Main loop ──────────────────────────────────────────────────────────────────
rows: list = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*72}")
    print(f"Dataset: {ds_name}  ({len(cfg['channels'])} signals)")
    print(f"{'='*72}")

    for sig in cfg["channels"]:
        csv_path = os.path.join(cfg["dir"], f"{sig}.csv")
        gt_ivs   = gt.get(sig, [])
        if not os.path.exists(csv_path) or not gt_ivs:
            print(f"  SKIP {sig}")
            continue

        print(f"\n  [{sig}]")
        data            = pd.read_csv(csv_path)
        values, timestamps = orion_to_internal(data)
        T               = len(values)
        sig_row         = {"dataset": ds_name, "signal": sig}

        # ── Baseline: line + LTR k=5 ──────────────────────────────────────
        sc5, ts5 = load_cached_scores(ds_name, sig, K5_DIR, "ltr")
        if sc5 is not None and ts5 is not None:
            T5  = min(len(sc5), len(ts5))
            bl  = f1_score(evt_detect(sc5[:T5], ts5[:T5]), gt_ivs)
            sig_row["baseline_f1"] = bl
            print(f"    baseline (line, LTR k5)      F1={bl:.4f}")

        # ── add(k5, k30) ──────────────────────────────────────────────────
        sc30, ts30 = load_cached_scores(ds_name, sig, K30_DIR, "ltr_k30")
        if sc5 is not None and sc30 is not None and ts5 is not None:
            T_c   = min(len(sc5), len(sc30), len(ts5))
            fused = normalize_01(sc5[:T_c]) + normalize_01(sc30[:T_c])
            f1_add = f1_score(evt_detect(fused, ts5[:T_c]), gt_ivs)
            sig_row["add_k5k30_f1"] = f1_add
            print(f"    add(k5,k30)                  F1={f1_add:.4f}")

        # ── Spectrogram + PatchCore ────────────────────────────────────────
        pc_ckpt  = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__spec_pc.pkl")
        pc_scores = None

        if os.path.exists(pc_ckpt):
            try:
                pc_scores = pickle.load(open(pc_ckpt, "rb"))["scores"]
                print(f"    [spec+PC]  cache")
            except Exception:
                pass

        if pc_scores is None:
            try:
                t0   = time.time()
                imgs = render_spectrogram_windows(values, WINDOW_SIZE, STEP_SIZE)
                pc   = get_patchcore()
                win_scores = pc.fit_and_score(imgs)
                pc_scores  = scores_to_timeseries(win_scores, T, WINDOW_SIZE, STEP_SIZE)
                pickle.dump({"scores": pc_scores, "timestamps": timestamps},
                            open(pc_ckpt, "wb"))
                print(f"    [spec+PC]  computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [spec+PC]  ERROR: {e}")
                import traceback; traceback.print_exc()

        if pc_scores is not None:
            T_ts = min(len(pc_scores), len(timestamps))
            f1pc = f1_score(evt_detect(pc_scores[:T_ts], timestamps[:T_ts]), gt_ivs)
            sig_row["spec_pc_f1"] = f1pc
            delta = f1pc - sig_row.get("baseline_f1", float("nan"))
            print(f"    spec+PC                      F1={f1pc:.4f}  ({delta:+.4f})")

        rows.append(sig_row)


# ── Save ───────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

# ── Print table ────────────────────────────────────────────────────────────────
ALL_COLS = ["baseline_f1", "add_k5k30_f1", "spec_pc_f1"]
LABELS   = {"baseline_f1": "line+LTR", "add_k5k30_f1": "add(k5,k30)", "spec_pc_f1": "spec+PC"}
COL_W    = 12

def _fmt(v) -> str:
    try:
        f = float(v)
        return "    NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "    NaN"

def _print_table(ds_name: str) -> None:
    sub  = results_df[results_df["dataset"] == ds_name]
    if sub.empty:
        return
    cols = [c for c in ALL_COLS if c in results_df.columns]
    sep  = 26 + COL_W * len(cols)
    print(f"\n{'='*sep}")
    print(f"=== {ds_name} ===")
    print("%-26s" % "Signal" + "".join("%*s" % (COL_W, LABELS[c]) for c in cols))
    print("-" * sep)
    for _, r in sub.iterrows():
        row = "%-26s" % r["signal"]
        for c in cols:
            row += "%*s" % (COL_W, _fmt(r.get(c)))
        print(row)
    print("-" * sep)
    avg = "%-26s" % "AVG"
    for c in cols:
        vals = sub[c].dropna()
        avg += "%*.4f" % (COL_W, vals.mean()) if len(vals) else "%*s" % (COL_W, "NaN")
    print(avg)

for ds_name in DATASET_CONFIGS:
    _print_table(ds_name)

# ── Weighted ALL ───────────────────────────────────────────────────────────────
cols       = [c for c in ALL_COLS if c in results_df.columns]
sep        = 26 + COL_W * len(cols)
ds_weights = {"NAB": 16, "SMAP": 13, "MSL": 11}

print(f"\n{'='*sep}")
print("=== CROSS-DATASET SUMMARY ===")
print("%-10s" % "Dataset" + "".join("%*s" % (COL_W, LABELS[c]) for c in cols))
print("-" * sep)

all_avgs: dict = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = "%-10s" % ds_name
    for c in cols:
        v = float(sub[c].mean()) if c in sub.columns and not sub[c].isna().all() else float("nan")
        row += "%*.4f" % (COL_W, v)
        all_avgs.setdefault(c, []).append((ds_name, v))
    print(row)

print("-" * sep)
overall = "%-10s" % "ALL (wt)"
for c in cols:
    vals = all_avgs.get(c, [])
    wt   = sum(ds_weights.get(ds, 0) * v for ds, v in vals if not np.isnan(v))
    tot  = sum(ds_weights.get(ds, 0) for ds, v in vals if not np.isnan(v))
    overall += "%*.4f" % (COL_W, wt / tot) if tot else "%*s" % (COL_W, "NaN")
print(overall)

print(f"\nResults saved → {out_csv}")
