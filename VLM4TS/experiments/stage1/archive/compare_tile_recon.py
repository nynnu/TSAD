"""2D Tile Reconstruction vs LTR baselines
==========================================

Hypothesis
----------
Stack 14 consecutive windows as rows in a 224x224 image (each row = 16px).
Mask the center row (target window, row 7). MAE reconstructs it from the
13 surrounding context rows.

This directly tests: "is this window consistent with its temporal neighbors?"
Normal  -> low reconstruction error
Anomaly -> high reconstruction error

Why this should work better than prior approaches
-------------------------------------------------
- Spectrogram + PatchCore: fragile fixed reference (first 20%)
- Spectrogram + PatchLTR: patch positions inconsistent across windows
- GAF diagonal recon: domain gap (GAF looks nothing like natural images)
- Tile recon: stacked waveforms look like textures -> MAE well-calibrated
              temporal context is baked INTO the image, not a separate step

Ablation columns
----------------
  baseline_f1    : line + LTR k=5      + EVT   [cached]
  add_k5k30_f1   : add(k5, k30)        + EVT   [cached, current best]
  tile_recon_f1  : 2D tile + MAE recon + EVT   [new]

Usage
-----
  python experiments/compare_tile_recon.py
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
    [sys.executable, "-m", "pip", "install", "timm", "open-clip-torch", "scipy", "transformers", "-q"],
    check=False,
)

import torch
from preprocessing.data_utils import orion_to_internal
from preprocessing.tile_encoder import render_tile_windows
from models.mae_tile_recon import MAE_TileRecon, scores_to_timeseries
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_SIZE   = int(WINDOW_SIZE / 4.0)   # 56
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_tile_recon")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

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

print(f"k=5  cache : {K5_DIR}")
print(f"k=30 cache : {K30_DIR}")
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


def load_ltr_cache(ds, sig, directory, suffix):
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

# ── Lazy model ─────────────────────────────────────────────────────────────────
_model = None

def get_model() -> MAE_TileRecon:
    global _model
    if _model is None:
        _model = MAE_TileRecon(device=DEVICE, n_iter=3, batch_size=16)
    return _model


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
        data               = pd.read_csv(csv_path)
        values, timestamps = orion_to_internal(data)
        T                  = len(values)
        sig_row            = {"dataset": ds_name, "signal": sig}

        # ── Baseline: line + LTR k=5 ─────────────────────────────────────
        sc5, ts5 = load_ltr_cache(ds_name, sig, K5_DIR, "ltr")
        if sc5 is not None and ts5 is not None:
            T5 = min(len(sc5), len(ts5))
            bl = f1_score(evt_detect(sc5[:T5], ts5[:T5]), gt_ivs)
            sig_row["baseline_f1"] = bl
            print(f"    baseline (line, LTR k5)        F1={bl:.4f}")

        # ── add(k5, k30) ─────────────────────────────────────────────────
        sc30, _ = load_ltr_cache(ds_name, sig, K30_DIR, "ltr_k30")
        if sc5 is not None and sc30 is not None and ts5 is not None:
            T_c    = min(len(sc5), len(sc30), len(ts5))
            fused  = normalize_01(sc5[:T_c]) + normalize_01(sc30[:T_c])
            f1_add = f1_score(evt_detect(fused, ts5[:T_c]), gt_ivs)
            sig_row["add_k5k30_f1"] = f1_add
            print(f"    add(k5,k30)                    F1={f1_add:.4f}")

        # ── Tile Reconstruction ───────────────────────────────────────────
        tile_ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__tile_recon.pkl")
        tile_scores = None

        if os.path.exists(tile_ckpt):
            try:
                tile_scores = pickle.load(open(tile_ckpt, "rb"))["scores"]
                print(f"    [tile_recon]  cache")
            except Exception:
                pass

        if tile_scores is None:
            try:
                t0         = time.time()
                imgs       = render_tile_windows(values, WINDOW_SIZE, STEP_SIZE)
                mdl        = get_model()
                win_scores = mdl.score_windows(imgs)
                tile_scores = scores_to_timeseries(
                    win_scores, T, WINDOW_SIZE, STEP_SIZE)
                pickle.dump({"scores": tile_scores, "timestamps": timestamps},
                            open(tile_ckpt, "wb"))
                print(f"    [tile_recon]  computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [tile_recon]  ERROR: {e}")
                import traceback; traceback.print_exc()

        if tile_scores is not None:
            T_ts = min(len(tile_scores), len(timestamps))
            f1tr = f1_score(evt_detect(tile_scores[:T_ts], timestamps[:T_ts]), gt_ivs)
            sig_row["tile_recon_f1"] = f1tr
            delta = f1tr - sig_row.get("baseline_f1", float("nan"))
            print(f"    tile_recon                     F1={f1tr:.4f}  ({delta:+.4f})")

        rows.append(sig_row)


# ── Save & print ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "add_k5k30_f1", "tile_recon_f1"]
LABELS   = {
    "baseline_f1":   "line+LTR",
    "add_k5k30_f1":  "add(k5,k30)",
    "tile_recon_f1": "tile_recon",
}
COL_W = 13


def _fmt(v):
    try:
        f = float(v)
        return "    NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "    NaN"


def _print_table(ds_name):
    sub  = results_df[results_df["dataset"] == ds_name]
    if sub.empty: return
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

# Stuck signals
print(f"\n{'='*sep}")
print("STUCK SIGNALS (baseline <= 0)")
for _, r in results_df[results_df.get("baseline_f1", pd.Series()).fillna(1) <= 0.001].iterrows():
    parts = [f"{r['dataset']}__{r['signal']:<12}"]
    for c in cols:
        parts.append(f"{LABELS[c]}={_fmt(r.get(c))}")
    print("  " + "  ".join(parts))

print(f"\nResults saved -> {out_csv}")
