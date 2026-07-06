"""STL + Mahalanobis scoring experiment.

Hypothesis
----------
LTR k-NN scoring fails for long anomalies (neighbors are also anomalous).
Mahalanobis distance from the GLOBAL feature distribution catches anomalies
regardless of duration — long anomalies are still outliers globally.

STL de-seasonalization removes cyclic variation before rendering, so MAE
features see only genuine deviations from the normal regime.

Ablation
--------
  baseline       : LTR k=5  (current, loaded from checkpoint)
  best           : add(k5, k30) (current best, loaded from checkpoint)
  mahal          : raw signal  + Mahalanobis
  stl_mahal      : STL residual + Mahalanobis  ← main hypothesis
  max_stl_k5     : max(stl_mahal, k5)          ← fusion: global + local
  max_stl_best   : max(stl_mahal, add(k5,k30)) ← fusion: global + current best

Checkpoints
-----------
  mahal     : results_stl_mahalanobis/checkpoints/{DS}__{sig}__mahal.pkl
  stl_mahal : results_stl_mahalanobis/checkpoints/{DS}__{sig}__stl_mahal.pkl
  k5        : results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl  (reused)
  k30       : results_ltr_multiscale/checkpoints/{DS}__{sig}__ltr_k30.pkl (reused)
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

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy", "transformers",
     "statsmodels", "scikit-learn", "--quiet"],
    check=True,
)

import torch

from models.mae_mahalanobis import MAE_Mahalanobis
from evaluation.evaluate import evaluate_intervals
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_stl_mahalanobis")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

# Existing checkpoint dirs (reused — no recomputation)
_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_CKPT_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)), _K5_CANDIDATES[-1])

_K30_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
]
K30_CKPT_DIR = next((p for p in _K30_CANDIDATES if os.path.isdir(p)), _K30_CANDIDATES[-1])

print(f"k5  ckpts : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR)) if os.path.isdir(K5_CKPT_DIR) else 'MISSING'} files)")
print(f"k30 ckpts : {K30_CKPT_DIR} ({len(os.listdir(K30_CKPT_DIR)) if os.path.isdir(K30_CKPT_DIR) else 'MISSING'} files)")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE,
    window_step_ratio=STEP_RATIO,
    agg_percent=0.25,
    patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224),
    dpi=100,
    standardize=True,
    smoothing_alpha=1.0,
    alpha=0.01,
    verbose=False,
)


# ── EVT threshold ─────────────────────────────────────────────────────────────

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
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def max_fuse(*arrays: np.ndarray) -> np.ndarray:
    T = min(len(a) for a in arrays)
    return np.maximum.reduce([normalize_01(a[:T]) for a in arrays])


def add_fuse(*arrays: np.ndarray) -> np.ndarray:
    T = min(len(a) for a in arrays)
    return sum(normalize_01(a[:T]) for a in arrays)


# ── Evaluation helpers ────────────────────────────────────────────────────────

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


def _ivs_alpha(scores: np.ndarray, timestamps: np.ndarray, alpha: float = 0.01) -> list:
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


# ── Model init ────────────────────────────────────────────────────────────────
print("\n[INFO] Loading models ...")
det_mahal     = MAE_Mahalanobis(**BASE_PARAMS, use_stl=False, pca_components=50)
det_stl_mahal = MAE_Mahalanobis(**BASE_PARAMS, use_stl=True,  pca_components=50)
# Share backbone weights between instances to avoid double GPU memory
det_stl_mahal.model = det_mahal.model
print("  Mahalanobis (raw) + Mahalanobis (STL residual) ready")

# ── Dataset config ────────────────────────────────────────────────────────────
gt        = load_gt(ANOM_CSV)
nab_files = sorted(f for f in os.listdir(NAB_DIR) if f.endswith(".csv"))
NAB_SIGS  = [f[:-4] for f in nab_files if gt.get(f[:-4])]
SMAP_SIGS = [
    "D-1", "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7",
    "F-1", "F-2", "F-3", "P-1", "T-1",
]
MSL_SIGS  = [
    "P-11", "T-12", "D-15", "C-1", "F-8", "F-7",
    "T-13", "D-16", "T-8", "P-14", "D-14",
]
DATASET_CONFIGS = {
    "NAB":  {"dir": NAB_DIR,  "channels": NAB_SIGS},
    "SMAP": {"dir": SMAP_DIR, "channels": SMAP_SIGS},
    "MSL":  {"dir": MSL_DIR,  "channels": MSL_SIGS},
}

# ── Experiment loop ───────────────────────────────────────────────────────────
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
        data    = pd.read_csv(csv_path)
        sig_row = {"dataset": ds_name, "signal": sig}

        k5_ckpt       = os.path.join(K5_CKPT_DIR,  f"{ds_name}__{sig}__ltr.pkl")
        k30_ckpt      = os.path.join(K30_CKPT_DIR, f"{ds_name}__{sig}__ltr_k30.pkl")
        mahal_ckpt    = os.path.join(CKPT_DIR,     f"{ds_name}__{sig}__mahal.pkl")
        stl_mahal_ckpt= os.path.join(CKPT_DIR,     f"{ds_name}__{sig}__stl_mahal.pkl")

        # ── Load / compute k5 ────────────────────────────────────
        s_k5 = timestamps = None
        if os.path.exists(k5_ckpt):
            try:
                c          = pickle.load(open(k5_ckpt, "rb"))
                s_k5       = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [k5      ] cache")
            except Exception as e:
                print(f"    [k5      ] load error: {e}")
        if s_k5 is None:
            print("    [k5      ] MISSING — skipping signal (run compare_ltr_multiscale first)")
            continue

        # ── Load / compute k30 ───────────────────────────────────
        s_k30 = None
        if os.path.exists(k30_ckpt):
            try:
                s_k30 = pickle.load(open(k30_ckpt, "rb"))["scores"]
                print(f"    [k30     ] cache")
            except Exception as e:
                print(f"    [k30     ] load error: {e}")

        # ── Compute / load Mahalanobis (raw signal) ───────────────
        t0 = time.time()
        if os.path.exists(mahal_ckpt):
            try:
                s_mahal = pickle.load(open(mahal_ckpt, "rb"))["scores"]
                print(f"    [mahal   ] cache  ({time.time()-t0:.1f}s)")
            except Exception:
                s_mahal = None
        else:
            s_mahal = None

        if s_mahal is None:
            try:
                s_mahal, _ = det_mahal.predict_scores(data)
                pickle.dump({"scores": s_mahal}, open(mahal_ckpt, "wb"))
                print(f"    [mahal   ] computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [mahal   ] ERROR: {e}"); s_mahal = None

        # ── Compute / load STL + Mahalanobis ─────────────────────
        t0 = time.time()
        if os.path.exists(stl_mahal_ckpt):
            try:
                s_stl = pickle.load(open(stl_mahal_ckpt, "rb"))["scores"]
                print(f"    [stl_mah ] cache  ({time.time()-t0:.1f}s)")
            except Exception:
                s_stl = None
        else:
            s_stl = None

        if s_stl is None:
            try:
                s_stl, _ = det_stl_mahal.predict_scores(data)
                pickle.dump({"scores": s_stl}, open(stl_mahal_ckpt, "wb"))
                print(f"    [stl_mah ] computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [stl_mah ] ERROR: {e}"); s_stl = None

        T        = len(s_k5)
        ts_trim  = timestamps[:T]

        # ── baseline: k5 + alpha=0.01 ────────────────────────────
        try:
            f1 = _eval(_ivs_alpha(s_k5, timestamps), gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["baseline_f1"] = f1
        print(f"    baseline       F1={f1:.4f}")

        # ── current best: add(k5, k30) + EVT ─────────────────────
        if s_k30 is not None:
            try:
                f1 = _eval(evt_detect(add_fuse(s_k5, s_k30), ts_trim), gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["best_f1"] = f1
            print(f"    add(k5,k30)    F1={f1:.4f}")

        # ── mahal (raw) + EVT ─────────────────────────────────────
        if s_mahal is not None:
            try:
                f1 = _eval(evt_detect(s_mahal[:T], ts_trim), gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["mahal_f1"] = f1
            delta = f1 - sig_row["baseline_f1"]
            print(f"    mahal          F1={f1:.4f}  ({delta:+.4f})")

        # ── STL + mahal + EVT  ← main ────────────────────────────
        if s_stl is not None:
            try:
                f1 = _eval(evt_detect(s_stl[:T], ts_trim), gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["stl_mahal_f1"] = f1
            delta = f1 - sig_row["baseline_f1"]
            print(f"    stl_mahal      F1={f1:.4f}  ({delta:+.4f})")

        # ── max(stl_mahal, k5) + EVT ──────────────────────────────
        if s_stl is not None:
            try:
                f1 = _eval(evt_detect(max_fuse(s_stl, s_k5), ts_trim), gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["max_stl_k5_f1"] = f1
            delta = f1 - sig_row["baseline_f1"]
            print(f"    max(stl,k5)    F1={f1:.4f}  ({delta:+.4f})")

        # ── max(stl_mahal, add(k5,k30)) + EVT ────────────────────
        if s_stl is not None and s_k30 is not None:
            try:
                f1 = _eval(evt_detect(max_fuse(s_stl, add_fuse(s_k5, s_k30)), ts_trim), gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["max_stl_best_f1"] = f1
            delta = f1 - sig_row["baseline_f1"]
            print(f"    max(stl,best)  F1={f1:.4f}  ({delta:+.4f})")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "best_f1", "mahal_f1", "stl_mahal_f1",
            "max_stl_k5_f1", "max_stl_best_f1"]
LABELS   = {
    "baseline_f1":    "LTR_k5",
    "best_f1":        "add(k5,k30)",
    "mahal_f1":       "Mahal(raw)",
    "stl_mahal_f1":   "STL+Mahal",
    "max_stl_k5_f1":  "max(stl,k5)",
    "max_stl_best_f1":"max(stl,best)",
}
COL_W = 14
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v: object) -> str:
    try:
        f = float(v)
        return "    NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "    NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    valid = [c for c in ALL_COLS if c in df.columns]
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — STL + Mahalanobis ===")
    print(f"{'='*SEP}")
    print(f"{'Signal':<36}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid))
    print("-" * SEP)
    for _, r in sub.iterrows():
        row = f"{r['signal']:<36}"
        for c in valid:
            row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)
    print("-" * SEP)
    avg = f"{'AVG':<36}"
    for c in valid:
        vals = sub[c].dropna()
        avg += f"{vals.mean():>{COL_W}.4f}" if len(vals) else f"{'NaN':>{COL_W}}"
    print(avg)


for ds_name in DATASET_CONFIGS:
    _print_table(ds_name, results_df)


# ── Cross-dataset summary ─────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("=== CROSS-DATASET SUMMARY ===")
print(f"{'='*SEP}")
valid_cols = [c for c in ALL_COLS if c in results_df.columns]
print(f"{'Dataset':<10}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid_cols))
print("-" * SEP)

all_avgs: dict = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = f"{ds_name:<10}"
    for c in valid_cols:
        v = float(sub[c].mean()) if c in sub.columns and not sub[c].isna().all() else float("nan")
        row += f"{v:>{COL_W}.4f}"
        all_avgs.setdefault(c, []).append(v)
    print(row)

print("-" * SEP)
overall = f"{'ALL':<10}"
for c in valid_cols:
    vals = [v for v in all_avgs.get(c, []) if not np.isnan(v)]
    overall += f"{np.mean(vals):>{COL_W}.4f}" if vals else f"{'NaN':>{COL_W}}"
print(overall)


# ── Key findings ──────────────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("KEY FINDINGS")
print(f"{'='*SEP}")
base = results_df["baseline_f1"].dropna().mean()
print(f"Baseline (LTR k=5)   : {base:.4f}")
for c, lbl in [
    ("best_f1",         "add(k5,k30) [prev best]"),
    ("mahal_f1",        "Mahal (raw signal)     "),
    ("stl_mahal_f1",    "STL + Mahal            "),
    ("max_stl_k5_f1",   "max(STL+Mahal, k5)    "),
    ("max_stl_best_f1", "max(STL+Mahal, best)  "),
]:
    if c in results_df.columns:
        v = results_df[c].dropna().mean()
        print(f"{lbl}: {v:.4f}  ({v-base:+.4f})")

# SMAP F-series spotlight (stuck-at-zero targets)
print("\nSMAP F-series (long-anomaly targets, currently stuck at F1=0):")
smap_f = results_df[
    (results_df["dataset"] == "SMAP") &
    (results_df["signal"].str.startswith("F"))
]
if not smap_f.empty:
    valid = [c for c in ALL_COLS if c in results_df.columns]
    print(f"{'Signal':<8}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid))
    for _, r in smap_f.iterrows():
        row = f"{r['signal']:<8}"
        for c in valid:
            row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)

print(f"\nResults saved → {out_csv}")
