"""
SMA × Multi-Scale LTR: Spectral Alignment + k=5/k=30 Fusion
=============================================================

Motivation
----------
Two orthogonal improvements have each shown positive signal:

  add(k5, k30)    → ALL F1=0.6256  (multi-scale handles long anomalies)
  SMA β=0.75      → ALL F1=0.6056  (spectral alignment fixes stuck signals)

SMA and multi-scale address DIFFERENT failure modes:
  - Multi-scale: solves long-duration anomaly contamination of local reference
  - SMA:         solves spectral gap between rendered TS images and MAE pretraining

If their gains are independent (different signal subsets benefit), combining
them should stack. This experiment tests that directly.

Ablation table
--------------
  add_k5_k30         : add(k5, k30) + EVT              [current best = 0.6256]
  add_sma_k5_k30     : add(SMA_k5, k30) + EVT          [SMA on k5 only]
  add_k5_sma_k30     : add(k5, SMA_k30) + EVT          [SMA on k30 only]
  add_sma_k5_sma_k30 : add(SMA_k5, SMA_k30) + EVT      [SMA on both ← main]
  max_sma_k5_sma_k30 : max(SMA_k5, SMA_k30) + EVT      [max fusion variant]

SMA beta fixed at 0.75 (best overall from compare_sma_ltr.py).

Checkpoint reuse (no recompute)
---------------------------------
  k5        : results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl
  k30       : results_ltr_multiscale/checkpoints/{DS}__{sig}__ltr_k30.pkl
  SMA_k5    : results_sma_ltr/checkpoints/{DS}__{sig}__ltr_sma075.pkl

New checkpoints (SMA_k30 only)
---------------------------------
  SMA_k30   : results_sma_multiscale/checkpoints/{DS}__{sig}__ltr_k30_sma075.pkl

Usage
-----
  python experiments/compare_sma_multiscale.py
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
    ["pip", "install", "timm", "open-clip-torch", "scipy", "transformers", "--quiet"],
    check=True,
)

import torch

from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.sma_transform import SMADataset
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)
from evaluation.evaluate import evaluate_intervals
from torch.utils.data import DataLoader

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results_sma_multiscale")
NEW_CKPT_DIR   = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(NEW_CKPT_DIR, exist_ok=True)

# k5 no-SMA
_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_CKPT_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)), _K5_CANDIDATES[-1])

# k30 no-SMA
_K30_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
]
K30_CKPT_DIR = next((p for p in _K30_CANDIDATES if os.path.isdir(p)), _K30_CANDIDATES[-1])

# k5 SMA β=0.75
_SMA_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_sma_ltr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_sma_ltr", "checkpoints"),
]
SMA_K5_CKPT_DIR = next((p for p in _SMA_K5_CANDIDATES if os.path.isdir(p)), _SMA_K5_CANDIDATES[-1])

print(f"k5       : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR))} files)")
print(f"k30      : {K30_CKPT_DIR}  ({len(os.listdir(K30_CKPT_DIR))} files)")
print(f"SMA k5   : {SMA_K5_CKPT_DIR}  ({len(os.listdir(SMA_K5_CKPT_DIR))} files)")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
K_LOCAL     = 5
K_MEDIUM    = 30
MIN_REF     = 5
SMA_BETA    = 0.75
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)


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
    if (scores.max() - scores.min()) < 1e-8:
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
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def add_fuse(*arrays: np.ndarray) -> np.ndarray:
    T = min(len(a) for a in arrays)
    return sum(normalize_01(a[:T]) for a in arrays)


def max_fuse(*arrays: np.ndarray) -> np.ndarray:
    T = min(len(a) for a in arrays)
    return np.maximum.reduce([normalize_01(a[:T]) for a in arrays])


# ── SMA-aware LTR detector (parametric k) ────────────────────────────────────

class SMA_LTR_Detector(ViT4TS_MAE):
    """LTR with configurable k and optional SMA preprocessing."""

    def __init__(self, *args, local_k: int = 5, sma_beta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_k  = local_k
        self.sma_beta = sma_beta

    def _run_inference(self, results_dir: str, base_series_id: str):
        base_ds = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_series_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(base_ds) == 0:
            return None

        dataset = SMADataset(base_ds, beta=self.sma_beta) if self.sma_beta > 0 else base_ds

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.model, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = self.image_size[0]
        ph = pw = h // self.patch_size

        anomaly_maps = []
        with torch.no_grad():
            for i in range(L):
                l_ref, _ = get_local_reference(large_embeds, i, self.local_k, MIN_REF)
                m_ref, _ = get_local_reference(mid_embeds,   i, self.local_k, MIN_REF)
                p_ref, _ = get_local_reference(patch_embeds, i, self.local_k, MIN_REF)

                m_l = compute_dissimilarity_with_ref(
                    large_embeds[i].unsqueeze(0).to(self.device), l_ref.to(self.device))
                m_m = compute_dissimilarity_with_ref(
                    mid_embeds[i].unsqueeze(0).to(self.device),   m_ref.to(self.device))
                m_p = compute_dissimilarity_with_ref(
                    patch_embeds[i].unsqueeze(0).to(self.device), p_ref.to(self.device))

                m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(self.device)
                m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(self.device)
                m_p = m_p.reshape((1, ph, pw)).to(self.device)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                          nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(h, h), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)


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


def _ivs_alpha(scores, timestamps, alpha=0.01):
    from models.model_utils import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


# ── Model init ────────────────────────────────────────────────────────────────
print("\n[INFO] Loading models (shared MAE backbone)...")
det_k30_sma = SMA_LTR_Detector(**BASE_PARAMS, local_k=K_MEDIUM, sma_beta=SMA_BETA)
print(f"  SMA_k30  : k={K_MEDIUM}, beta={SMA_BETA}  (only new inference needed)")

# ── Dataset config ────────────────────────────────────────────────────────────
gt = load_gt(ANOM_CSV)

nab_files = sorted(f for f in os.listdir(NAB_DIR) if f.endswith(".csv"))
NAB_SIGS  = [f[:-4] for f in nab_files if gt.get(f[:-4])]

SMAP_SIGS = [
    "D-1", "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7",
    "F-1", "F-2", "F-3", "P-1", "T-1",
]
MSL_SIGS = [
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

        k5_ckpt      = os.path.join(K5_CKPT_DIR,      f"{ds_name}__{sig}__ltr.pkl")
        k30_ckpt     = os.path.join(K30_CKPT_DIR,     f"{ds_name}__{sig}__ltr_k30.pkl")
        sma_k5_ckpt  = os.path.join(SMA_K5_CKPT_DIR,  f"{ds_name}__{sig}__ltr_sma075.pkl")
        sma_k30_ckpt = os.path.join(NEW_CKPT_DIR,     f"{ds_name}__{sig}__ltr_k30_sma075.pkl")

        # ── Load k5 (no-SMA) ──────────────────────────────────────
        try:
            c          = pickle.load(open(k5_ckpt, "rb"))
            s_k5       = c["scores"]
            timestamps = c["timestamps"]
        except Exception as e:
            print(f"    [k5] ERROR: {e}"); continue

        # ── Load k30 (no-SMA) ─────────────────────────────────────
        try:
            s_k30 = pickle.load(open(k30_ckpt, "rb"))["scores"]
        except Exception as e:
            print(f"    [k30] ERROR: {e}"); continue

        # ── Load SMA k5 (β=0.75) ──────────────────────────────────
        try:
            s_sma_k5 = pickle.load(open(sma_k5_ckpt, "rb"))["scores"]
        except Exception as e:
            print(f"    [SMA_k5] ERROR: {e}"); s_sma_k5 = None

        # ── Compute SMA k30 (β=0.75) — only new inference ─────────
        t0 = time.time()
        try:
            if os.path.exists(sma_k30_ckpt):
                s_sma_k30 = pickle.load(open(sma_k30_ckpt, "rb"))["scores"]
                print(f"    [SMA_k30] cache  ({time.time()-t0:.1f}s)")
            else:
                s_sma_k30, _ = det_k30_sma.predict_scores(data)
                pickle.dump({"scores": s_sma_k30}, open(sma_k30_ckpt, "wb"))
                print(f"    [SMA_k30] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [SMA_k30] ERROR: {e}"); s_sma_k30 = None

        T       = min(len(s_k5), len(s_k30))
        ts_trim = timestamps[:T]

        # ── baseline: k5 alpha=0.01 ───────────────────────────────
        try:
            ivs = _ivs_alpha(s_k5, timestamps)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["baseline_f1"] = f1
        print(f"    baseline          F1={f1:.4f}")

        # ── add(k5, k30) — current best ───────────────────────────
        try:
            s = add_fuse(s_k5, s_k30)
            ivs = evt_detect(s, ts_trim[:len(s)])
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["add_k5_k30_f1"] = f1
        print(f"    add(k5,k30)       F1={f1:.4f}  ({f1-sig_row['baseline_f1']:+.4f})")

        # ── add(SMA_k5, k30) ─────────────────────────────────────
        if s_sma_k5 is not None:
            try:
                s = add_fuse(s_sma_k5, s_k30)
                ivs = evt_detect(s, ts_trim[:len(s)])
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["add_smak5_k30_f1"] = f1
            print(f"    add(SMA_k5,k30)   F1={f1:.4f}  ({f1-sig_row['baseline_f1']:+.4f})")

        # ── add(k5, SMA_k30) ─────────────────────────────────────
        if s_sma_k30 is not None:
            try:
                s = add_fuse(s_k5, s_sma_k30)
                ivs = evt_detect(s, ts_trim[:len(s)])
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["add_k5_smak30_f1"] = f1
            print(f"    add(k5,SMA_k30)   F1={f1:.4f}  ({f1-sig_row['baseline_f1']:+.4f})")

        # ── add(SMA_k5, SMA_k30) ← main ──────────────────────────
        if s_sma_k5 is not None and s_sma_k30 is not None:
            try:
                s = add_fuse(s_sma_k5, s_sma_k30)
                ivs = evt_detect(s, ts_trim[:len(s)])
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["add_sma_both_f1"] = f1
            delta = f1 - sig_row["baseline_f1"]
            print(f"    add(SMA_k5,SMA_k30) F1={f1:.4f}  ({delta:+.4f})  ← main")

        # ── max(SMA_k5, SMA_k30) ─────────────────────────────────
        if s_sma_k5 is not None and s_sma_k30 is not None:
            try:
                s = max_fuse(s_sma_k5, s_sma_k30)
                ivs = evt_detect(s, ts_trim[:len(s)])
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["max_sma_both_f1"] = f1
            print(f"    max(SMA_k5,SMA_k30) F1={f1:.4f}  ({f1-sig_row['baseline_f1']:+.4f})")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "add_k5_k30_f1", "add_smak5_k30_f1",
            "add_k5_smak30_f1", "add_sma_both_f1", "max_sma_both_f1"]
LABELS = {
    "baseline_f1":       "base(k5)",
    "add_k5_k30_f1":     "add(k5,k30)",
    "add_smak5_k30_f1":  "add(S·k5,k30)",
    "add_k5_smak30_f1":  "add(k5,S·k30)",
    "add_sma_both_f1":   "add(S·k5,S·k30)",
    "max_sma_both_f1":   "max(S·k5,S·k30)",
}
COL_W = 16
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v: object) -> str:
    try:
        f = float(v)
        return "     NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "     NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    valid = [c for c in ALL_COLS if c in df.columns]
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — SMA × Multi-Scale ===")
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
valid_cols = [c for c in ALL_COLS if c in results_df.columns]
print(f"\n{'='*SEP}")
print("=== CROSS-DATASET SUMMARY ===")
print(f"{'='*SEP}")
print(f"{'Dataset':<10}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid_cols))
print("-" * SEP)

all_avgs: dict = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = f"{ds_name:<10}"
    for c in valid_cols:
        v = float(sub[c].mean()) if c in sub.columns and not sub[c].isna().all() \
            else float("nan")
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

base     = results_df["baseline_f1"].dropna().mean()
prev_best = 0.6256
print(f"baseline (k5, alpha=0.01)    : {base:.4f}")
print(f"previous best (add(k5,k30))  : {prev_best:.4f}")

for c, lbl in [("add_k5_k30_f1",    "add(k5,k30)            "),
               ("add_smak5_k30_f1",  "add(SMA·k5, k30)       "),
               ("add_k5_smak30_f1",  "add(k5, SMA·k30)       "),
               ("add_sma_both_f1",   "add(SMA·k5, SMA·k30)   "),
               ("max_sma_both_f1",   "max(SMA·k5, SMA·k30)   ")]:
    if c in results_df.columns:
        v = np.mean([results_df[results_df["dataset"]==ds][c].mean()
                     for ds in DATASET_CONFIGS])
        marker = " ← NEW BEST" if v > prev_best else ""
        print(f"{lbl}: {v:.4f}  ({v-prev_best:+.4f} vs prev_best){marker}")

# Spotlight: signals fixed by SMA that were 0 before
print("\nSMAP F/T series (previously stuck signals):")
target = results_df[
    (results_df["dataset"] == "SMAP") &
    (results_df["signal"].isin(["F-1", "F-2", "F-3", "T-1"]))
]
if not target.empty:
    print(f"{'Signal':<8}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid_cols
                                      if c in results_df.columns))
    for _, r in target.iterrows():
        row = f"{r['signal']:<8}"
        for c in valid_cols:
            row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)

print(f"\nResults saved → {out_csv}")
