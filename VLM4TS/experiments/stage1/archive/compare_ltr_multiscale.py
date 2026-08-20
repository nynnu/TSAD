"""
Multi-Scale LTR: k=5 (local) + k=30 (medium) — Max Fusion
===========================================================

Problem
-------
LTR k=5 misses long-duration anomalies (e.g. SMAP E-series).
When an anomaly spans > k windows, ALL k neighbors are anomalous
→ reference IS the anomaly → dissimilarity ≈ 0 → miss.

Solution
--------
Run LTR at two temporal scales and fuse with max:

  s_k5   : LTR k=5  — captures short/contextual anomalies
            reference = 5 nearest temporal neighbors
  s_k30  : LTR k=30 — captures medium/long-duration anomalies
            reference = 30 nearest temporal neighbors
            k=30 covers 30×56 = 1680 timesteps of context

  s_max  = max(norm(s_k5), norm(s_k30))   ← main hypothesis
            fires if EITHER scale detects something

Why max, not add?
  Short anomaly : s_k5 high, s_k30 low  → max = s_k5  ✅
  Long anomaly  : s_k5 ≈ 0, s_k30 high → max = s_k30 ✅
  Normal        : both low              → max low      ✅
  Add would mix both scales indiscriminately and amplify noise.

Ablation
--------
  baseline        : LTR k=5  + alpha=0.01     (existing benchmark)
  k5_evt          : LTR k=5  + EVT            (EVT baseline)
  k30_evt         : LTR k=30 + EVT            (k=30 alone)
  max_k5_k30      : max(norm(k5), norm(k30)) + EVT   ← main
  add_k5_k30      : norm(k5) + norm(k30) + EVT       (add comparison)
  max_recon       : max(norm(k5), norm(k30)) + 0.3*norm(Recon) + EVT

Checkpoints
-----------
  k=5  (reused) : results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl
  k=30 (new)    : results_ltr_multiscale/checkpoints/{DS}__{sig}__ltr_k30.pkl
  Recon (reused): results_ltr_recon_additive/checkpoints/{DS}__{sig}__recon.pkl

Usage
-----
  python experiments/compare_ltr_multiscale.py
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

RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results_ltr_multiscale")
K30_CKPT_DIR  = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(K30_CKPT_DIR, exist_ok=True)

# k=5 checkpoints (reuse MGMR)
_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_CKPT_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)),
                   _K5_CANDIDATES[-1])
os.makedirs(K5_CKPT_DIR, exist_ok=True)
print(f"LTR k=5  ckpt : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR))} files)")

# Recon checkpoints (reuse additive experiment)
_RECON_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_recon_additive", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_recon_additive", "checkpoints"),
]
RECON_CKPT_DIR = next((p for p in _RECON_CANDIDATES if os.path.isdir(p)),
                      _RECON_CANDIDATES[-1])
print(f"Recon    ckpt : {RECON_CKPT_DIR}  ({len(os.listdir(RECON_CKPT_DIR))} files)")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

K_LOCAL  = 5    # existing LTR
K_MEDIUM = 30   # new — covers 30×56 = 1680 timesteps of context

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


def max_fuse(*arrays: np.ndarray) -> np.ndarray:
    """Element-wise max over normalized arrays (length-aligned)."""
    T = min(len(a) for a in arrays)
    return np.maximum.reduce([normalize_01(a[:T]) for a in arrays])


def add_fuse(*arrays: np.ndarray) -> np.ndarray:
    """Sum of normalized arrays (length-aligned)."""
    T = min(len(a) for a in arrays)
    return sum(normalize_01(a[:T]) for a in arrays)


# ── LTR detector (parametric k) ───────────────────────────────────────────────

class LTR_Detector(ViT4TS_MAE):
    """LTR with configurable k. Inherits ViT4TS_MAE; only _run_inference differs."""

    def __init__(self, *args, local_k: int = 5, min_ref: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_k = local_k
        self.min_ref = min_ref

    def _run_inference(self, results_dir: str, base_series_id: str):
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_series_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return None

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
                l_ref, _ = get_local_reference(large_embeds, i, self.local_k, self.min_ref)
                m_ref, _ = get_local_reference(mid_embeds,   i, self.local_k, self.min_ref)
                p_ref, _ = get_local_reference(patch_embeds, i, self.local_k, self.min_ref)

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


def _ivs_alpha(scores: np.ndarray, timestamps: np.ndarray, alpha: float = 0.01) -> list:
    from models.model_utils import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


# ── Model init ────────────────────────────────────────────────────────────────
print("\n[INFO] Loading models ...")
det_k5  = LTR_Detector(**BASE_PARAMS, local_k=K_LOCAL,  min_ref=5)
det_k30 = LTR_Detector(**BASE_PARAMS, local_k=K_MEDIUM, min_ref=5)
# Share the same ViT backbone weights — both models loaded the same checkpoint
print(f"  LTR k={K_LOCAL}  : local  reference ({K_LOCAL*56} ts context)")
print(f"  LTR k={K_MEDIUM} : medium reference ({K_MEDIUM*56} ts context)")

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

        k5_ckpt    = os.path.join(K5_CKPT_DIR,    f"{ds_name}__{sig}__ltr.pkl")
        k30_ckpt   = os.path.join(K30_CKPT_DIR,   f"{ds_name}__{sig}__ltr_k30.pkl")
        recon_ckpt = os.path.join(RECON_CKPT_DIR,  f"{ds_name}__{sig}__recon.pkl")

        # ── s_k5 ──────────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(k5_ckpt):
                c          = pickle.load(open(k5_ckpt, "rb"))
                s_k5       = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [k=5 ] cache  ({time.time()-t0:.1f}s)")
            else:
                s_k5, timestamps = det_k5.predict_scores(data)
                pickle.dump({"scores": s_k5, "timestamps": timestamps},
                            open(k5_ckpt, "wb"))
                print(f"    [k=5 ] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [k=5 ] ERROR: {e}"); s_k5 = timestamps = None

        # ── s_k30 ─────────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(k30_ckpt):
                c    = pickle.load(open(k30_ckpt, "rb"))
                s_k30 = c["scores"]
                print(f"    [k=30] cache  ({time.time()-t0:.1f}s)")
            else:
                s_k30, _ = det_k30.predict_scores(data)
                pickle.dump({"scores": s_k30}, open(k30_ckpt, "wb"))
                print(f"    [k=30] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [k=30] ERROR: {e}"); s_k30 = None

        if s_k5 is None or s_k30 is None:
            continue

        # ── s_recon (optional) ────────────────────────────────────
        s_recon = None
        if os.path.exists(recon_ckpt):
            try:
                s_recon = pickle.load(open(recon_ckpt, "rb"))["scores"]
            except Exception:
                pass

        T       = min(len(s_k5), len(s_k30))
        ts_trim = timestamps[:T]

        # ── baseline: k=5 + alpha=0.01 ────────────────────────────
        try:
            ivs = _ivs_alpha(s_k5, timestamps)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["baseline_f1"] = f1
        print(f"    baseline      F1={f1:.4f}  ivs={len(ivs)}")

        # ── k5 + EVT ──────────────────────────────────────────────
        try:
            ivs = evt_detect(s_k5[:T], ts_trim)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["k5_evt_f1"] = f1
        print(f"    k5_evt        F1={f1:.4f}  ivs={len(ivs)}")

        # ── k30 + EVT ─────────────────────────────────────────────
        try:
            ivs = evt_detect(s_k30[:T], ts_trim)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["k30_evt_f1"] = f1
        delta = f1 - sig_row["baseline_f1"]
        print(f"    k30_evt       F1={f1:.4f}  ivs={len(ivs)}  ({delta:+.4f} vs base)")

        # ── max(k5, k30) + EVT  ← main ───────────────────────────
        try:
            s_max = max_fuse(s_k5, s_k30)
            ivs   = evt_detect(s_max, ts_trim)
            f1    = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["max_k5_k30_f1"] = f1
        delta = f1 - sig_row["baseline_f1"]
        print(f"    max(k5,k30)   F1={f1:.4f}  ivs={len(ivs)}  ({delta:+.4f} vs base)")

        # ── add(k5, k30) + EVT ────────────────────────────────────
        try:
            s_add = add_fuse(s_k5, s_k30)
            ivs   = evt_detect(s_add, ts_trim)
            f1    = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["add_k5_k30_f1"] = f1
        print(f"    add(k5,k30)   F1={f1:.4f}  ivs={len(ivs)}")

        # ── max(k5, k30) + 0.3*Recon + EVT ───────────────────────
        # max_fuse already returns values in [0,1].
        # Add Recon on top directly (same pattern as additive experiment).
        if s_recon is not None:
            try:
                s_max = max_fuse(s_k5, s_k30)           # [0, 1]
                s_mr  = s_max + 0.3 * normalize_01(s_recon[:T])  # [0, 1.3]
                ivs   = evt_detect(s_mr, ts_trim)
                f1    = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["max_recon_f1"] = f1
            delta = f1 - sig_row["baseline_f1"]
            print(f"    max+Recon     F1={f1:.4f}  ivs={len(ivs)}  ({delta:+.4f} vs base)")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "k5_evt_f1", "k30_evt_f1",
            "max_k5_k30_f1", "add_k5_k30_f1", "max_recon_f1"]
LABELS   = {
    "baseline_f1":   "LTR_k5",
    "k5_evt_f1":     "k5+EVT",
    "k30_evt_f1":    "k30+EVT",
    "max_k5_k30_f1": "max(k5,k30)",
    "add_k5_k30_f1": "add(k5,k30)",
    "max_recon_f1":  "max+Recon",
}
COL_W = 13
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v: object) -> str:
    try:
        f = float(v)
        return "   NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "   NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — Multi-Scale LTR ===")
    print(f"{'='*SEP}")
    print(f"{'Signal':<36}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in ALL_COLS
                                       if c in df.columns))
    print("-" * SEP)
    for _, r in sub.iterrows():
        row = f"{r['signal']:<36}"
        for c in ALL_COLS:
            if c in df.columns:
                row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)
    print("-" * SEP)
    avg = f"{'AVG':<36}"
    for c in ALL_COLS:
        if c in df.columns:
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

base  = results_df["baseline_f1"].dropna().mean()
print(f"baseline (LTR k=5, alpha=0.01) : {base:.4f}")

for c, lbl in [("k30_evt_f1",    "LTR k=30 alone       "),
               ("max_k5_k30_f1", "max(k5, k30)          "),
               ("add_k5_k30_f1", "add(k5, k30)          "),
               ("max_recon_f1",  "max(k5,k30)+0.3·Recon ")]:
    if c in results_df.columns:
        v = results_df[c].dropna().mean()
        print(f"{lbl}: {v:.4f}  ({v-base:+.4f})")

# SMAP E-series spotlight
print("\nSMAP E-series  (long-duration anomaly targets):")
smap_e = results_df[
    (results_df["dataset"] == "SMAP") &
    (results_df["signal"].str.startswith("E"))
]
if not smap_e.empty:
    print(f"{'Signal':<8}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid_cols))
    for _, r in smap_e.iterrows():
        row = f"{r['signal']:<8}"
        for c in valid_cols:
            row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)

print(f"\nResults saved → {out_csv}")
