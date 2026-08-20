"""
DINOv2 LTR: Backbone Swap + Length-Aware k_medium
==================================================

Hypothesis
----------
DINOv2 ViT-B-14 learns richer structural features than MAE ViT-B-16 for
natural-image-style time-series plots, because its self-supervised objective
(self-distillation) produces spatially discriminative patch tokens superior
to MAE's reconstruction-driven tokens (UniFormaly, CVPR 2024).

Combined with length-aware k_medium that prevents MSL regression from noisy
k=30 references on short signals, this should improve ALL F1 over the
current best (add(k5,k30) MAE = 0.6256).

Design
------
  DINOv2 ViT-B-14 : patch_size=14, ph=pw=16, N=256 patches
  LTR k=5         : local reference, 5×56=280 ts context
  LTR k=km        : length-aware medium reference
                    k_medium = min(30, max(6, n_windows // 5))
  Fusion          : add(norm(k5), norm(k_medium)) + EVT

Ablations
---------
  baseline_f1  : MAE k=5, alpha=0.01 (legacy reference)
  mae_add_f1   : add(MAE_k5, MAE_k30) + EVT  (current best ~0.6256)
  dino_k5_f1   : DINOv2 k=5 + EVT
  dino_add_f1  : DINOv2 add(k5, k_medium) + EVT

Checkpoints
-----------
  DINOv2 k=5   : results_dino_ltr/checkpoints/{DS}__{sig}__dino_k5.pkl
                 keys: scores, timestamps
  DINOv2 k=km  : results_dino_ltr/checkpoints/{DS}__{sig}__dino_km.pkl
                 keys: scores, k_medium

Usage
-----
  python experiments/compare_dino_ltr.py
"""

from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
import tempfile
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
import torch.nn.functional as F

from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.preprocess import preprocess_time_series, draw_windowed_images, apply_ewma
from preprocessing.data_utils import orion_to_internal
from models.dino_vision import DINO_AD
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps, align_anomaly_vector
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)
from evaluation.evaluate import evaluate_intervals
from torch.utils.data import DataLoader

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────────────────────
PATCH_SIZE  = 14
PH = PW     = 224 // PATCH_SIZE  # 16
WINDOW_SIZE = 224
STEP_SIZE   = int(WINDOW_SIZE / 4.0)  # 56
STEP_RATIO  = 4.0
AGG_PERCENT = 0.25
K_LOCAL     = 5
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_dino_ltr")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# MAE reference checkpoints (no inference needed — pure fusion)
_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
MAE_K5_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)), _K5_CANDIDATES[-1])

_K30_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
]
MAE_K30_DIR = next((p for p in _K30_CANDIDATES if os.path.isdir(p)), _K30_CANDIDATES[-1])

_RECON_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_recon_additive", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_recon_additive", "checkpoints"),
]
RECON_DIR = next((p for p in _RECON_CANDIDATES if os.path.isdir(p)), _RECON_CANDIDATES[-1])

_count = lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0
print(f"MAE k=5  ckpt : {MAE_K5_DIR}  ({_count(MAE_K5_DIR)} files)")
print(f"MAE k=30 ckpt : {MAE_K30_DIR}  ({_count(MAE_K30_DIR)} files)")
print(f"Recon    ckpt : {RECON_DIR}  ({_count(RECON_DIR)} files)")
print(f"DINO     ckpt : {CKPT_DIR}")

# ── Load DINOv2 ───────────────────────────────────────────────────────────────
print("\n[INFO] Loading DINOv2 ViT-B-14 ...")
dino = DINO_AD(model_name="dinov2_vitb14", device=DEVICE, image_size=(224, 224))
dino.eval()
print("[INFO] DINOv2 ready.")

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


# ── DINOv2 LTR inference ─────────────────────────────────────────────────────

def run_ltr_dino(data: pd.DataFrame, local_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Run DINOv2 LTR with a fixed k and return (aligned_scores, timestamps)."""
    values, timestamps = orion_to_internal(data)
    T_full      = len(values)
    values_proc = preprocess_time_series(values)
    values_proc = apply_ewma(values_proc, 1.0)

    plot_params = ("-", 1, "*", 0.1, "black", (0, 1))

    with tempfile.TemporaryDirectory() as tmp:
        success = draw_windowed_images(
            base_series_id="series",
            save_path=tmp,
            time_series=values_proc,
            time_points=np.arange(len(values_proc)),
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            override=True,
            save_image=False,
            image_size=(224, 224),
            dpi=100,
            plot_params=plot_params,
        )
        if not success:
            return np.zeros(T_full), timestamps

        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="series",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return np.zeros(T_full), timestamps

        loader = DataLoader(dataset, batch_size=20, shuffle=False)
        large_embeds, mid_embeds, patch_embeds, large_mask, mid_mask, _ = \
            build_ordered_embeddings(dino, loader, PATCH_SIZE, DEVICE)

        L = large_embeds.shape[0]
        anomaly_maps = []

        with torch.no_grad():
            for i in range(L):
                l_ref, _ = get_local_reference(large_embeds, i, local_k, 5)
                m_ref, _ = get_local_reference(mid_embeds,   i, local_k, 5)
                p_ref, _ = get_local_reference(patch_embeds, i, local_k, 5)

                m_l = compute_dissimilarity_with_ref(
                    large_embeds[i].unsqueeze(0).to(DEVICE), l_ref.to(DEVICE))
                m_m = compute_dissimilarity_with_ref(
                    mid_embeds[i].unsqueeze(0).to(DEVICE),   m_ref.to(DEVICE))
                m_p = compute_dissimilarity_with_ref(
                    patch_embeds[i].unsqueeze(0).to(DEVICE), p_ref.to(DEVICE))

                m_l = harmonic_aggregation((1, PH, PW), m_l, large_mask).to(DEVICE)
                m_m = harmonic_aggregation((1, PH, PW), m_m, mid_mask).to(DEVICE)
                m_p = m_p.reshape((1, PH, PW)).to(DEVICE)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                         nan=0., posinf=0., neginf=0.)
                score = F.interpolate(
                    score.unsqueeze(1), size=(224, 224), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr   = torch.stack(anomaly_maps, dim=0).numpy()
        raw_scores = stitch_anomaly_maps(maps_arr, STEP_RATIO, AGG_PERCENT)

    n_windows = int((T_full - WINDOW_SIZE) / STEP_SIZE) + 1
    aligned   = align_anomaly_vector(raw_scores, T_full, WINDOW_SIZE, STEP_SIZE, n_windows)
    return aligned, timestamps


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

        # Compute n_windows for length-aware k_medium
        values, _ = orion_to_internal(data)
        T_full    = len(values)
        n_windows = int((T_full - WINDOW_SIZE) / STEP_SIZE) + 1
        k_medium  = max(K_LOCAL + 1, n_windows // 5)
        print(f"    T={T_full}  n_windows={n_windows}  k_medium={k_medium}")

        # Checkpoint paths
        mae_k5_ckpt  = os.path.join(MAE_K5_DIR,  f"{ds_name}__{sig}__ltr.pkl")
        mae_k30_ckpt = os.path.join(MAE_K30_DIR, f"{ds_name}__{sig}__ltr_k30.pkl")
        recon_ckpt   = os.path.join(RECON_DIR,   f"{ds_name}__{sig}__recon.pkl")
        dino_k5_ckpt = os.path.join(CKPT_DIR,    f"{ds_name}__{sig}__dino_k5.pkl")
        dino_km_ckpt = os.path.join(CKPT_DIR,    f"{ds_name}__{sig}__dino_km.pkl")

        # ── baseline: MAE k=5, alpha=0.01 ────────────────────────────────────
        s_mae_k5 = timestamps = None
        if os.path.exists(mae_k5_ckpt):
            try:
                c          = pickle.load(open(mae_k5_ckpt, "rb"))
                s_mae_k5   = c["scores"]
                timestamps = c["timestamps"]
            except Exception as e:
                print(f"    [MAE k5] load error: {e}")

        if s_mae_k5 is not None:
            try:
                ivs = _ivs_alpha(s_mae_k5, timestamps)
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["baseline_f1"] = f1
            print(f"    baseline (MAE k5, alpha)  F1={f1:.4f}")

        # ── mae_add: add(MAE_k5, MAE_k30) + EVT — current best reference ─────
        s_mae_k30 = None
        if os.path.exists(mae_k30_ckpt):
            try:
                s_mae_k30 = pickle.load(open(mae_k30_ckpt, "rb"))["scores"]
            except Exception:
                pass

        if s_mae_k5 is not None and s_mae_k30 is not None:
            try:
                T      = min(len(s_mae_k5), len(s_mae_k30))
                ts_t   = timestamps[:T]
                s_add  = add_fuse(s_mae_k5, s_mae_k30)
                ivs    = evt_detect(s_add, ts_t)
                f1     = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["mae_add_f1"] = f1
            print(f"    mae_add (k5+k30, EVT)     F1={f1:.4f}")

        # ── DINOv2 k=5 ────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(dino_k5_ckpt):
                c          = pickle.load(open(dino_k5_ckpt, "rb"))
                s_dino_k5  = c["scores"]
                d_ts       = c["timestamps"]
                print(f"    [DINO k5]  cache  ({time.time()-t0:.1f}s)")
            else:
                s_dino_k5, d_ts = run_ltr_dino(data, K_LOCAL)
                pickle.dump({"scores": s_dino_k5, "timestamps": d_ts},
                            open(dino_k5_ckpt, "wb"))
                print(f"    [DINO k5]  computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [DINO k5]  ERROR: {e}")
            s_dino_k5 = d_ts = None

        if s_dino_k5 is not None:
            try:
                ivs = evt_detect(s_dino_k5, d_ts)
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["dino_k5_f1"] = f1
            print(f"    dino_k5 (EVT)             F1={f1:.4f}")

        # ── DINOv2 k=k_medium (length-aware) ─────────────────────────────────
        t0 = time.time()
        try:
            # Check if cached k_medium matches the current k_medium
            if os.path.exists(dino_km_ckpt):
                c = pickle.load(open(dino_km_ckpt, "rb"))
                if c.get("k_medium") == k_medium:
                    s_dino_km = c["scores"]
                    print(f"    [DINO km]  cache k={k_medium}  ({time.time()-t0:.1f}s)")
                else:
                    raise ValueError(f"cached k={c.get('k_medium')} != current k={k_medium}")
            else:
                raise FileNotFoundError
        except Exception:
            try:
                s_dino_km, _ = run_ltr_dino(data, k_medium)
                pickle.dump({"scores": s_dino_km, "k_medium": k_medium},
                            open(dino_km_ckpt, "wb"))
                print(f"    [DINO km]  computed k={k_medium}  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [DINO km]  ERROR: {e}")
                s_dino_km = None

        # ── DINOv2 add(k5, k_medium) + EVT ───────────────────────────────────
        s_dino_add = None
        if s_dino_k5 is not None and s_dino_km is not None:
            try:
                T          = min(len(s_dino_k5), len(s_dino_km))
                ts_t       = d_ts[:T]
                s_dino_add = add_fuse(s_dino_k5, s_dino_km)
                ivs        = evt_detect(s_dino_add, ts_t)
                f1         = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["dino_add_f1"] = f1
            delta_mae = f1 - sig_row.get("mae_add_f1", float("nan"))
            print(f"    dino_add (k5+km, EVT)     F1={f1:.4f}  ({delta_mae:+.4f} vs mae_add)")

        # ── DINOv2 add + 0.3×Recon + EVT ─────────────────────────────────────
        s_recon = None
        if os.path.exists(recon_ckpt):
            try:
                c       = pickle.load(open(recon_ckpt, "rb"))
                s_recon = c.get("scores", c.get("mean"))
            except Exception:
                pass

        if s_dino_add is not None and s_recon is not None:
            try:
                T    = min(len(s_dino_add), len(s_recon))
                ts_t = d_ts[:T]
                s_f  = normalize_01(s_dino_add[:T]) + 0.3 * normalize_01(s_recon[:T])
                ivs  = evt_detect(s_f, ts_t)
                f1   = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["dino_add_recon_f1"] = f1
            delta = f1 - sig_row.get("dino_add_f1", float("nan"))
            print(f"    dino_add+Recon (EVT)      F1={f1:.4f}  ({delta:+.4f} vs dino_add)")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "mae_add_f1", "dino_k5_f1", "dino_add_f1", "dino_add_recon_f1"]
LABELS   = {
    "baseline_f1":      "MAE_k5",
    "mae_add_f1":       "mae_add",
    "dino_k5_f1":       "DINO_k5",
    "dino_add_f1":      "DINO_add",
    "dino_add_recon_f1": "DINO+Recon",
}
COL_W = 12
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v: object) -> str:
    try:
        f = float(v)
        return "  NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "  NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    valid = [c for c in ALL_COLS if c in df.columns]
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — DINOv2 LTR ===")
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

if "baseline_f1" in results_df.columns:
    base = results_df["baseline_f1"].dropna().mean()
    print(f"baseline (MAE k=5, alpha=0.01) : {base:.4f}")

if "mae_add_f1" in results_df.columns:
    v = results_df["mae_add_f1"].dropna().mean()
    print(f"mae_add (current best ref)     : {v:.4f}")

if "dino_k5_f1" in results_df.columns:
    v    = results_df["dino_k5_f1"].dropna().mean()
    ref  = results_df["mae_add_f1"].dropna().mean() if "mae_add_f1" in results_df.columns else float("nan")
    print(f"DINOv2 k=5                     : {v:.4f}  ({v-ref:+.4f} vs mae_add)")

if "dino_add_f1" in results_df.columns:
    v   = results_df["dino_add_f1"].dropna().mean()
    ref = results_df["mae_add_f1"].dropna().mean() if "mae_add_f1" in results_df.columns else float("nan")
    print(f"DINOv2 add(k5,km) length-aware : {v:.4f}  ({v-ref:+.4f} vs mae_add)")

if "dino_add_recon_f1" in results_df.columns:
    v   = results_df["dino_add_recon_f1"].dropna().mean()
    ref = results_df["mae_add_f1"].dropna().mean() if "mae_add_f1" in results_df.columns else float("nan")
    print(f"DINOv2 add + 0.3×Recon         : {v:.4f}  ({v-ref:+.4f} vs mae_add)")

# SMAP E-series spotlight (long-duration anomaly targets)
smap_e = results_df[
    (results_df["dataset"] == "SMAP") &
    (results_df["signal"].str.startswith("E"))
]
if not smap_e.empty:
    print("\nSMAP E-series (long-duration anomaly targets):")
    print(f"{'Signal':<8}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid_cols if c in results_df.columns))
    for _, r in smap_e.iterrows():
        row = f"{r['signal']:<8}"
        for c in valid_cols:
            if c in results_df.columns:
                row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)

# MSL short-signal spotlight
msl_short = results_df[
    (results_df["dataset"] == "MSL") &
    (results_df["signal"].isin(["T-13", "D-16", "F-7"]))
]
if not msl_short.empty:
    print("\nMSL short signals (regression targets from k=30 noise):")
    print(f"{'Signal':<8}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid_cols if c in results_df.columns))
    for _, r in msl_short.iterrows():
        row = f"{r['signal']:<8}"
        for c in valid_cols:
            if c in results_df.columns:
                row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)

print(f"\nResults saved → {out_csv}")
