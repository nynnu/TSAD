"""
Fixed Preprocessing: Per-Window Local Normalization
====================================================

Hypothesis
----------
Current baseline uses GLOBAL preprocessing on the full time series before
any windowing:
  1. Global linear detrend (scipy.detrend) — distorts level-shift anomalies
  2. Global min-max normalization         — spike anomalies become the max,
                                            compressing all normal windows to
                                            near-zero → visually flat → missed
  3. y-axis locked to [0, 1]              — within-window relative variation lost

Fix: normalize each window LOCALLY (detrend + minmax per window), let
matplotlib auto-scale the y-axis.  The CLIP/MAE features capture line SHAPE,
not absolute amplitude, so local scale should be strictly better.

Expected beneficiaries
  • Spike anomalies  : no longer dominate global max → normal windows get
                       full dynamic range in the image
  • Level-shift      : global detrend no longer warps baseline around shift
  • Stuck signals    : SMAP F-1, F-3, MSL D-15 (F1=0.0 in baseline)

Design
------
  Subclass LTR_Detector (from compare_ltr_multiscale); override predict_scores
  only for the preprocessing step.  All scoring (LTR k=5, k=30, EVT, fusion)
  is unchanged so we isolate the preprocessing effect.

  Variants:
    k5_fixed   : LTR k=5  + local-norm preprocessing
    k30_fixed  : LTR k=30 + local-norm preprocessing
    add_fixed  : add(norm(k5_fixed), norm(k30_fixed)) + EVT  ← main

  Baseline (reused from existing checkpoints, no recompute):
    baseline_f1 : LTR k=5 + global preprocessing  (original)
    add_k5_k30  : add(k5, k30) + EVT              (current best 0.6256)

Checkpoints
-----------
  results_fixed_preproc/checkpoints/{DS}__{sig}__k5_fixed.pkl
  results_fixed_preproc/checkpoints/{DS}__{sig}__k30_fixed.pkl
  keys: scores, timestamps

Reuses (read-only, never overwritten):
  results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl         (k=5 baseline)
  results_ltr_multiscale/checkpoints/{DS}__{sig}__ltr_k30.pkl (k=30 baseline)

Usage
-----
  python experiments/compare_fixed_preprocessing.py
"""

from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
import tempfile
import time
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import genpareto

# ── Path setup ────────────────────────────────────────────────────────────────
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
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — faster, no display needed
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils import orion_to_internal, intervals_from_indices
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps, compute_detection_intervals
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(PROJECT_ROOT, "data")
NAB_DIR   = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR  = os.path.join(DATA_DIR, "SMAP")
MSL_DIR   = os.path.join(DATA_DIR, "MSL")
ANOM_CSV  = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_fixed_preproc")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

# Baseline checkpoint dirs (read-only)
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

print(f"Baseline k=5  ckpt : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR)) if os.path.isdir(K5_CKPT_DIR) else 0} files)")
print(f"Baseline k=30 ckpt : {K30_CKPT_DIR}  ({len(os.listdir(K30_CKPT_DIR)) if os.path.isdir(K30_CKPT_DIR) else 0} files)")

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
IMAGE_SIZE  = (224, 224)
PATCH_SIZE  = 16
DPI         = 100
K_LOCAL     = 5
K_MEDIUM    = 30
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=PATCH_SIZE,
    model_name="vit_base_patch16_224.mae",
    image_size=IMAGE_SIZE, dpi=DPI,
    standardize=False,   # we handle preprocessing ourselves
    smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

_TO_TENSOR = T.ToTensor()


# ══════════════════════════════════════════════════════════════════════════════
# Fixed preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_window(window: np.ndarray) -> np.ndarray:
    """Local detrend + local min-max for a single window."""
    w = detrend(window.astype(float))       # remove local linear trend
    lo, hi = w.min(), w.max()
    if hi - lo > 1e-8:
        return (w - lo) / (hi - lo)
    return np.zeros_like(w)                 # flat window → zeros


def _render_window(window_vals: np.ndarray, window_time: np.ndarray,
                   image_size: tuple, dpi: int) -> np.ndarray:
    """
    Render one window as an image tensor [C, H, W].

    Changes vs baseline draw_image():
      • window_vals is already locally normalized (no global scale)
      • y_scale=None  → matplotlib auto-scales to local data range
        (was hardcoded (0,1) in baseline)
      • figure is created fresh each call but with Agg backend (no GUI overhead)
    """
    H, W = image_size
    fig_w = W / dpi
    fig_h = H / dpi

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.plot(window_time, window_vals, "-", linewidth=1, color="black")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)

    buf = BytesIO()
    fig.savefig(buf, format="png", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img_pil = Image.open(buf).convert("RGB")
    return _TO_TENSOR(img_pil).cpu().numpy()   # [C, H, W]


def build_windows_local_norm(
    values: np.ndarray,
    window_size: int,
    step_size: int,
    image_size: tuple,
    dpi: int,
) -> np.ndarray:
    """
    Slide a window over `values`, normalize each window locally,
    render to image.  Returns [N_windows, C, H, W].

    This is the fixed replacement for the global-preprocess + draw_windowed_images
    pipeline in baseline preprocess.py.
    """
    imgs = []
    time_pts = np.arange(len(values), dtype=float)

    for start in range(0, len(values) - window_size + 1, step_size):
        end     = start + window_size
        w_vals  = _normalize_window(values[start:end])   # ← LOCAL norm
        w_time  = time_pts[start:end]
        img     = _render_window(w_vals, w_time, image_size, dpi)
        imgs.append(img)

    return np.stack(imgs, axis=0)   # [N, C, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# LTR detector with fixed preprocessing
# ══════════════════════════════════════════════════════════════════════════════

class LTR_FixedPreproc(ViT4TS_MAE):
    """
    MAE + LTR with per-window local normalization.

    Only predict_scores() is overridden — everything else (LTR scoring,
    harmonic aggregation, stitching) is identical to the baseline.
    """

    def __init__(self, *args, local_k: int = 5, min_ref: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_k = local_k
        self.min_ref = min_ref

    def predict_scores(self, data: pd.DataFrame):
        values, timestamps = orion_to_internal(data)
        T_full    = len(values)
        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        # ── Fixed preprocessing: per-window local norm (no global detrend/minmax)
        imgs = build_windows_local_norm(
            values, self.window_size, step_size, self.image_size, self.dpi
        )
        # imgs: [N, C, H, W]

        if imgs.shape[0] == 0:
            warnings.warn("No windows generated.")
            return np.zeros(T_full), timestamps

        with tempfile.TemporaryDirectory() as tmp:
            npy_path = os.path.join(tmp, "series_line_img.npy")
            np.save(npy_path, imgs)

            dataset = CLIPTimeSeriesDataset(
                results_dir=tmp, base_series_id="series",
                sample_size=None, no_anomaly=True, plot_type="line",
            )
            loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            # ── Single-pass encoding (no double-encode bug) ───────────────────
            (large_embeds, mid_embeds, patch_embeds,
             large_mask, mid_mask, _) = build_ordered_embeddings(
                self.model, loader, self.patch_size, self.device
            )

        h  = self.image_size[0]
        ph = pw = h // self.patch_size
        L  = large_embeds.shape[0]

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

                score = torch.nan_to_num(
                    (m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(h, h), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        raw      = stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)

        # align to original length
        from models.model_utils import align_anomaly_vector
        aligned = align_anomaly_vector(raw, T_full, self.window_size, step_size, n_windows)
        return aligned, timestamps


# ══════════════════════════════════════════════════════════════════════════════
# EVT + score helpers  (identical to compare_ltr_multiscale.py)
# ══════════════════════════════════════════════════════════════════════════════

def evt_threshold(scores: np.ndarray, q_init=EVT_Q_INIT, target_fpr=EVT_FPR) -> float:
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


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-8 else (arr - lo) / (hi - lo)


def _ivs_alpha(scores, timestamps, alpha=0.01) -> list:
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


def _eval(detected, gt_ivs) -> float:
    return evaluate_intervals(gt_ivs, detected)["F1"]


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Model init (single load — shared across k=5 and k=30)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[INFO] Loading MAE model ...")
det_k5  = LTR_FixedPreproc(**BASE_PARAMS, local_k=K_LOCAL,  min_ref=5)
det_k30 = LTR_FixedPreproc(**BASE_PARAMS, local_k=K_MEDIUM, min_ref=5)
# Share the underlying MAE weights (both loaded the same checkpoint)
det_k30.model = det_k5.model
print(f"  LTR k={K_LOCAL}  : local  reference")
print(f"  LTR k={K_MEDIUM} : medium reference")
print(f"  Preprocessing: per-window local detrend + minmax  (FIXED)")
print(f"  y-axis       : auto-scale per window               (FIXED)")

# ══════════════════════════════════════════════════════════════════════════════
# Dataset config
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# Main experiment loop
# ══════════════════════════════════════════════════════════════════════════════
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

        # checkpoint paths
        k5_fixed_ckpt  = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__k5_fixed.pkl")
        k30_fixed_ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__k30_fixed.pkl")

        # baseline checkpoint paths (read-only)
        k5_base_ckpt  = os.path.join(K5_CKPT_DIR,  f"{ds_name}__{sig}__ltr.pkl")
        k30_base_ckpt = os.path.join(K30_CKPT_DIR, f"{ds_name}__{sig}__ltr_k30.pkl")

        # ── Load baseline scores (no recompute) ───────────────────────────────
        s_k5_base = s_k30_base = timestamps_base = None
        if os.path.exists(k5_base_ckpt):
            try:
                c = pickle.load(open(k5_base_ckpt, "rb"))
                s_k5_base      = c["scores"]
                timestamps_base = c["timestamps"]
                print(f"    [base k=5 ] loaded from cache")
            except Exception as e:
                print(f"    [base k=5 ] load error: {e}")

        if os.path.exists(k30_base_ckpt):
            try:
                c = pickle.load(open(k30_base_ckpt, "rb"))
                s_k30_base = c["scores"]
                print(f"    [base k=30] loaded from cache")
            except Exception as e:
                print(f"    [base k=30] load error: {e}")

        # ── k=5 FIXED ─────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(k5_fixed_ckpt):
                c          = pickle.load(open(k5_fixed_ckpt, "rb"))
                s_k5_fix   = c["scores"]
                timestamps  = c["timestamps"]
                print(f"    [fix k=5 ] cache  ({time.time()-t0:.1f}s)")
            else:
                s_k5_fix, timestamps = det_k5.predict_scores(data)
                pickle.dump({"scores": s_k5_fix, "timestamps": timestamps},
                            open(k5_fixed_ckpt, "wb"))
                print(f"    [fix k=5 ] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [fix k=5 ] ERROR: {e}")
            import traceback; traceback.print_exc()
            s_k5_fix = timestamps = None

        # ── k=30 FIXED ────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(k30_fixed_ckpt):
                c          = pickle.load(open(k30_fixed_ckpt, "rb"))
                s_k30_fix  = c["scores"]
                print(f"    [fix k=30] cache  ({time.time()-t0:.1f}s)")
            else:
                s_k30_fix, _ = det_k30.predict_scores(data)
                pickle.dump({"scores": s_k30_fix},
                            open(k30_fixed_ckpt, "wb"))
                print(f"    [fix k=30] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [fix k=30] ERROR: {e}")
            s_k30_fix = None

        if s_k5_fix is None or s_k30_fix is None:
            continue

        T       = min(len(s_k5_fix), len(s_k30_fix))
        ts_trim = timestamps[:T]

        # ── Baseline F1 (from existing k=5 scores) ────────────────────────────
        if s_k5_base is not None and timestamps_base is not None:
            try:
                ivs = _ivs_alpha(s_k5_base, timestamps_base)
                sig_row["baseline_f1"] = _eval(ivs, gt_ivs)
                print(f"    baseline (k5 global)   F1={sig_row['baseline_f1']:.4f}")
            except Exception:
                sig_row["baseline_f1"] = float("nan")

        # ── Baseline add(k5,k30) = current best ───────────────────────────────
        if s_k5_base is not None and s_k30_base is not None:
            try:
                T_b   = min(len(s_k5_base), len(s_k30_base))
                s_add_base = normalize_01(s_k5_base[:T_b]) + normalize_01(s_k30_base[:T_b])
                ivs   = evt_detect(s_add_base, (timestamps_base or timestamps)[:T_b])
                sig_row["add_base_f1"] = _eval(ivs, gt_ivs)
                print(f"    add(k5,k30) global     F1={sig_row['add_base_f1']:.4f}  [current best]")
            except Exception:
                sig_row["add_base_f1"] = float("nan")

        # ── Fixed k=5 ─────────────────────────────────────────────────────────
        try:
            ivs = evt_detect(s_k5_fix[:T], ts_trim)
            sig_row["k5_fixed_f1"] = _eval(ivs, gt_ivs)
            delta = sig_row["k5_fixed_f1"] - sig_row.get("baseline_f1", float("nan"))
            print(f"    k5  fixed              F1={sig_row['k5_fixed_f1']:.4f}  ({delta:+.4f} vs base)")
        except Exception:
            sig_row["k5_fixed_f1"] = float("nan")

        # ── Fixed k=30 ────────────────────────────────────────────────────────
        try:
            ivs = evt_detect(s_k30_fix[:T], ts_trim)
            sig_row["k30_fixed_f1"] = _eval(ivs, gt_ivs)
            print(f"    k30 fixed              F1={sig_row['k30_fixed_f1']:.4f}")
        except Exception:
            sig_row["k30_fixed_f1"] = float("nan")

        # ── add(k5_fixed, k30_fixed) + EVT  ← main ───────────────────────────
        try:
            s_add_fix = normalize_01(s_k5_fix[:T]) + normalize_01(s_k30_fix[:T])
            ivs       = evt_detect(s_add_fix, ts_trim)
            sig_row["add_fixed_f1"] = _eval(ivs, gt_ivs)
            delta = sig_row["add_fixed_f1"] - sig_row.get("add_base_f1", float("nan"))
            print(f"    add(k5,k30) fixed      F1={sig_row['add_fixed_f1']:.4f}  ({delta:+.4f} vs best)")
        except Exception:
            sig_row["add_fixed_f1"] = float("nan")

        rows.append(sig_row)


# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "add_base_f1", "k5_fixed_f1", "k30_fixed_f1", "add_fixed_f1"]
LABELS   = {
    "baseline_f1":   "k5_global",
    "add_base_f1":   "add_global(best)",
    "k5_fixed_f1":   "k5_fixed",
    "k30_fixed_f1":  "k30_fixed",
    "add_fixed_f1":  "add_fixed",
}
COL_W = 18
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v) -> str:
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
    print(f"=== {ds_name} — Fixed Preprocessing ===")
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


# Cross-dataset summary
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

# Key findings
print(f"\n{'='*SEP}")
print("KEY FINDINGS")
print(f"{'='*SEP}")
base_mean = results_df["baseline_f1"].dropna().mean() if "baseline_f1" in results_df.columns else float("nan")
best_mean = results_df["add_base_f1"].dropna().mean() if "add_base_f1" in results_df.columns else float("nan")
fix_mean  = results_df["add_fixed_f1"].dropna().mean() if "add_fixed_f1" in results_df.columns else float("nan")

print(f"k5 global (baseline)         : {base_mean:.4f}")
print(f"add(k5,k30) global (cur best): {best_mean:.4f}")
print(f"add(k5,k30) FIXED            : {fix_mean:.4f}  ({fix_mean - best_mean:+.4f} vs best)")

# Spotlight: stuck signals
stuck = ["SMAP__F-1", "SMAP__F-3", "MSL__D-15"]
print("\nStuck signals (F1=0.0 in baseline):")
for sig_key in stuck:
    ds, sig = sig_key.split("__")
    row = results_df[(results_df["dataset"] == ds) & (results_df["signal"] == sig)]
    if not row.empty:
        r = row.iloc[0]
        base = _fmt(r.get("baseline_f1"))
        best = _fmt(r.get("add_base_f1"))
        fix  = _fmt(r.get("add_fixed_f1"))
        print(f"  {sig_key:<15}  base={base}  cur_best={best}  fixed={fix}")

print(f"\nResults saved → {out_csv}")
