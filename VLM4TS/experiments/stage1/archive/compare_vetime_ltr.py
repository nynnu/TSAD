"""
VETime Encoding × LTR k=5 — Frequency-Aware Feature Comparison
===============================================================

Motivation:
  RGB experiment showed SMAP E-3 (LTR=0, Recon=0) gets F1=0.667 with RGB encoding.
  E-series anomalies are subtle contextual shifts that grayscale LTR misses but
  frequency-decomposed channels can capture.

Encoding strategies tested:
  grayscale  : R=G=B=original                  (current baseline)
  vetime     : R=original, G=trend (MA×2), B=residual
  freq_split : R=original, G=lowpass (MA), B=highpass (X-MA)
  trend_vol  : R=original, G=deviation (X-MA), B=rolling_std

All strategies use the same LTR k=5 scoring (3-scale cosine dissimilarity).
The image encoding is the only variable.

Why this works:
  - B channel (residual/highpass) makes subtle frequency shifts visible
  - ViT patches see both original signal AND its frequency decomposition
  - E-series contextual anomalies create larger residual variance → LTR detects

Ablation:
  grayscale_ltr  : current baseline (LTR k=5 on grayscale)
  vetime_ltr     : LTR k=5 on VETime-encoded images
  freq_split_ltr : LTR k=5 on freq_split images
  trend_vol_ltr  : LTR k=5 on trend+volatility images
  vetime_ltr_recon: vetime_ltr + 0.3*Recon (best fusion from additive experiment)

Checkpoints: results_vetime_ltr/checkpoints/{DS}__{sig}__{encoding}__ltr.pkl
             Recon reused from results_ltr_recon_additive/checkpoints/

Usage:
  python experiments/compare_vetime_ltr.py
"""

from __future__ import annotations

import ast
import io
import os
import pickle
import sys
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import genpareto

# ── Path setup ───────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy", "transformers", "Pillow", "--quiet"],
    check=True,
)

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from preprocessing.preprocess import preprocess_time_series, apply_ewma
from preprocessing.data_utils import orion_to_internal
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from models.mae_vision import MAE_AD
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps, align_anomaly_vector
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)
from evaluation.evaluate import evaluate_intervals
from torch.utils.data import DataLoader, Dataset

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(PROJECT_ROOT, "data")
NAB_DIR   = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR  = os.path.join(DATA_DIR, "SMAP")
MSL_DIR   = os.path.join(DATA_DIR, "MSL")
ANOM_CSV  = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results_vetime_ltr")
CKPT_DIR     = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

# Recon checkpoints from previous experiment
_RECON_CKPT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_recon_additive", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_recon_additive", "checkpoints"),
]
RECON_CKPT_DIR = next(
    (p for p in _RECON_CKPT_CANDIDATES if os.path.isdir(p) and os.listdir(p)),
    None,
)
print(f"Recon ckpt   : {RECON_CKPT_DIR}  "
      f"({'%d files' % len(os.listdir(RECON_CKPT_DIR)) if RECON_CKPT_DIR else 'NOT FOUND'})")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
STEP_SIZE   = int(WINDOW_SIZE / STEP_RATIO)   # 56
PATCH_SIZE  = 16
GRID_DIM    = 14
BATCH_SIZE  = 16
LOCAL_K     = 5
MIN_REF     = 5
IMAGE_SIZE  = 224
DPI         = 100
MA_WINDOW   = 11   # must be odd

EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

ENCODINGS   = ["grayscale", "vetime", "freq_split", "trend_vol"]

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=PATCH_SIZE,
    model_name="vit_base_patch16_224.mae",
    image_size=(IMAGE_SIZE, IMAGE_SIZE), dpi=DPI,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

_INET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================================
# Multi-channel image encoder (4 strategies)
# ============================================================================

def _ma(x: np.ndarray, window: int) -> np.ndarray:
    pad    = window // 2
    padded = np.pad(x, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(x)]


def _rolling_std(x: np.ndarray, window: int = MA_WINDOW) -> np.ndarray:
    pad = window // 2
    return np.array([
        x[max(0, i - pad):min(len(x), i + pad + 1)].std()
        for i in range(len(x))
    ], dtype=np.float32)


def _render_channel(values: np.ndarray, y_min: float, y_max: float,
                    image_size: int = IMAGE_SIZE, dpi: int = DPI) -> np.ndarray:
    """1-D array → [H, W] float32 in [0, 1], white bg / black line."""
    fig_size = image_size / dpi
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.plot(values, color="black", linewidth=1.5)
    ax.set_xlim(0, max(len(values) - 1, 1))
    if abs(y_max - y_min) < 1e-8:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    ax.set_ylim(y_min, y_max)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    buf.close()
    return arr


def encode_window(window: np.ndarray, g_min: float, g_max: float,
                  strategy: str = "grayscale") -> np.ndarray:
    """
    Returns [3, H, W] float32 in [0, 1].
    g_min/g_max should be global series min/max.
    """
    w = np.asarray(window, dtype=np.float32)

    if strategy == "grayscale":
        ch = _render_channel(w, g_min, g_max)
        return np.stack([ch, ch, ch])

    elif strategy == "vetime":
        ch_r = _render_channel(w, g_min, g_max)
        ma2  = _ma(_ma(w, MA_WINDOW), MA_WINDOW)       # double MA = trend
        ch_g = _render_channel(ma2, g_min, g_max)
        res  = w - ma2
        r_abs = max(float(np.abs(res).max()), 1e-8)
        ch_b  = _render_channel(res, -r_abs, r_abs)
        return np.stack([ch_r, ch_g, ch_b])

    elif strategy == "freq_split":
        ch_r = _render_channel(w, g_min, g_max)
        lp   = _ma(w, MA_WINDOW)                        # lowpass
        ch_g = _render_channel(lp, g_min, g_max)
        hp   = w - lp                                   # highpass
        hp_abs = max(float(np.abs(hp).max()), 1e-8)
        ch_b = _render_channel(hp, -hp_abs, hp_abs)
        return np.stack([ch_r, ch_g, ch_b])

    elif strategy == "trend_vol":
        ch_r  = _render_channel(w, g_min, g_max)
        ma    = _ma(w, MA_WINDOW)
        dev   = w - ma
        d_abs = max(float(np.abs(dev).max()), 1e-8)
        ch_g  = _render_channel(dev, -d_abs, d_abs)
        rstd  = _rolling_std(w)
        ch_b  = _render_channel(rstd, 0.0, max(float(rstd.max()), 1e-8))
        return np.stack([ch_r, ch_g, ch_b])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# Simple tensor dataset for encoded windows
# ============================================================================

class EncodedWindowDataset(Dataset):
    """Pre-encoded windows as a torch Dataset for DataLoader."""

    def __init__(self, windows_tensor: torch.Tensor):
        self.windows = windows_tensor   # [N, 3, H, W]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {"img": self.windows[idx], "window_id": idx}


# ============================================================================
# LTR k=5 scoring on encoded images
# ============================================================================

@torch.no_grad()
def _encode_images_to_embeddings(windows_tensor: torch.Tensor, mae_ad: MAE_AD,
                                  patch_size: int, device: torch.device):
    """Run MAE encoder on pre-rendered windows → ordered embeddings."""
    dataset = EncodedWindowDataset(windows_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return build_ordered_embeddings(mae_ad, loader, patch_size, device)


def compute_ltr_scores(series: np.ndarray, mae_ad: MAE_AD,
                        strategy: str, device: torch.device,
                        window_size: int = WINDOW_SIZE,
                        step_size: int = STEP_SIZE,
                        local_k: int = LOCAL_K,
                        min_ref: int = MIN_REF,
                        agg_percent: float = 0.25,
                        image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Compute LTR k=5 anomaly scores for a full series.

    1. Slide window → encode each window with `strategy`
    2. MAE encoder → embeddings [L, 3-scale, patches, 768]
    3. LTR k=5: compare each window to local median reference
    4. Stitch anomaly maps → 1-D score vector
    """
    T = len(series)
    g_min, g_max = float(series.min()), float(series.max())
    window_starts = list(range(0, T - window_size + 1, step_size))
    L = len(window_starts)

    if L == 0:
        return np.zeros(T, dtype=np.float32)

    # Render all windows
    imgs = np.stack([
        encode_window(series[ws:ws + window_size], g_min, g_max, strategy)
        for ws in window_starts
    ])   # [L, 3, H, W]

    imgs_t = torch.from_numpy(imgs).float()

    # Normalize with ImageNet stats
    imgs_norm = (imgs_t - _INET_MEAN) / _INET_STD

    # Get MAE embeddings
    dataset = EncodedWindowDataset(imgs_norm)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    (large_embeds, mid_embeds, patch_embeds,
     large_mask, mid_mask, _) = build_ordered_embeddings(mae_ad, loader, PATCH_SIZE, device)

    h = w = image_size
    ph = h // PATCH_SIZE
    pw = w // PATCH_SIZE

    # LTR scoring
    anomaly_maps = []
    with torch.no_grad():
        for i in range(L):
            l_ref, _ = get_local_reference(large_embeds, i, local_k, min_ref)
            m_ref, _ = get_local_reference(mid_embeds,   i, local_k, min_ref)
            p_ref, _ = get_local_reference(patch_embeds, i, local_k, min_ref)

            m_l = compute_dissimilarity_with_ref(
                large_embeds[i].unsqueeze(0).to(device), l_ref.to(device))
            m_m = compute_dissimilarity_with_ref(
                mid_embeds[i].unsqueeze(0).to(device),   m_ref.to(device))
            m_p = compute_dissimilarity_with_ref(
                patch_embeds[i].unsqueeze(0).to(device), p_ref.to(device))

            m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(device)
            m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(device)
            m_p = m_p.reshape((1, ph, pw)).to(device)

            score = torch.nan_to_num(
                (m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.)
            score = F.interpolate(
                score.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
            anomaly_maps.append(score.squeeze(0).detach().cpu())

    maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
    ltr_1d   = stitch_anomaly_maps(maps_arr, STEP_RATIO, agg_percent)
    return align_anomaly_vector(ltr_1d, T, window_size, step_size, L)


# ============================================================================
# Helpers: EVT / normalise / evaluate
# ============================================================================

def evt_threshold(scores: np.ndarray, q_init: float = EVT_Q_INIT,
                  target_fpr: float = EVT_FPR) -> float:
    u           = float(np.percentile(scores, q_init * 100.0))
    exceedances = scores[scores > u] - u
    fallback    = float(np.percentile(scores, (1.0 - target_fpr) * 100.0))
    if len(exceedances) < 10:
        return fallback
    try:
        from scipy.stats import genpareto
        c, _, scale = genpareto.fit(exceedances, floc=0)
        p_cond  = min(target_fpr / max(1.0 - q_init, 1e-9), 1.0 - 1e-9)
        excess  = genpareto.ppf(1.0 - p_cond, c, loc=0, scale=scale)
        thr     = u + max(0.0, excess)
        return thr if u <= thr <= scores.max() else fallback
    except Exception:
        return fallback


def evt_detect(scores: np.ndarray, timestamps: np.ndarray) -> list:
    if (scores.max() - scores.min()) < 1e-8:
        return []
    threshold = evt_threshold(scores)
    flags     = scores > threshold
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


def ivs_alpha(scores: np.ndarray, timestamps: np.ndarray, alpha: float = 0.01) -> list:
    from models.model_utils import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def eval_f1(detected: list, gt_ivs: list) -> float:
    return evaluate_intervals(gt_ivs, detected)["F1"]


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


# ============================================================================
# Model init
# ============================================================================
print("\n[INFO] Loading MAE model ...")
mae_ad = ViT4TS_MAE(**BASE_PARAMS)
backbone = mae_ad.model   # timm MAE ViT-B/16
backbone.to(DEVICE).eval()
print(f"  Encodings : {ENCODINGS}")
print(f"  LTR k={LOCAL_K}  device={DEVICE}")

# ============================================================================
# Dataset config
# ============================================================================
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

# ============================================================================
# Experiment loop
# ============================================================================
rows: list = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}  ({len(cfg['channels'])} signals)")
    print(f"{'='*70}")

    for sig in cfg["channels"]:
        csv_path = os.path.join(cfg["dir"], f"{sig}.csv")
        gt_ivs   = gt.get(sig, [])
        if not os.path.exists(csv_path) or not gt_ivs:
            print(f"  SKIP {sig}")
            continue

        data    = pd.read_csv(csv_path)
        values, timestamps = orion_to_internal(data)
        values = preprocess_time_series(values)
        values = apply_ewma(values, 1.0)

        print(f"\n  [{sig}]  T={len(values)}")
        sig_row = {"dataset": ds_name, "signal": sig}

        enc_scores = {}

        for enc in ENCODINGS:
            ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__{enc}__ltr.pkl")
            t0   = time.time()
            try:
                if os.path.exists(ckpt):
                    c = pickle.load(open(ckpt, "rb"))
                    s = c["scores"]
                    print(f"    [{enc:12s}] cache hit  ({time.time()-t0:.1f}s)")
                else:
                    s = compute_ltr_scores(values, backbone, enc, DEVICE)
                    pickle.dump({"scores": s, "timestamps": timestamps},
                                open(ckpt, "wb"))
                    print(f"    [{enc:12s}] computed   ({time.time()-t0:.1f}s)")

                # Threshold with alpha=0.01 (same as baseline)
                ivs  = ivs_alpha(s, timestamps, alpha=0.01)
                f1   = eval_f1(ivs, gt_ivs)
                enc_scores[enc] = s
                sig_row[f"{enc}_f1"] = f1
                print(f"               F1={f1:.4f}  ivs={len(ivs)}")

            except Exception as e:
                print(f"    [{enc}] ERROR: {e}")
                sig_row[f"{enc}_f1"] = float("nan")

        # ── Best encoding + Recon fusion (vetime + 0.3*Recon) ──────────────
        if RECON_CKPT_DIR:
            recon_ckpt = os.path.join(RECON_CKPT_DIR, f"{ds_name}__{sig}__recon.pkl")
            if os.path.exists(recon_ckpt) and "vetime" in enc_scores:
                try:
                    s_recon = pickle.load(open(recon_ckpt, "rb"))["scores"]
                    s_vt    = enc_scores["vetime"]
                    T       = min(len(s_vt), len(s_recon))
                    ts_t    = timestamps[:T]
                    fused   = normalize_01(s_vt[:T]) + 0.3 * normalize_01(s_recon[:T])
                    ivs_f   = evt_detect(fused, ts_t)
                    f1_f    = eval_f1(ivs_f, gt_ivs)
                    sig_row["vetime_recon_f1"] = f1_f
                    delta   = f1_f - sig_row.get("grayscale_f1", 0)
                    print(f"    [vetime+Recon ] F1={f1_f:.4f}  ivs={len(ivs_f)}  "
                          f"(vs gray: {delta:+.4f})")
                except Exception as e:
                    print(f"    [vetime+Recon] ERROR: {e}")
                    sig_row["vetime_recon_f1"] = float("nan")

        rows.append(sig_row)


# ============================================================================
# Results table
# ============================================================================
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

COLS   = [f"{e}_f1" for e in ENCODINGS] + ["vetime_recon_f1"]
LABELS = {f"{e}_f1": e for e in ENCODINGS}
LABELS["vetime_recon_f1"] = "VT+Rec"
COL_W  = 12
SEP    = 36 + COL_W * len(COLS)


def _fmt(v):
    try:
        f = float(v)
        return "   NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "   NaN"


print(f"\n{'='*SEP}")
print("=== CROSS-DATASET SUMMARY ===")
print(f"{'='*SEP}")
print(f"{'Dataset':<10}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in COLS))
print("-" * SEP)

all_avgs = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = f"{ds_name:<10}"
    for c in COLS:
        v = sub[c].mean() if c in sub.columns else float("nan")
        row += f"{v:>{COL_W}.4f}"
        all_avgs.setdefault(c, []).append(v)
    print(row)

print("-" * SEP)
overall = f"{'ALL':<10}"
for c in COLS:
    vals = [v for v in all_avgs.get(c, []) if not np.isnan(v)]
    overall += f"{np.mean(vals):>{COL_W}.4f}" if vals else f"{'NaN':>{COL_W}}"
print(overall)

print(f"\nResults → {out_csv}")
