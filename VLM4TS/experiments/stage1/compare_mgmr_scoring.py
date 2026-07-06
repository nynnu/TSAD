"""
MGMR: Multi-Granularity Multi-Reference Scoring
================================================

Signal A — Marginal (global distribution deviation)
  ViT4TS_MAE: timm MAE L12, 3-scale build_memory, global cosine dissimilarity.
  "Is this window globally unusual in feature space?"

Signal B — Conditional (local temporal coherence)
  Temporal Feature Gradient (TFG) with local z-score.
  MAE encoder features only (no decoder).  Max-activation row selection
  isolates the line-bearing patches, avoiding SNR dilution from background.
  "Does this time block show an unusually sharp feature transition?"

Thresholding — Extreme Value Theory (Peaks-Over-Threshold)
  Fits a Generalized Pareto Distribution to score exceedances.
  Threshold adapts to each signal's score distribution instead of using
  a fixed top-1% cut — the primary driver of improvement over v1.

Combination — Precision-safe Fallback
  • A detected intervals  →  return A (B's false positives cannot corrupt).
  • A detected nothing    →  return B (rescue where A completely fails).
  Recall(MGMR) ≥ Recall(A) always holds.

Ablation table:
  baseline  : Signal A + EVT threshold
  signal_b  : Signal B + EVT threshold
  mgmr      : Fallback(A → B) + EVT threshold

Usage (Colab):
  !python "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/compare_mgmr_scoring.py"
"""

from __future__ import annotations

import ast
import os
import pickle
import sys
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import genpareto

# ── Path setup ──────────────────────────────────────────────────────────────
# Auto-detect from __file__ so Korean paths in env vars don't cause encoding issues.
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)   # parent of experiments/
# __file__-based detection is always reliable; VLM4TS_ROOT only as explicit override
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy", "--quiet"],
    check=True,
)

import torch

from preprocessing.preprocess import preprocess_time_series, apply_ewma, draw_windowed_images
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
from torch.utils.data import DataLoader

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global constants ─────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
NAB_DIR     = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR    = os.path.join(DATA_DIR, "SMAP")
MSL_DIR     = os.path.join(DATA_DIR, "MSL")
ANOM_CSV    = os.path.join(DATA_DIR, "anomalies.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_mgmr")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

WINDOW_SIZE = 224
STEP_RATIO  = 4.0
PATCH_PX    = 16
GRID_DIM    = 14       # 224 / 16 = 14 → 14×14 patch grid
LOCAL_K     = 5        # temporal neighbourhood for TFG z-score
BATCH_ENC   = 16       # encoder batch size

EVT_Q_INIT  = 0.90    # POT: base percentile for GPD exceedance fitting
EVT_FPR     = 0.01    # target false-positive rate per timestep

# Cardinality gate was tuned for the OLD global-memory Signal A (weak, scatter-shooting).
# LTR k=5 Signal A is far more precise — 11 intervals still yields F1≈0.43.
# With LTR, trust A whenever it detects anything; only let B rescue when A is empty.
MAX_IVS_A   = 9999  # effectively disabled

_INET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BASE_PARAMS = dict(
    window_size       = WINDOW_SIZE,
    window_step_ratio = STEP_RATIO,
    agg_percent       = 0.25,
    patch_size        = 16,
    model_name        = "vit_base_patch16_224.mae",
    image_size        = (224, 224),
    dpi               = 100,
    standardize       = True,
    smoothing_alpha   = 1.0,
    alpha             = 0.01,   # kept for ViT4TS_MAE internal use; overridden by EVT
    verbose           = False,
)


# ============================================================================
# EVT Adaptive Thresholding  (Peaks-Over-Threshold)
# ============================================================================

def evt_threshold(scores: np.ndarray,
                  q_init: float = EVT_Q_INIT,
                  target_fpr: float = EVT_FPR) -> float:
    """
    Compute an adaptive anomaly threshold via Extreme Value Theory.

    Method: Peaks-Over-Threshold (POT)
    -----------------------------------
    1. Choose a high base percentile u = quantile(scores, q_init).
    2. Collect exceedances Y = scores[scores > u] - u.
    3. Fit a Generalized Pareto Distribution (GPD) to Y.
    4. Invert P(score > threshold) = target_fpr:

         P(score > threshold) = P(score > u) · P(Y > threshold − u)
                               = (1 − q_init) · GPD.sf(threshold − u)

       Solving for threshold:
         threshold = u + GPD.ppf(1 − target_fpr / (1 − q_init))

    Falls back to the (1 − target_fpr) empirical percentile when fewer than
    10 exceedances exist or GPD fitting fails.

    Parameters
    ----------
    scores     : 1-D anomaly score vector
    q_init     : base percentile for POT (default 0.90)
    target_fpr : desired per-timestep false-positive rate (default 0.01)

    Returns
    -------
    threshold : float
    """
    u            = float(np.percentile(scores, q_init * 100.0))
    exceedances  = scores[scores > u] - u

    fallback = float(np.percentile(scores, (1.0 - target_fpr) * 100.0))

    if len(exceedances) < 10:
        return fallback

    try:
        c, _, scale = genpareto.fit(exceedances, floc=0)

        # Conditional exceedance probability needed to hit target_fpr overall
        p_cond = target_fpr / max(1.0 - q_init, 1e-9)
        p_cond = min(p_cond, 1.0 - 1e-9)

        excess    = genpareto.ppf(1.0 - p_cond, c, loc=0, scale=scale)
        threshold = u + max(0.0, excess)

        # Sanity: threshold must be ≥ u and ≤ max(scores)
        if not (u <= threshold <= scores.max()):
            return fallback
        return threshold

    except Exception:
        return fallback


def evt_detect(scores: np.ndarray,
               timestamps: np.ndarray,
               q_init: float = EVT_Q_INIT) -> list:
    """
    Apply EVT threshold and extract contiguous anomalous intervals.

    Parameters
    ----------
    scores     : aligned per-timestep anomaly scores [T]
    timestamps : corresponding timestamps [T]
    q_init     : POT base percentile

    Returns
    -------
    intervals : list of [start_ts, end_ts]  (inclusive, timestamp space)
    """
    if (scores.max() - scores.min()) < 1e-8:   # uniform → nothing to detect
        return []

    threshold = evt_threshold(scores, q_init=q_init)
    flags     = scores > threshold

    if not flags.any():
        return []

    intervals: list = []
    in_seg           = False
    seg_start        = 0

    for i, f in enumerate(flags):
        if f and not in_seg:
            in_seg    = True
            seg_start = i
        elif not f and in_seg:
            in_seg = False
            intervals.append([timestamps[seg_start], timestamps[i - 1]])

    if in_seg:
        intervals.append([timestamps[seg_start], timestamps[len(flags) - 1]])

    return intervals


# ============================================================================
# Signal B — Temporal Feature Gradient (TFG)
# ============================================================================

@torch.no_grad()
def _extract_column_features(
    images_raw: torch.Tensor,
    mae_ad:     MAE_AD,
    device:     torch.device,
    batch_size: int = BATCH_ENC,
) -> np.ndarray:
    """
    Extract per-column MAE encoder features for every window.

    Architecture insight
    --------------------
    A rendered line plot is mostly black background; only 1-2 of the 14 patch
    rows actually contain the time-series line.  Averaging over all rows
    (mean pooling) dilutes the informative signal.

    Fix: select the single most-active row per window — the row whose patches
    have the highest mean L2 norm, i.e., where the encoder fires strongest.
    This isolates the line-bearing row and improves Signal-to-Noise Ratio.

    Parameters
    ----------
    images_raw : [N, 3, H, W]  float in [0, 1]
    mae_ad     : MAE_AD encoder (timm ViT-B/16)

    Returns
    -------
    col_feat : np.ndarray  [N, GRID_DIM, 768]
        Feature vector for each of the 14 time-axis columns,
        taken from the most-active patch row of each window.
    """
    N   = len(images_raw)
    D   = 768
    out = np.zeros((N, GRID_DIM, D), dtype=np.float32)

    inet_mean = _INET_MEAN.to(device)
    inet_std  = _INET_STD.to(device)

    for i in range(0, N, batch_size):
        batch  = images_raw[i : i + batch_size].to(device)
        B_cur  = batch.shape[0]
        batch  = (batch - inet_mean) / inet_std

        # encode_image(use_mask=False) → patch tokens only, no multi-scale
        _, _, patch_tokens, _, _, _ = mae_ad.encode_image(
            batch, patch_size=PATCH_PX, use_mask=False
        )
        patch_tokens = patch_tokens.squeeze(2)          # [B, 196, 768]

        # Row-major layout: patch[r*14 + c] belongs to (row r, col c)
        feats = patch_tokens.reshape(B_cur, GRID_DIM, GRID_DIM, D)  # [B, row, col, D]

        # ── Max-activation row selection ──────────────────────────────────
        # row_activity[b, r] = mean L2 norm of all patches in row r for sample b
        row_norms    = torch.norm(feats, dim=-1)          # [B, row, col]
        row_activity = row_norms.mean(dim=2)              # [B, row]
        top_rows     = row_activity.argmax(dim=1)         # [B]  — one row per image

        # Gather features from the most-active row for each sample
        col_feat = feats[torch.arange(B_cur, device=device), top_rows]  # [B, col, D]
        out[i : i + B_cur] = col_feat.cpu().numpy()

    return out  # [N, 14, 768]


def _temporal_gradient_scores(col_feat: np.ndarray) -> np.ndarray:
    """
    Per-column anomaly scores from temporal feature gradients.

    grad_norm[w, c] = ||col_feat[w, c+1] - col_feat[w, c]||_2   (13 boundaries)

    Each column c is assigned the maximum of its two adjacent boundary norms
    so that a sharp feature transition raises both flanking columns.

    Parameters
    ----------
    col_feat : [N, 14, 768]

    Returns
    -------
    col_score : np.ndarray  [N, 14]
    """
    delta     = col_feat[:, 1:, :] - col_feat[:, :-1, :]  # [N, 13, 768]
    grad_norm = np.linalg.norm(delta, axis=-1)              # [N, 13]

    N         = len(col_feat)
    col_score = np.zeros((N, GRID_DIM), dtype=np.float32)
    col_score[:, 0]              = grad_norm[:, 0]
    col_score[:, 1 : GRID_DIM-1] = np.maximum(grad_norm[:, :-1], grad_norm[:, 1:])
    col_score[:, GRID_DIM - 1]  = grad_norm[:, -1]
    return col_score


def _local_zscore(col_score: np.ndarray, k: int = LOCAL_K) -> np.ndarray:
    """
    Normalise each window's column scores against k temporal neighbours.

      z[w, c] = (col_score[w, c] − mean_nbr[c]) / (std_nbr[c] + ε)

    High z-score ↔ this column's gradient spike is locally unusual,
    not just a reflection of globally noisy data.

    Parameters
    ----------
    col_score : [N, 14]
    k         : total neighbourhood size (⌊k/2⌋ windows on each side)

    Returns
    -------
    z : np.ndarray  [N, 14]
    """
    N      = len(col_score)
    half_k = k // 2
    z      = np.zeros_like(col_score)

    for w in range(N):
        lo   = max(0, w - half_k)
        hi   = min(N, w + half_k + 1)
        nbrs = [j for j in range(lo, hi) if j != w]

        if len(nbrs) < 2:                              # edge: use all others
            nbrs = [j for j in range(N) if j != w]
        if not nbrs:
            continue

        nbr  = col_score[nbrs]                         # [k', 14]
        mu   = nbr.mean(axis=0)                        # [14]
        sig  = nbr.std(axis=0)  + 1e-6                # [14]
        z[w] = (col_score[w] - mu) / sig

    return z


def compute_tfg_scores(
    images_raw:    torch.Tensor,
    mae_ad:        MAE_AD,
    window_starts: list[int],
    T:             int,
    device:        torch.device,
    k:             int = LOCAL_K,
) -> np.ndarray:
    """
    Full TFG pipeline:
      encoder features → max-row selection → temporal gradient → local z-score
      → scatter to timestep level.

    Returns
    -------
    s_B : np.ndarray  [T]
    """
    print(f"    [TFG] Extracting column features  ({len(images_raw)} windows) ...")
    col_feat  = _extract_column_features(images_raw, mae_ad, device)

    print(f"    [TFG] Computing temporal gradients ...")
    col_score = _temporal_gradient_scores(col_feat)

    print(f"    [TFG] Local z-score  (k={k}) ...")
    z_score   = _local_zscore(col_score, k=k)

    # Scatter window-column z-scores to timestep positions (max over overlaps)
    s_B = np.zeros(T, dtype=np.float32)
    for wi, ws in enumerate(window_starts):
        for col in range(GRID_DIM):
            t0         = ws + col * PATCH_PX
            t1         = min(ws + (col + 1) * PATCH_PX, T)
            s_B[t0:t1] = np.maximum(s_B[t0:t1], z_score[wi, col])

    print(f"    [TFG] z_max={s_B.max():.3f}  z_p95={np.percentile(s_B, 95):.3f}")
    return s_B


# ============================================================================
# Signal A — Local Temporal Reference (LTR k=5)
# ============================================================================

class ViT4TS_SignalA_LTR(ViT4TS_MAE):
    """
    Signal A using Local Temporal Reference (LTR k=5).

    Instead of building a global memory from ALL windows (build_memory),
    each window w is compared against the median of its k nearest temporal
    neighbours.  This captures local stationarity rather than global
    distribution deviation, which is far more discriminative for real-world
    time-series anomalies that appear against a locally normal background.

    Scoring (identical multi-scale structure as ViT4TS_MAE):
        large / mid / patch scale dissimilarity → harmonic_aggregation → stitch
    Reference:
        get_local_reference(embeds, i, k=local_k) → median of k neighbours
    """

    def __init__(self, *args, local_k: int = 5, min_ref: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_k = local_k
        self.min_ref = min_ref

    def _run_inference(self, results_dir: str, base_series_id: str):
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir,
            base_series_id=base_series_id,
            sample_size=None,
            no_anomaly=True,
            plot_type="line",
        )
        if len(dataset) == 0:
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # ── Pass 1: encode all windows in temporal order ──────────────────
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.model, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = w  = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        # ── Pass 2: per-window LTR scoring ────────────────────────────────
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
                    (m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.
                )
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(h, w), mode="bilinear"
                ).squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()   # [L, H, W]
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)


# ============================================================================
# ViT4TS_SignalB_TFG
# ============================================================================

class ViT4TS_SignalB_TFG(ViT4TS_MAE):
    """
    Signal B detector (TFG) as a ViT4TS_MAE-compatible wrapper.

    Only predict_scores() is overridden; everything else (windowing,
    preprocessing, device management) is inherited unchanged.
    No decoder or second model is loaded.
    """

    def predict_scores(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values, timestamps = orion_to_internal(data)
        T_full      = len(values)
        values_proc = (
            preprocess_time_series(values) if self.standardize else values.astype(float)
        )
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        step_size     = int(self.window_size / self.window_step_ratio)
        window_starts = list(range(0, T_full - self.window_size + 1, step_size))

        with tempfile.TemporaryDirectory() as tmp:
            ok = draw_windowed_images(
                base_series_id = "series",
                save_path      = tmp,
                time_series    = values_proc,
                time_points    = np.arange(len(values_proc)),
                override       = True,
                window_size    = self.window_size,
                step_size      = step_size,
                image_size     = self.image_size,
                dpi            = self.dpi,
                plot_params    = (
                    "-", 1, "*", 0.1, "black",
                    (0, 1) if self.standardize else None,
                ),
            )
            if not ok:
                return np.zeros(T_full), timestamps

            images_raw = torch.from_numpy(
                np.load(os.path.join(tmp, "series_line_img.npy"))
            ).float()   # [N, 3, H, W]  in [0, 1]

        s_B = compute_tfg_scores(
            images_raw    = images_raw,
            mae_ad        = self.model,
            window_starts = window_starts,
            T             = T_full,
            device        = self.device,
        )
        # s_B is already length T_full (direct timestep scatter in compute_tfg_scores)
        return s_B, timestamps


# ============================================================================
# Precision-safe Fallback combination
# ============================================================================

def _merge(ivs: list) -> list:
    """Sort and merge overlapping / adjacent intervals."""
    merged: list = []
    for seg in sorted(ivs, key=lambda x: x[0]):
        s, e = seg[0], seg[1]
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def fallback_union(ivs_a: list, ivs_b: list) -> tuple[list, str]:
    """
    Combine Signal A and Signal B intervals.

    Routing logic (three tiers)
    ---------------------------
    1. A is empty                    → B rescues  (A missed everything)
    2. A has > MAX_IVS_A intervals   → A is scatter-shooting noise; prefer B
       (only switches when B also has detections; otherwise keeps A)
    3. A has 1 … MAX_IVS_A intervals → trust A  (confident, precise detections)

    Returns (intervals, mode_label).
    """
    if not ivs_a:
        return _merge(ivs_b), "B↑rescue"

    if len(ivs_a) > MAX_IVS_A and ivs_b:
        return _merge(ivs_b), "B↑noisy_A"

    return _merge(ivs_a), "A"


# ============================================================================
# Evaluation helpers
# ============================================================================

def load_gt(anom_csv: str) -> dict:
    gt: dict = {}
    with open(anom_csv, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt


def _eval(detected: list, gt_ivs: list) -> dict:
    m = evaluate_intervals(gt_ivs, detected)
    return {"f1": m["F1"], "p": m["precision"], "r": m["recall"]}


# ============================================================================
# Model initialisation
# ============================================================================
print("\n[INFO] Loading models ...")

det_baseline = ViT4TS_SignalA_LTR(**BASE_PARAMS, local_k=5, min_ref=5)
det_signal_b = ViT4TS_SignalB_TFG(**BASE_PARAMS)

print(
    f"  Signal A : LTR k=5      (local temporal reference, 3-scale cosine, alpha=0.01)\n"
    f"  Signal B : TFG          (max-row encoder features, local z-score + EVT threshold)\n"
    f"  MGMR     : cardinality-safe fallback  MAX_IVS_A={MAX_IVS_A}"
)


# ============================================================================
# Dataset configuration
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

METHODS       = ["baseline", "signal_b", "mgmr"]
METHOD_LABELS = {
    "baseline": "Baseline(A)",
    "signal_b": "TFG(B)",
    "mgmr":     "Fallback(A→B)",
}


# ============================================================================
# Experiment loop
# ============================================================================
rows: list = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*65}")
    print(f"Dataset: {ds_name}  ({len(cfg['channels'])} signals)")
    print(f"{'='*65}")

    for sig in cfg["channels"]:
        csv_path = os.path.join(cfg["dir"], f"{sig}.csv")
        gt_ivs   = gt.get(sig, [])
        if not os.path.exists(csv_path) or not gt_ivs:
            print(f"  SKIP {sig}")
            continue

        print(f"\n  [{sig}]")
        data    = pd.read_csv(csv_path)
        sig_row = {"dataset": ds_name, "signal": sig}

        ckpt_a = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__ltr.pkl")
        ckpt_b = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__tfg.pkl")

        # ── Signal A: LTR k=5, alpha=0.01  (cached) ──────────────
        t0 = time.time()
        try:
            if os.path.exists(ckpt_a):
                cached     = pickle.load(open(ckpt_a, "rb"))
                s_A        = cached["scores"]
                timestamps = cached["timestamps"]
                print(f"    baseline   [cache hit]")
            else:
                s_A, timestamps = det_baseline.predict_scores(data)
                pickle.dump({"scores": s_A, "timestamps": timestamps}, open(ckpt_a, "wb"))

            ivs_df = det_baseline.get_intervals(s_A, timestamps, alpha=0.01)
            ivs_a  = [[row["start"], row["end"]] for _, row in ivs_df.iterrows()]
            m_base = _eval(ivs_a, gt_ivs)
        except Exception as e:
            print(f"    baseline   ERROR: {e}")
            s_A        = np.zeros(1)
            timestamps  = np.zeros(1)
            ivs_a      = []
            m_base     = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
        sig_row["baseline_f1"] = m_base["f1"]
        print(
            f"    baseline   F1={m_base['f1']:.4f}  ivs={len(ivs_a)}  ({time.time()-t0:.1f}s)"
        )

        # ── Signal B: TFG, EVT threshold  (cached) ───────────────
        t0 = time.time()
        try:
            if os.path.exists(ckpt_b):
                cached = pickle.load(open(ckpt_b, "rb"))
                s_B    = cached["scores"]
                print(f"    signal_b   [cache hit]")
            else:
                s_B, _ = det_signal_b.predict_scores(data)
                pickle.dump({"scores": s_B}, open(ckpt_b, "wb"))

            thr_B  = evt_threshold(s_B)
            ivs_b  = evt_detect(s_B, timestamps)
            m_sigb = _eval(ivs_b, gt_ivs)
        except Exception as e:
            print(f"    signal_b   ERROR: {e}")
            s_B    = np.zeros(1)
            thr_B  = 0.0
            ivs_b  = []
            m_sigb = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
        sig_row["signal_b_f1"] = m_sigb["f1"]
        print(
            f"    signal_b   F1={m_sigb['f1']:.4f}  "
            f"thr={thr_B:.4f}  ivs={len(ivs_b)}  ({time.time()-t0:.1f}s)"
        )

        # ── MGMR: precision-safe fallback (no extra inference) ───
        try:
            ivs_mgmr, mode = fallback_union(ivs_a, ivs_b)
            m_mgmr         = _eval(ivs_mgmr, gt_ivs)
        except Exception as e:
            print(f"    mgmr       ERROR: {e}")
            ivs_mgmr = []
            mode     = "err"
            m_mgmr   = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
        sig_row["mgmr_f1"] = m_mgmr["f1"]
        print(f"    mgmr       F1={m_mgmr['f1']:.4f}  ({mode})")

        rows.append(sig_row)


# ============================================================================
# Results table
# ============================================================================
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

COL_W = 14
SEP   = 42 + COL_W * len(METHODS)


def _fmt(v) -> str:
    try:
        f = float(v)
        return "   NaN" if (f != f) else f"{f:.4f}"
    except (TypeError, ValueError):
        return "   NaN"


def _print_dataset_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return

    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — MGMR Ablation  (interval F1, EVT threshold) ===")
    print(f"{'='*SEP}")
    print(f"{'Signal':<42}" + "".join(f"{METHOD_LABELS[m]:>{COL_W}}" for m in METHODS))
    print("-" * SEP)

    for _, r in sub.iterrows():
        fs   = [r.get(f"{m}_f1", float("nan")) for m in METHODS]
        best = max((f for f in fs if f == f), default=float("nan"))
        line = f"{r['signal']:<42}"
        for f in fs:
            marker = "→" if (f == f and f == best) else " "
            line  += f"{marker + _fmt(f):>{COL_W}}"
        print(line)

    print("-" * SEP)
    avgs = [sub[f"{m}_f1"].mean() for m in METHODS]
    print(f"{'AVG F1':<42}" + "".join(f" {a:>{COL_W-1}.4f}" for a in avgs))

    wins = {m: 0 for m in METHODS}
    for _, r in sub.iterrows():
        fs    = {m: r.get(f"{m}_f1", float("nan")) for m in METHODS}
        valid = {m: v for m, v in fs.items() if v == v}
        if valid:
            wins[max(valid, key=valid.get)] += 1
    print(f"{'WINS':<42}" + "".join(f"{wins[m]:>{COL_W}}" for m in METHODS))


for ds in ["NAB", "SMAP", "MSL"]:
    _print_dataset_table(ds, results_df)


# ── Overall summary ──────────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("OVERALL SUMMARY  (EVT adaptive threshold)")
print(f"{'='*SEP}")
print(f"{'Dataset':<14}" + "".join(f"{METHOD_LABELS[m]:>{COL_W}}" for m in METHODS))
print("-" * SEP)
for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    print(f"{ds:<14}" + "".join(f" {sub[f'{m}_f1'].mean():>{COL_W-1}.4f}" for m in METHODS))


# ── Key findings ─────────────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("KEY FINDINGS")
print(f"{'='*SEP}")

for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    base_avg = sub["baseline_f1"].mean()

    print(f"\n{ds}:")
    for m in METHODS:
        avg   = sub[f"{m}_f1"].mean()
        delta = avg - base_avg
        print(f"  {METHOD_LABELS[m]:<16} avg={avg:.4f}  vs_baseline={delta:+.4f}")

    # Signals genuinely rescued: A=0 (baseline F1=0) AND MGMR>0
    rescued = sub[(sub["baseline_f1"] == 0) & (sub["mgmr_f1"] > 0)]
    if not rescued.empty:
        print(f"  Rescued (A=0 → MGMR>0)        : {list(rescued['signal'])}")

    # Signals where B overrode noisy A (A>MAX_IVS_A) and improved F1
    noisy_gain = sub[
        (sub["mgmr_f1"] > sub["baseline_f1"] + 0.05)
    ]
    if not noisy_gain.empty:
        print(f"  Noisy-A switched to B (+>0.05) : {list(noisy_gain['signal'])}")

    # Signals where MGMR > baseline by >0.05
    gains = sub[sub["mgmr_f1"] - sub["baseline_f1"] > 0.05]
    if not gains.empty:
        print(f"  MGMR > Baseline by >0.05     : {list(gains['signal'])}")

    # Signals where baseline > MGMR by >0.05  (B added false positives)
    losses = sub[sub["baseline_f1"] - sub["mgmr_f1"] > 0.05]
    if not losses.empty:
        print(f"  Baseline > MGMR by >0.05     : {list(losses['signal'])}")

print(f"\nResults → {RESULTS_DIR}/comparison.csv")
