"""
Triple Fusion: LTR k=5 + TFG + MAE Recon
==========================================

Hypothesis
----------
Three complementary anomaly signals:

  LTR   : cross-window cosine dissimilarity vs local temporal reference (k=5)
           Catches abrupt transitions and spike anomalies.
           Fails when anomaly spans > k windows (neighbor reference is contaminated).

  TFG   : within-window temporal feature gradient, locally z-scored
           Detects irregular temporal dynamics INSIDE a single window.
           Orthogonal to LTR: works even when all LTR neighbors are anomalous.
           compare_score_fusion.py confirmed: SMAP E-3 0.0→0.4, T-1 0.0→0.21.

  Recon : MAE pixel-space reconstruction error (n_iter=5 random masks)
           Catches signals that deviate from the MAE visual prior.
           compare_ltr_recon_additive.py: LTR + 0.3*Recon = 0.6084 (current best).

Triple fusion score:
  s = normalize_01(LTR) + w_tfg * normalize_01(TFG) + w_recon * normalize_01(Recon)
  Threshold via EVT (GPD Peaks-Over-Threshold, q_init=0.90, fpr=0.01).

Checkpoint reuse (all three signals cached from previous experiments):
  LTR  : {MGMR_CKPT}/{DS}__{sig}__ltr.pkl    keys: scores, timestamps
  TFG  : {MGMR_CKPT}/{DS}__{sig}__tfg.pkl    key:  scores
  Recon: {RECON_CKPT}/{DS}__{sig}__recon.pkl  key:  scores

  Models are loaded lazily only when a checkpoint is missing.

Ablations
---------
  baseline    : LTR k=5                         (reference)
  ltr_recon   : LTR + 0.3*Recon                 (current best = 0.6084)
  ltr_tfg_w01 : LTR + 0.1*TFG
  ltr_tfg_w02 : LTR + 0.2*TFG
  triple_t1r3 : LTR + 0.1*TFG + 0.3*Recon      [main hypothesis]
  triple_t2r3 : LTR + 0.2*TFG + 0.3*Recon
  triple_t1r5 : LTR + 0.1*TFG + 0.5*Recon
  triple_t2r5 : LTR + 0.2*TFG + 0.5*Recon

Usage (Lightning AI):
  python experiments/compare_triple_fusion.py
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

# ── Path setup ────────────────────────────────────────────────────────────────
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
    ["pip", "install", "timm", "open-clip-torch", "scipy", "transformers", "--quiet"],
    check=True,
)

import torch

from preprocessing.preprocess import preprocess_time_series, apply_ewma, draw_windowed_images
from preprocessing.data_utils import orion_to_internal
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from models.mae_vision import MAE_AD
from models.vit4ts_mae import ViT4TS_MAE
from models.mae_recon_ad import MAE_Recon
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(PROJECT_ROOT, "data")
NAB_DIR   = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR  = os.path.join(DATA_DIR, "SMAP")
MSL_DIR   = os.path.join(DATA_DIR, "MSL")
ANOM_CSV  = os.path.join(DATA_DIR, "anomalies.csv")

# LTR + TFG checkpoints (from compare_score_fusion / compare_mgmr_scoring)
_MGMR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
MGMR_CKPT_DIR = next(
    (p for p in _MGMR_CANDIDATES if os.path.isdir(p)),
    _MGMR_CANDIDATES[-1],
)
os.makedirs(MGMR_CKPT_DIR, exist_ok=True)
print(f"MGMR ckpt  : {MGMR_CKPT_DIR}  ({len(os.listdir(MGMR_CKPT_DIR))} files)")

# Recon checkpoints (from compare_ltr_recon_additive)
_RECON_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_recon_additive", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_recon_additive", "checkpoints"),
]
RECON_CKPT_DIR = next(
    (p for p in _RECON_CANDIDATES if os.path.isdir(p)),
    _RECON_CANDIDATES[-1],
)
os.makedirs(RECON_CKPT_DIR, exist_ok=True)
print(f"Recon ckpt : {RECON_CKPT_DIR}  ({len(os.listdir(RECON_CKPT_DIR))} files)")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_triple_fusion")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
STEP_SIZE   = int(WINDOW_SIZE / STEP_RATIO)   # 56
PATCH_PX    = 16
GRID_DIM    = 14
LOCAL_K     = 5
MIN_REF     = 5
AGG_PERCENT = 0.25
BATCH_ENC   = 16

EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

_INET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=AGG_PERCENT, patch_size=PATCH_PX,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

# ── Ablation grid: (name, w_tfg, w_recon) ─────────────────────────────────────
ABLATIONS = [
    ("baseline",    0.0, 0.0),
    ("ltr_recon",   0.0, 0.3),
    ("ltr_tfg_w01", 0.1, 0.0),
    ("ltr_tfg_w02", 0.2, 0.0),
    ("triple_t1r3", 0.1, 0.3),
    ("triple_t2r3", 0.2, 0.3),
    ("triple_t1r5", 0.1, 0.5),
    ("triple_t2r5", 0.2, 0.5),
]


# ============================================================================
# EVT threshold
# ============================================================================

def evt_threshold(scores: np.ndarray, q_init: float = EVT_Q_INIT,
                  target_fpr: float = EVT_FPR) -> float:
    u           = float(np.percentile(scores, q_init * 100.0))
    exceedances = scores[scores > u] - u
    fallback    = float(np.percentile(scores, (1.0 - target_fpr) * 100.0))
    if len(exceedances) < 10:
        return fallback
    try:
        c, _, scale = genpareto.fit(exceedances, floc=0)
        p_cond  = min(target_fpr / max(1.0 - q_init, 1e-9), 1.0 - 1e-9)
        excess  = genpareto.ppf(1.0 - p_cond, c, loc=0, scale=scale)
        thr     = u + max(0.0, excess)
        if not (u <= thr <= scores.max()):
            return fallback
        return thr
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


# ============================================================================
# Score helpers
# ============================================================================

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def triple_fuse(s_ltr: np.ndarray, s_tfg: np.ndarray, s_recon: np.ndarray,
                w_tfg: float, w_recon: float) -> np.ndarray:
    T      = min(len(s_ltr), len(s_tfg), len(s_recon))
    result = normalize_01(s_ltr[:T])
    if w_tfg > 0:
        result = result + w_tfg   * normalize_01(s_tfg[:T])
    if w_recon > 0:
        result = result + w_recon * normalize_01(s_recon[:T])
    return result


# ============================================================================
# TFG helper functions  (from compare_score_fusion.py)
# ============================================================================

@torch.no_grad()
def _extract_column_features(images_raw: torch.Tensor, mae_ad, device) -> np.ndarray:
    """Extract the most-active-row column features per window.

    images_raw : [N, 3, H, W]  raw float images (NOT ImageNet-normalised)
    Returns    : [N, GRID_DIM, 768]
    """
    N, D = len(images_raw), 768
    out  = np.zeros((N, GRID_DIM, D), dtype=np.float32)
    inet_mean = _INET_MEAN.to(device)
    inet_std  = _INET_STD.to(device)

    for i in range(0, N, BATCH_ENC):
        batch = images_raw[i:i + BATCH_ENC].to(device)
        B_cur = batch.shape[0]
        batch = (batch - inet_mean) / inet_std

        _, _, patch_tokens, _, _, _ = mae_ad.encode_image(batch, patch_size=PATCH_PX, use_mask=False)
        patch_tokens = patch_tokens.squeeze(2)                             # [B, 196, 768]
        feats        = patch_tokens.reshape(B_cur, GRID_DIM, GRID_DIM, D) # [B, 14, 14, 768]
        row_norms    = torch.norm(feats, dim=-1)                           # [B, 14, 14]
        row_activity = row_norms.mean(dim=2)                               # [B, 14]
        top_rows     = row_activity.argmax(dim=1)                          # [B]
        col_feat     = feats[torch.arange(B_cur, device=device), top_rows] # [B, 14, 768]
        out[i:i + B_cur] = col_feat.cpu().numpy()

    return out


def _temporal_gradient_scores(col_feat: np.ndarray) -> np.ndarray:
    """Per-window, per-column gradient magnitude.  Returns [N, GRID_DIM]."""
    delta      = col_feat[:, 1:, :] - col_feat[:, :-1, :]   # [N, 13, 768]
    grad_norm  = np.linalg.norm(delta, axis=-1)               # [N, 13]
    N          = len(col_feat)
    col_score  = np.zeros((N, GRID_DIM), dtype=np.float32)
    col_score[:, 0]              = grad_norm[:, 0]
    col_score[:, 1:GRID_DIM - 1] = np.maximum(grad_norm[:, :-1], grad_norm[:, 1:])
    col_score[:, GRID_DIM - 1]   = grad_norm[:, -1]
    return col_score


def _local_zscore(col_score: np.ndarray, k: int = LOCAL_K) -> np.ndarray:
    """Normalise each window's score relative to its k nearest temporal neighbours."""
    N, half_k = len(col_score), k // 2
    z = np.zeros_like(col_score)
    for w in range(N):
        lo   = max(0, w - half_k)
        hi   = min(N, w + half_k + 1)
        nbrs = [j for j in range(lo, hi) if j != w]
        if len(nbrs) < 2:
            nbrs = [j for j in range(N) if j != w]
        if not nbrs:
            continue
        nbr  = col_score[nbrs]
        z[w] = (col_score[w] - nbr.mean(axis=0)) / (nbr.std(axis=0) + 1e-6)
    return z


# ============================================================================
# Lazy model holders
# ============================================================================

_backbone_wrapper = None
_recon_model      = None


def _get_backbone() -> ViT4TS_MAE:
    global _backbone_wrapper
    if _backbone_wrapper is None:
        print("\n[INFO] Loading MAE backbone (shared for LTR + TFG) ...")
        _backbone_wrapper = ViT4TS_MAE(**BASE_PARAMS)
        _backbone_wrapper.model.eval()
    return _backbone_wrapper


def _get_recon() -> MAE_Recon:
    global _recon_model
    if _recon_model is None:
        print("\n[INFO] Loading MAE_Recon ...")
        _recon_model = MAE_Recon(device=DEVICE, n_iter=5, mask_ratio=0.5)
    return _recon_model


# ============================================================================
# Score computation functions
# ============================================================================

def compute_ltr_scores(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """LTR k=5.  Returns (scores [T], timestamps [T])."""
    backbone = _get_backbone()
    values, timestamps = orion_to_internal(data)
    T_full      = len(values)
    values_proc = apply_ewma(preprocess_time_series(values), 1.0)
    step_size   = STEP_SIZE
    n_windows   = int((T_full - WINDOW_SIZE) / step_size) + 1

    with tempfile.TemporaryDirectory() as tmp:
        draw_windowed_images(
            base_series_id="series", save_path=tmp,
            time_series=values_proc, time_points=np.arange(len(values_proc)),
            window_size=WINDOW_SIZE, step_size=step_size,
            override=True, image_size=(224, 224), dpi=100,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
        )
        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="series",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        loader = DataLoader(dataset, batch_size=BATCH_ENC, shuffle=False)

        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            backbone.model, loader, PATCH_PX, DEVICE,
        )
        L  = large_embeds.shape[0]
        ph = pw = 224 // PATCH_PX   # 14

        anomaly_maps = []
        with torch.no_grad():
            for i in range(L):
                l_ref, _ = get_local_reference(large_embeds, i, LOCAL_K, MIN_REF)
                m_ref, _ = get_local_reference(mid_embeds,   i, LOCAL_K, MIN_REF)
                p_ref, _ = get_local_reference(patch_embeds, i, LOCAL_K, MIN_REF)

                m_l = compute_dissimilarity_with_ref(
                    large_embeds[i].unsqueeze(0).to(DEVICE), l_ref.to(DEVICE))
                m_m = compute_dissimilarity_with_ref(
                    mid_embeds[i].unsqueeze(0).to(DEVICE), m_ref.to(DEVICE))
                m_p = compute_dissimilarity_with_ref(
                    patch_embeds[i].unsqueeze(0).to(DEVICE), p_ref.to(DEVICE))

                m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(DEVICE)
                m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(DEVICE)
                m_p = m_p.reshape((1, ph, pw)).to(DEVICE)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                         nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(224, 224), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        ltr_1d   = stitch_anomaly_maps(maps_arr, STEP_RATIO, AGG_PERCENT)

    score_ltr = align_anomaly_vector(ltr_1d, T_full, WINDOW_SIZE, step_size, n_windows)
    return score_ltr, timestamps


def compute_tfg_scores(data: pd.DataFrame) -> np.ndarray:
    """TFG (Temporal Feature Gradient).  Returns scores [T]."""
    backbone = _get_backbone()
    values, timestamps = orion_to_internal(data)
    T_full      = len(values)
    values_proc = apply_ewma(preprocess_time_series(values), 1.0)
    step_size   = STEP_SIZE
    window_starts = list(range(0, T_full - WINDOW_SIZE + 1, step_size))

    with tempfile.TemporaryDirectory() as tmp:
        ok = draw_windowed_images(
            base_series_id="series", save_path=tmp,
            time_series=values_proc, time_points=np.arange(len(values_proc)),
            window_size=WINDOW_SIZE, step_size=step_size,
            override=True, image_size=(224, 224), dpi=100,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
            # Note: no save_image=False — keeps series_line_img.npy in tmp
        )
        if not ok:
            return np.zeros(T_full, dtype=np.float32)
        npy_path   = os.path.join(tmp, "series_line_img.npy")
        images_raw = torch.from_numpy(np.load(npy_path)).float()

    col_feat  = _extract_column_features(images_raw, backbone.model, DEVICE)
    col_score = _temporal_gradient_scores(col_feat)
    z_score   = _local_zscore(col_score, k=LOCAL_K)

    s_tfg = np.zeros(T_full, dtype=np.float32)
    for wi, ws in enumerate(window_starts):
        for col in range(GRID_DIM):
            t0 = ws + col * PATCH_PX
            t1 = min(t0 + PATCH_PX, T_full)
            s_tfg[t0:t1] = np.maximum(s_tfg[t0:t1], z_score[wi, col])

    return s_tfg


def compute_recon_scores(data: pd.DataFrame) -> np.ndarray:
    """MAE pixel-space reconstruction error.  Returns scores [T]."""
    recon_model = _get_recon()
    values, timestamps = orion_to_internal(data)
    T_full      = len(values)
    values_proc = apply_ewma(preprocess_time_series(values), 1.0)
    step_size   = STEP_SIZE
    window_starts = list(range(0, T_full - WINDOW_SIZE + 1, step_size))

    with tempfile.TemporaryDirectory() as tmp:
        ok = draw_windowed_images(
            base_series_id="series", save_path=tmp,
            time_series=values_proc, time_points=np.arange(len(values_proc)),
            window_size=WINDOW_SIZE, step_size=step_size,
            override=True, save_image=False,
            image_size=(224, 224), dpi=100,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
        )
        if not ok:
            return np.zeros(T_full, dtype=np.float32)

        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="series",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        loader  = DataLoader(dataset, batch_size=BATCH_ENC, shuffle=False)
        raw, wids = [], []
        for batch in loader:
            errs = recon_model.score_batch(batch["img"])
            raw.extend(errs.tolist())
            wids.extend(batch["window_id"].tolist())

    order        = sorted(range(len(wids)), key=lambda i: wids[i])
    recon_scores = np.array([raw[i] for i in order], dtype=np.float32)

    s_recon = np.zeros(T_full, dtype=np.float32)
    for wi in range(min(len(window_starts), len(recon_scores))):
        ws = window_starts[wi]
        t1 = min(ws + WINDOW_SIZE, T_full)
        s_recon[ws:t1] = np.maximum(s_recon[ws:t1], recon_scores[wi])

    return s_recon


# ============================================================================
# Evaluation helpers
# ============================================================================

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


# ============================================================================
# Dataset config
# ============================================================================

gt       = load_gt(ANOM_CSV)
nab_files = sorted(f for f in os.listdir(NAB_DIR) if f.endswith(".csv"))
NAB_SIGS  = [f[:-4] for f in nab_files if gt.get(f[:-4])]
SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
              "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
              "T-13","D-16","T-8","P-14","D-14"]

DATASET_CONFIGS = {
    "NAB":  {"dir": NAB_DIR,  "sigs": NAB_SIGS},
    "SMAP": {"dir": SMAP_DIR, "sigs": SMAP_SIGS},
    "MSL":  {"dir": MSL_DIR,  "sigs": MSL_SIGS},
}


# ============================================================================
# Main experiment loop
# ============================================================================

rows: list = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*70}\nDataset: {ds_name}  ({len(cfg['sigs'])} signals)\n{'='*70}")

    for sig in cfg["sigs"]:
        csv_path = os.path.join(cfg["dir"], f"{sig}.csv")
        gt_ivs   = gt.get(sig, [])
        if not os.path.exists(csv_path) or not gt_ivs:
            print(f"  SKIP {sig}")
            continue

        print(f"\n  [{sig}]")
        data    = pd.read_csv(csv_path)
        sig_row = {"dataset": ds_name, "signal": sig}

        ltr_ckpt   = os.path.join(MGMR_CKPT_DIR,  f"{ds_name}__{sig}__ltr.pkl")
        tfg_ckpt   = os.path.join(MGMR_CKPT_DIR,  f"{ds_name}__{sig}__tfg.pkl")
        recon_ckpt = os.path.join(RECON_CKPT_DIR, f"{ds_name}__{sig}__recon.pkl")

        # ── Signal A: LTR ────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(ltr_ckpt):
                c          = pickle.load(open(ltr_ckpt, "rb"))
                s_ltr      = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [LTR]   cache hit  ({time.time()-t0:.1f}s)")
            else:
                s_ltr, timestamps = compute_ltr_scores(data)
                pickle.dump({"scores": s_ltr, "timestamps": timestamps},
                            open(ltr_ckpt, "wb"))
                print(f"    [LTR]   computed   ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [LTR]   ERROR: {e}")
            continue

        # ── Signal B: TFG ────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(tfg_ckpt):
                c     = pickle.load(open(tfg_ckpt, "rb"))
                s_tfg = c["scores"]
                print(f"    [TFG]   cache hit  ({time.time()-t0:.1f}s)")
            else:
                s_tfg = compute_tfg_scores(data)
                pickle.dump({"scores": s_tfg}, open(tfg_ckpt, "wb"))
                print(f"    [TFG]   computed   ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [TFG]   ERROR: {e}")
            continue

        # ── Signal C: Recon ──────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(recon_ckpt):
                c       = pickle.load(open(recon_ckpt, "rb"))
                # key is 'scores' (from compare_ltr_recon_additive)
                # fall back to 'mean' (from compare_recon_discrepancy)
                s_recon = c.get("scores", c.get("mean"))
                if s_recon is None:
                    raise KeyError(f"Neither 'scores' nor 'mean' found in {recon_ckpt}")
                print(f"    [Recon] cache hit  ({time.time()-t0:.1f}s)")
            else:
                s_recon = compute_recon_scores(data)
                pickle.dump({"scores": s_recon}, open(recon_ckpt, "wb"))
                print(f"    [Recon] computed   ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [Recon] ERROR: {e}")
            continue

        # ── Align to common length ────────────────────────────────────
        T        = min(len(s_ltr), len(s_tfg), len(s_recon))
        ts_trim  = timestamps[:T]

        # ── Ablation fusions ─────────────────────────────────────────
        for name, w_tfg, w_recon in ABLATIONS:
            try:
                s_fused   = triple_fuse(s_ltr, s_tfg, s_recon, w_tfg, w_recon)
                ivs       = evt_detect(s_fused, ts_trim)
                f1        = _eval(ivs, gt_ivs)
            except Exception as e:
                print(f"    {name} ERROR: {e}")
                f1 = float("nan")
            sig_row[f"{name}_f1"] = f1

        # Print per-signal summary
        base = sig_row.get("baseline_f1", float("nan"))
        best = max((v for k, v in sig_row.items() if k.endswith("_f1") and v == v),
                   default=float("nan"))
        best_name = max(
            ((k, v) for k, v in sig_row.items() if k.endswith("_f1") and v == v),
            key=lambda kv: kv[1], default=("?", float("nan"))
        )[0].replace("_f1", "")
        print(f"    baseline={base:.4f}  best={best:.4f} ({best_name})")

        rows.append(sig_row)


# ============================================================================
# Results table
# ============================================================================

results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

COLS   = [name for name, _, _ in ABLATIONS]
COL_W  = 13
SEP    = 36 + COL_W * len(COLS)


def _fmt(v):
    try:
        f = float(v)
        return "  NaN" if (f != f) else f"{f:.4f}"
    except Exception:
        return "  NaN"


def _print_table(ds_name: str, df: pd.DataFrame):
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — Triple Fusion Ablation ===")
    print(f"{'='*SEP}")
    print(f"{'Signal':<36}" + "".join(f"{c:>{COL_W}}" for c in COLS))
    print("-" * SEP)
    for _, r in sub.iterrows():
        fs   = [r.get(f"{c}_f1", float("nan")) for c in COLS]
        best = max((f for f in fs if f == f), default=float("nan"))
        line = f"{r['signal']:<36}"
        for f in fs:
            mark = "→" if (f == f and abs(f - best) < 1e-9) else " "
            line += f"{mark + _fmt(f):>{COL_W}}"
        print(line)
    print("-" * SEP)
    avgs = [sub[f"{c}_f1"].mean() for c in COLS]
    print(f"{'AVG F1':<36}" + "".join(f" {a:>{COL_W-1}.4f}" for a in avgs))


for ds in ["NAB", "SMAP", "MSL"]:
    _print_table(ds, results_df)


# ── Cross-dataset summary ────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("OVERALL SUMMARY  (Triple Fusion: norm(LTR) + w_tfg*norm(TFG) + w_recon*norm(Recon))")
print(f"{'='*SEP}")
print(f"{'Dataset':<14}" + "".join(f"{c:>{COL_W}}" for c in COLS))
print("-" * SEP)

all_f1: dict[str, list] = {c: [] for c in COLS}
for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    row = f"{ds:<14}"
    for c in COLS:
        v = sub[f"{c}_f1"].mean() if f"{c}_f1" in sub else float("nan")
        row += f" {v:>{COL_W-1}.4f}"
        if not np.isnan(v):
            all_f1[c].append(v)
    print(row)

print("-" * SEP)
overall = f"{'ALL':<14}"
for c in COLS:
    vals = all_f1[c]
    overall += f" {np.mean(vals):>{COL_W-1}.4f}" if vals else f"{'NaN':>{COL_W}}"
print(overall)

print(f"\nResults saved → {out_csv}")
