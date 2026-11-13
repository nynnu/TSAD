"""
LTR k=5 + MAE Reconstruction Error — Additive Score Fusion
===========================================================

Problem with product fusion (ltr_x_recon):
  LTR = 0 on some SMAP signals (F-2, T-1) → product = 0 regardless of Recon.
  → Recon's correct detection is silenced.

Fix: additive fusion before thresholding.
  s_combined = normalize_01(s_LTR) + w * normalize_01(s_Recon)

Why this should work:
  - SMAP F-2: LTR≈0 but Recon=1.0 → combined still fires on Recon peak
  - NAB/MSL: LTR dominates (strong signal); Recon adds marginal precision boost
  - No zero-product problem

Ablation:
  baseline    : LTR k=5 + alpha=0.01             (reference, same as compare_mgmr)
  recon_only  : Recon + EVT                       (upper bound on Recon alone)
  ltr_w03     : norm(LTR) + 0.3*norm(Recon) + EVT
  ltr_w05     : norm(LTR) + 0.5*norm(Recon) + EVT
  ltr_w07     : norm(LTR) + 0.7*norm(Recon) + EVT
  ltr_w10     : norm(LTR) + 1.0*norm(Recon) + EVT

Checkpoints:
  LTR  : reuses results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl
  Recon: results_ltr_recon_additive/checkpoints/{DS}__{sig}__recon.pkl  (new)

Usage (Lightning AI / Colab):
  python experiments/compare_ltr_recon_additive.py
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
    ["pip", "install", "timm", "open-clip-torch", "scipy", "transformers", "--quiet"],
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
from models.mae_recon_ad import MAE_Recon
from evaluation.evaluate import evaluate_intervals
from torch.utils.data import DataLoader

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
NAB_DIR     = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR    = os.path.join(DATA_DIR, "SMAP")
MSL_DIR     = os.path.join(DATA_DIR, "MSL")
ANOM_CSV    = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results_ltr_recon_additive")
RECON_CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(RECON_CKPT_DIR, exist_ok=True)

# LTR checkpoints: try the pre-existing synced location first, then local
_LTR_CKPT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
LTR_CKPT_DIR = next(
    (p for p in _LTR_CKPT_CANDIDATES if os.path.isdir(p)),
    _LTR_CKPT_CANDIDATES[-1],   # fallback: local (will be created below)
)
os.makedirs(LTR_CKPT_DIR, exist_ok=True)
print(f"LTR ckpt dir : {LTR_CKPT_DIR}  ({len(os.listdir(LTR_CKPT_DIR))} files)")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 224
STEP_RATIO     = 4.0
STEP_SIZE      = int(WINDOW_SIZE / STEP_RATIO)   # 56
PATCH_PX       = 16
GRID_DIM       = 14
BATCH_ENC      = 16

EVT_Q_INIT     = 0.90
EVT_FPR        = 0.01

FUSION_WEIGHTS = [0.3, 0.5, 0.7, 1.0]

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

_INET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================================
# EVT threshold (identical to compare_score_fusion.py)
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


def evt_detect(scores: np.ndarray, timestamps: np.ndarray,
               q_init: float = EVT_Q_INIT) -> list:
    if (scores.max() - scores.min()) < 1e-8:
        return []
    threshold = evt_threshold(scores, q_init=q_init)
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
# Normalisation / fusion helpers
# ============================================================================

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def additive_fusion(s_ltr: np.ndarray, s_recon: np.ndarray, w: float) -> np.ndarray:
    """normalize_01(LTR) + w * normalize_01(Recon), length-aligned."""
    T = min(len(s_ltr), len(s_recon))
    return normalize_01(s_ltr[:T]) + w * normalize_01(s_recon[:T])


# ============================================================================
# Signal A — LTR k=5  (reuses MGMR checkpoints)
# ============================================================================

class ViT4TS_SignalA_LTR(ViT4TS_MAE):
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
        h  = w  = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size
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
                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())
        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)


# ============================================================================
# Signal B — MAE Reconstruction Error
# ============================================================================

def compute_recon_scores(data: pd.DataFrame, recon_model: MAE_Recon,
                         window_size: int = WINDOW_SIZE,
                         step_size: int = STEP_SIZE,
                         standardize: bool = True,
                         smoothing_alpha: float = 1.0,
                         image_size: tuple = (224, 224),
                         dpi: int = 100) -> tuple:
    """
    Compute per-timestep reconstruction error scores.

    Per-window MAE recon error → aggregate to per-timestep by max-pooling
    overlapping windows.
    """
    values, timestamps = orion_to_internal(data)
    T_full = len(values)

    values_proc = preprocess_time_series(values) if standardize else values.astype(float)
    values_proc = apply_ewma(values_proc, smoothing_alpha)

    window_starts = list(range(0, T_full - window_size + 1, step_size))
    n_windows     = len(window_starts)

    with tempfile.TemporaryDirectory() as tmp:
        ok = draw_windowed_images(
            base_series_id="series", save_path=tmp,
            time_series=values_proc, time_points=np.arange(len(values_proc)),
            override=True, window_size=window_size, step_size=step_size,
            image_size=image_size, dpi=dpi,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1) if standardize else None),
        )
        if not ok:
            return np.zeros(T_full), timestamps

        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="series",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return np.zeros(T_full), timestamps

        loader = DataLoader(dataset, batch_size=BATCH_ENC, shuffle=False)
        recon_raw, recon_wids = [], []
        for batch in loader:
            imgs = batch["img"]
            wids = batch["window_id"]
            errs = recon_model.score_batch(imgs)   # [B] per-window MSE
            recon_raw.extend(errs.tolist())
            recon_wids.extend(wids.tolist())

    # Sort by window_id and map to timesteps
    order         = sorted(range(len(recon_wids)), key=lambda i: recon_wids[i])
    recon_scores  = np.array([recon_raw[i] for i in order], dtype=np.float32)

    # Aggregate: each window covers [ws, ws+window_size), take max across overlaps
    s_recon = np.zeros(T_full, dtype=np.float32)
    for wi in range(min(n_windows, len(recon_scores))):
        ws = window_starts[wi]
        t0 = ws
        t1 = min(ws + window_size, T_full)
        s_recon[t0:t1] = np.maximum(s_recon[t0:t1], recon_scores[wi])

    return s_recon, timestamps


# ============================================================================
# Evaluation helpers
# ============================================================================

def load_gt(anom_csv):
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


def _eval(detected, gt_ivs):
    m = evaluate_intervals(gt_ivs, detected)
    return {"f1": m["F1"], "p": m["precision"], "r": m["recall"]}


def _ivs_from_scores(scores, timestamps, alpha=0.01):
    from models.model_utils import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


# ============================================================================
# Model init
# ============================================================================
print("\n[INFO] Loading models ...")
det_ltr   = ViT4TS_SignalA_LTR(**BASE_PARAMS, local_k=5, min_ref=5)
mae_recon = MAE_Recon(device=DEVICE, n_iter=5, mask_ratio=0.5)
print(
    f"  Signal A : LTR k=5  (3-scale cosine, local temporal reference)\n"
    f"  Signal B : MAE Recon (pixel-space reconstruction error, n_iter=5)\n"
    f"  Fusion   : norm(LTR) + w*norm(Recon), EVT threshold\n"
    f"  Weights  : {FUSION_WEIGHTS}"
)


# ============================================================================
# Dataset config  (same signal set as compare_mgmr_scoring.py)
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

METHODS = ["baseline", "recon_only"] + [f"w{str(w).replace('.','')}" for w in FUSION_WEIGHTS]


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

        print(f"\n  [{sig}]")
        data    = pd.read_csv(csv_path)
        sig_row = {"dataset": ds_name, "signal": sig}

        ltr_ckpt   = os.path.join(LTR_CKPT_DIR,   f"{ds_name}__{sig}__ltr.pkl")
        recon_ckpt = os.path.join(RECON_CKPT_DIR, f"{ds_name}__{sig}__recon.pkl")

        # ── Signal A: LTR scores ──────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(ltr_ckpt):
                c          = pickle.load(open(ltr_ckpt, "rb"))
                s_ltr      = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [LTR] cache hit  ({time.time()-t0:.1f}s)")
            else:
                s_ltr, timestamps = det_ltr.predict_scores(data)
                pickle.dump({"scores": s_ltr, "timestamps": timestamps},
                            open(ltr_ckpt, "wb"))
                print(f"    [LTR] computed   ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [LTR] ERROR: {e}")
            s_ltr = timestamps = None

        # ── Signal B: Recon scores ────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(recon_ckpt):
                c       = pickle.load(open(recon_ckpt, "rb"))
                s_recon = c["scores"]
                print(f"    [Recon] cache hit  ({time.time()-t0:.1f}s)")
            else:
                s_recon, _ = compute_recon_scores(data, mae_recon)
                pickle.dump({"scores": s_recon}, open(recon_ckpt, "wb"))
                print(f"    [Recon] computed   ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [Recon] ERROR: {e}")
            s_recon = None

        if s_ltr is None or s_recon is None:
            continue

        # ── Baseline: LTR + alpha=0.01 ────────────────────────────
        try:
            ivs_base = _ivs_from_scores(s_ltr, timestamps, alpha=0.01)
            m_base   = _eval(ivs_base, gt_ivs)
        except Exception as e:
            print(f"    baseline ERROR: {e}")
            m_base = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
        sig_row["baseline_f1"] = m_base["f1"]
        print(f"    baseline     F1={m_base['f1']:.4f}  ivs={len(ivs_base)}")

        # ── Recon only + EVT ──────────────────────────────────────
        T        = min(len(s_ltr), len(s_recon))
        ts_trim  = timestamps[:T]
        try:
            ivs_recon = evt_detect(s_recon[:T], ts_trim)
            m_recon   = _eval(ivs_recon, gt_ivs)
        except Exception as e:
            print(f"    recon_only ERROR: {e}")
            m_recon = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
        sig_row["recon_only_f1"] = m_recon["f1"]
        delta = m_recon["f1"] - m_base["f1"]
        print(f"    recon_only   F1={m_recon['f1']:.4f}  ivs={len(ivs_recon)}  ({delta:+.4f})")

        # ── Additive fusion (multiple w values) ───────────────────
        for w in FUSION_WEIGHTS:
            key = f"w{str(w).replace('.', '')}"
            try:
                s_fused = additive_fusion(s_ltr, s_recon, w)
                ivs_f   = evt_detect(s_fused, ts_trim)
                m_f     = _eval(ivs_f, gt_ivs)
            except Exception as e:
                print(f"    {key} ERROR: {e}")
                m_f = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
            sig_row[f"{key}_f1"] = m_f["f1"]
            delta = m_f["f1"] - m_base["f1"]
            print(f"    LTR+{w}*Rec  F1={m_f['f1']:.4f}  ivs={len(ivs_f)}  ({delta:+.4f})")

        rows.append(sig_row)


# ============================================================================
# Results table
# ============================================================================
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

COL_W = 12
COLS  = ["baseline_f1", "recon_only_f1"] + [f"w{str(w).replace('.','')}_f1" for w in FUSION_WEIGHTS]
LABELS = {
    "baseline_f1":  "LTR_k5",
    "recon_only_f1":"Recon",
}
for w in FUSION_WEIGHTS:
    LABELS[f"w{str(w).replace('.','')}_f1"] = f"LTR+{w}R"

SEP = 36 + COL_W * len(COLS)


def _fmt(v):
    try:
        f = float(v)
        return "   NaN" if (f != f) else f"{f:.4f}"
    except Exception:
        return "   NaN"


def _print_table(ds_name, df):
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — LTR + Recon Additive Fusion ===")
    print(f"{'='*SEP}")
    header = f"{'Signal':<36}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in COLS)
    print(header)
    print("-" * SEP)
    for _, r in sub.iterrows():
        row = f"{r['signal']:<36}" + "".join(f"{_fmt(r.get(c, float('nan'))):>{COL_W}}" for c in COLS)
        print(row)
    print("-" * SEP)
    avg_row = f"{'AVG':<36}"
    for c in COLS:
        vals = sub[c].dropna()
        avg_row += f"{vals.mean():>{COL_W}.4f}" if len(vals) else f"{'NaN':>{COL_W}}"
    print(avg_row)


for ds_name in DATASET_CONFIGS:
    _print_table(ds_name, results_df)


# ============================================================================
# Cross-dataset summary
# ============================================================================
print(f"\n{'='*SEP}")
print("=== CROSS-DATASET SUMMARY (avg F1 per method) ===")
print(f"{'='*SEP}")
print(f"{'Dataset':<10}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in COLS))
print("-" * SEP)

all_avgs = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = f"{ds_name:<10}"
    for c in COLS:
        v = sub[c].mean() if c in sub else float("nan")
        row += f"{v:>{COL_W}.4f}"
        all_avgs.setdefault(c, []).append(v)
    print(row)

print("-" * SEP)
overall = f"{'ALL':<10}"
for c in COLS:
    vals = [v for v in all_avgs.get(c, []) if not np.isnan(v)]
    overall += f"{np.mean(vals):>{COL_W}.4f}" if vals else f"{'NaN':>{COL_W}}"
print(overall)

print(f"\nResults saved → {out_csv}")
