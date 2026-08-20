"""
Residual Encoding + LTR
========================
Hypothesis
----------
Rendering the RESIDUAL of a signal (signal minus trend/seasonality) instead of
the raw signal creates a more discriminative image for MAE:

  Normal windows  → near-zero residual → homogeneous, low-energy images
                    → LTR reference is tightly clustered
  Anomaly windows → large residual spike / level-shift → stands out clearly

Principles (universality check):
  - STL decomposition works on any time series (principled, not dataset-specific)
  - Period is auto-detected via ACF, so no hand-tuning per dataset
  - Residual is stationary by construction → LTR stationarity assumption is satisfied
  - Works for trend anomalies, seasonal-break anomalies, point anomalies

Decomposition
-------------
  1. Detect dominant period via ACF peak (capped at len//4)
  2. If period ≥ 4 and signal long enough: STL (robust=True)
  3. Fallback: subtract centered moving average (window = window_size // 4)

Conditions
----------
  sma_k5_k30    : add(SMA·k5, k30) + EVT       ← current best (0.6391), loaded from ckpts
  resid_k5      : LTR k=5 on residual images
  resid_add     : add(resid_k5, resid_k30) + EVT
  resid_sma_add : add(SMA·resid_k5, resid_k30) + EVT   ← main hypothesis

Usage
-----
  python experiments/compare_residual_encoding.py
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
from scipy.signal import find_peaks

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

print(f"Project root : {PROJECT_ROOT}")

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy", "statsmodels", "--quiet"],
    check=True,
)

import torch
from torch.utils.data import DataLoader

from preprocessing.preprocess import preprocess_time_series, draw_windowed_images, apply_ewma
from preprocessing.data_utils import orion_to_internal
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.sma_transform import SMADataset
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps, align_anomaly_vector
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

def _first_existing(*candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[-1]

SMA_K5_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_sma_ltr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_sma_ltr", "checkpoints"),
)
K30_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
)
K5_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_residual_encoding")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"SMA k5  : {SMA_K5_DIR}")
print(f"k30     : {K30_DIR}")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
STEP_SIZE   = int(WINDOW_SIZE / STEP_RATIO)
PATCH_SIZE  = 16
LOCAL_K     = 5
MEDIUM_K    = 30
MIN_REF     = 5
SMA_BETA    = 0.75
EVT_Q       = 0.90
EVT_FPR     = 0.01

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=PATCH_SIZE,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)


# ── Residual decomposition ────────────────────────────────────────────────────

def detect_period(values: np.ndarray, min_period: int = 4) -> int:
    """Dominant period from ACF peak. Returns 0 if no significant periodicity."""
    x       = values - values.mean()
    var     = float(np.dot(x, x))
    if var < 1e-10:
        return 0
    max_lag = min(len(x) // 4, 500)
    if max_lag < min_period:
        return 0
    acf = np.array([float(np.dot(x[:-lag], x[lag:])) / var
                    for lag in range(1, max_lag + 1)])
    peaks, _ = find_peaks(acf, height=0.15, distance=min_period)
    if len(peaks) == 0:
        return 0
    return int(peaks[0]) + 1   # lag offset: index 0 = lag 1


def compute_residual(values: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    Decompose values and return the residual component.

    Strategy:
      1. Detect dominant period via ACF.
      2. If period ≥ 4 and series has ≥ 2 full periods: STL (robust).
      3. Fallback: subtract centered moving-average trend (window = W//4).
    """
    period = detect_period(values)

    if period >= 4 and len(values) >= 2 * period:
        try:
            from statsmodels.tsa.seasonal import STL
            result = STL(values, period=period, robust=True).fit()
            residual = result.resid.astype(np.float32)
            # Re-standardize residual so renderer gets unit-variance input
            std = residual.std()
            if std > 1e-8:
                residual = residual / std
            return residual
        except Exception:
            pass  # fall through to MA fallback

    # MA trend removal — window = window_size // 4 (principled: removes trends
    # longer than 1/4 of the render window, preserves shorter structure)
    k    = max(window_size // 4, 5)
    half = k // 2
    pad  = np.pad(values, (half, half), mode="edge")
    trend = np.convolve(pad, np.ones(k) / k, mode="valid")[:len(values)]
    residual = (values - trend).astype(np.float32)
    std = residual.std()
    if std > 1e-8:
        residual = residual / std
    return residual


# ── EVT ───────────────────────────────────────────────────────────────────────

def evt_threshold(scores: np.ndarray) -> float:
    u           = float(np.percentile(scores, EVT_Q * 100))
    exceedances = scores[scores > u] - u
    fallback    = float(np.percentile(scores, (1.0 - EVT_FPR) * 100))
    if len(exceedances) < 10:
        return fallback
    try:
        c, _, scale = genpareto.fit(exceedances, floc=0)
        p_cond = min(EVT_FPR / max(1.0 - EVT_Q, 1e-9), 1.0 - 1e-9)
        thr    = u + max(0.0, float(genpareto.ppf(1.0 - p_cond, c, loc=0, scale=scale)))
        return thr if u <= thr <= scores.max() else fallback
    except Exception:
        return fallback

def evt_detect(scores: np.ndarray, timestamps: np.ndarray) -> list:
    if scores.max() - scores.min() < 1e-8:
        return []
    thr   = evt_threshold(scores)
    flags = scores > thr
    if not flags.any():
        return []
    ivs, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg:
            in_seg = True; s = i
        elif not f and in_seg:
            in_seg = False; ivs.append([timestamps[s], timestamps[i - 1]])
    if in_seg:
        ivs.append([timestamps[s], timestamps[len(flags) - 1]])
    return ivs

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-8 else (arr - lo) / (hi - lo)


# ── Residual LTR detector ─────────────────────────────────────────────────────

class ResidualLTR(ViT4TS_MAE):
    """LTR detector that renders the RESIDUAL signal instead of raw signal."""

    def __init__(self, *args, local_k: int = LOCAL_K, min_ref: int = MIN_REF,
                 use_sma: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_k = local_k
        self.min_ref = min_ref
        self.use_sma = use_sma

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values_raw, timestamps = orion_to_internal(data)
        T_full = len(values_raw)

        # Standard preprocessing
        values_proc = preprocess_time_series(values_raw) if self.standardize \
                      else values_raw.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        # ── Residual decomposition ────────────────────────────────────────────
        period   = detect_period(values_proc)
        residual = compute_residual(values_proc, self.window_size)
        if self.verbose:
            print(f"  period={period}  residual std={residual.std():.4f}")

        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        with tempfile.TemporaryDirectory() as tmp:
            plot_params = ("-", 1, "*", 0.1, "black", None)  # no fixed y-range for residual
            ok = draw_windowed_images(
                base_series_id="series", save_path=tmp,
                time_series=residual, time_points=np.arange(T_full),
                window_size=self.window_size, step_size=step_size,
                override=True, save_image=False,
                image_size=self.image_size, dpi=self.dpi,
                plot_params=plot_params,
            )
            if not ok:
                return np.zeros(T_full), timestamps

            dataset = CLIPTimeSeriesDataset(
                results_dir=tmp, base_series_id="series",
                sample_size=None, no_anomaly=True, plot_type="line",
            )
            if self.use_sma:
                dataset = SMADataset(dataset, beta=SMA_BETA)

            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            (large_e, mid_e, patch_e,
             l_mask, m_mask, _) = build_ordered_embeddings(
                self.model, loader, self.patch_size, self.device)

        L  = large_e.shape[0]
        ph = pw = self.image_size[0] // self.patch_size
        maps = []
        with torch.no_grad():
            for i in range(L):
                l_ref, _ = get_local_reference(large_e, i, self.local_k, self.min_ref)
                m_ref, _ = get_local_reference(mid_e,   i, self.local_k, self.min_ref)
                p_ref, _ = get_local_reference(patch_e, i, self.local_k, self.min_ref)
                m_l = compute_dissimilarity_with_ref(
                    large_e[i].unsqueeze(0).to(self.device), l_ref.to(self.device))
                m_m = compute_dissimilarity_with_ref(
                    mid_e[i].unsqueeze(0).to(self.device),   m_ref.to(self.device))
                m_p = compute_dissimilarity_with_ref(
                    patch_e[i].unsqueeze(0).to(self.device), p_ref.to(self.device))
                m_l = harmonic_aggregation((1, ph, pw), m_l, l_mask).to(self.device)
                m_m = harmonic_aggregation((1, ph, pw), m_m, m_mask).to(self.device)
                m_p = m_p.reshape((1, ph, pw)).to(self.device)
                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=self.image_size, mode="bilinear").squeeze(1)
                maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(maps, dim=0).numpy()
        raw_scores = stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)
        aligned    = align_anomaly_vector(raw_scores, T_full, self.window_size, step_size, n_windows)
        return aligned, timestamps


# ── Model init ────────────────────────────────────────────────────────────────
print("\n[INFO] Loading models ...")
det_k5       = ResidualLTR(**BASE_PARAMS, local_k=LOCAL_K,  use_sma=False)
det_k30      = ResidualLTR(**BASE_PARAMS, local_k=MEDIUM_K, use_sma=False)
det_sma_k5   = ResidualLTR(**BASE_PARAMS, local_k=LOCAL_K,  use_sma=True)
print("  Models loaded.")


# ── Dataset config ────────────────────────────────────────────────────────────

def load_gt():
    gt = {}
    with open(ANOM_CSV, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt

gt       = load_gt()
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
NAB_SIGS = sorted(f[:-4] for f in os.listdir(NAB_DIR)
                  if f.endswith(".csv") and gt.get(f[:-4]))

SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
             "T-13","D-16","T-8","P-14","D-14"]

DATASET_CONFIGS = {
    "NAB":  {"dir": NAB_DIR,                         "sigs": NAB_SIGS},
    "SMAP": {"dir": os.path.join(DATA_DIR, "SMAP"),  "sigs": SMAP_SIGS},
    "MSL":  {"dir": os.path.join(DATA_DIR, "MSL"),   "sigs": MSL_SIGS},
}


# ── Reference score loader ────────────────────────────────────────────────────

def load_ref_scores(ds, sig):
    """Load SMA·k5 and k30 scores for current-best reference: add(SMA·k5, k30)."""
    p_sma = os.path.join(SMA_K5_DIR, f"{ds}__{sig}__ltr_sma075.pkl")
    p_k30 = os.path.join(K30_DIR,    f"{ds}__{sig}__ltr_k30.pkl")
    p_k5  = os.path.join(K5_DIR,     f"{ds}__{sig}__ltr.pkl")
    s_sma = pickle.load(open(p_sma, "rb"))["scores"] if os.path.exists(p_sma) else None
    s_k30 = pickle.load(open(p_k30, "rb"))["scores"] if os.path.exists(p_k30) else None
    ts    = pickle.load(open(p_k5,  "rb"))["timestamps"] if os.path.exists(p_k5) else None
    return s_sma, s_k30, ts


# ── Experiment loop ───────────────────────────────────────────────────────────
rows: list = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}  ({len(cfg['sigs'])} signals)")
    print(f"{'='*60}")

    for sig in cfg["sigs"]:
        csv_path = os.path.join(cfg["dir"], f"{sig}.csv")
        gt_ivs   = gt.get(sig, [])
        if not os.path.exists(csv_path) or not gt_ivs:
            continue

        print(f"\n  [{sig}]")
        t0   = time.time()
        data = pd.read_csv(csv_path)
        row  = {"dataset": ds_name, "signal": sig}

        # ── Reference: add(SMA·k5, k30) ─────────────────────────────────────
        s_sma, s_k30, ts_ref = load_ref_scores(ds_name, sig)
        if s_sma is not None and s_k30 is not None and ts_ref is not None:
            T = min(len(s_sma), len(s_k30), len(ts_ref))
            s_ref = normalize_01(s_sma[:T]) + normalize_01(s_k30[:T])
            row["sma_k5_k30_f1"] = evaluate_intervals(
                gt_ivs, evt_detect(s_ref, ts_ref[:T]))["F1"]
        else:
            row["sma_k5_k30_f1"] = float("nan")

        # ── Residual k5 ──────────────────────────────────────────────────────
        ckpt_k5 = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__resid_k5.pkl")
        try:
            if os.path.exists(ckpt_k5):
                c      = pickle.load(open(ckpt_k5, "rb"))
                s_rk5  = c["scores"]; ts = c["timestamps"]
            else:
                s_rk5, ts = det_k5.predict_scores(data)
                pickle.dump({"scores": s_rk5, "timestamps": ts}, open(ckpt_k5, "wb"))
            row["resid_k5_f1"] = evaluate_intervals(
                gt_ivs, evt_detect(s_rk5, ts))["F1"]
        except Exception as e:
            print(f"    resid_k5 ERROR: {e}"); s_rk5 = ts = None
            row["resid_k5_f1"] = float("nan")

        # ── Residual k30 ─────────────────────────────────────────────────────
        ckpt_k30 = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__resid_k30.pkl")
        try:
            if os.path.exists(ckpt_k30):
                c      = pickle.load(open(ckpt_k30, "rb"))
                s_rk30 = c["scores"]
            else:
                s_rk30, _ = det_k30.predict_scores(data)
                pickle.dump({"scores": s_rk30, "timestamps": ts}, open(ckpt_k30, "wb"))
            # fused: add(resid_k5, resid_k30)
            if s_rk5 is not None and ts is not None:
                T = min(len(s_rk5), len(s_rk30), len(ts))
                s_radd = normalize_01(s_rk5[:T]) + normalize_01(s_rk30[:T])
                row["resid_add_f1"] = evaluate_intervals(
                    gt_ivs, evt_detect(s_radd, ts[:T]))["F1"]
            else:
                row["resid_add_f1"] = float("nan")
        except Exception as e:
            print(f"    resid_k30 ERROR: {e}"); s_rk30 = None
            row["resid_add_f1"] = float("nan")

        # ── SMA·Residual k5 + Residual k30 ───────────────────────────────────
        ckpt_sma = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__resid_sma_k5.pkl")
        try:
            if os.path.exists(ckpt_sma):
                c        = pickle.load(open(ckpt_sma, "rb"))
                s_rsma   = c["scores"]
            else:
                s_rsma, _ = det_sma_k5.predict_scores(data)
                pickle.dump({"scores": s_rsma, "timestamps": ts}, open(ckpt_sma, "wb"))
            if s_rk30 is not None and ts is not None:
                T = min(len(s_rsma), len(s_rk30), len(ts))
                s_final = normalize_01(s_rsma[:T]) + normalize_01(s_rk30[:T])
                row["resid_sma_add_f1"] = evaluate_intervals(
                    gt_ivs, evt_detect(s_final, ts[:T]))["F1"]
            else:
                row["resid_sma_add_f1"] = float("nan")
        except Exception as e:
            print(f"    resid_sma_k5 ERROR: {e}")
            row["resid_sma_add_f1"] = float("nan")

        rows.append(row)
        elapsed = time.time() - t0
        print(f"    ref={row['sma_k5_k30_f1']:.4f}  "
              f"resid_k5={row['resid_k5_f1']:.4f}  "
              f"resid_add={row['resid_add_f1']:.4f}  "
              f"resid_sma_add={row['resid_sma_add_f1']:.4f}  "
              f"({elapsed:.0f}s)")


# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "residual_encoding.csv"), index=False)

sig_counts = {"NAB": 16, "SMAP": 13, "MSL": 11}
F1_COLS    = ["sma_k5_k30_f1", "resid_k5_f1", "resid_add_f1", "resid_sma_add_f1"]
LABELS     = ["sma_k5_k30", "resid_k5", "resid_add", "resid_sma_add"]

COL_W = 14
print(f"\n{'='*80}")
print("SUMMARY — Residual Encoding vs Current Best")
print(f"{'='*80}")
print(f"{'Dataset':<10}" + "".join(f"{l:>{COL_W}}" for l in LABELS))
print("-" * (10 + COL_W * len(LABELS)))

wtd = {c: 0.0 for c in F1_COLS}
for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty: continue
    avgs = [sub[c].mean() for c in F1_COLS]
    print(f"{ds:<10}" + "".join(f"{a:>{COL_W}.4f}" for a in avgs))
    for c, a in zip(F1_COLS, avgs):
        wtd[c] += a * sig_counts[ds]

total = sum(sig_counts.values())
print("-" * (10 + COL_W * len(LABELS)))
print(f"{'ALL (wtd)':<10}" + "".join(f"{wtd[c]/total:>{COL_W}.4f}" for c in F1_COLS))

best_col = max(F1_COLS, key=lambda c: wtd[c])
delta    = (wtd[best_col] - wtd["sma_k5_k30_f1"]) / total
print(f"\nBest new method: {best_col.replace('_f1','')}  "
      f"(ALL = {wtd[best_col]/total:.4f},  Δ vs current best = {delta:+.4f})")
print(f"Results → {RESULTS_DIR}/residual_encoding.csv")
