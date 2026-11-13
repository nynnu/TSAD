"""
Residual Encoding — Focused Probe on Diagnostic Signals
=========================================================
Tests only the 5 signals that determine whether residual encoding
is worth pursuing:

  SMAP F-1  : stays 0? → residual can't help invisible anomalies
  SMAP F-3  : stays 0? → residual can't help stuck sensor
  SMAP F-2  : improves/holds? → long-duration anomaly
  SMAP E-3  : improves/holds? → partial fix signal
  MSL  T-12 : stable or collapses? → threshold-sensitive signal

Run:
  python experiments/probe_residual_encoding.py
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

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

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
    build_ordered_embeddings, get_local_reference, compute_dissimilarity_with_ref,
)
from evaluation.evaluate import evaluate_intervals

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Probe targets ─────────────────────────────────────────────────────────────
PROBE_SIGNALS = [
    ("SMAP", "F-1"),
    ("SMAP", "F-3"),
    ("SMAP", "F-2"),
    ("SMAP", "E-3"),
    ("MSL",  "T-12"),
]

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")
DS_DIRS  = {
    "SMAP": os.path.join(DATA_DIR, "SMAP"),
    "MSL":  os.path.join(DATA_DIR, "MSL"),
}

CKPT_DIR = os.path.join(PROJECT_ROOT, "results_residual_encoding", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Reference checkpoint dirs ─────────────────────────────────────────────────
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

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
PATCH_SIZE  = 16
LOCAL_K     = 5
MEDIUM_K    = 30
MIN_REF     = 5
SMA_BETA    = 0.75
EVT_Q, EVT_FPR = 0.90, 0.01

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
    x   = values - values.mean()
    var = float(np.dot(x, x))
    if var < 1e-10:
        return 0
    max_lag = min(len(x) // 4, 500)
    if max_lag < min_period:
        return 0
    acf = np.array([float(np.dot(x[:-lag], x[lag:])) / var
                    for lag in range(1, max_lag + 1)])
    peaks, _ = find_peaks(acf, height=0.15, distance=min_period)
    return int(peaks[0]) + 1 if len(peaks) > 0 else 0

def compute_residual(values: np.ndarray) -> np.ndarray:
    period = detect_period(values)
    if period >= 4 and len(values) >= 2 * period:
        try:
            from statsmodels.tsa.seasonal import STL
            res = STL(values, period=period, robust=True).fit().resid.astype(np.float32)
            std = res.std()
            return res / std if std > 1e-8 else res
        except Exception:
            pass
    k    = max(WINDOW_SIZE // 4, 5)
    half = k // 2
    pad  = np.pad(values, (half, half), mode="edge")
    trend = np.convolve(pad, np.ones(k) / k, mode="valid")[:len(values)]
    res   = (values - trend).astype(np.float32)
    std   = res.std()
    return res / std if std > 1e-8 else res

# ── EVT ───────────────────────────────────────────────────────────────────────

def evt_detect(scores, timestamps):
    if scores.max() - scores.min() < 1e-8:
        return []
    u    = float(np.percentile(scores, EVT_Q * 100))
    exc  = scores[scores > u] - u
    fb   = float(np.percentile(scores, (1 - EVT_FPR) * 100))
    thr  = fb
    if len(exc) >= 10:
        try:
            c, _, sc = genpareto.fit(exc, floc=0)
            p = min(EVT_FPR / max(1 - EVT_Q, 1e-9), 1 - 1e-9)
            t = u + max(0., float(genpareto.ppf(1 - p, c, loc=0, scale=sc)))
            thr = t if u <= t <= scores.max() else fb
        except Exception:
            pass
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

def normalize_01(a):
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-8 else (a - lo) / (hi - lo)

# ── Residual LTR ──────────────────────────────────────────────────────────────

class ResidualLTR(ViT4TS_MAE):
    def __init__(self, *args, local_k=LOCAL_K, use_sma=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_k = local_k
        self.use_sma = use_sma

    def predict_scores(self, data):
        values_raw, timestamps = orion_to_internal(data)
        T_full      = len(values_raw)
        values_proc = preprocess_time_series(values_raw) if self.standardize \
                      else values_raw.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        period   = detect_period(values_proc)
        residual = compute_residual(values_proc)
        print(f"    period={period}  resid_std={residual.std():.3f}  "
              f"({'STL' if period >= 4 else 'MA-fallback'})")

        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        with tempfile.TemporaryDirectory() as tmp:
            ok = draw_windowed_images(
                base_series_id="series", save_path=tmp,
                time_series=residual, time_points=np.arange(T_full),
                window_size=self.window_size, step_size=step_size,
                override=True, save_image=False,
                image_size=self.image_size, dpi=self.dpi,
                plot_params=("-", 1, "*", 0.1, "black", None),
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
            (le, me, pe, lm, mm, _) = build_ordered_embeddings(
                self.model, loader, self.patch_size, self.device)

        L = le.shape[0]; ph = pw = self.image_size[0] // self.patch_size
        maps = []
        with torch.no_grad():
            for i in range(L):
                lr, _ = get_local_reference(le, i, self.local_k, MIN_REF)
                mr, _ = get_local_reference(me, i, self.local_k, MIN_REF)
                pr, _ = get_local_reference(pe, i, self.local_k, MIN_REF)
                ml = harmonic_aggregation((1, ph, pw),
                    compute_dissimilarity_with_ref(le[i].unsqueeze(0).to(self.device), lr.to(self.device)), lm).to(self.device)
                mm_ = harmonic_aggregation((1, ph, pw),
                    compute_dissimilarity_with_ref(me[i].unsqueeze(0).to(self.device), mr.to(self.device)), mm).to(self.device)
                mp = compute_dissimilarity_with_ref(
                    pe[i].unsqueeze(0).to(self.device), pr.to(self.device)).reshape((1, ph, pw)).to(self.device)
                sc = torch.nan_to_num((ml + mm_ + mp) / 3., nan=0., posinf=0., neginf=0.)
                sc = torch.nn.functional.interpolate(
                    sc.unsqueeze(1), size=self.image_size, mode="bilinear").squeeze(1)
                maps.append(sc.squeeze(0).detach().cpu())

        maps_arr = torch.stack(maps, dim=0).numpy()
        raw      = stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)
        aligned  = align_anomaly_vector(raw, T_full, self.window_size, step_size, n_windows)
        return aligned, timestamps

# ── Load ground truth ─────────────────────────────────────────────────────────
gt = {}
with open(ANOM_CSV, encoding="utf-8") as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(",", 1)
        if len(parts) == 2:
            try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
            except Exception: pass

# ── Models ────────────────────────────────────────────────────────────────────
print("\n[INFO] Loading models ...")
det_k5  = ResidualLTR(**BASE_PARAMS, local_k=LOCAL_K,  use_sma=False)
det_k30 = ResidualLTR(**BASE_PARAMS, local_k=MEDIUM_K, use_sma=False)
det_sma = ResidualLTR(**BASE_PARAMS, local_k=LOCAL_K,  use_sma=True)
print("  Done.\n")

# ── Probe loop ────────────────────────────────────────────────────────────────
print("=" * 60)
print("RESIDUAL ENCODING PROBE — 5 diagnostic signals")
print("=" * 60)

rows = []
for ds_name, sig in PROBE_SIGNALS:
    csv_path = os.path.join(DS_DIRS[ds_name], f"{sig}.csv")
    gt_ivs   = gt.get(sig, [])
    print(f"\n[{ds_name} {sig}]")
    t0   = time.time()
    data = pd.read_csv(csv_path)
    row  = {"dataset": ds_name, "signal": sig}

    # ── Reference: add(SMA·k5, k30) ──────────────────────────────────────────
    p_sma = os.path.join(SMA_K5_DIR, f"{ds_name}__{sig}__ltr_sma075.pkl")
    p_k30 = os.path.join(K30_DIR,    f"{ds_name}__{sig}__ltr_k30.pkl")
    p_k5  = os.path.join(K5_DIR,     f"{ds_name}__{sig}__ltr.pkl")
    if all(os.path.exists(p) for p in [p_sma, p_k30, p_k5]):
        s_sma = pickle.load(open(p_sma, "rb"))["scores"]
        s_k30 = pickle.load(open(p_k30, "rb"))["scores"]
        ts_r  = pickle.load(open(p_k5,  "rb"))["timestamps"]
        T     = min(len(s_sma), len(s_k30), len(ts_r))
        s_ref = normalize_01(s_sma[:T]) + normalize_01(s_k30[:T])
        row["ref_f1"] = evaluate_intervals(gt_ivs, evt_detect(s_ref, ts_r[:T]))["F1"]
    else:
        row["ref_f1"] = float("nan")

    # ── resid_k5 ─────────────────────────────────────────────────────────────
    ck5 = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__resid_k5.pkl")
    if os.path.exists(ck5):
        c = pickle.load(open(ck5, "rb")); s_rk5 = c["scores"]; ts = c["timestamps"]
        print("    [k5] cache hit")
    else:
        s_rk5, ts = det_k5.predict_scores(data)
        pickle.dump({"scores": s_rk5, "timestamps": ts}, open(ck5, "wb"))
    row["resid_k5_f1"] = evaluate_intervals(gt_ivs, evt_detect(s_rk5, ts))["F1"]

    # ── resid_k30 ────────────────────────────────────────────────────────────
    ck30 = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__resid_k30.pkl")
    if os.path.exists(ck30):
        c = pickle.load(open(ck30, "rb")); s_rk30 = c["scores"]
        print("    [k30] cache hit")
    else:
        s_rk30, _ = det_k30.predict_scores(data)
        pickle.dump({"scores": s_rk30, "timestamps": ts}, open(ck30, "wb"))
    T = min(len(s_rk5), len(s_rk30), len(ts))
    row["resid_add_f1"] = evaluate_intervals(
        gt_ivs, evt_detect(normalize_01(s_rk5[:T]) + normalize_01(s_rk30[:T]), ts[:T]))["F1"]

    # ── SMA·resid_k5 + resid_k30 ─────────────────────────────────────────────
    csma = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__resid_sma_k5.pkl")
    if os.path.exists(csma):
        c = pickle.load(open(csma, "rb")); s_rsma = c["scores"]
        print("    [sma] cache hit")
    else:
        s_rsma, _ = det_sma.predict_scores(data)
        pickle.dump({"scores": s_rsma, "timestamps": ts}, open(csma, "wb"))
    T = min(len(s_rsma), len(s_rk30), len(ts))
    row["resid_sma_add_f1"] = evaluate_intervals(
        gt_ivs, evt_detect(normalize_01(s_rsma[:T]) + normalize_01(s_rk30[:T]), ts[:T]))["F1"]

    rows.append(row)
    print(f"  → ref={row['ref_f1']:.4f}  resid_k5={row['resid_k5_f1']:.4f}  "
          f"resid_add={row['resid_add_f1']:.4f}  resid_sma_add={row['resid_sma_add_f1']:.4f}  "
          f"({time.time()-t0:.0f}s)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("PROBE SUMMARY")
print(f"{'='*65}")
print(f"{'Signal':<12} {'ref':>8} {'resid_k5':>10} {'resid_add':>11} {'resid_sma':>11}  verdict")
print("-" * 65)
for r in rows:
    best_resid = max(r["resid_k5_f1"], r["resid_add_f1"], r["resid_sma_add_f1"])
    delta      = best_resid - r["ref_f1"]
    verdict    = ("BETTER ✓" if delta > 0.05 else
                  "SAME"     if abs(delta) <= 0.05 else
                  "WORSE ✗")
    print(f"{r['dataset']+' '+r['signal']:<12} "
          f"{r['ref_f1']:>8.4f} {r['resid_k5_f1']:>10.4f} "
          f"{r['resid_add_f1']:>11.4f} {r['resid_sma_add_f1']:>11.4f}  {verdict}")
