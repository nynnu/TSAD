"""
MAE LTR with Length-Aware k_medium
====================================

Problem
-------
add(k5, k30) regresses on short MSL signals (T-13, D-16, F-7).
Root cause: n_windows ≈ 60 for these signals → k=30 reference
covers half the signal → reference IS the anomaly → dissimilarity ≈ 0.

Fix
---
  k_medium = max(K_LOCAL + 1, n_windows // 5)

  The reference covers 20% of the signal length, scaling proportionally.
  No arbitrary upper cap — longer signals get a proportionally larger context.

  Example:
    MSL T-13  : n_windows=60  → k_medium = max(6, 12)  = 12
    SMAP F-2  : n_windows=280 → k_medium = max(6, 56)  = 56
    NAB long  : n_windows=350 → k_medium = max(6, 70)  = 70

Checkpoint reuse
----------------
  k=5  : results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl         (always reused)
  k=30 : results_ltr_multiscale/checkpoints/{DS}__{sig}__ltr_k30.pkl
          → reused when k_medium == 30 (most signals)
  k=km : results_mae_kmedium/checkpoints/{DS}__{sig}__ltr_km.pkl
          → computed only when k_medium < 30 (short signals)

Ablations
---------
  mae_add_f1    : add(k5, k30_fixed) + EVT   [current best = 0.6256]
  mae_km_f1     : add(k5, k_medium)  + EVT   [hypothesis: fixes MSL regression]

Usage
-----
  python experiments/compare_mae_kmedium.py
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
from preprocessing.data_utils import orion_to_internal
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

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_SIZE   = int(WINDOW_SIZE / 4.0)   # 56
STEP_RATIO  = 4.0
AGG_PERCENT = 0.25
PATCH_SIZE  = 16
PH = PW     = WINDOW_SIZE // PATCH_SIZE  # 14
K_LOCAL     = 5
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=AGG_PERCENT, patch_size=PATCH_SIZE,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mae_kmedium")
KM_CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(KM_CKPT_DIR, exist_ok=True)

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

_count = lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0
print(f"k=5  ckpt : {K5_CKPT_DIR}  ({_count(K5_CKPT_DIR)} files)")
print(f"k=30 ckpt : {K30_CKPT_DIR}  ({_count(K30_CKPT_DIR)} files)")
print(f"k=km ckpt : {KM_CKPT_DIR}")


# ── LTR detector (parametric k) ───────────────────────────────────────────────

class LTR_Detector(ViT4TS_MAE):
    def __init__(self, *args, local_k: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_k = local_k

    def _run_inference(self, results_dir: str, base_series_id: str):
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_series_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return None

        loader = DataLoader(dataset, batch_size=20, shuffle=False)
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.model, loader, self.patch_size, self.device
        )

        L = large_embeds.shape[0]
        anomaly_maps = []
        with torch.no_grad():
            for i in range(L):
                l_ref, _ = get_local_reference(large_embeds, i, self.local_k, 5)
                m_ref, _ = get_local_reference(mid_embeds,   i, self.local_k, 5)
                p_ref, _ = get_local_reference(patch_embeds, i, self.local_k, 5)

                m_l = compute_dissimilarity_with_ref(
                    large_embeds[i].unsqueeze(0).to(self.device), l_ref.to(self.device))
                m_m = compute_dissimilarity_with_ref(
                    mid_embeds[i].unsqueeze(0).to(self.device),   m_ref.to(self.device))
                m_p = compute_dissimilarity_with_ref(
                    patch_embeds[i].unsqueeze(0).to(self.device), p_ref.to(self.device))

                m_l = harmonic_aggregation((1, PH, PW), m_l, large_mask).to(self.device)
                m_m = harmonic_aggregation((1, PH, PW), m_m, mid_mask).to(self.device)
                m_p = m_p.reshape((1, PH, PW)).to(self.device)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                         nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(224, 224), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)


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
    if scores.max() - scores.min() < 1e-8:
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


def add_fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    T = min(len(a), len(b))
    return normalize_01(a[:T]) + normalize_01(b[:T])


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


# ── Lazy model load (only if k_medium < 30 inference is needed) ──────────────
_detector: LTR_Detector | None = None


def get_detector(k: int) -> LTR_Detector:
    global _detector
    if _detector is None:
        print("\n[INFO] Loading MAE model (needed for k_medium < 30 signals) ...")
        _detector = LTR_Detector(**BASE_PARAMS, local_k=k)
        print("[INFO] MAE model ready.")
    _detector.local_k = k
    return _detector


# ── Dataset config ────────────────────────────────────────────────────────────
gt = load_gt(ANOM_CSV)

nab_files = sorted(f for f in os.listdir(NAB_DIR) if f.endswith(".csv"))
NAB_SIGS  = [f[:-4] for f in nab_files if gt.get(f[:-4])]
SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
              "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
              "T-13","D-16","T-8","P-14","D-14"]

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

        # Compute length-aware k_medium
        values, _ = orion_to_internal(data)
        T_full    = len(values)
        n_windows = int((T_full - WINDOW_SIZE) / STEP_SIZE) + 1
        k_medium  = max(K_LOCAL + 1, n_windows // 5)
        print(f"    T={T_full}  n_windows={n_windows}  k_medium={k_medium}")
        sig_row["k_medium"] = k_medium

        k5_ckpt  = os.path.join(K5_CKPT_DIR,  f"{ds_name}__{sig}__ltr.pkl")
        k30_ckpt = os.path.join(K30_CKPT_DIR, f"{ds_name}__{sig}__ltr_k30.pkl")
        km_ckpt  = os.path.join(KM_CKPT_DIR,  f"{ds_name}__{sig}__ltr_km.pkl")

        # ── Load or compute k=5 ──────────────────────────────────────────────
        s_k5 = timestamps = None
        t0 = time.time()
        if os.path.exists(k5_ckpt):
            try:
                c          = pickle.load(open(k5_ckpt, "rb"))
                s_k5       = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [k5]  cache  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [k5] load error: {e}")

        if s_k5 is None:
            try:
                det = get_detector(K_LOCAL)
                s_k5, timestamps = det.predict_scores(data)
                os.makedirs(K5_CKPT_DIR, exist_ok=True)
                pickle.dump({"scores": s_k5, "timestamps": timestamps},
                            open(k5_ckpt, "wb"))
                print(f"    [k5]  computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [k5]  ERROR: {e}")

        if s_k5 is None:
            print(f"    SKIP — k=5 failed")
            continue

        # ── baseline: k=5, alpha=0.01 ─────────────────────────────────────
        try:
            ivs = _ivs_alpha(s_k5, timestamps)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["baseline_f1"] = f1
        print(f"    baseline (k5, alpha)      F1={f1:.4f}")

        # ── mae_add: add(k5, k30_fixed) — current best reference ──────────
        s_k30 = None
        if os.path.exists(k30_ckpt):
            try:
                s_k30 = pickle.load(open(k30_ckpt, "rb"))["scores"]
            except Exception:
                pass

        if s_k30 is not None:
            try:
                T     = min(len(s_k5), len(s_k30))
                ts_t  = timestamps[:T]
                s_add = add_fuse(s_k5, s_k30)
                ivs   = evt_detect(s_add, ts_t)
                f1    = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["mae_add_f1"] = f1
            print(f"    mae_add (k5+k30, EVT)     F1={f1:.4f}")

        # ── mae_km: add(k5, k_medium) — length-aware ──────────────────────
        # If k_medium == 30: reuse k=30 checkpoint directly
        # If k_medium < 30:  load or compute new km checkpoint
        t0 = time.time()
        s_km = None

        if k_medium == 30 and s_k30 is not None:
            s_km = s_k30
            print(f"    [km={k_medium}] reused k30 checkpoint")
        else:
            # Try loading cached km checkpoint
            if os.path.exists(km_ckpt):
                try:
                    c = pickle.load(open(km_ckpt, "rb"))
                    if c.get("k_medium") == k_medium:
                        s_km = c["scores"]
                        print(f"    [km={k_medium}] cache  ({time.time()-t0:.1f}s)")
                    else:
                        print(f"    [km={k_medium}] stale cache (k={c.get('k_medium')}), recomputing")
                except Exception:
                    pass

            if s_km is None:
                try:
                    det = get_detector(k_medium)
                    s_km, _ = det.predict_scores(data)
                    pickle.dump({"scores": s_km, "k_medium": k_medium},
                                open(km_ckpt, "wb"))
                    print(f"    [km={k_medium}] computed ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"    [km={k_medium}] ERROR: {e}")

        if s_km is not None:
            try:
                T     = min(len(s_k5), len(s_km))
                ts_t  = timestamps[:T]
                s_add = add_fuse(s_k5, s_km)
                ivs   = evt_detect(s_add, ts_t)
                f1    = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["mae_km_f1"] = f1
            delta = f1 - sig_row.get("mae_add_f1", float("nan"))
            print(f"    mae_km (k5+km, EVT)       F1={f1:.4f}  ({delta:+.4f} vs mae_add)")

        rows.append(sig_row)


# ── Results ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "mae_add_f1", "mae_km_f1"]
LABELS   = {"baseline_f1": "baseline", "mae_add_f1": "add(k5,k30)", "mae_km_f1": "add(k5,km)"}
COL_W    = 13
SEP      = 36 + COL_W * len(ALL_COLS)


def _fmt(v) -> str:
    try:
        f = float(v)
        return "   NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "   NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub  = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    cols = [c for c in ALL_COLS if c in df.columns]
    sep  = 26 + COL_W * len(cols)
    print(f"\n{'='*sep}")
    print(f"=== {ds_name} — MAE k_medium (length-aware) ===")
    print(f"{'='*sep}")
    print(f"{'Signal':<20}{'k_med':>6}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in cols))
    print("-" * sep)
    for _, r in sub.iterrows():
        row = f"{r['signal']:<20}{int(r.get('k_medium', 30)):>6}"
        for c in cols:
            row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)
    print("-" * sep)
    avg = f"{'AVG':<26}"
    for c in cols:
        vals = sub[c].dropna()
        avg += f"{vals.mean():>{COL_W}.4f}" if len(vals) else f"{'NaN':>{COL_W}}"
    print(avg)


for ds_name in DATASET_CONFIGS:
    _print_table(ds_name, results_df)

# ── Cross-dataset summary ─────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("=== CROSS-DATASET SUMMARY ===")
print(f"{'='*SEP}")
ds_weights  = {"NAB": 16, "SMAP": 13, "MSL": 11}
print(f"{'Dataset':<10}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in ALL_COLS))
print("-" * SEP)

all_avgs: dict = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = f"{ds_name:<10}"
    for c in ALL_COLS:
        v = float(sub[c].mean()) if c in sub.columns and not sub[c].isna().all() else float("nan")
        row += f"{v:>{COL_W}.4f}"
        all_avgs.setdefault(c, []).append((ds_name, v))
    print(row)

print("-" * SEP)
overall = f"{'ALL (wt)':<10}"
for c in ALL_COLS:
    vals = all_avgs.get(c, [])
    wt   = sum(ds_weights.get(ds, 0) * v for ds, v in vals if not np.isnan(v))
    tot  = sum(ds_weights.get(ds, 0) for ds, v in vals if not np.isnan(v))
    overall += f"{wt/tot:>{COL_W}.4f}" if tot else f"{'NaN':>{COL_W}}"
print(overall)

# ── Key findings ──────────────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("KEY FINDINGS")
print(f"{'='*SEP}")

if "mae_add_f1" in results_df and "mae_km_f1" in results_df:
    # Per-dataset delta
    for ds_name in DATASET_CONFIGS:
        sub     = results_df[results_df["dataset"] == ds_name]
        add_avg = sub["mae_add_f1"].dropna().mean()
        km_avg  = sub["mae_km_f1"].dropna().mean()
        print(f"  {ds_name:<6}  add(k5,k30)={add_avg:.4f}  add(k5,km)={km_avg:.4f}  Δ={km_avg-add_avg:+.4f}")

    # Spotlight: short MSL signals
    msl_short = results_df[
        (results_df["dataset"] == "MSL") &
        (results_df["signal"].isin(["T-13", "D-16", "F-7"]))
    ]
    if not msl_short.empty:
        print("\nMSL short signals (regression targets):")
        for _, r in msl_short.iterrows():
            add_v = r.get("mae_add_f1", float("nan"))
            km_v  = r.get("mae_km_f1",  float("nan"))
            print(f"  {r['signal']}  k_med={int(r.get('k_medium',30))}  "
                  f"add(k30)={add_v:.4f}  add(km)={km_v:.4f}  Δ={km_v-add_v:+.4f}")

print(f"\nResults saved → {out_csv}")
