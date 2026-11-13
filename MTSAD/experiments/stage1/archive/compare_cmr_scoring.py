"""
compare_cmr_scoring.py
======================
Central Memory Reference (CMR) scoring.

Why GMD failed
--------------
GMD (distance from global mean) fails for NAB because normal cloud-metric
signals are MULTIMODAL in feature space — a CPU signal has "idle", "moderate",
"busy" operating states, and the global mean sits in the middle of all of them.
Random normal windows end up far from the mean, creating false positives.

Core insight from GMD results
------------------------------
MSL anomalies (D-15, T-1, P-11) are STRUCTURAL changes: anomalous windows
form a SMALL, SEPARATE cluster in feature space, while normal windows form
the DENSE CORE. Any method that builds its reference from the DENSE CORE
correctly excludes the anomaly cluster.

The fix: CMR
------------
Select the most central windows as the reference memory.
"Central" = lowest mean pairwise distance to all other windows.
= windows that have MANY similar neighbors = NORMAL windows.

  memory = top-p% most central windows  (default p=50%)

Score each window by its distance to the nearest memory entry:
  s_i = min_j ||PCA(f_i) − PCA(m_j)||₂

Why this fixes SCORER_BROKEN (D-15, T-1):
  - Anomalous windows are PERIPHERAL (far from all others) → not in memory
  - Their distance to nearest NORMAL memory entry is HIGH → correct score

Why this handles MULTIMODAL normals (NAB):
  - Each normal cluster (low-CPU, mid-CPU, high-CPU) contributes central members
  - Memory covers ALL normal operating states
  - Anomalous spike window is far from ALL normal clusters → far from memory → HIGH

Why this differs from LTR:
  - LTR reference = temporal neighbors → contaminated when anomaly is sustained
  - CMR reference = density-selected central windows → contaminated only if
    anomaly rate > 50% (not the case in any of our signals)

Why this differs from GMD:
  - GMD: distance from single global mean → fails for multimodal normals
  - CMR: nearest-neighbor to distributed memory → handles multimodal naturally

PCA step: reduce 768→min(N-2, 64) dims to denoise before distance computation.
          Large N (SMAP ~200 windows): d=64 captures dominant structure.
          Small N (iio 19 windows): d=17 avoids rank-deficiency.

Feature cache
-------------
Raw features are expensive to extract (MAE inference). This script saves
features to a shared cache so all future scoring experiments can reuse them:
  results_feature_cache/checkpoints/{DS}__{sig}__features.pkl
    keys: features [N, 768], timestamps [T_full], T_full, N_windows

Conditions
----------
  ltr_add      : add(k5, k30) current best — loaded from CSV (no rerun)
  gmd          : global mean distance — loaded from GMD checkpoints (no rerun)
  cmr          : Central Memory Reference (new)
  max_ltr_cmr  : max(norm(ltr_k5), norm(cmr)) — exploration

Expected gains
--------------
  D-15 (sep=2.247, SCORER_BROKEN)  → peripheral anomaly, far from central memory ✓
  T-1  (sep=1.200, weak)            → peripheral, but overlap possible
  iio  (sep=2.596, threshold issue) → 2 anomalous windows far from 9-window memory ✓
  NAB multimodal                     → memory covers all normal states ✓
  F-1/F-3/T-13/E-5 (FEATURE_BLIND) → no gain (features can't separate)

Usage
-----
  python experiments/compare_cmr_scoring.py
"""

from __future__ import annotations

import ast
import os
import pickle
import sys
import subprocess
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import genpareto
from sklearn.decomposition import PCA

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

subprocess.run(["pip", "install", "timm", "open-clip-torch", "--quiet"], check=True)

import torch
from torch.utils.data import DataLoader

from preprocessing.preprocess import preprocess_time_series, apply_ewma, draw_windowed_images
from preprocessing.data_utils import orion_to_internal
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from models.mae_vision import MAE_AD
from models.model_utils_local_v2 import build_ordered_embeddings
from evaluation.evaluate import evaluate_intervals

print(f"Project root: {PROJECT_ROOT}")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV      = os.path.join(DATA_DIR, "anomalies.csv")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results_cmr_scoring")
CKPT_DIR      = os.path.join(RESULTS_DIR, "checkpoints")
FEAT_CACHE    = os.path.join(PROJECT_ROOT, "results_feature_cache", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FEAT_CACHE, exist_ok=True)

# Reference CSVs (no rerun needed)
LTR_CSV = os.path.join(
    PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "comparison.csv"
)
GMD_CSV = os.path.join(PROJECT_ROOT, "results_gmd_scoring", "comparison.csv")

def _first_existing(*candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[-1]

K5_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)

# ── Signal lists ──────────────────────────────────────────────────────────────
SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
             "T-13","D-16","T-8","P-14","D-14"]

def _load_gt():
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

def _load_nab_sigs(gt):
    nab_dir = os.path.join(DATA_DIR, "realAWSCloudwatch")
    return sorted(f[:-4] for f in os.listdir(nab_dir)
                  if f.endswith(".csv") and gt.get(f[:-4]))

gt = _load_gt()
NAB_SIGS = _load_nab_sigs(gt)

DATASET_CONFIGS = {
    "NAB":  {"dir": os.path.join(DATA_DIR, "realAWSCloudwatch"), "sigs": NAB_SIGS},
    "SMAP": {"dir": os.path.join(DATA_DIR, "SMAP"),             "sigs": SMAP_SIGS},
    "MSL":  {"dir": os.path.join(DATA_DIR, "MSL"),              "sigs": MSL_SIGS},
}

WINDOW_SIZE = 224
STEP_RATIO  = 4.0
STEP_SIZE   = int(WINDOW_SIZE / STEP_RATIO)   # 56
PATCH_SIZE  = 16
BATCH_SIZE  = 20

# ── EVT ───────────────────────────────────────────────────────────────────────
EVT_Q   = 0.90
EVT_FPR = 0.01

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
            in_seg = True; seg_start = i
        elif not f and in_seg:
            in_seg = False
            ivs.append([timestamps[seg_start], timestamps[i - 1]])
    if in_seg:
        ivs.append([timestamps[seg_start], timestamps[len(flags) - 1]])
    return ivs

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-8 else (arr - lo) / (hi - lo)

# ── Core CMR ─────────────────────────────────────────────────────────────────
def compute_cmr_scores(feats: np.ndarray, memory_pct: float = 0.5) -> np.ndarray:
    """
    feats: [N, 768]  mean-pooled MAE patch tokens
    Returns: [N,]    min distance to central memory in PCA space

    Central memory = top-p% windows with lowest mean pairwise distance
    to all other windows.  These are the NORMAL windows (majority + dense).
    Score = nearest-neighbor distance to this normal-only reference.

    Unlike LTR: reference is not contaminated by temporal anomaly neighbors.
    Unlike GMD: multimodal normals are handled — each mode contributes to memory.
    """
    N, D = feats.shape
    d = max(2, min(N - 2, 64))

    pca = PCA(n_components=d, random_state=0)
    reduced = pca.fit_transform(feats)   # [N, d]

    # Pairwise distances in PCA space
    pw_dists = cdist(reduced, reduced, metric="euclidean")  # [N, N]

    # Centrality: mean distance to all others (lower = more central = more normal)
    centrality = pw_dists.mean(axis=1)   # [N,]

    # Memory: most central p% windows
    k_mem = max(1, int(N * memory_pct))
    mem_idx  = np.argsort(centrality)[:k_mem]
    memory   = reduced[mem_idx]          # [k_mem, d]

    # Score: distance to nearest memory entry
    scores = cdist(reduced, memory, metric="euclidean").min(axis=1)  # [N,]
    return scores


def window_scores_to_ts(window_scores: np.ndarray, T_full: int,
                         window_size: int, step_size: int) -> np.ndarray:
    aligned = np.zeros(T_full, dtype=np.float64)
    counts  = np.zeros(T_full, dtype=np.float64)
    for i, s in enumerate(window_scores):
        start = i * step_size
        end   = min(start + window_size, T_full)
        aligned[start:end] += s
        counts[start:end]  += 1.0
    return aligned / np.maximum(counts, 1.0)


# ── Feature extraction with caching ──────────────────────────────────────────
def get_features(mae, ds_name: str, sig: str, csv_path: str,
                 device: torch.device) -> dict | None:
    """
    Returns {'features': [N,768], 'timestamps': [T], 'T_full': int, 'N': int}
    Caches to FEAT_CACHE so any future scorer can reuse without inference.
    """
    cache_path = os.path.join(FEAT_CACHE, f"{ds_name}__{sig}__features.pkl")
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))

    data = pd.read_csv(csv_path)
    values_raw, timestamps = orion_to_internal(data)
    T_full = len(values_raw)

    values_proc = preprocess_time_series(values_raw)
    values_proc = apply_ewma(values_proc, 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        draw_windowed_images(
            base_series_id="series",
            save_path=tmp,
            time_series=values_proc,
            time_points=np.arange(T_full),
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            override=True,
            save_image=False,
            image_size=(224, 224),
            dpi=100,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
        )
        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="series",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return None
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        _, _, patch_embeds, _, _, _ = build_ordered_embeddings(
            mae, loader, PATCH_SIZE, device
        )

    features = patch_embeds.mean(dim=1).numpy()   # [N, 768]
    result = {"features": features, "timestamps": timestamps,
              "T_full": T_full, "N": features.shape[0]}
    pickle.dump(result, open(cache_path, "wb"))
    return result


# ── Load baselines ────────────────────────────────────────────────────────────
ltr_baseline = {}
if os.path.exists(LTR_CSV):
    for _, r in pd.read_csv(LTR_CSV).iterrows():
        ltr_baseline[(r["dataset"], r["signal"])] = r.get("add_k5_k30_f1", float("nan"))

gmd_baseline = {}
if os.path.exists(GMD_CSV):
    for _, r in pd.read_csv(GMD_CSV).iterrows():
        gmd_baseline[(r["dataset"], r["signal"])] = r.get("gmd_f1", float("nan"))

def load_k5_scores(ds, sig):
    p = os.path.join(K5_CKPT_DIR, f"{ds}__{sig}__ltr.pkl")
    if not os.path.exists(p):
        return None, None
    c = pickle.load(open(p, "rb"))
    return c["scores"], c["timestamps"]


# ── Main loop ─────────────────────────────────────────────────────────────────
print("Loading MAE model ...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae    = MAE_AD(model_name="vit_base_patch16_224.mae", device=DEVICE)
mae.eval()
print(f"Done. Device={DEVICE}\n")

rows = []

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}  ({len(cfg['sigs'])} signals)")
    print(f"{'='*60}")

    for sig in cfg["sigs"]:
        gt_ivs = gt.get(sig, [])
        if not gt_ivs:
            continue

        csv_path = os.path.join(cfg["dir"], sig + ".csv")
        if not os.path.exists(csv_path):
            print(f"  SKIP {sig} (missing CSV)")
            continue

        print(f"  {sig} ...", end=" ", flush=True)
        feat_data = get_features(mae, ds_name, sig, csv_path, DEVICE)
        if feat_data is None:
            print("FAILED")
            continue

        features   = feat_data["features"]    # [N, 768]
        timestamps = feat_data["timestamps"]
        T_full     = feat_data["T_full"]

        # ── CMR scoring ───────────────────────────────────────────────────────
        win_scores_cmr = compute_cmr_scores(features, memory_pct=0.5)
        scores_cmr     = window_scores_to_ts(win_scores_cmr, T_full, WINDOW_SIZE, STEP_SIZE)
        T = min(len(scores_cmr), len(timestamps))
        scores_cmr = scores_cmr[:T]
        ts         = timestamps[:T]

        f1_cmr = evaluate_intervals(gt_ivs, evt_detect(scores_cmr, ts))["F1"]

        # ── max(ltr_k5, cmr) fusion ───────────────────────────────────────────
        k5_scores, k5_ts = load_k5_scores(ds_name, sig)
        f1_max_cmr_ltr = float("nan")
        if k5_scores is not None:
            Tk = min(T, len(k5_scores))
            s  = np.maximum(normalize_01(scores_cmr[:Tk]),
                            normalize_01(k5_scores[:Tk]))
            f1_max_cmr_ltr = evaluate_intervals(gt_ivs, evt_detect(s, ts[:Tk]))["F1"]

        # ── add(ltr_k5, cmr) fusion ───────────────────────────────────────────
        f1_add_cmr_ltr = float("nan")
        if k5_scores is not None:
            s  = normalize_01(scores_cmr[:Tk]) + normalize_01(k5_scores[:Tk])
            f1_add_cmr_ltr = evaluate_intervals(gt_ivs, evt_detect(s, ts[:Tk]))["F1"]

        # ── Baselines ─────────────────────────────────────────────────────────
        f1_ltr = ltr_baseline.get((ds_name, sig), float("nan"))
        f1_gmd = gmd_baseline.get((ds_name, sig), float("nan"))

        row = {
            "dataset": ds_name, "signal": sig,
            "ltr_add_f1": f1_ltr,
            "gmd_f1":     f1_gmd,
            "cmr_f1":     f1_cmr,
            "max_cmr_ltr_f1": f1_max_cmr_ltr,
            "add_cmr_ltr_f1": f1_add_cmr_ltr,
        }
        rows.append(row)

        delta = f1_cmr - f1_ltr if not np.isnan(f1_ltr) else float("nan")
        print(f"ltr={f1_ltr:.3f}  gmd={f1_gmd:.3f}  cmr={f1_cmr:.3f}  "
              f"max_cmr_ltr={f1_max_cmr_ltr:.3f}  "
              f"add_cmr_ltr={f1_add_cmr_ltr:.3f}  Δcmr={delta:+.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

SIG_COUNTS = {"NAB": 16, "SMAP": 13, "MSL": 11}
COLS       = ["ltr_add_f1", "gmd_f1", "cmr_f1", "max_cmr_ltr_f1", "add_cmr_ltr_f1"]
COL_W      = 14

print(f"\n{'='*80}")
print("SUMMARY — Central Memory Reference vs baselines")
print(f"{'='*80}")
print(f"{'Dataset':<10}" + "".join(f"{c:>{COL_W}}" for c in COLS))
print("-" * (10 + COL_W * len(COLS)))

all_wtd = {c: 0.0 for c in COLS}
total_n = 0

for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    avgs = [sub[c].mean() for c in COLS]
    print(f"{ds:<10}" + "".join(f"{a:>{COL_W}.4f}" for a in avgs))
    n = SIG_COUNTS[ds]
    for c, a in zip(COLS, avgs):
        all_wtd[c] += a * n
    total_n += n

print("-" * (10 + COL_W * len(COLS)))
print(f"{'ALL (wtd)':<10}" +
      "".join(f"{all_wtd[c]/total_n:>{COL_W}.4f}" for c in COLS))
print(f"\nResults → {RESULTS_DIR}/comparison.csv")
print(f"Features cached → {FEAT_CACHE}/")

# ── Diagnostic spotlight ──────────────────────────────────────────────────────
SPOTLIGHT = [
    ("NAB",  "iio_us-east-1_i-a2eb1cd9_NetworkIn", "scorer sep=2.60"),
    ("SMAP", "T-1",   "scorer_inv sep=1.20"),
    ("MSL",  "D-15",  "scorer_inv sep=2.25"),
    ("SMAP", "F-3",   "blind(0 wins)"),
    ("MSL",  "T-13",  "blind sep=0.96"),
    ("SMAP", "E-5",   "blind sep=0.84"),
    ("SMAP", "F-1",   "blind sep=0.00"),
]
print(f"\n{'='*80}")
print("Spotlight — previously zero/near-zero signals")
print(f"{'='*80}")
print(f"{'Signal':<44} {'ltr':>6} {'gmd':>6} {'cmr':>6} {'max':>6} {'add':>6}  notes")
print("-" * 84)
for ds, sig, note in SPOTLIGHT:
    sub = results_df[(results_df["dataset"] == ds) & (results_df["signal"] == sig)]
    if sub.empty:
        continue
    r = sub.iloc[0]
    print(f"{ds}/{sig:<40} "
          f"{r['ltr_add_f1']:>6.3f} {r['gmd_f1']:>6.3f} {r['cmr_f1']:>6.3f} "
          f"{r['max_cmr_ltr_f1']:>6.3f} {r['add_cmr_ltr_f1']:>6.3f}  {note}")
