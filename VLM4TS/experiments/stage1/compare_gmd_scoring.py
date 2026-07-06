"""
compare_gmd_scoring.py
======================
Global Mahalanobis Distance (GMD) scoring — replaces LTR.

Why LTR fails
-------------
LTR uses k temporal neighbors as the normal reference. When anomaly duration
exceeds k*step windows, local neighbors are ALSO anomalous → dissimilarity
collapses → score inverts (confirmed: D-15 sep=2.247, T-1 INVERTED, iio sep=2.596 F1=0).

GMD fixes this
--------------
Build ONE global normality model from ALL N windows of the signal.
Anomalies are a small minority, so the global mean is dominated by normal windows.
No temporal locality assumption.

Algorithm
---------
  1. Extract MAE patch features for all N windows: F ∈ R^{N×768}
  2. PCA to d = min(N−2, 64)  — denoises without whitening
  3. μ = mean(F_pca)
  4. s_i = ‖F_pca_i − μ‖₂     (L2 from global mean; NOT whitened Mahalanobis)
     Rationale: sep_ratio was computed as L2/L2 and already showed 2.24–2.60
     for discriminable signals. Whitening suppresses high-variance directions
     and would reduce scores for large structural anomalies (D-15).
  5. Align per-window scores to T_full via average-overlap
  6. EVT threshold → anomaly intervals

Conditions
----------
  ltr_add   : add(k5, k30) — current best, loaded from existing CSV (no rerun)
  gmd       : Global Mahalanobis Distance (new inference)
  gmd_add   : gmd + normalize(ltr_k5) — exploration whether complementary

Expected gains
--------------
  iio  (sep=2.596, LTR failed)  → GMD should score anomaly high
  D-15 (sep=2.247, LTR inverted) → GMD global mean unaffected by local inversion
  T-1  (sep=1.200, LTR inverted) → weak separation, uncertain
  F-1/F-3/T-13/E-5 (feature-blind) → no gain expected

Checkpoint cache
----------------
  results_gmd_scoring/checkpoints/{DS}__{sig}__gmd.pkl
    keys: scores (T_full,), timestamps (T_full,)

Usage
-----
  python experiments/compare_gmd_scoring.py
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
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV    = os.path.join(DATA_DIR, "anomalies.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_gmd_scoring")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# LTR baseline CSV (existing — no rerun)
LTR_CSV = os.path.join(
    PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "comparison.csv"
)

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

# ── Inference parameters (must match LTR for fair comparison) ─────────────────
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

# ── Core GMD ──────────────────────────────────────────────────────────────────
def compute_gmd_scores(feats: np.ndarray) -> np.ndarray:
    """
    feats: [N, 768]  (mean-pooled MAE patch tokens)
    returns: [N,]    L2 distance from global mean after PCA denoising

    Design rationale:
    - PCA denoises the 768-dim space (many dims are noise for N < 768)
    - d = min(N-2, 64): safe rank, keeps dominant structure
    - L2 from mean (NOT whitened): preserves scale of anomaly directions
      (whitening would suppress high-variance directions where structural
       anomalies like stuck sensors live — measured sep_ratio drop for D-15)
    - Global mean is contaminated by ≤13% anomalies (worst case T-1),
      which shifts μ toward anomalies → slightly reduces scores, acceptable
    """
    N, D = feats.shape
    d = max(2, min(N - 2, 64))

    pca = PCA(n_components=d, random_state=0)
    reduced = pca.fit_transform(feats)   # [N, d]

    mu     = reduced.mean(axis=0)        # [d]
    scores = np.linalg.norm(reduced - mu, axis=1)  # [N,]
    return scores


def window_scores_to_ts(window_scores: np.ndarray, T_full: int,
                         window_size: int, step_size: int) -> np.ndarray:
    """Average-overlap projection of per-window scalars to per-timepoint array."""
    aligned = np.zeros(T_full, dtype=np.float64)
    counts  = np.zeros(T_full, dtype=np.float64)
    for i, s in enumerate(window_scores):
        start = i * step_size
        end   = min(start + window_size, T_full)
        aligned[start:end] += s
        counts[start:end]  += 1.0
    return aligned / np.maximum(counts, 1.0)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(mae, values_proc: np.ndarray, T_full: int,
                     device: torch.device) -> tuple[np.ndarray, list]:
    """
    Returns mean-pooled patch features [N, 768] and sorted window_ids [N].
    Uses the same preprocessing + draw_windowed_images pipeline as LTR,
    ensuring features are comparable.
    """
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
            results_dir=tmp,
            base_series_id="series",
            sample_size=None,
            no_anomaly=True,
            plot_type="line",
        )
        if len(dataset) == 0:
            return None, None

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        _, _, patch_embeds, _, _, wids = build_ordered_embeddings(
            mae, loader, PATCH_SIZE, device
        )

    # patch_embeds: [N, 196, 768] → mean-pool → [N, 768]
    feats = patch_embeds.mean(dim=1).numpy()
    return feats, wids


# ── Per-signal GMD pipeline ───────────────────────────────────────────────────
def run_gmd(mae, ds_name: str, sig: str, csv_path: str,
            device: torch.device) -> dict | None:
    """
    Compute GMD scores for one signal. Returns dict with scores + timestamps,
    or loads from cache if available.
    """
    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__gmd.pkl")

    if os.path.exists(ckpt_path):
        return pickle.load(open(ckpt_path, "rb"))

    data = pd.read_csv(csv_path)
    values_raw, timestamps = orion_to_internal(data)
    T_full = len(values_raw)

    values_proc = preprocess_time_series(values_raw)
    values_proc = apply_ewma(values_proc, 1.0)

    feats, wids = extract_features(mae, values_proc, T_full, device)
    if feats is None:
        warnings.warn(f"No images for {ds_name} {sig}")
        return None

    N = feats.shape[0]
    window_scores = compute_gmd_scores(feats)  # [N,]
    scores_ts     = window_scores_to_ts(window_scores, T_full, WINDOW_SIZE, STEP_SIZE)

    result = {"scores": scores_ts, "timestamps": timestamps,
              "window_scores": window_scores, "N_windows": N}
    pickle.dump(result, open(ckpt_path, "wb"))
    return result


# ── Load LTR baseline ─────────────────────────────────────────────────────────
ltr_baseline = {}   # (ds, sig) → add_k5_k30_f1
if os.path.exists(LTR_CSV):
    ltr_df = pd.read_csv(LTR_CSV)
    for _, row in ltr_df.iterrows():
        ltr_baseline[(row["dataset"], row["signal"])] = row.get("add_k5_k30_f1", float("nan"))

# Also load k5 scores for gmd_add fusion
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
        result = run_gmd(mae, ds_name, sig, csv_path, DEVICE)
        if result is None:
            print("FAILED")
            continue

        scores_gmd = result["scores"]
        timestamps  = result["timestamps"]
        T = min(len(scores_gmd), len(timestamps))
        scores_gmd = scores_gmd[:T]
        timestamps  = timestamps[:T]

        # ── Condition 1: GMD alone ────────────────────────────────────────────
        f1_gmd = evaluate_intervals(gt_ivs, evt_detect(scores_gmd, timestamps))["F1"]

        # ── Condition 2: GMD + LTR k5 (additive fusion) ───────────────────────
        k5_scores, k5_ts = load_k5_scores(ds_name, sig)
        f1_gmd_add = float("nan")
        if k5_scores is not None:
            Tk = min(T, len(k5_scores))
            s  = normalize_01(scores_gmd[:Tk]) + normalize_01(k5_scores[:Tk])
            f1_gmd_add = evaluate_intervals(gt_ivs, evt_detect(s, timestamps[:Tk]))["F1"]

        # ── Baseline ──────────────────────────────────────────────────────────
        f1_ltr = ltr_baseline.get((ds_name, sig), float("nan"))

        row = {
            "dataset": ds_name, "signal": sig,
            "ltr_add_f1": f1_ltr,
            "gmd_f1": f1_gmd,
            "gmd_add_k5_f1": f1_gmd_add,
        }
        rows.append(row)

        delta = f1_gmd - f1_ltr if not np.isnan(f1_ltr) else float("nan")
        print(f"ltr_add={f1_ltr:.4f}  gmd={f1_gmd:.4f}  "
              f"gmd+k5={f1_gmd_add:.4f}  Δgmd={delta:+.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

SIG_COUNTS = {"NAB": 16, "SMAP": 13, "MSL": 11}
COLS       = ["ltr_add_f1", "gmd_f1", "gmd_add_k5_f1"]
COL_W      = 16

print(f"\n{'='*75}")
print("SUMMARY — Global Mahalanobis Distance vs LTR baseline")
print(f"{'='*75}")
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

# ── Zero-F1 signal spotlight ──────────────────────────────────────────────────
ZERO_SIGNALS = [
    ("NAB",  "iio_us-east-1_i-a2eb1cd9_NetworkIn"),
    ("SMAP", "F-3"), ("SMAP", "T-1"), ("SMAP", "E-5"),
    ("MSL",  "T-13"), ("MSL", "D-15"), ("SMAP", "F-1"),
]
print(f"\n{'='*60}")
print("Previously zero/near-zero signals spotlight")
print(f"{'='*60}")
print(f"{'Signal':<40} {'ltr_add':>8} {'gmd':>8} {'gmd+k5':>8}  diagnosis")
print("-" * 76)
diagnoses = {
    "iio_us-east-1_i-a2eb1cd9_NetworkIn": "scorer(sep=2.60)",
    "F-3": "feature_blind(0 anom wins)",
    "T-1": "scorer_inverted(sep=1.20)",
    "E-5": "feature_blind(sep=0.84)",
    "T-13": "feature_blind(sep=0.96)",
    "D-15": "scorer_inverted(sep=2.25)",
    "F-1": "feature_blind(sep=0.00)",
}
for ds, sig in ZERO_SIGNALS:
    sub = results_df[(results_df["dataset"] == ds) & (results_df["signal"] == sig)]
    if sub.empty:
        continue
    r   = sub.iloc[0]
    tag = f"{ds}/{sig}"
    print(f"{tag:<40} {r['ltr_add_f1']:>8.4f} {r['gmd_f1']:>8.4f} "
          f"{r['gmd_add_k5_f1']:>8.4f}  {diagnoses.get(sig, '')}")
