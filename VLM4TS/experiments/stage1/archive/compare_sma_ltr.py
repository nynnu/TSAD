"""
SMA × LTR: Spectral Magnitude Alignment preprocessing for LTR k=5
==================================================================

Hypothesis (Proposal 2 core test)
----------------------------------
MAE was pretrained on natural images whose 2D power spectrum follows ~1/f².
Rendered time-series line-plot images have a flat/high-frequency spectrum.
This spectral gap causes MAE features to be suboptimal for TSAD.

SMA corrects ONLY the amplitude spectrum (phase = structural signal is preserved).
If the hypothesis is correct, SMA → better MAE features → better LTR F1.

Algorithm recap
---------------
  For each image channel:
    A_new = (1-beta)*|FFT(img)| + beta*target_1/f^alpha
    img_new = IFFT(A_new * exp(i * phase(FFT(img))))

  beta=0.0 : no-op (baseline)
  beta=0.5 : half-blend (gentle alignment)
  beta=1.0 : full alignment to 1/f^2 spectrum

Ablation table
--------------
  ltr_min_noSMA   : LTR k=5 min-cosine, no SMA   (reuse existing checkpoints)
  ltr_min_sma025  : LTR k=5 + SMA beta=0.25
  ltr_min_sma050  : LTR k=5 + SMA beta=0.50
  ltr_min_sma075  : LTR k=5 + SMA beta=0.75
  ltr_min_sma100  : LTR k=5 + SMA beta=1.00

All use:
  - same MAE backbone (vit_base_patch16_224.mae)
  - same k=5 local temporal reference
  - same EVT thresholding
  - same grayscale line-plot rendering

Checkpoints
-----------
  noSMA  (reuse) : results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl
  sma025  (new)  : results_sma_ltr/checkpoints/{DS}__{sig}__ltr_sma025.pkl
  sma050  (new)  : results_sma_ltr/checkpoints/{DS}__{sig}__ltr_sma050.pkl
  sma075  (new)  : results_sma_ltr/checkpoints/{DS}__{sig}__ltr_sma075.pkl
  sma100  (new)  : results_sma_ltr/checkpoints/{DS}__{sig}__ltr_sma100.pkl

Usage
-----
  python experiments/compare_sma_ltr.py
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
from preprocessing.sma_transform import SMADataset
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_sma_ltr")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_CKPT_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)),
                   _K5_CANDIDATES[-1])
print(f"LTR k=5 no-SMA ckpt : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR))} files)")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
K_LOCAL     = 5
MIN_REF     = 5
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

SMA_BETAS   = [0.25, 0.50, 0.75, 1.00]    # beta values to sweep

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)


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


# ── Score helpers ─────────────────────────────────────────────────────────────

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ── SMA-aware LTR detector ────────────────────────────────────────────────────

class SMA_LTR_Detector(ViT4TS_MAE):
    """LTR k=5 with optional SMA preprocessing (beta=0 → identical to baseline)."""

    def __init__(self, *args, sma_beta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sma_beta = sma_beta

    def _run_inference(self, results_dir: str, base_series_id: str):
        base_ds = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_series_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(base_ds) == 0:
            return None

        # Wrap with SMA if beta > 0
        dataset = SMADataset(base_ds, beta=self.sma_beta) if self.sma_beta > 0 else base_ds

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.model, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = self.image_size[0]
        ph = pw = h // self.patch_size

        anomaly_maps = []
        with torch.no_grad():
            for i in range(L):
                l_ref, _ = get_local_reference(large_embeds,  i, K_LOCAL, MIN_REF)
                m_ref, _ = get_local_reference(mid_embeds,    i, K_LOCAL, MIN_REF)
                p_ref, _ = get_local_reference(patch_embeds,  i, K_LOCAL, MIN_REF)

                m_l = compute_dissimilarity_with_ref(
                    large_embeds[i].unsqueeze(0).to(self.device), l_ref.to(self.device))
                m_m = compute_dissimilarity_with_ref(
                    mid_embeds[i].unsqueeze(0).to(self.device),   m_ref.to(self.device))
                m_p = compute_dissimilarity_with_ref(
                    patch_embeds[i].unsqueeze(0).to(self.device), p_ref.to(self.device))

                m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(self.device)
                m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(self.device)
                m_p = m_p.reshape((1, ph, pw)).to(self.device)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                          nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(h, h), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)


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


def _ivs_alpha(scores, timestamps, alpha=0.01):
    from models.model_utils import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


# ── Model init ────────────────────────────────────────────────────────────────
print("\n[INFO] Loading models (shared MAE backbone)...")
det_nosma = SMA_LTR_Detector(**BASE_PARAMS, sma_beta=0.0)
sma_dets  = {b: SMA_LTR_Detector(**BASE_PARAMS, sma_beta=b) for b in SMA_BETAS}
for det in sma_dets.values():
    det.model = det_nosma.model   # share backbone weights
print(f"  SMA betas tested : {SMA_BETAS}")

# ── Dataset config ────────────────────────────────────────────────────────────
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

BETA_KEYS = {b: f"sma{int(b*100):03d}" for b in SMA_BETAS}   # e.g. 0.25 → "sma025"


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

        nosma_ckpt = os.path.join(K5_CKPT_DIR, f"{ds_name}__{sig}__ltr.pkl")

        # ── baseline: no SMA (reuse) ──────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(nosma_ckpt):
                c          = pickle.load(open(nosma_ckpt, "rb"))
                s_nosma    = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [no-SMA] cache  ({time.time()-t0:.1f}s)")
            else:
                s_nosma, timestamps = det_nosma.predict_scores(data)
                pickle.dump({"scores": s_nosma, "timestamps": timestamps},
                            open(nosma_ckpt, "wb"))
                print(f"    [no-SMA] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [no-SMA] ERROR: {e}"); continue

        T       = len(s_nosma)
        ts_trim = timestamps[:T]

        # baseline F1 (alpha=0.01)
        try:
            ivs = _ivs_alpha(s_nosma, timestamps)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["nosma_alpha_f1"] = f1
        print(f"    no-SMA alpha  F1={f1:.4f}")

        # baseline F1 (EVT)
        try:
            ivs = evt_detect(s_nosma[:T], ts_trim)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["nosma_evt_f1"] = f1
        print(f"    no-SMA EVT    F1={f1:.4f}")

        # ── SMA betas ─────────────────────────────────────────────
        for beta in SMA_BETAS:
            key       = BETA_KEYS[beta]
            sma_ckpt  = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__ltr_{key}.pkl")
            col_name  = f"{key}_evt_f1"

            t0 = time.time()
            try:
                if os.path.exists(sma_ckpt):
                    s_sma = pickle.load(open(sma_ckpt, "rb"))["scores"]
                    print(f"    [beta={beta:.2f}] cache  ({time.time()-t0:.1f}s)")
                else:
                    s_sma, _ = sma_dets[beta].predict_scores(data)
                    pickle.dump({"scores": s_sma}, open(sma_ckpt, "wb"))
                    print(f"    [beta={beta:.2f}] computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [beta={beta:.2f}] ERROR: {e}"); continue

            try:
                T2  = min(T, len(s_sma))
                ivs = evt_detect(s_sma[:T2], ts_trim[:T2])
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0

            sig_row[col_name] = f1
            delta = f1 - sig_row["nosma_alpha_f1"]
            print(f"    SMA b={beta:.2f}       F1={f1:.4f}  ({delta:+.4f} vs base)")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

BASE_COLS = ["nosma_alpha_f1", "nosma_evt_f1"]
SMA_COLS  = [f"{BETA_KEYS[b]}_evt_f1" for b in SMA_BETAS]
ALL_COLS  = BASE_COLS + SMA_COLS

LABELS = {"nosma_alpha_f1": "noSMA_α", "nosma_evt_f1": "noSMA_EVT"}
LABELS.update({f"{BETA_KEYS[b]}_evt_f1": f"β={b:.2f}" for b in SMA_BETAS})

COL_W = 11
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v: object) -> str:
    try:
        f = float(v)
        return "  NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "  NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    valid = [c for c in ALL_COLS if c in df.columns]
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — SMA beta sweep ===")
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


# ── Cross-dataset summary ─────────────────────────────────────────────────────
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
        v = float(sub[c].mean()) if c in sub.columns and not sub[c].isna().all() \
            else float("nan")
        row += f"{v:>{COL_W}.4f}"
        all_avgs.setdefault(c, []).append(v)
    print(row)

print("-" * SEP)
overall = f"{'ALL':<10}"
for c in valid_cols:
    vals = [v for v in all_avgs.get(c, []) if not np.isnan(v)]
    overall += f"{np.mean(vals):>{COL_W}.4f}" if vals else f"{'NaN':>{COL_W}}"
print(overall)


# ── Key findings ──────────────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("KEY FINDINGS  (SMA hypothesis test)")
print(f"{'='*SEP}")

base = results_df["nosma_alpha_f1"].dropna().mean()
print(f"baseline (no SMA, alpha=0.01) : {base:.4f}")
for b in SMA_BETAS:
    c = f"{BETA_KEYS[b]}_evt_f1"
    if c in results_df.columns:
        v = results_df[c].dropna().mean()
        print(f"SMA beta={b:.2f} + EVT          : {v:.4f}  ({v-base:+.4f})")

# Spotlight: SMAP F-series (target stuck/subtle signals)
print("\nSMAP F-series (target stuck/subtle signals):")
smap_f = results_df[
    (results_df["dataset"] == "SMAP") &
    (results_df["signal"].str.startswith("F"))
]
if not smap_f.empty:
    print(f"{'Signal':<8}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in valid_cols
                                      if c in results_df.columns))
    for _, r in smap_f.iterrows():
        row = f"{r['signal']:<8}"
        for c in valid_cols:
            row += f"{_fmt(r.get(c)):>{COL_W}}"
        print(row)

# Verdict: hypothesis supported or not
best_sma_col = max(SMA_COLS, key=lambda c: results_df[c].dropna().mean()
                   if c in results_df.columns else -1)
best_beta    = SMA_BETAS[SMA_COLS.index(best_sma_col)]
best_val     = results_df[best_sma_col].dropna().mean() if best_sma_col in results_df.columns else 0
print(f"\nBest SMA beta  : {best_beta:.2f}  (ALL F1={best_val:.4f})")
print(f"Hypothesis     : {'SUPPORTED (+delta>0.005)' if best_val - base > 0.005 else 'NOT SUPPORTED (delta<0.005)'}")

print(f"\nResults saved → {out_csv}")
