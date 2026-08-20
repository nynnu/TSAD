"""
LTR k=5 + Recon Mean + Recon Std — Variance-Augmented Fusion
=============================================================

Hypothesis
----------
MAE reconstruction error across n_iter=5 random masks produces two signals:
  s_mean : avg MSE over masks  (how anomalous a window looks on average)
  s_std  : std MSE over masks  (how unstable reconstruction is across masks)

Why std is informative
----------------------
With mask_ratio=0.5 (98/196 patches masked per iteration), a point/spike
anomaly occupies a small region: some masks cover it (high patch loss), others
miss it (low patch loss) → high variance across iterations → high std.

Normal windows reconstruct consistently regardless of which patches are masked
→ low std.

Important scope limitation: std is NOT expected to help long-duration anomalies
(e.g. SMAP E-series). When the entire window is anomalous, the anomaly region
is covered by nearly every mask → reconstruction error is consistently high or
consistently low → std stays low. std primarily provides complementary signal
for POINT/SPIKE anomalies that Recon_mean already catches but with extra
discrimination signal, and for cases where LTR fires weakly.

Ablation
--------
  baseline         : LTR k=5 + alpha=0.01  (identical to compare_mgmr)
  ltr_mean_w03     : norm(LTR) + 0.3*norm(mean) + EVT        (our current best)
  ltr_std_w05      : norm(LTR) + 0.5*norm(std)  + EVT        (std only)
  ltr_std_w10      : norm(LTR) + 1.0*norm(std)  + EVT
  ltr_m03_s05      : norm(LTR) + 0.3*norm(mean) + 0.5*norm(std) + EVT
  ltr_m03_s10      : norm(LTR) + 0.3*norm(mean) + 1.0*norm(std) + EVT
  std_only         : norm(std) + EVT  (std signal alone — diagnostic)

Checkpoints
-----------
  LTR  (reused) : results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl
                  keys: scores [T], timestamps [T]
  Recon var (new): results_recon_variance/checkpoints/{DS}__{sig}__recon_var.pkl
                   keys: mean [T], std [T]
                   (timestamps come from LTR checkpoint; not duplicated here)

Usage
-----
  python experiments/compare_recon_variance.py
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

from preprocessing.preprocess import preprocess_time_series, apply_ewma, draw_windowed_images
from preprocessing.data_utils import orion_to_internal
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps
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
DATA_DIR  = os.path.join(PROJECT_ROOT, "data")
NAB_DIR   = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR  = os.path.join(DATA_DIR, "SMAP")
MSL_DIR   = os.path.join(DATA_DIR, "MSL")
ANOM_CSV  = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results_recon_variance")
RECON_CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(RECON_CKPT_DIR, exist_ok=True)

# LTR checkpoints: try synced location first, then local fallback
_LTR_CKPT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
LTR_CKPT_DIR = next(
    (p for p in _LTR_CKPT_CANDIDATES if os.path.isdir(p)),
    _LTR_CKPT_CANDIDATES[-1],
)
os.makedirs(LTR_CKPT_DIR, exist_ok=True)
print(f"LTR ckpt dir : {LTR_CKPT_DIR}  ({len(os.listdir(LTR_CKPT_DIR))} files)")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
STEP_SIZE   = int(WINDOW_SIZE / STEP_RATIO)   # 56
BATCH_ENC   = 16
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)


# ── EVT threshold ─────────────────────────────────────────────────────────────

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
    thr   = evt_threshold(scores)
    flags = scores > thr
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


# ── Fusion helpers ────────────────────────────────────────────────────────────

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def fuse(s_ltr: np.ndarray, s_mean: np.ndarray, s_std: np.ndarray,
         w_mean: float, w_std: float) -> np.ndarray:
    """norm(LTR) + w_mean*norm(mean) + w_std*norm(std), length-aligned."""
    T = min(len(s_ltr), len(s_mean), len(s_std))
    return (normalize_01(s_ltr[:T])
            + w_mean * normalize_01(s_mean[:T])
            + w_std  * normalize_01(s_std[:T]))


# ── Signal A: LTR k=5 (reuses MGMR checkpoints) ──────────────────────────────

class _LTR_Detector(ViT4TS_MAE):
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
        ph = pw = h // self.patch_size
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
                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                         nan=0., posinf=0., neginf=0.)
                score = torch.nn.functional.interpolate(
                    score.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())
        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)


# ── Signal B: Recon mean + std ────────────────────────────────────────────────

def compute_recon_var_scores(
    data: pd.DataFrame,
    recon_model: MAE_Recon,
    window_size: int = WINDOW_SIZE,
    step_size:   int = STEP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (s_mean, s_std, timestamps) — per-timestep scores aggregated by
    max-pooling over overlapping windows.
    """
    values, timestamps = orion_to_internal(data)
    T_full = len(values)

    values_proc  = preprocess_time_series(values)
    values_proc  = apply_ewma(values_proc, 1.0)
    window_starts = list(range(0, T_full - window_size + 1, step_size))
    n_windows     = len(window_starts)

    with tempfile.TemporaryDirectory() as tmp:
        ok = draw_windowed_images(
            base_series_id="series", save_path=tmp,
            time_series=values_proc, time_points=np.arange(len(values_proc)),
            override=True, window_size=window_size, step_size=step_size,
            image_size=(224, 224), dpi=100,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
        )
        if not ok:
            return np.zeros(T_full), np.zeros(T_full), timestamps

        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="series",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return np.zeros(T_full), np.zeros(T_full), timestamps

        loader = DataLoader(dataset, batch_size=BATCH_ENC, shuffle=False)
        raw_mean, raw_std, raw_wids = [], [], []
        for batch in loader:
            imgs = batch["img"]
            wids = batch["window_id"]
            m, s = recon_model.score_batch(imgs, return_std=True)  # both [B]
            raw_mean.extend(m.tolist())
            raw_std.extend(s.tolist())
            raw_wids.extend(wids.tolist())

    order        = sorted(range(len(raw_wids)), key=lambda i: raw_wids[i])
    scores_mean  = np.array([raw_mean[i] for i in order], dtype=np.float32)
    scores_std   = np.array([raw_std[i]  for i in order], dtype=np.float32)

    s_mean = np.zeros(T_full, dtype=np.float32)
    s_std  = np.zeros(T_full, dtype=np.float32)
    for wi in range(min(n_windows, len(scores_mean))):
        ws = window_starts[wi]
        t1 = min(ws + window_size, T_full)
        s_mean[ws:t1] = np.maximum(s_mean[ws:t1], scores_mean[wi])
        s_std[ws:t1]  = np.maximum(s_std[ws:t1],  scores_std[wi])

    return s_mean, s_std, timestamps


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


def _eval(detected: list, gt_ivs: list) -> dict:
    m = evaluate_intervals(gt_ivs, detected)
    return {"f1": m["F1"], "p": m["precision"], "r": m["recall"]}


def _ivs_from_scores(scores, timestamps, alpha=0.01):
    from models.model_utils import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


# ── Model init ────────────────────────────────────────────────────────────────
print("\n[INFO] Loading models ...")
det_ltr   = _LTR_Detector(**BASE_PARAMS, local_k=5, min_ref=5)
mae_recon = MAE_Recon(device=DEVICE, n_iter=5, mask_ratio=0.5)
print(
    f"  Signal A : LTR k=5\n"
    f"  Signal B : MAE Recon  mean + std  (n_iter=5)\n"
    f"  Ablation : 7 fusion configurations\n"
    f"  Threshold: EVT (q=0.90, fpr=0.01)"
)

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

# Fusion configurations: (label, w_mean, w_std)
FUSIONS = [
    ("ltr_mean_w03", 0.3, 0.0),   # current best from additive experiment
    ("ltr_std_w05",  0.0, 0.5),   # std-only addition
    ("ltr_std_w10",  0.0, 1.0),
    ("ltr_m03_s05",  0.3, 0.5),   # mean + std combined
    ("ltr_m03_s10",  0.3, 1.0),
]

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

        ltr_ckpt   = os.path.join(LTR_CKPT_DIR,   f"{ds_name}__{sig}__ltr.pkl")
        recon_ckpt = os.path.join(RECON_CKPT_DIR,  f"{ds_name}__{sig}__recon_var.pkl")

        # ── Signal A: LTR ─────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(ltr_ckpt):
                c          = pickle.load(open(ltr_ckpt, "rb"))
                s_ltr      = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [LTR  ] cache  ({time.time()-t0:.1f}s)")
            else:
                s_ltr, timestamps = det_ltr.predict_scores(data)
                pickle.dump({"scores": s_ltr, "timestamps": timestamps},
                            open(ltr_ckpt, "wb"))
                print(f"    [LTR  ] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [LTR  ] ERROR: {e}")
            s_ltr = timestamps = None

        # ── Signal B: Recon mean + std ────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(recon_ckpt):
                c      = pickle.load(open(recon_ckpt, "rb"))
                s_mean = c["mean"]
                s_std  = c["std"]
                print(f"    [Recon] cache  ({time.time()-t0:.1f}s)")
            else:
                s_mean, s_std, _ = compute_recon_var_scores(data, mae_recon)
                pickle.dump({"mean": s_mean, "std": s_std},
                            open(recon_ckpt, "wb"))
                print(f"    [Recon] computed ({time.time()-t0:.1f}s)"
                      f"  mean_max={s_mean.max():.4f}  std_max={s_std.max():.4f}")
        except Exception as e:
            print(f"    [Recon] ERROR: {e}")
            s_mean = s_std = None

        if s_ltr is None or s_mean is None:
            continue

        T       = min(len(s_ltr), len(s_mean), len(s_std))
        ts_trim = timestamps[:T]

        # ── Baseline: LTR + alpha=0.01 ────────────────────────────
        try:
            ivs_base = _ivs_from_scores(s_ltr, timestamps, alpha=0.01)
            m_base   = _eval(ivs_base, gt_ivs)
        except Exception as e:
            print(f"    baseline ERROR: {e}")
            m_base = {"f1": 0.0, "p": 0.0, "r": 0.0}
        sig_row["baseline_f1"] = m_base["f1"]
        print(f"    baseline      F1={m_base['f1']:.4f}  ivs={len(ivs_base)}")

        # ── std_only diagnostic ───────────────────────────────────
        try:
            ivs_std = evt_detect(s_std[:T], ts_trim)
            m_std   = _eval(ivs_std, gt_ivs)
        except Exception as e:
            print(f"    std_only ERROR: {e}")
            m_std = {"f1": 0.0, "p": 0.0, "r": 0.0}
        sig_row["std_only_f1"] = m_std["f1"]
        print(f"    std_only      F1={m_std['f1']:.4f}  ivs={len(ivs_std)}"
              f"  (vs base: {m_std['f1']-m_base['f1']:+.4f})")

        # ── Fusion ablation ───────────────────────────────────────
        for label, w_m, w_s in FUSIONS:
            try:
                s_fused = fuse(s_ltr, s_mean, s_std, w_m, w_s)
                ivs_f   = evt_detect(s_fused, ts_trim)
                m_f     = _eval(ivs_f, gt_ivs)
            except Exception as e:
                print(f"    {label} ERROR: {e}")
                m_f = {"f1": 0.0, "p": 0.0, "r": 0.0}
            sig_row[f"{label}_f1"] = m_f["f1"]
            delta = m_f["f1"] - m_base["f1"]
            print(f"    {label:<16} F1={m_f['f1']:.4f}  ivs={len(ivs_f)}"
                  f"  ({delta:+.4f})")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = (["baseline_f1", "std_only_f1"]
            + [f"{lbl}_f1" for lbl, _, _ in FUSIONS])
LABELS   = {
    "baseline_f1":   "LTR_k5",
    "std_only_f1":   "Std_only",
    "ltr_mean_w03_f1": "LTR+0.3M",
    "ltr_std_w05_f1":  "LTR+0.5S",
    "ltr_std_w10_f1":  "LTR+1.0S",
    "ltr_m03_s05_f1":  "L+0.3M+0.5S",
    "ltr_m03_s10_f1":  "L+0.3M+1.0S",
}
COL_W = 14
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v: object) -> str:
    try:
        f = float(v)
        return "     NaN" if (f != f) else f"{f:.4f}"
    except Exception:
        return "     NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — Recon Variance Fusion ===")
    print(f"{'='*SEP}")
    header = f"{'Signal':<36}" + "".join(
        f"{LABELS.get(c, c):>{COL_W}}" for c in ALL_COLS)
    print(header)
    print("-" * SEP)
    for _, r in sub.iterrows():
        row = f"{r['signal']:<36}" + "".join(
            f"{_fmt(r.get(c, float('nan'))):>{COL_W}}" for c in ALL_COLS)
        print(row)
    print("-" * SEP)
    avg_row = f"{'AVG':<36}"
    for c in ALL_COLS:
        vals = sub[c].dropna() if c in sub else pd.Series([], dtype=float)
        avg_row += f"{vals.mean():>{COL_W}.4f}" if len(vals) else f"{'NaN':>{COL_W}}"
    print(avg_row)


for ds_name in DATASET_CONFIGS:
    _print_table(ds_name, results_df)


# ── Cross-dataset summary ─────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("=== CROSS-DATASET SUMMARY ===")
print(f"{'='*SEP}")
print(f"{'Dataset':<10}" + "".join(
    f"{LABELS.get(c, c):>{COL_W}}" for c in ALL_COLS))
print("-" * SEP)

all_avgs: dict = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = f"{ds_name:<10}"
    for c in ALL_COLS:
        v = float(sub[c].mean()) if c in sub.columns else float("nan")
        row += f"{v:>{COL_W}.4f}"
        all_avgs.setdefault(c, []).append(v)
    print(row)

print("-" * SEP)
overall = f"{'ALL':<10}"
for c in ALL_COLS:
    vals = [v for v in all_avgs.get(c, []) if not np.isnan(v)]
    overall += f"{np.mean(vals):>{COL_W}.4f}" if vals else f"{'NaN':>{COL_W}}"
print(overall)

# ── Key findings ──────────────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("KEY FINDINGS — does Recon std add value over Recon mean?")
print(f"Note: std primarily helps SPIKE anomalies, not long-duration.")
print(f"{'='*SEP}")

base_vals = results_df["baseline_f1"].dropna()
print(f"Baseline (LTR k=5)    avg F1 = {base_vals.mean():.4f}")

for c, lbl in [("ltr_mean_w03_f1", "LTR+0.3*mean        "),
               ("ltr_std_w05_f1",  "LTR+0.5*std         "),
               ("ltr_std_w10_f1",  "LTR+1.0*std         "),
               ("ltr_m03_s05_f1",  "LTR+0.3*mean+0.5*std"),
               ("ltr_m03_s10_f1",  "LTR+0.3*mean+1.0*std")]:
    if c in results_df.columns:
        v = results_df[c].dropna().mean()
        print(f"{lbl:<32} avg F1 = {v:.4f}  ({v-base_vals.mean():+.4f} vs baseline)")

# SMAP E-series spotlight (the critical failure cases)
print("\nSMAP E-series spotlight (critical long-duration cases):")
smap_e = results_df[
    (results_df["dataset"] == "SMAP") &
    (results_df["signal"].str.startswith("E"))
]
if not smap_e.empty:
    print(f"{'Signal':<8}", end="")
    for c in ALL_COLS:
        print(f"{LABELS.get(c, c):>{COL_W}}", end="")
    print()
    for _, r in smap_e.iterrows():
        print(f"{r['signal']:<8}", end="")
        for c in ALL_COLS:
            print(f"{_fmt(r.get(c, float('nan'))):>{COL_W}}", end="")
        print()

print(f"\nResults saved → {out_csv}")
