"""MAE + LP Residual Scoring Experiment

Hypothesis
----------
Friend's DINOv2 + LP resid_sum achieves SMAP=0.9762.
Is the gain from LP scoring, or from DINOv2 backbone?

Test: apply LP residual scoring to OUR MAE embeddings (same backbone as baseline).
  - If MAE+LP ≈ DINOv2+LP  →  scoring method is key, backbone doesn't matter
  - If MAE+LP << DINOv2+LP →  DINOv2 backbone is essential

Pipeline
--------
  raw signal
    → global preprocessing (same as baseline: detrend + minmax → [0,1])
    → sliding windows (size=224, step=56, same as baseline)
    → render as 224×224 line plot (ylim=(0,1), same as baseline)
    → MAE ViT-B/16 forward_features → mean pool → 768-dim vector
    → AR(p) Ridge regression → prediction residuals → anomaly scores
    → EVT thresholding

LP Variants
-----------
  lp_p5_sum   : AR order 5,  resid_sum = Σ|e_t[d] - ê_t[d]|  ← friend's method
  lp_p10_sum  : AR order 10, resid_sum
  lp_p20_sum  : AR order 20, resid_sum
  lp_p5_norm  : AR order 5,  resid_norm = ||e_t - ê_t||₂
  lp_p10_norm : AR order 10, resid_norm

Caches
------
  embeddings: results_lp_resid/embeddings/{DS}__{sig}__embeds.pkl
  scores:     results_lp_resid/checkpoints/{DS}__{sig}__{variant}.pkl
  (re-run safe: always resumes from cache)
"""
from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
import time
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from scipy.signal import detrend as scipy_detrend
from scipy.stats import genpareto
from sklearn.linear_model import Ridge

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT   = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy",
     "scikit-learn", "transformers", "--quiet"],
    check=True,
)

import torch
import timm

from evaluation.evaluate import evaluate_intervals
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

K5_CKPT_DIR = next((p for p in [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
] if os.path.isdir(p)), "")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_lp_resid")
EMBED_DIR   = os.path.join(RESULTS_DIR, "embeddings")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
for d in [RESULTS_DIR, EMBED_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"k5 ckpts  : {K5_CKPT_DIR}")

# ── Constants (must match baseline) ───────────────────────────────────────────
WINDOW_SIZE = 224
STEP_SIZE   = 56          # WINDOW_SIZE / 4
IMAGE_SIZE  = (224, 224)
DPI         = 100
FIG_W = FIG_H = IMAGE_SIZE[0] / DPI   # 2.24 inches

EVT_Q_INIT = 0.90
EVT_FPR    = 0.01

AR_ORDERS = [5, 10, 20]
LP_MODES  = ["sum", "norm"]          # sum = friend's method

# ── MAE model ─────────────────────────────────────────────────────────────────
print("\n[INFO] Loading MAE model ...")
_mae = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
_mae = _mae.eval().to(DEVICE)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


# ── Preprocessing (global — same as baseline) ─────────────────────────────────
def global_preprocess(values: np.ndarray) -> np.ndarray:
    v = scipy_detrend(values.astype(float))
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-8:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


# ── Window rendering ──────────────────────────────────────────────────────────
def render_windows(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Slide window, render each as 224×224 line plot (ylim=(0,1) — same as baseline).
    Returns: imgs (N, 3, 224, 224) uint8,  start_indices list
    """
    imgs, starts = [], []
    t = np.arange(WINDOW_SIZE)
    start = 0
    while start + WINDOW_SIZE <= len(values):
        window = values[start:start + WINDOW_SIZE]
        fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), dpi=DPI)
        ax.plot(t, window, '-', lw=1, color='black')
        ax.set_ylim(0.0, 1.0)
        ax.axis('off')
        fig.tight_layout(pad=0)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = np.array(
            PILImage.open(buf).convert('RGB').resize(IMAGE_SIZE)
        ).transpose(2, 0, 1)                   # (3, H, W) uint8
        imgs.append(img)
        starts.append(start)
        start += STEP_SIZE
    return np.stack(imgs, axis=0), starts


def window_center_ts(timestamps: np.ndarray, starts: list[int]) -> np.ndarray:
    center = WINDOW_SIZE // 2
    return np.array([timestamps[s + center] for s in starts])


# ── Embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(imgs: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """
    imgs: (N, 3, 224, 224) uint8
    → MAE forward_features → mean-pool tokens → (N, 768) float32
    """
    results = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.from_numpy(imgs[i:i + batch_size]).float().to(DEVICE) / 255.0
        batch = (batch - _MEAN) / _STD
        with torch.no_grad():
            feats = _mae.forward_features(batch)   # (B, N_tok, 768) or (B, 768)
            if feats.ndim == 3:
                pooled = feats.mean(dim=1)         # mean pool all tokens
            else:
                pooled = feats
        results.append(pooled.cpu().float().numpy())
    return np.vstack(results)  # (N, 768)


# ── LP Residual scoring ───────────────────────────────────────────────────────
def lp_residual_scores(embeds: np.ndarray, ar_order: int, mode: str) -> np.ndarray:
    """
    AR(p) Ridge regression on embedding time series.
    mode='sum'  : Σ|e_t[d] - ê_t[d]|          ← friend's resid_sum
    mode='norm' : ||e_t - ê_t||₂               ← L2 residual norm
    Returns (N,) float32 anomaly scores.
    """
    N, D = embeds.shape
    scores = np.zeros(N, dtype=np.float32)
    if N <= ar_order + 2:
        return scores

    # Design matrix: X[i] = concat(e_{i-1}, ..., e_{i-p})  →  predict e_i
    X = np.hstack([embeds[ar_order - k - 1:N - k - 1] for k in range(ar_order)])
    y = embeds[ar_order:]                          # (N-p, 768)

    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(X, y)
    residuals = y - reg.predict(X)                 # (N-p, 768)

    s = np.abs(residuals).sum(axis=1) if mode == "sum" \
        else np.linalg.norm(residuals, axis=1)

    scores[ar_order:] = s
    scores[:ar_order] = s.mean()
    return scores


# ── EVT & helpers ─────────────────────────────────────────────────────────────
def evt_threshold(scores, q=EVT_Q_INIT, fpr=EVT_FPR):
    u = float(np.percentile(scores, q * 100))
    exc = scores[scores > u] - u
    fallback = float(np.percentile(scores, (1 - fpr) * 100))
    if len(exc) < 10:
        return fallback
    try:
        c, _, scale = genpareto.fit(exc, floc=0)
        p_cond = min(fpr / max(1 - q, 1e-9), 1 - 1e-9)
        excess = genpareto.ppf(1 - p_cond, c, loc=0, scale=scale)
        thr = u + max(0.0, excess)
        return thr if u <= thr <= scores.max() else fallback
    except Exception:
        return fallback


def evt_detect(scores, timestamps):
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
        intervals.append([timestamps[seg_start], timestamps[-1]])
    return intervals


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt


def _eval(detected, gt_ivs):
    return evaluate_intervals(gt_ivs, detected)["F1"]


def _ivs_alpha(scores, timestamps, alpha=0.01):
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


def load_pkl(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ── Datasets ──────────────────────────────────────────────────────────────────
NAB_SIGNALS = [
    "ec2_cpu_utilization_24ae8d", "ec2_cpu_utilization_53ea38",
    "ec2_cpu_utilization_5f5533", "ec2_cpu_utilization_77c1ca",
    "ec2_cpu_utilization_825cc2", "ec2_cpu_utilization_ac20cd",
    "ec2_cpu_utilization_fe7f93", "ec2_disk_write_bytes_1ef3de",
    "ec2_disk_write_bytes_c0d644", "ec2_network_in_257a54",
    "ec2_network_in_5abac7", "elb_request_count_8c0756",
    "grok_asg_anomaly", "iio_us-east-1_i-a2eb1cd9_NetworkIn",
    "rds_cpu_utilization_cc0c53", "rds_cpu_utilization_e47b3b",
]
SMAP_SIGNALS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
                "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGNALS  = ["P-11","T-12","D-15","C-1","F-8","F-7","T-13",
                "D-16","T-8","P-14","D-14"]

DATASETS = [("NAB", NAB_SIGNALS), ("SMAP", SMAP_SIGNALS), ("MSL", MSL_SIGNALS)]
GT = load_gt(ANOM_CSV)

# ── LP variant names ──────────────────────────────────────────────────────────
LP_VARIANTS = [(p, m) for p in AR_ORDERS for m in LP_MODES]

# ── Main loop ─────────────────────────────────────────────────────────────────
records = []

for ds, signals in DATASETS:
    print(f"\n{'='*72}")
    print(f"Dataset: {ds}  ({len(signals)} signals)")
    print(f"{'='*72}")

    csv_dir = {"NAB": "realAWSCloudwatch", "SMAP": "SMAP", "MSL": "MSL"}[ds]
    csv_dir = os.path.join(DATA_DIR, csv_dir)

    for sig in signals:
        gt_key = sig if ds == "NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig, []))

        # baseline k5 scores (for comparison)
        k5_pkl = load_pkl(os.path.join(K5_CKPT_DIR, f"{ds}__{sig}__ltr.pkl"))
        if k5_pkl is None:
            print(f"  [{sig}] SKIP — k5 checkpoint missing")
            continue
        k5_scores = np.array(k5_pkl["scores"])
        k5_ts     = np.array(k5_pkl["timestamps"])
        f1_base   = _eval(_ivs_alpha(k5_scores, k5_ts), gt_ivs)

        print(f"\n  [{sig}]")

        # ── Embeddings ────────────────────────────────────────────────────────
        embed_path = os.path.join(EMBED_DIR, f"{ds}__{sig}__embeds.pkl")
        embed_pkl  = load_pkl(embed_path)

        if embed_pkl is not None:
            embeds = embed_pkl["embeddings"]
            win_ts = embed_pkl["timestamps"]
            print(f"    [embeds] loaded from cache  {embeds.shape}")
        else:
            csv_path = os.path.join(csv_dir, f"{sig}.csv")
            if not os.path.exists(csv_path):
                print(f"    SKIP — CSV missing: {csv_path}")
                continue

            df_raw   = pd.read_csv(csv_path)
            ts_col   = next(c for c in df_raw.columns if "time" in c.lower())
            val_col  = next(c for c in df_raw.columns if c != ts_col)
            raw_ts   = pd.to_datetime(df_raw[ts_col]).astype(np.int64).to_numpy()
            raw_vals = df_raw[val_col].to_numpy(dtype=float)

            proc_vals = global_preprocess(raw_vals)

            t0 = time.time()
            imgs, starts = render_windows(proc_vals)
            win_ts = window_center_ts(raw_ts, starts)
            print(f"    [render] {time.time()-t0:.1f}s  {imgs.shape}")

            t0 = time.time()
            embeds = extract_embeddings(imgs)
            print(f"    [encode] {time.time()-t0:.1f}s  {embeds.shape}")

            save_pkl({"embeddings": embeds, "timestamps": win_ts}, embed_path)

        # ── LP scoring ────────────────────────────────────────────────────────
        print(f"    baseline (LTR k5)  F1={f1_base:.4f}")

        row = dict(dataset=ds, signal=sig, baseline_f1=f1_base)

        for p, mode in LP_VARIANTS:
            variant = f"lp_p{p}_{mode}"
            ckpt_path = os.path.join(CKPT_DIR, f"{ds}__{sig}__{variant}.pkl")
            cached    = load_pkl(ckpt_path)

            if cached is not None:
                lp_scores = np.array(cached["scores"])
            else:
                lp_scores = lp_residual_scores(embeds, ar_order=p, mode=mode)
                save_pkl({"scores": lp_scores, "timestamps": win_ts}, ckpt_path)

            T   = min(len(lp_scores), len(win_ts))
            ivs = evt_detect(lp_scores[:T], win_ts[:T])
            f1  = _eval(ivs, gt_ivs)
            row[variant] = f1

            marker = " ← friend's method" if (p == 5 and mode == "sum") else ""
            print(f"    {variant:<18}  F1={f1:.4f}  (Δ={f1-f1_base:+.4f}){marker}")

        records.append(row)

# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
lp_cols = [f"lp_p{p}_{m}" for p, m in LP_VARIANTS]
lp_cols = [c for c in lp_cols if c in df.columns]

W = 16
print("\n\n" + "=" * (8 + 12 + W * len(lp_cols)))
print("CROSS-DATASET SUMMARY")
print("=" * (8 + 12 + W * len(lp_cols)))
hdr = f"{'Dataset':<8}{'baseline':>12}" + "".join(f"{c:>{W}}" for c in lp_cols)
print(hdr)
print("-" * len(hdr))
for ds in ["NAB", "SMAP", "MSL"]:
    sub = df[df.dataset == ds]
    print(f"{ds:<8}{sub.baseline_f1.mean():>12.4f}" +
          "".join(f"{sub[c].mean():>{W}.4f}" for c in lp_cols))
print("-" * len(hdr))
print(f"{'ALL':<8}{df.baseline_f1.mean():>12.4f}" +
      "".join(f"{df[c].mean():>{W}.4f}" for c in lp_cols))

# ── Friend comparison ──────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("KEY COMPARISON: LP scoring on MAE vs DINOv2+LP (friend)")
print("=" * 60)
smap = df[df.dataset == "SMAP"]
best_lp_smap = max((smap[c].mean(), c) for c in lp_cols)
print(f"  Friend DINOv2+LP  SMAP = 0.9762  ALL ≈ 0.6834")
print(f"  MAE+LP best SMAP  = {best_lp_smap[0]:.4f}  ({best_lp_smap[1]})")
print(f"  MAE baseline SMAP = {smap.baseline_f1.mean():.4f}")
print()
if best_lp_smap[0] > 0.85:
    print("  → SCORING METHOD is the key. MAE+LP ≈ DINOv2+LP.")
    print("    DINOv2 backbone not necessary.")
elif best_lp_smap[0] > 0.70:
    print("  → LP scoring helps significantly, but DINOv2 provides additional gain.")
    print("    Both backbone and scoring contribute.")
else:
    print("  → DINOv2 BACKBONE appears to be key. MAE+LP is much lower.")
    print("    DINOv2 features more suitable for LP prediction in SMAP domain.")

# ── Stuck signals ──────────────────────────────────────────────────────────────
print("\n\nStuck signals (baseline F1=0):")
stuck = df[df.baseline_f1 == 0.0]
for _, row in stuck.iterrows():
    vals = "  ".join(f"{c.split('_',1)[1]}={row[c]:.3f}" for c in lp_cols)
    print(f"  {row.dataset}_{row.signal:<8}  {vals}")

out_csv = os.path.join(RESULTS_DIR, "comparison.csv")
df.to_csv(out_csv, index=False)
print(f"\nResults saved → {out_csv}")
