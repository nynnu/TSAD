"""
CDE-LTR: Delay Embedding Density Maps + MAE + Local Temporal Reference
========================================================================
Hypothesis: Phase-space density maps (Takens delay embedding) are more
discriminative than raw line plots for structural anomalies:

  Normal windows  → spread attractor  → diffuse density map
  Stuck sensor    → collapsed point   → concentrated density map
  Missing segment → empty phase space → near-zero density map
  Outlier spike   → point far from attractor → isolated density blip

τ = first ACF zero-crossing per signal (capped at W//8 = 28).

Scoring: same LTR k=5 pipeline as baseline, but input images are
CDE density maps instead of line plots.

Conditions
----------
  baseline : LTR k=5 on line plot images     (loads existing checkpoints)
  cde_k5   : LTR k=5 on CDE density maps     (new inference)

Usage
-----
  python experiments/compare_cde_ltr.py
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
from scipy.ndimage import gaussian_filter
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

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy", "--quiet"], check=True
)

import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.preprocess import preprocess_time_series, apply_ewma
from preprocessing.data_utils import orion_to_internal
from models.mae_vision import MAE_AD
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps
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

K5_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_cde_ltr")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Baseline k5 : {K5_CKPT_DIR}")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
STEP_SIZE   = int(WINDOW_SIZE / STEP_RATIO)   # 56
IMAGE_SIZE  = 224
PATCH_SIZE  = 16
LOCAL_K     = 5
MIN_REF     = 5
EVT_Q       = 0.90
EVT_FPR     = 0.01

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=PATCH_SIZE,
    model_name="vit_base_patch16_224.mae",
    image_size=(IMAGE_SIZE, IMAGE_SIZE), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

# ── ACF-based τ ───────────────────────────────────────────────────────────────

def compute_tau(x: np.ndarray, max_lag_frac: float = 0.25) -> int:
    x_c  = x - x.mean()
    var  = float(np.dot(x_c, x_c))
    if var < 1e-10:
        return 1
    max_lag = max(2, int(len(x) * max_lag_frac))
    for tau in range(1, max_lag):
        acf = float(np.dot(x_c[:-tau], x_c[tau:])) / var
        if acf <= 0:
            return tau
    return max(1, max_lag // 4)


# ── CDE density map ───────────────────────────────────────────────────────────

def cde_image(window: np.ndarray, tau: int,
              x_range: tuple[float, float],
              size: int = IMAGE_SIZE,
              sigma_px: float | None = None) -> np.ndarray:
    """Delay-embedding density map → float32 [H, W] in [0, 1]."""
    tau  = min(tau, len(window) - 2)
    u    = window[:-tau]
    v    = window[tau:]
    n    = len(u)
    x_min, x_max = x_range
    span = x_max - x_min + 1e-8
    u_px = np.clip(((u - x_min) / span * (size - 1)).astype(int), 0, size - 1)
    v_px = np.clip(((v - x_min) / span * (size - 1)).astype(int), 0, size - 1)
    H = np.zeros((size, size), dtype=np.float32)
    np.add.at(H, (v_px, u_px), 1.0)
    if sigma_px is None:
        std_px   = max(1.0, float(np.std(u_px)))
        sigma_px = max(3.0, (n ** (-1.0 / 6.0)) * std_px)
    density = gaussian_filter(H, sigma=sigma_px)
    if density.max() > 1e-10:
        density /= density.max()
    return density


def build_cde_images(values: np.ndarray, tau: int,
                     x_range: tuple[float, float],
                     window_size: int = WINDOW_SIZE,
                     step_size: int = STEP_SIZE,
                     image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Render CDE density maps for all windows → [N, 3, H, W] float32."""
    n_wins = (len(values) - window_size) // step_size + 1
    imgs   = np.empty((n_wins, 3, image_size, image_size), dtype=np.float32)
    for i in range(n_wins):
        start = i * step_size
        density = cde_image(values[start: start + window_size], tau, x_range, image_size)
        imgs[i, 0] = density
        imgs[i, 1] = density
        imgs[i, 2] = density
    return imgs


# ── In-memory dataset for CDE images ─────────────────────────────────────────

class CDEDataset(Dataset):
    """Wraps a pre-built [N, 3, H, W] float32 array for MAE encoding."""
    def __init__(self, imgs: np.ndarray):
        self.imgs = torch.from_numpy(imgs).float()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return {"img": self.imgs[idx], "cls_name": "cde", "window_id": idx}


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
            in_seg = True; seg_start = i
        elif not f and in_seg:
            in_seg = False
            ivs.append([timestamps[seg_start], timestamps[i - 1]])
    if in_seg:
        ivs.append([timestamps[seg_start], timestamps[len(flags) - 1]])
    return ivs


# ── LTR scoring over embeddings ───────────────────────────────────────────────

def ltr_score_from_embeddings(
    large_embeds, mid_embeds, patch_embeds,
    large_mask, mid_mask,
    model, local_k: int = LOCAL_K, min_ref: int = MIN_REF,
    image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE,
    window_step_ratio: float = STEP_RATIO,
) -> np.ndarray:
    L    = large_embeds.shape[0]
    ph   = pw = image_size // patch_size
    maps = []
    with torch.no_grad():
        for i in range(L):
            l_ref, _ = get_local_reference(large_embeds, i, local_k, min_ref)
            m_ref, _ = get_local_reference(mid_embeds,   i, local_k, min_ref)
            p_ref, _ = get_local_reference(patch_embeds, i, local_k, min_ref)
            m_l = compute_dissimilarity_with_ref(
                large_embeds[i].unsqueeze(0).to(DEVICE), l_ref.to(DEVICE))
            m_m = compute_dissimilarity_with_ref(
                mid_embeds[i].unsqueeze(0).to(DEVICE),   m_ref.to(DEVICE))
            m_p = compute_dissimilarity_with_ref(
                patch_embeds[i].unsqueeze(0).to(DEVICE), p_ref.to(DEVICE))
            m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(DEVICE)
            m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(DEVICE)
            m_p = m_p.reshape((1, ph, pw)).to(DEVICE)
            score = torch.nan_to_num((m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.)
            score = torch.nn.functional.interpolate(
                score.unsqueeze(1), size=(image_size, image_size), mode="bilinear"
            ).squeeze(1)
            maps.append(score.squeeze(0).detach().cpu())
    maps_arr = torch.stack(maps, dim=0).numpy()
    return stitch_anomaly_maps(maps_arr, window_step_ratio, agg_percent=0.25)


# ── Model ─────────────────────────────────────────────────────────────────────

print("\n[INFO] Loading MAE model ...")
mae_model = MAE_AD(model_name="vit_base_patch16_224.mae", device=DEVICE)
mae_model.eval()
print("  Model loaded.")

# Baseline LTR detector (for loading checkpoints)
det_baseline = ViT4TS_MAE(**BASE_PARAMS)


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

gt = load_gt()

NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
NAB_SIGS = sorted(f[:-4] for f in os.listdir(NAB_DIR) if f.endswith(".csv") and gt.get(f[:-4]))

SMAP_SIGS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"]
MSL_SIGS  = ["P-11","T-12","D-15","C-1","F-8","F-7",
             "T-13","D-16","T-8","P-14","D-14"]

DATASET_CONFIGS = {
    "NAB":  {"dir": NAB_DIR,                          "sigs": NAB_SIGS},
    "SMAP": {"dir": os.path.join(DATA_DIR, "SMAP"),   "sigs": SMAP_SIGS},
    "MSL":  {"dir": os.path.join(DATA_DIR, "MSL"),    "sigs": MSL_SIGS},
}

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
        t0 = time.time()

        data       = pd.read_csv(csv_path)
        values_raw, timestamps = orion_to_internal(data)
        T_full     = len(values_raw)

        values_proc = preprocess_time_series(values_raw)
        values_proc = apply_ewma(values_proc, 1.0)

        row = {"dataset": ds_name, "signal": sig}

        # ── Baseline: load existing LTR k=5 checkpoint ──────────────────────
        ckpt_base = os.path.join(K5_CKPT_DIR, f"{ds_name}__{sig}__ltr.pkl")
        try:
            if os.path.exists(ckpt_base):
                c   = pickle.load(open(ckpt_base, "rb"))
                s_b = c["scores"]
                ts  = c["timestamps"]
            else:
                s_b, ts = det_baseline.predict_scores(data)
                pickle.dump({"scores": s_b, "timestamps": ts}, open(ckpt_base, "wb"))
            ivs = evt_detect(s_b, ts)
            row["baseline_f1"] = evaluate_intervals(gt_ivs, ivs)["F1"]
        except Exception as e:
            print(f"    baseline ERROR: {e}")
            row["baseline_f1"] = float("nan")
            ts = timestamps

        # ── CDE-LTR: render density maps → MAE → LTR k=5 ────────────────────
        ckpt_cde = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__cde_k5.pkl")
        try:
            if os.path.exists(ckpt_cde):
                c_cde  = pickle.load(open(ckpt_cde, "rb"))
                s_cde  = c_cde["scores"]
                ts_cde = c_cde["timestamps"]
                print(f"    [CDE] cache hit  ({time.time()-t0:.1f}s)")
            else:
                # τ from full-signal ACF, capped at W//8
                tau_acf = compute_tau(values_proc)
                tau     = min(tau_acf, WINDOW_SIZE // 8)
                print(f"    τ_acf={tau_acf}  τ_used={tau}")

                x_range = (float(values_proc.min()), float(values_proc.max()))
                imgs    = build_cde_images(values_proc, tau, x_range)   # [N, 3, H, W]
                print(f"    CDE images: {imgs.shape}")

                dataset = CDEDataset(imgs)
                loader  = DataLoader(dataset, batch_size=16, shuffle=False)

                (large_e, mid_e, patch_e,
                 l_mask, m_mask, _) = build_ordered_embeddings(
                    mae_model, loader, PATCH_SIZE, DEVICE)

                s_cde  = ltr_score_from_embeddings(
                    large_e, mid_e, patch_e, l_mask, m_mask, mae_model)
                ts_cde = timestamps

                pickle.dump({"scores": s_cde, "timestamps": ts_cde},
                            open(ckpt_cde, "wb"))
                print(f"    [CDE] computed  ({time.time()-t0:.1f}s)")

            T_min = min(len(s_cde), len(ts_cde))
            ivs_cde = evt_detect(s_cde[:T_min], ts_cde[:T_min])
            row["cde_k5_f1"] = evaluate_intervals(gt_ivs, ivs_cde)["F1"]

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    [CDE] ERROR: {e}")
            row["cde_k5_f1"] = float("nan")

        delta = row.get("cde_k5_f1", float("nan")) - row.get("baseline_f1", float("nan"))
        print(f"    baseline F1={row['baseline_f1']:.4f}  "
              f"cde_k5 F1={row['cde_k5_f1']:.4f}  (Δ={delta:+.4f})")
        rows.append(row)


# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "cde_ltr.csv"), index=False)

sig_counts = {"NAB": 16, "SMAP": 13, "MSL": 11}

print(f"\n{'='*60}")
print("SUMMARY — CDE-LTR vs Baseline")
print(f"{'='*60}")
print(f"{'Dataset':<10}  {'baseline':>10}  {'cde_k5':>10}  {'Δ':>8}")
print("-" * 44)

wtd = {"baseline": 0.0, "cde_k5": 0.0}
for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    b = sub["baseline_f1"].mean()
    c = sub["cde_k5_f1"].mean()
    print(f"{ds:<10}  {b:>10.4f}  {c:>10.4f}  {c-b:>+8.4f}")
    w = sig_counts[ds]
    wtd["baseline"] += b * w
    wtd["cde_k5"]   += c * w

total = sum(sig_counts.values())
print("-" * 44)
print(f"{'ALL (wtd)':<10}  {wtd['baseline']/total:>10.4f}  "
      f"{wtd['cde_k5']/total:>10.4f}  "
      f"{(wtd['cde_k5']-wtd['baseline'])/total:>+8.4f}")
print(f"\nResults → {RESULTS_DIR}/cde_ltr.csv")
