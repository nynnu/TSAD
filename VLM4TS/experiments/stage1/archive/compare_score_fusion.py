"""
Score-level Fusion: LTR k=5 + TFG
===================================

Problem with decision-level fallback:
  - LTR Signal A always fires (alpha=0.01 guarantees ≥1 interval)
  - B rescue condition (ivs_A == 0) almost never triggers
  - E-3 case: A=1 wrong interval → B (F1=0.5) can never rescue

Fix: combine at SCORE level before thresholding.

  s_combined[t] = normalize_01(s_A)[t] + w * normalize_01(s_B)[t]

Effects:
  - baseline=0 cases (A wrong): B spike raises combined score → correct peak
  - baseline>0 cases (A correct): A dominates; B adds marginal precision boost
  - No threshold re-tuning: EVT applied to s_combined (adaptive)

Ablation:
  baseline  : LTR k=5 + alpha=0.01    (reference)
  fusion_w1 : LTR + 0.1*TFG + EVT
  fusion_w2 : LTR + 0.2*TFG + EVT
  fusion_w3 : LTR + 0.3*TFG + EVT
  fusion_w5 : LTR + 0.5*TFG + EVT

Checkpoint:  reuses results_mgmr/checkpoints/ from compare_mgmr_scoring.py
             (same LTR / TFG score caches, no re-inference needed)

Usage (Colab):
  !python "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/compare_score_fusion.py"
"""

from __future__ import annotations

import ast
import os
import pickle
import sys
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import genpareto

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "preprocessing"))

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

subprocess.run(["pip", "install", "timm", "open-clip-torch", "scipy", "--quiet"], check=True)

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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_score_fusion")
CKPT_DIR    = os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints")  # reuse
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
PATCH_PX    = 16
GRID_DIM    = 14
LOCAL_K     = 5
BATCH_ENC   = 16
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

FUSION_WEIGHTS = [0.1, 0.2, 0.3, 0.5]   # w in: norm(A) + w*norm(B)

_INET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BASE_PARAMS = dict(
    window_size=WINDOW_SIZE, window_step_ratio=STEP_RATIO,
    agg_percent=0.25, patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)


# ============================================================================
# EVT threshold
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
# Normalisation helpers
# ============================================================================

def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def score_fusion(s_A: np.ndarray, s_B: np.ndarray, w: float) -> np.ndarray:
    """normalize_01(A) + w * normalize_01(B), length-aligned."""
    T = min(len(s_A), len(s_B))
    return normalize_01(s_A[:T]) + w * normalize_01(s_B[:T])


# ============================================================================
# Signal A — LTR k=5  (identical to compare_mgmr_scoring.py)
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
# Signal B — TFG  (identical to compare_mgmr_scoring.py)
# ============================================================================

@torch.no_grad()
def _extract_column_features(images_raw, mae_ad, device, batch_size=BATCH_ENC):
    N, D = len(images_raw), 768
    out  = np.zeros((N, GRID_DIM, D), dtype=np.float32)
    inet_mean = _INET_MEAN.to(device)
    inet_std  = _INET_STD.to(device)
    for i in range(0, N, batch_size):
        batch  = images_raw[i:i+batch_size].to(device)
        B_cur  = batch.shape[0]
        batch  = (batch - inet_mean) / inet_std
        _, _, patch_tokens, _, _, _ = mae_ad.encode_image(batch, patch_size=PATCH_PX, use_mask=False)
        patch_tokens = patch_tokens.squeeze(2)
        feats        = patch_tokens.reshape(B_cur, GRID_DIM, GRID_DIM, D)
        row_norms    = torch.norm(feats, dim=-1)
        row_activity = row_norms.mean(dim=2)
        top_rows     = row_activity.argmax(dim=1)
        col_feat     = feats[torch.arange(B_cur, device=device), top_rows]
        out[i:i+B_cur] = col_feat.cpu().numpy()
    return out


def _temporal_gradient_scores(col_feat):
    delta     = col_feat[:, 1:, :] - col_feat[:, :-1, :]
    grad_norm = np.linalg.norm(delta, axis=-1)
    N         = len(col_feat)
    col_score = np.zeros((N, GRID_DIM), dtype=np.float32)
    col_score[:, 0]              = grad_norm[:, 0]
    col_score[:, 1:GRID_DIM-1]  = np.maximum(grad_norm[:, :-1], grad_norm[:, 1:])
    col_score[:, GRID_DIM - 1]  = grad_norm[:, -1]
    return col_score


def _local_zscore(col_score, k=LOCAL_K):
    N, half_k = len(col_score), k // 2
    z = np.zeros_like(col_score)
    for w in range(N):
        lo   = max(0, w - half_k)
        hi   = min(N, w + half_k + 1)
        nbrs = [j for j in range(lo, hi) if j != w]
        if len(nbrs) < 2:
            nbrs = [j for j in range(N) if j != w]
        if not nbrs:
            continue
        nbr  = col_score[nbrs]
        z[w] = (col_score[w] - nbr.mean(axis=0)) / (nbr.std(axis=0) + 1e-6)
    return z


class ViT4TS_SignalB_TFG(ViT4TS_MAE):
    def predict_scores(self, data: pd.DataFrame):
        values, timestamps = orion_to_internal(data)
        T_full      = len(values)
        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)
        step_size     = int(self.window_size / self.window_step_ratio)
        window_starts = list(range(0, T_full - self.window_size + 1, step_size))
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ok = draw_windowed_images(
                base_series_id="series", save_path=tmp,
                time_series=values_proc, time_points=np.arange(len(values_proc)),
                override=True, window_size=self.window_size, step_size=step_size,
                image_size=self.image_size, dpi=self.dpi,
                plot_params=("-", 1, "*", 0.1, "black", (0, 1) if self.standardize else None),
            )
            if not ok:
                return np.zeros(T_full), timestamps
            images_raw = torch.from_numpy(
                np.load(os.path.join(tmp, "series_line_img.npy"))
            ).float()
        col_feat  = _extract_column_features(images_raw, self.model, self.device)
        col_score = _temporal_gradient_scores(col_feat)
        z_score   = _local_zscore(col_score)
        s_B = np.zeros(T_full, dtype=np.float32)
        for wi, ws in enumerate(window_starts):
            for col in range(GRID_DIM):
                t0 = ws + col * PATCH_PX
                t1 = min(ws + (col + 1) * PATCH_PX, T_full)
                s_B[t0:t1] = np.maximum(s_B[t0:t1], z_score[wi, col])
        return s_B, timestamps


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
    """alpha-percentile threshold → interval list."""
    from models.model_utils import compute_detection_intervals
    from preprocessing.data_utils import intervals_from_indices
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


# ============================================================================
# Model init
# ============================================================================
print("\n[INFO] Loading models ...")
det_ltr = ViT4TS_SignalA_LTR(**BASE_PARAMS, local_k=5, min_ref=5)
det_tfg = ViT4TS_SignalB_TFG(**BASE_PARAMS)
print(
    f"  Signal A : LTR k=5  (local temporal reference, 3-scale cosine)\n"
    f"  Signal B : TFG      (max-row encoder, temporal gradient, local z-score)\n"
    f"  Fusion   : norm(A) + w*norm(B), EVT threshold\n"
    f"  Weights  : {FUSION_WEIGHTS}"
)


# ============================================================================
# Dataset config
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

METHODS = ["baseline"] + [f"w{str(w).replace('.','')}" for w in FUSION_WEIGHTS]
METHOD_LABELS = {"baseline": "LTR_k5(A)"}
for w in FUSION_WEIGHTS:
    METHOD_LABELS[f"w{str(w).replace('.', '')}"] = f"A+{w}B"


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

        ckpt_a = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__ltr.pkl")
        ckpt_b = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__tfg.pkl")

        # ── Load / compute Signal A scores ───────────────────────
        t0 = time.time()
        try:
            if os.path.exists(ckpt_a):
                c          = pickle.load(open(ckpt_a, "rb"))
                s_A        = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [A] cache hit  ({time.time()-t0:.1f}s)")
            else:
                s_A, timestamps = det_ltr.predict_scores(data)
                pickle.dump({"scores": s_A, "timestamps": timestamps}, open(ckpt_a, "wb"))
                print(f"    [A] computed   ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [A] ERROR: {e}")
            s_A = timestamps = None

        # ── Load / compute Signal B scores ───────────────────────
        t0 = time.time()
        try:
            if os.path.exists(ckpt_b):
                c   = pickle.load(open(ckpt_b, "rb"))
                s_B = c["scores"]
                print(f"    [B] cache hit  ({time.time()-t0:.1f}s)")
            else:
                s_B, _ = det_tfg.predict_scores(data)
                pickle.dump({"scores": s_B}, open(ckpt_b, "wb"))
                print(f"    [B] computed   ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [B] ERROR: {e}")
            s_B = None

        if s_A is None or s_B is None:
            continue

        # ── Baseline: LTR + alpha=0.01 ───────────────────────────
        try:
            ivs_base   = _ivs_from_scores(s_A, timestamps, alpha=0.01)
            m_base     = _eval(ivs_base, gt_ivs)
        except Exception as e:
            print(f"    baseline ERROR: {e}")
            m_base = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
        sig_row["baseline_f1"] = m_base["f1"]
        print(f"    baseline   F1={m_base['f1']:.4f}  ivs={len(ivs_base)}")

        # ── Score-level fusion  (multiple w values) ──────────────
        T = min(len(s_A), len(s_B))
        ts_trim = timestamps[:T]
        for w in FUSION_WEIGHTS:
            key = f"w{str(w).replace('.', '')}"
            try:
                s_fused = score_fusion(s_A, s_B, w)
                ivs_f   = evt_detect(s_fused, ts_trim)
                m_f     = _eval(ivs_f, gt_ivs)
            except Exception as e:
                print(f"    {key} ERROR: {e}")
                m_f = {"f1": float("nan"), "p": float("nan"), "r": float("nan")}
            sig_row[f"{key}_f1"] = m_f["f1"]
            delta = m_f["f1"] - m_base["f1"]
            print(f"    A+{w}*B     F1={m_f['f1']:.4f}  ivs={len(ivs_f)}  ({delta:+.4f})")

        rows.append(sig_row)


# ============================================================================
# Results table
# ============================================================================
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, "score_fusion.csv"), index=False)

COL_W = 12
SEP   = 38 + COL_W * len(METHODS)


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
    print(f"=== {ds_name} — Score Fusion Ablation ===")
    print(f"{'='*SEP}")
    print(f"{'Signal':<38}" + "".join(f"{METHOD_LABELS[m]:>{COL_W}}" for m in METHODS))
    print("-" * SEP)
    for _, r in sub.iterrows():
        fs   = [r.get(f"{m}_f1", float("nan")) for m in METHODS]
        best = max((f for f in fs if f == f), default=float("nan"))
        line = f"{r['signal']:<38}"
        for f in fs:
            mark  = "→" if (f == f and f == best) else " "
            line += f"{mark + _fmt(f):>{COL_W}}"
        print(line)
    print("-" * SEP)
    avgs = [sub[f"{m}_f1"].mean() for m in METHODS]
    print(f"{'AVG F1':<38}" + "".join(f" {a:>{COL_W-1}.4f}" for a in avgs))


for ds in ["NAB", "SMAP", "MSL"]:
    _print_table(ds, results_df)


# ── Overall summary ───────────────────────────────────────────────────────────
print(f"\n{'='*SEP}")
print("OVERALL SUMMARY  (score-level fusion: norm(A) + w*norm(B), EVT threshold)")
print(f"{'='*SEP}")
print(f"{'Dataset':<14}" + "".join(f"{METHOD_LABELS[m]:>{COL_W}}" for m in METHODS))
print("-" * SEP)
for ds in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df["dataset"] == ds]
    if sub.empty:
        continue
    print(f"{ds:<14}" + "".join(f" {sub[f'{m}_f1'].mean():>{COL_W-1}.4f}" for m in METHODS))

print(f"\nResults → {RESULTS_DIR}/score_fusion.csv")
