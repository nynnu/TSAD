"""Recurrence Plot + MAE: Two-Component Anomaly Scoring
=======================================================

Hypothesis
----------
Line plots are information-sparse (~94% background).
RP creates binary-like textures with diagonal line structures
that look closer to natural images → MAE features more discriminative.

Two complementary components address the full anomaly taxonomy:

  Component A: Within-window RP + LTR k=5
    - RP(i,j) captures internal self-similarity of each window
    - LTR compares RP structure to temporal neighbors
    - Detects: point anomalies, transitional anomalies
    - Misses:  sustained mean shifts (self-similarity unchanged)

  Component B: Cross Recurrence Plot (CRP) density scoring
    - CRP(window_i, reference) measures cross-similarity to normal reference
    - Reference = median of first N_REF windows (assumed pre-anomaly normal)
    - Score = 1 - CRP_density (low cross-recurrence = anomalous)
    - Detects: mean shifts, level changes, sustained anomalies
    - No MAE needed — pure signal-level comparison

  Combined: add(norm(score_A), norm(score_B))

Ablation columns
----------------
  baseline_f1  : line plot + LTR k=5 + EVT          [cached]
  rp_ltr_f1    : RP + LTR k=5 + EVT                 [Component A]
  crp_f1       : CRP density + EVT                   [Component B]
  rp_crp_f1    : add(rp_ltr, crp) + EVT             [combined]

Key signals to watch
--------------------
  rp_ltr vs baseline   → does RP visualization help for LTR?
  crp_f1 vs baseline   → does CRP catch the stuck signals (F-1, T-13)?
  rp_crp vs all        → is the combined score universally better?

Usage
-----
  python experiments/compare_rp_mae.py
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

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

subprocess.run(
    ["pip", "install", "timm", "open-clip-torch", "scipy", "transformers", "--quiet"],
    check=True,
)

import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing.data_utils import orion_to_internal
from preprocessing.rp_encoder import render_rp_windows, compute_crp
from models.mae_vision import MAE_AD
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 224
STEP_SIZE    = int(WINDOW_SIZE / 4.0)   # 56
STEP_RATIO   = 4.0
AGG_PERCENT  = 0.25
PATCH_SIZE   = 16
PH = PW      = WINDOW_SIZE // PATCH_SIZE  # 14
K_LOCAL      = 5
N_REF        = 10    # first N_REF windows used as CRP reference
SIGMA_SCALE  = 1.0   # RP Gaussian bandwidth (z-scored units)
EVT_Q_INIT   = 0.90
EVT_FPR      = 0.01

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_rp_mae")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_LINE_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)), _K5_CANDIDATES[-1])

_count = lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0
print(f"line k=5 cache : {K5_LINE_DIR}  ({_count(K5_LINE_DIR)} files)")
print(f"ckpt dir       : {CKPT_DIR}")


# ── In-memory dataset ─────────────────────────────────────────────────────────
class ArrayDataset(Dataset):
    def __init__(self, imgs: np.ndarray):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return {
            "img":       torch.from_numpy(self.imgs[idx]).float(),
            "cls_name":  "",
            "window_id": idx,
        }


# ── MAE model (lazy) ──────────────────────────────────────────────────────────
_mae_model = None

def get_mae() -> MAE_AD:
    global _mae_model
    if _mae_model is None:
        print("\n[INFO] Loading MAE ViT-B/16 ...")
        _mae_model = MAE_AD(
            model_name="vit_base_patch16_224.mae",
            device=DEVICE,
            image_size=(224, 224),
        )
        _mae_model.eval()
        print("[INFO] MAE ready.")
    return _mae_model


# ── Component A: RP + LTR ──────────────────────────────────────────────────────
def run_rp_ltr(imgs: np.ndarray, k: int) -> np.ndarray:
    """LTR k scoring on RP images. Returns stitched 1-D score vector."""
    loader = DataLoader(ArrayDataset(imgs), batch_size=20, shuffle=False)
    model  = get_mae()

    (large_embeds, mid_embeds, patch_embeds,
     large_mask, mid_mask, _) = build_ordered_embeddings(
        model, loader, PATCH_SIZE, DEVICE
    )

    L = large_embeds.shape[0]
    anomaly_maps = []
    with torch.no_grad():
        for i in range(L):
            l_ref, _ = get_local_reference(large_embeds, i, k, 5)
            m_ref, _ = get_local_reference(mid_embeds,   i, k, 5)
            p_ref, _ = get_local_reference(patch_embeds, i, k, 5)

            m_l = compute_dissimilarity_with_ref(
                large_embeds[i].unsqueeze(0).to(DEVICE), l_ref.to(DEVICE))
            m_m = compute_dissimilarity_with_ref(
                mid_embeds[i].unsqueeze(0).to(DEVICE),   m_ref.to(DEVICE))
            m_p = compute_dissimilarity_with_ref(
                patch_embeds[i].unsqueeze(0).to(DEVICE), p_ref.to(DEVICE))

            m_l = harmonic_aggregation((1, PH, PW), m_l, large_mask).to(DEVICE)
            m_m = harmonic_aggregation((1, PH, PW), m_m, mid_mask).to(DEVICE)
            m_p = m_p.reshape((1, PH, PW)).to(DEVICE)

            score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                     nan=0., posinf=0., neginf=0.)
            score = torch.nn.functional.interpolate(
                score.unsqueeze(1), size=(224, 224), mode="bilinear"
            ).squeeze(1)
            anomaly_maps.append(score.squeeze(0).detach().cpu())

    maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
    return stitch_anomaly_maps(maps_arr, STEP_RATIO, AGG_PERCENT)


# ── Component B: CRP density scoring ──────────────────────────────────────────
def compute_crp_scores(values: np.ndarray) -> np.ndarray:
    """Cross Recurrence Plot density scoring against a fixed normal reference.

    Reference = median of the first N_REF windows (raw values).
    Normalization uses GLOBAL mean/std — preserves inter-window mean shifts.
    Per-window z-score would erase mean shifts (both windows look identical
    after normalization), so we must use global statistics here.

    Score(i) = 1 - CRP_density(window_i, reference)
             = fraction of non-recurrent points → high = anomalous.

    Returns a T-length time series score vector.
    """
    T      = len(values)
    starts = list(range(0, T - WINDOW_SIZE + 1, STEP_SIZE))
    N      = len(starts)

    # Global normalization — preserves mean shift between windows
    g_mu  = values.mean()
    g_std = values.std() + 1e-8

    # Build reference: median of first N_REF windows (raw, before normalization)
    n_ref     = min(N_REF, N)
    ref_wins  = [values[starts[j] : starts[j] + WINDOW_SIZE] for j in range(n_ref)]
    reference = np.median(np.stack(ref_wins, axis=0), axis=0)   # (WINDOW_SIZE,)
    ref_norm  = (reference - g_mu) / g_std

    # Score each window by CRP density against reference
    win_scores = np.empty(N, dtype=np.float32)
    for i, s in enumerate(starts):
        win      = values[s : s + WINDOW_SIZE]
        win_norm = (win - g_mu) / g_std                          # global normalization
        diff     = win_norm[:, None] - ref_norm[None, :]         # (W, W)
        crp      = np.exp(-diff ** 2 / (2.0 * SIGMA_SCALE ** 2))
        density       = crp.mean()
        win_scores[i] = 1.0 - density                           # high = anomalous

    # Stitch to time-series length (mean aggregation)
    out   = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)
    for i, s in enumerate(starts):
        e = min(s + WINDOW_SIZE, T)
        out[s:e]   += win_scores[i]
        count[s:e] += 1.0
    return out / np.maximum(count, 1.0)


# ── EVT ───────────────────────────────────────────────────────────────────────
def evt_threshold(scores: np.ndarray,
                  q_init: float = EVT_Q_INIT,
                  fpr: float = EVT_FPR) -> float:
    u  = float(np.percentile(scores, q_init * 100))
    ex = scores[scores > u] - u
    fb = float(np.percentile(scores, (1 - fpr) * 100))
    if len(ex) < 10:
        return fb
    try:
        c, _, sc = genpareto.fit(ex, floc=0)
        p   = min(fpr / max(1 - q_init, 1e-9), 1 - 1e-9)
        thr = u + max(0., genpareto.ppf(1 - p, c, loc=0, scale=sc))
        return thr if u <= thr <= scores.max() else fb
    except Exception:
        return fb


def evt_detect(scores: np.ndarray, timestamps: np.ndarray) -> list:
    if scores.max() - scores.min() < 1e-8:
        return []
    flags = scores > evt_threshold(scores)
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


def normalize_01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-8 else (a - lo) / (hi - lo)


def f1_score(detected: list, gt_ivs: list) -> float:
    return evaluate_intervals(gt_ivs, detected)["F1"]


# ── GT loading ────────────────────────────────────────────────────────────────
def load_gt(path: str) -> dict:
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


def baseline_f1_from_cache(ds: str, sig: str, gt_ivs: list):
    p = os.path.join(K5_LINE_DIR, f"{ds}__{sig}__ltr.pkl")
    if not os.path.exists(p):
        return None
    try:
        c = pickle.load(open(p, "rb"))
        return f1_score(evt_detect(c["scores"], c["timestamps"]), gt_ivs)
    except Exception:
        return None


# ── Dataset config ─────────────────────────────────────────────────────────────
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

# ── Main loop ──────────────────────────────────────────────────────────────────
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
        data           = pd.read_csv(csv_path)
        values, timestamps = orion_to_internal(data)
        T              = len(values)
        sig_row        = {"dataset": ds_name, "signal": sig}

        # ── Baseline ──────────────────────────────────────────────────────
        bl = baseline_f1_from_cache(ds_name, sig, gt_ivs)
        if bl is not None:
            sig_row["baseline_f1"] = bl
            print(f"    baseline (line, LTR k5)    F1={bl:.4f}")

        # ── Component A: RP + LTR ──────────────────────────────────────────
        rp_ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__rp_ltr.pkl")
        rp_scores = None
        t0 = time.time()

        if os.path.exists(rp_ckpt):
            try:
                rp_scores = pickle.load(open(rp_ckpt, "rb"))["scores"]
                print(f"    [RP+LTR]  cache")
            except Exception:
                pass

        if rp_scores is None:
            try:
                imgs      = render_rp_windows(values, WINDOW_SIZE, STEP_SIZE, SIGMA_SCALE)
                rp_scores = run_rp_ltr(imgs, K_LOCAL)
                pickle.dump({"scores": rp_scores, "timestamps": timestamps},
                            open(rp_ckpt, "wb"))
                print(f"    [RP+LTR]  computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [RP+LTR]  ERROR: {e}")

        if rp_scores is not None:
            T_ts = min(len(rp_scores), len(timestamps))
            f1a  = f1_score(evt_detect(rp_scores[:T_ts], timestamps[:T_ts]), gt_ivs)
            sig_row["rp_ltr_f1"] = f1a
            delta = f1a - sig_row.get("baseline_f1", float("nan"))
            print(f"    rp_ltr  (RP, LTR k5)        F1={f1a:.4f}  ({delta:+.4f})")

        # ── Component B: CRP density ───────────────────────────────────────
        crp_ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__crp.pkl")
        crp_scores = None

        if os.path.exists(crp_ckpt):
            try:
                crp_scores = pickle.load(open(crp_ckpt, "rb"))["scores"]
                print(f"    [CRP]     cache")
            except Exception:
                pass

        if crp_scores is None:
            try:
                t0         = time.time()
                crp_scores = compute_crp_scores(values)
                pickle.dump({"scores": crp_scores}, open(crp_ckpt, "wb"))
                print(f"    [CRP]     computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [CRP]     ERROR: {e}")

        if crp_scores is not None:
            T_ts = min(len(crp_scores), len(timestamps))
            f1b  = f1_score(evt_detect(crp_scores[:T_ts], timestamps[:T_ts]), gt_ivs)
            sig_row["crp_f1"] = f1b
            delta = f1b - sig_row.get("baseline_f1", float("nan"))
            print(f"    crp     (CRP density)       F1={f1b:.4f}  ({delta:+.4f})")

        # ── Combined: add(rp_ltr, crp) ─────────────────────────────────────
        if rp_scores is not None and crp_scores is not None:
            T_c    = min(len(rp_scores), len(crp_scores), len(timestamps))
            fused  = normalize_01(rp_scores[:T_c]) + normalize_01(crp_scores[:T_c])
            f1_ab  = f1_score(evt_detect(fused, timestamps[:T_c]), gt_ivs)
            sig_row["rp_crp_f1"] = f1_ab
            delta  = f1_ab - sig_row.get("baseline_f1", float("nan"))
            print(f"    rp+crp  (RP+LTR + CRP)      F1={f1_ab:.4f}  ({delta:+.4f})")

        rows.append(sig_row)


# ── Save & print ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "rp_ltr_f1", "crp_f1", "rp_crp_f1"]
LABELS   = {
    "baseline_f1": "line+LTR",
    "rp_ltr_f1":   "RP+LTR",
    "crp_f1":      "CRP",
    "rp_crp_f1":   "RP+CRP",
}
COL_W = 11


def _fmt(v) -> str:
    try:
        f = float(v)
        return "   NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "   NaN"


def _print_table(ds_name: str) -> None:
    sub  = results_df[results_df["dataset"] == ds_name]
    if sub.empty:
        return
    cols = [c for c in ALL_COLS if c in results_df.columns]
    sep  = 26 + COL_W * len(cols)
    print(f"\n{'='*sep}")
    print(f"=== {ds_name} ===")
    print(f"{'='*sep}")
    print(f"{'Signal':<26}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in cols))
    print("-" * sep)
    for _, r in sub.iterrows():
        row = f"{r['signal']:<26}"
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
    _print_table(ds_name)

# ── Weighted ALL ───────────────────────────────────────────────────────────────
cols       = [c for c in ALL_COLS if c in results_df.columns]
sep        = 26 + COL_W * len(cols)
ds_weights = {"NAB": 16, "SMAP": 13, "MSL": 11}

print(f"\n{'='*sep}")
print("=== CROSS-DATASET SUMMARY ===")
print(f"{'='*sep}")
print(f"{'Dataset':<10}" + "".join(f"{LABELS[c]:>{COL_W}}" for c in cols))
print("-" * sep)

all_avgs: dict = {}
for ds_name in DATASET_CONFIGS:
    sub = results_df[results_df["dataset"] == ds_name]
    row = f"{ds_name:<10}"
    for c in cols:
        v = float(sub[c].mean()) if c in sub.columns and not sub[c].isna().all() else float("nan")
        row += f"{v:>{COL_W}.4f}"
        all_avgs.setdefault(c, []).append((ds_name, v))
    print(row)

print("-" * sep)
overall = f"{'ALL (wt)':<10}"
for c in cols:
    vals = all_avgs.get(c, [])
    wt   = sum(ds_weights.get(ds, 0) * v for ds, v in vals if not np.isnan(v))
    tot  = sum(ds_weights.get(ds, 0) for ds, v in vals if not np.isnan(v))
    overall += f"{wt/tot:>{COL_W}.4f}" if tot else f"{'NaN':>{COL_W}}"
print(overall)

# ── Stuck signals ─────────────────────────────────────────────────────────────
print(f"\n{'='*sep}")
print("STUCK SIGNALS (baseline ≤ 0.0)")
print(f"{'='*sep}")
stuck_sigs = results_df[results_df.get("baseline_f1", pd.Series()).fillna(1) <= 0.001]
for _, r in stuck_sigs.iterrows():
    parts = [f"{r['dataset']}__{r['signal']:<12}"]
    for c in cols:
        v = r.get(c, float("nan"))
        parts.append(f"{LABELS[c]}={_fmt(v)}")
    print("  " + "  ".join(parts))

print(f"\nResults saved → {out_csv}")
