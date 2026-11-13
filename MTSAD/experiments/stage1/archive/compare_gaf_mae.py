"""GAF + MAE: Visualization Ablation
=====================================

Hypothesis
----------
Line-plot images are information-sparse: ~94% of each 16x16 ViT patch is
white background. MAE features are dominated by background noise, so the
actual anomaly signal is diluted.

GAF (Gramian Angular Field) encodes temporal correlations into a dense
224x224 matrix — every pixel carries information. An anomaly at time t
perturbs the entire t-th row AND column, making it spatially extensive
and far more visible to MAE patch embeddings.

Experiment
----------
  Same MAE backbone, same LTR k=5 scoring, same EVT threshold.
  Only the input image changes: line plot → GAF.
  This isolates the effect of visualization from all other factors.

Ablations
---------
  baseline_f1  : line plot + LTR k=5 + EVT   [reused from MGMR cache]
  gaf_f1       : GAF       + LTR k=5 + EVT   [new]
  gaf_add_f1   : GAF       + add(k5, k30_gaf) + EVT   [multi-scale with GAF]

Key signals to watch
--------------------
  SMAP F-1, F-3  : always 0.0 — mean-shift anomaly invisible in line plot
  MSL T-13, D-16 : always 0.0 — pattern shift invisible in line plot
  If GAF fixes even one of these → visualization hypothesis confirmed.

Usage
-----
  python experiments/compare_gaf_mae.py
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT   = os.environ.get("VLM4TS_ROOT", "").strip()
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
from torch.utils.data import Dataset, DataLoader
from preprocessing.data_utils import orion_to_internal
from preprocessing.gaf_encoder import render_gaf_windows
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

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_SIZE   = int(WINDOW_SIZE / 4.0)   # 56
STEP_RATIO  = 4.0
AGG_PERCENT = 0.25
PATCH_SIZE  = 16
PH = PW     = WINDOW_SIZE // PATCH_SIZE  # 14
K_LOCAL     = 5
K_MEDIUM    = 30
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_gaf_mae")
CKPT_DIR     = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# Reuse line-plot k=5 checkpoints for baseline_f1
_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_LINE_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)), _K5_CANDIDATES[-1])

_count = lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0
print(f"line k=5 cache : {K5_LINE_DIR}  ({_count(K5_LINE_DIR)} files)")
print(f"GAF   ckpt dir : {CKPT_DIR}")


# ── In-memory image dataset (no disk I/O) ─────────────────────────────────────
class GAFDataset(Dataset):
    """Wraps a pre-rendered [N, 3, H, W] array for DataLoader consumption."""

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


# ── MAE model (lazy load) ─────────────────────────────────────────────────────
_mae_model = None

def get_mae():
    global _mae_model
    if _mae_model is None:
        print("\n[INFO] Loading MAE ViT-B/16 ...")
        from models.mae_vision import MAE_AD
        _mae_model = MAE_AD(
            model_name="vit_base_patch16_224.mae",
            device=DEVICE,
            image_size=(224, 224),
        )
        _mae_model.eval()
        print("[INFO] MAE ready.")
    return _mae_model


# ── LTR inference on a GAFDataset ─────────────────────────────────────────────
def run_ltr_on_dataset(dataset: GAFDataset, k: int) -> np.ndarray:
    """Run LTR k scoring on pre-rendered images. Returns patch-level score map."""
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
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


# ── EVT threshold + detection ──────────────────────────────────────────────────
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


def add_fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    T = min(len(a), len(b))
    return normalize_01(a[:T]) + normalize_01(b[:T])


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


def f1(detected, gt_ivs):
    return evaluate_intervals(gt_ivs, detected)["F1"]


def baseline_f1_from_cache(ds: str, sig: str, gt_ivs: list) -> float | None:
    """Compute baseline using cached line-plot k=5 scores + EVT."""
    p = os.path.join(K5_LINE_DIR, f"{ds}__{sig}__ltr.pkl")
    if not os.path.exists(p):
        return None
    try:
        c  = pickle.load(open(p, "rb"))
        s5 = c["scores"]; ts = c["timestamps"]
        return f1(evt_detect(s5, ts), gt_ivs)
    except Exception:
        return None


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

# ── Main experiment loop ───────────────────────────────────────────────────────
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
        data   = pd.read_csv(csv_path)
        values, timestamps = orion_to_internal(data)
        sig_row = {"dataset": ds_name, "signal": sig}

        # ── baseline: cached line-plot k=5 + EVT ─────────────────────────
        bl = baseline_f1_from_cache(ds_name, sig, gt_ivs)
        if bl is not None:
            sig_row["baseline_f1"] = bl
            print(f"    baseline (line, k5, EVT)    F1={bl:.4f}")

        # ── GAF k=5 checkpoint ────────────────────────────────────────────
        gaf_k5_ckpt  = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__gaf_k5.pkl")
        gaf_k30_ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__gaf_k30.pkl")

        # Load or compute GAF k=5 scores
        gaf_scores_k5 = None
        t0 = time.time()
        if os.path.exists(gaf_k5_ckpt):
            try:
                c = pickle.load(open(gaf_k5_ckpt, "rb"))
                gaf_scores_k5 = c["scores"]
                print(f"    [GAF k5]  cache  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [GAF k5]  load error: {e}")

        if gaf_scores_k5 is None:
            try:
                imgs    = render_gaf_windows(values, WINDOW_SIZE, STEP_SIZE)
                dataset = GAFDataset(imgs)
                gaf_scores_k5 = run_ltr_on_dataset(dataset, K_LOCAL)
                pickle.dump({"scores": gaf_scores_k5, "timestamps": timestamps},
                            open(gaf_k5_ckpt, "wb"))
                print(f"    [GAF k5]  computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [GAF k5]  ERROR: {e}")

        if gaf_scores_k5 is None:
            print(f"    SKIP — GAF k=5 failed")
            rows.append(sig_row)
            continue

        # ── gaf_f1: GAF k=5 + EVT ─────────────────────────────────────────
        T_ts   = min(len(gaf_scores_k5), len(timestamps))
        ts_use = timestamps[:T_ts]
        gf1    = f1(evt_detect(gaf_scores_k5, ts_use), gt_ivs)
        sig_row["gaf_f1"] = gf1
        delta = gf1 - sig_row.get("baseline_f1", float("nan"))
        print(f"    gaf   (GAF,  k5, EVT)      F1={gf1:.4f}  ({delta:+.4f} vs baseline)")

        # ── GAF k=30: load or compute ──────────────────────────────────────
        gaf_scores_k30 = None
        t0 = time.time()
        if os.path.exists(gaf_k30_ckpt):
            try:
                c = pickle.load(open(gaf_k30_ckpt, "rb"))
                gaf_scores_k30 = c["scores"]
                print(f"    [GAF k30] cache  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [GAF k30] load error: {e}")

        if gaf_scores_k30 is None:
            try:
                imgs    = render_gaf_windows(values, WINDOW_SIZE, STEP_SIZE)
                dataset = GAFDataset(imgs)
                gaf_scores_k30 = run_ltr_on_dataset(dataset, K_MEDIUM)
                pickle.dump({"scores": gaf_scores_k30},
                            open(gaf_k30_ckpt, "wb"))
                print(f"    [GAF k30] computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [GAF k30] ERROR: {e}")

        if gaf_scores_k30 is not None:
            fused  = add_fuse(gaf_scores_k5, gaf_scores_k30)
            T_f    = min(len(fused), len(timestamps))
            gaf_add = f1(evt_detect(fused, timestamps[:T_f]), gt_ivs)
            sig_row["gaf_add_f1"] = gaf_add
            print(f"    gaf_add(k5+k30, EVT)       F1={gaf_add:.4f}")

        rows.append(sig_row)


# ── Save CSV ──────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

# ── Print results table ───────────────────────────────────────────────────────
ALL_COLS = ["baseline_f1", "gaf_f1", "gaf_add_f1"]
LABELS   = {"baseline_f1": "baseline", "gaf_f1": "GAF(k5)", "gaf_add_f1": "GAF add(k5,k30)"}
COL_W    = 16


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
    print(f"=== {ds_name} — GAF vs Line Plot ===")
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

# ── Weighted ALL F1 summary ───────────────────────────────────────────────────
ds_weights = {"NAB": 16, "SMAP": 13, "MSL": 11}
cols = [c for c in ALL_COLS if c in results_df.columns]
sep  = 26 + COL_W * len(cols)

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

# ── Key signals to watch ──────────────────────────────────────────────────────
print(f"\n{'='*sep}")
print("KEY SIGNALS TO WATCH (stuck at 0.0 with line plot)")
print(f"{'='*sep}")
stuck = ["F-1", "F-3", "T-13", "D-16"]
for _, r in results_df[results_df["signal"].isin(stuck)].iterrows():
    bl  = r.get("baseline_f1", float("nan"))
    gf1 = r.get("gaf_f1",      float("nan"))
    d   = gf1 - bl if not np.isnan(gf1) and not np.isnan(bl) else float("nan")
    print(f"  {r['dataset']}__{r['signal']:<10}  baseline={bl:.4f}  GAF={gf1:.4f}  delta={d:+.4f}")

print(f"\nResults saved → {out_csv}")
