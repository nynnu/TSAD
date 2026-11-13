"""GAF + Diagonal Masked Reconstruction
========================================

Research question
-----------------
Can MAE predict the signal values (GAF diagonal) from the correlation
structure (GAF off-diagonal)?  If yes → anomaly when prediction fails.

Ablation design (all use the same MAE backbone, same EVT threshold):
  baseline_f1   : line plot + LTR k=5          [existing, no new inference]
  gaf_rand_f1   : GAF      + random masking    [tests GAF alone]
  gaf_diag_f1   : GAF      + diagonal masking  [full proposal]

Isolating variables:
  baseline   → gaf_rand  :  visualization changes (line → GAF), same scoring
  gaf_rand   → gaf_diag  :  masking strategy changes, same visualization
  baseline   → gaf_diag  :  combined effect

Key signals to watch
--------------------
  SMAP F-1, F-3, MSL T-13, D-16 — stuck at 0.0 with all LTR variants.
  If diagonal masking fixes even one: hypothesis confirmed.

Usage
-----
  python experiments/compare_gaf_diag_recon.py
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
from preprocessing.data_utils import orion_to_internal
from preprocessing.gaf_encoder import render_gaf_windows
from models.mae_diag_recon import MAE_DiagRecon, window_scores_to_timeseries
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_SIZE   = int(WINDOW_SIZE / 4.0)  # 56
EVT_Q_INIT  = 0.90
EVT_FPR     = 0.01

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NAB_DIR  = os.path.join(DATA_DIR, "realAWSCloudwatch")
SMAP_DIR = os.path.join(DATA_DIR, "SMAP")
MSL_DIR  = os.path.join(DATA_DIR, "MSL")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_gaf_diag_recon")
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


# ── Lazy model load ────────────────────────────────────────────────────────────
_scorer: MAE_DiagRecon | None = None


def get_scorer() -> MAE_DiagRecon:
    global _scorer
    if _scorer is None:
        _scorer = MAE_DiagRecon(device=DEVICE, diag_band=1, n_iter=3)
    return _scorer


# ── Random-masking baseline on GAF (uses same MAE_DiagRecon infrastructure) ──
def score_gaf_random(imgs: np.ndarray, n_iter: int = 3, seed: int = 0) -> np.ndarray:
    """Standard random-masked MAE reconstruction on GAF images.

    Same model, same GAF encoding, but random masking (not diagonal).
    This isolates the effect of masking strategy.
    """
    try:
        from transformers import ViTMAEForPreTraining
    except ImportError:
        return None

    model = get_scorer().model   # reuse already-loaded model

    N = len(imgs)
    all_scores = np.zeros((n_iter, N), dtype=np.float32)

    with torch.no_grad():
        for it in range(n_iter):
            torch.manual_seed(seed + it)
            iter_scores = []
            for start in range(0, N, 16):
                batch_np = imgs[start : start + 16]
                images   = torch.from_numpy(batch_np).to(DEVICE)
                B        = len(images)

                out    = model(pixel_values=images)
                pred   = out.logits                     # [B, 196, p^2*3]
                mask   = out.mask                       # [B, 196]  1=masked
                target = model.patchify(images)
                if model.config.norm_pix_loss:
                    mu  = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mu) / (var + 1e-6).sqrt()
                patch_mse = (pred - target).pow(2).mean(dim=-1)  # [B, 196]
                n_masked  = mask.sum(dim=1).clamp(min=1)
                per_img   = (patch_mse * mask).sum(dim=1) / n_masked
                iter_scores.append(per_img.cpu().numpy())
            all_scores[it] = np.concatenate(iter_scores)

    return all_scores.mean(axis=0)


# ── Baseline: cached line-plot k=5 + EVT ──────────────────────────────────────
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

        # ── baseline ──────────────────────────────────────────────────────
        bl = baseline_f1_from_cache(ds_name, sig, gt_ivs)
        if bl is not None:
            sig_row["baseline_f1"] = bl
            print(f"    baseline (line, LTR k5)    F1={bl:.4f}")

        # ── Render GAF (shared for both rand and diag) ────────────────────
        rand_ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__gaf_rand.pkl")
        diag_ckpt = os.path.join(CKPT_DIR, f"{ds_name}__{sig}__gaf_diag.pkl")

        # Render GAF images (compute once, reuse for both scorers)
        t0   = time.time()
        imgs = render_gaf_windows(values, WINDOW_SIZE, STEP_SIZE)
        print(f"    GAF rendered: {len(imgs)} windows  ({time.time()-t0:.1f}s)")

        # ── gaf_rand: random masking on GAF ───────────────────────────────
        rand_scores_ts = None
        if os.path.exists(rand_ckpt):
            try:
                c = pickle.load(open(rand_ckpt, "rb"))
                rand_scores_ts = c["scores_ts"]
                print(f"    [gaf_rand] cache")
            except Exception:
                pass

        if rand_scores_ts is None:
            try:
                t0         = time.time()
                win_scores = score_gaf_random(imgs)
                rand_scores_ts = window_scores_to_timeseries(
                    win_scores, T, WINDOW_SIZE, STEP_SIZE)
                pickle.dump({"scores_ts": rand_scores_ts}, open(rand_ckpt, "wb"))
                print(f"    [gaf_rand] computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [gaf_rand] ERROR: {e}")

        if rand_scores_ts is not None:
            T_ts = min(len(rand_scores_ts), len(timestamps))
            f1r  = f1_score(evt_detect(rand_scores_ts[:T_ts], timestamps[:T_ts]), gt_ivs)
            sig_row["gaf_rand_f1"] = f1r
            delta = f1r - sig_row.get("baseline_f1", float("nan"))
            print(f"    gaf_rand  (GAF, rand mask)  F1={f1r:.4f}  ({delta:+.4f})")

        # ── gaf_diag: diagonal masking on GAF ─────────────────────────────
        diag_scores_ts = None
        if os.path.exists(diag_ckpt):
            try:
                c = pickle.load(open(diag_ckpt, "rb"))
                diag_scores_ts = c["scores_ts"]
                print(f"    [gaf_diag] cache")
            except Exception:
                pass

        if diag_scores_ts is None:
            try:
                t0         = time.time()
                scorer     = get_scorer()
                win_scores = scorer.score_windows(imgs)
                diag_scores_ts = window_scores_to_timeseries(
                    win_scores, T, WINDOW_SIZE, STEP_SIZE)
                pickle.dump({"scores_ts": diag_scores_ts}, open(diag_ckpt, "wb"))
                print(f"    [gaf_diag] computed ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    [gaf_diag] ERROR: {e}")

        if diag_scores_ts is not None:
            T_ts = min(len(diag_scores_ts), len(timestamps))
            f1d  = f1_score(evt_detect(diag_scores_ts[:T_ts], timestamps[:T_ts]), gt_ivs)
            sig_row["gaf_diag_f1"] = f1d
            delta = f1d - sig_row.get("baseline_f1", float("nan"))
            print(f"    gaf_diag  (GAF, diag mask)  F1={f1d:.4f}  ({delta:+.4f})")

        rows.append(sig_row)


# ── Save & print ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["baseline_f1", "gaf_rand_f1", "gaf_diag_f1"]
LABELS   = {
    "baseline_f1":  "line+LTR",
    "gaf_rand_f1":  "GAF+rand",
    "gaf_diag_f1":  "GAF+diag",
}
COL_W = 12


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

# ── Weighted ALL summary ───────────────────────────────────────────────────────
cols = [c for c in ALL_COLS if c in results_df.columns]
sep  = 26 + COL_W * len(cols)
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

# ── Key signals ────────────────────────────────────────────────────────────────
print(f"\n{'='*sep}")
print("STUCK SIGNALS (all prior methods = 0.0)")
print(f"{'='*sep}")
stuck = ["F-1", "F-3", "T-13", "D-16"]
for _, r in results_df[results_df["signal"].isin(stuck)].iterrows():
    bl = r.get("baseline_f1", float("nan"))
    gd = r.get("gaf_diag_f1", float("nan"))
    d  = gd - bl if not (np.isnan(gd) or np.isnan(bl)) else float("nan")
    print(f"  {r['dataset']}__{r['signal']:<10}  baseline={bl:.4f}  "
          f"GAF+diag={gd:.4f}  delta={d:+.4f}")

print(f"\nResults saved → {out_csv}")
