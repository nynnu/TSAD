"""
Patch Scoring Variants: min-cosine vs mean-cosine vs min-euclidean
==================================================================

Hypothesis (Proposal 3)
-----------------------
The current LTR scoring uses nearest-neighbor cosine dissimilarity (min over
reference patches). Two alternative aggregations may be more discriminative:

  min_cos  : 0.5 * min(1 - cosine_sim)     ← current baseline (LTR k=5)
             Nearest-neighbor match; forgiving if one patch coincidentally aligns.

  mean_cos : 0.5 * mean(1 - cosine_sim)    ← H1: mean aggregation
             All patches must deviate from reference to score high.
             Expected to reduce false negatives from accidental patch matches.

  min_euc  : min(euclidean_dist)            ← H2: euclidean distance
             Unlike cosine, captures embedding magnitude changes.
             Hypothesis: MAE encoder magnitude encodes anomaly severity.

Morphological motivation (SMAP F-1, F-3)
-----------------------------------------
  F-1 (len=11453, anom_len=101): anom has same mean/std as normal — only
       subtle pattern shift. Cosine direction may be identical; euclidean
       magnitude difference might be the only signal.
  F-3 (len=11256, anom_len=41): stuck sensor (std=0.03 vs normal 0.58).
       All patches in the anomaly window are similar → mean cosine dissim to
       reference may differ from min because the reference has diverse patches.

Ablation table
--------------
  ltr_k5_min   : current best per-signal (reuse existing checkpoints)
  ltr_k5_mean  : mean cosine aggregation (new checkpoint)
  ltr_k5_euc   : euclidean distance (new checkpoint)
  add_k5_mean  : add(ltr_k5_min, ltr_k5_mean) + EVT  — do the two signals complement?
  add_mean_euc : add(ltr_k5_mean, ltr_k5_euc) + EVT

All scoring variants use:
  - same MAE backbone (vit_base_patch16_224.mae)
  - same k=5 local reference window
  - same EVT thresholding

Checkpoints
-----------
  ltr_k5_min  (reuse) : results_mgmr/checkpoints/{DS}__{sig}__ltr.pkl
  ltr_k5_mean  (new)  : results_patch_scoring/checkpoints/{DS}__{sig}__ltr_mean.pkl
  ltr_k5_euc   (new)  : results_patch_scoring/checkpoints/{DS}__{sig}__ltr_euc.pkl

Usage
-----
  python experiments/compare_patch_scoring.py
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
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import harmonic_aggregation, stitch_anomaly_maps
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
    compute_dissimilarity_mean,
    compute_dissimilarity_euclidean,
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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_patch_scoring")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

_K5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
]
K5_CKPT_DIR = next((p for p in _K5_CANDIDATES if os.path.isdir(p)),
                   _K5_CANDIDATES[-1])
print(f"LTR k=5 min ckpt : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR))} files)")

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224
STEP_RATIO  = 4.0
K_LOCAL     = 5
MIN_REF     = 5
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


def add_fuse(*arrays: np.ndarray) -> np.ndarray:
    T = min(len(a) for a in arrays)
    return sum(normalize_01(a[:T]) for a in arrays)


# ── Parametric LTR detector (scoring_fn controls aggregation) ────────────────

class PatchScoringDetector(ViT4TS_MAE):
    """LTR k=5 with configurable patch aggregation function."""

    def __init__(self, *args, scoring: str = "min", **kwargs):
        super().__init__(*args, **kwargs)
        assert scoring in ("min", "mean", "euc"), f"Unknown scoring: {scoring}"
        self.scoring = scoring

    def _dissim(self, token, ref):
        if self.scoring == "min":
            return compute_dissimilarity_with_ref(token, ref)
        elif self.scoring == "mean":
            return compute_dissimilarity_mean(token, ref)
        else:
            return compute_dissimilarity_euclidean(token, ref)

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
        h  = self.image_size[0]
        ph = pw = h // self.patch_size

        anomaly_maps = []
        with torch.no_grad():
            for i in range(L):
                l_ref, _ = get_local_reference(large_embeds,  i, K_LOCAL, MIN_REF)
                m_ref, _ = get_local_reference(mid_embeds,    i, K_LOCAL, MIN_REF)
                p_ref, _ = get_local_reference(patch_embeds,  i, K_LOCAL, MIN_REF)

                m_l = self._dissim(large_embeds[i].unsqueeze(0).to(self.device),
                                   l_ref.to(self.device))
                m_m = self._dissim(mid_embeds[i].unsqueeze(0).to(self.device),
                                   m_ref.to(self.device))
                m_p = self._dissim(patch_embeds[i].unsqueeze(0).to(self.device),
                                   p_ref.to(self.device))

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
det_min  = PatchScoringDetector(**BASE_PARAMS, scoring="min")
det_mean = PatchScoringDetector(**BASE_PARAMS, scoring="mean")
det_euc  = PatchScoringDetector(**BASE_PARAMS, scoring="euc")
# Share backbone weights
det_mean.model = det_min.model
det_euc.model  = det_min.model
print("  min_cos  : nearest-neighbor cosine dissimilarity (LTR baseline)")
print("  mean_cos : mean cosine dissimilarity over reference patches")
print("  min_euc  : nearest-neighbor Euclidean distance")

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

        k5_ckpt   = os.path.join(K5_CKPT_DIR, f"{ds_name}__{sig}__ltr.pkl")
        mean_ckpt = os.path.join(CKPT_DIR,     f"{ds_name}__{sig}__ltr_mean.pkl")
        euc_ckpt  = os.path.join(CKPT_DIR,     f"{ds_name}__{sig}__ltr_euc.pkl")

        # ── s_min (reuse existing LTR k=5) ───────────────────────
        t0 = time.time()
        try:
            if os.path.exists(k5_ckpt):
                c          = pickle.load(open(k5_ckpt, "rb"))
                s_min      = c["scores"]
                timestamps = c["timestamps"]
                print(f"    [min ] cache  ({time.time()-t0:.1f}s)")
            else:
                s_min, timestamps = det_min.predict_scores(data)
                pickle.dump({"scores": s_min, "timestamps": timestamps},
                            open(k5_ckpt, "wb"))
                print(f"    [min ] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [min ] ERROR: {e}"); continue

        # ── s_mean ────────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(mean_ckpt):
                s_mean = pickle.load(open(mean_ckpt, "rb"))["scores"]
                print(f"    [mean] cache  ({time.time()-t0:.1f}s)")
            else:
                s_mean, _ = det_mean.predict_scores(data)
                pickle.dump({"scores": s_mean}, open(mean_ckpt, "wb"))
                print(f"    [mean] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [mean] ERROR: {e}"); s_mean = None

        # ── s_euc ─────────────────────────────────────────────────
        t0 = time.time()
        try:
            if os.path.exists(euc_ckpt):
                s_euc = pickle.load(open(euc_ckpt, "rb"))["scores"]
                print(f"    [euc ] cache  ({time.time()-t0:.1f}s)")
            else:
                s_euc, _ = det_euc.predict_scores(data)
                pickle.dump({"scores": s_euc}, open(euc_ckpt, "wb"))
                print(f"    [euc ] computed ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [euc ] ERROR: {e}"); s_euc = None

        T       = len(s_min)
        ts_trim = timestamps[:T]

        # ── baseline: min_cos + alpha=0.01 ────────────────────────
        try:
            ivs = _ivs_alpha(s_min, timestamps)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["min_alpha_f1"] = f1
        print(f"    min_alpha     F1={f1:.4f}  ivs={len(ivs)}")

        # ── min_cos + EVT ─────────────────────────────────────────
        try:
            ivs = evt_detect(s_min[:T], ts_trim)
            f1  = _eval(ivs, gt_ivs)
        except Exception:
            f1 = 0.0
        sig_row["min_evt_f1"] = f1
        print(f"    min_evt       F1={f1:.4f}  ivs={len(ivs)}")

        # ── mean_cos + EVT ────────────────────────────────────────
        if s_mean is not None:
            try:
                T2  = min(T, len(s_mean))
                ivs = evt_detect(s_mean[:T2], ts_trim[:T2])
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["mean_evt_f1"] = f1
            delta = f1 - sig_row["min_alpha_f1"]
            print(f"    mean_evt      F1={f1:.4f}  ivs={len(ivs)}  ({delta:+.4f} vs base)")

        # ── min_euc + EVT ─────────────────────────────────────────
        if s_euc is not None:
            try:
                T2  = min(T, len(s_euc))
                ivs = evt_detect(s_euc[:T2], ts_trim[:T2])
                f1  = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["euc_evt_f1"] = f1
            delta = f1 - sig_row["min_alpha_f1"]
            print(f"    euc_evt       F1={f1:.4f}  ivs={len(ivs)}  ({delta:+.4f} vs base)")

        # ── add(min, mean) + EVT ──────────────────────────────────
        if s_mean is not None:
            try:
                s_add = add_fuse(s_min, s_mean)
                ivs   = evt_detect(s_add, ts_trim[:len(s_add)])
                f1    = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["add_min_mean_f1"] = f1
            print(f"    add(min,mean) F1={f1:.4f}  ivs={len(ivs)}")

        # ── add(mean, euc) + EVT ──────────────────────────────────
        if s_mean is not None and s_euc is not None:
            try:
                s_add = add_fuse(s_mean, s_euc)
                ivs   = evt_detect(s_add, ts_trim[:len(s_add)])
                f1    = _eval(ivs, gt_ivs)
            except Exception:
                f1 = 0.0
            sig_row["add_mean_euc_f1"] = f1
            print(f"    add(mean,euc) F1={f1:.4f}  ivs={len(ivs)}")

        rows.append(sig_row)


# ── Results table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
out_csv    = os.path.join(RESULTS_DIR, "comparison.csv")
results_df.to_csv(out_csv, index=False)

ALL_COLS = ["min_alpha_f1", "min_evt_f1", "mean_evt_f1",
            "euc_evt_f1", "add_min_mean_f1", "add_mean_euc_f1"]
LABELS = {
    "min_alpha_f1":   "min+alpha",
    "min_evt_f1":     "min+EVT",
    "mean_evt_f1":    "mean+EVT",
    "euc_evt_f1":     "euc+EVT",
    "add_min_mean_f1": "add(min,mean)",
    "add_mean_euc_f1": "add(mean,euc)",
}
COL_W = 14
SEP   = 36 + COL_W * len(ALL_COLS)


def _fmt(v: object) -> str:
    try:
        f = float(v)
        return "   NaN" if f != f else f"{f:.4f}"
    except Exception:
        return "   NaN"


def _print_table(ds_name: str, df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == ds_name]
    if sub.empty:
        return
    print(f"\n{'='*SEP}")
    print(f"=== {ds_name} — Patch Scoring Variants ===")
    print(f"{'='*SEP}")
    valid = [c for c in ALL_COLS if c in df.columns]
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
print(f"\n{'='*SEP}")
print("=== CROSS-DATASET SUMMARY ===")
print(f"{'='*SEP}")
valid_cols = [c for c in ALL_COLS if c in results_df.columns]
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
print("KEY FINDINGS")
print(f"{'='*SEP}")

base = results_df["min_alpha_f1"].dropna().mean()
print(f"baseline (min_cos k=5, alpha=0.01) : {base:.4f}")
for c, lbl in [("min_evt_f1",     "min_cos + EVT            "),
               ("mean_evt_f1",    "mean_cos + EVT           "),
               ("euc_evt_f1",     "min_euc + EVT            "),
               ("add_min_mean_f1","add(min,mean) + EVT      "),
               ("add_mean_euc_f1","add(mean,euc) + EVT      ")]:
    if c in results_df.columns:
        v = results_df[c].dropna().mean()
        print(f"{lbl}: {v:.4f}  ({v-base:+.4f})")

# Spotlight: SMAP F-series (target stuck signals)
print("\nSMAP F-series  (target stuck/subtle signals):")
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

print(f"\nResults saved → {out_csv}")
