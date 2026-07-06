"""Adaptive Fusion experiment: ADF-based scorer selection.

Hypothesis
----------
SMAP signals are non-stationary (slow drifts, regime shifts) → Mahalanobis on
global embedding distribution is the best scorer.
NAB/MSL signals are stationary (spikes relative to stable baseline) → LTR
temporal reference (add k5+k30) is the best scorer.

An Augmented Dickey-Fuller (ADF) test on the raw signal automatically detects
stationarity — no manual per-dataset tuning, satisfies the Universality Principle.

Methods compared
----------------
  baseline      : LTR k=5 (from existing checkpoints)
  add(k5,k30)   : additive fusion (from existing checkpoints)
  Mahal(raw)    : Mahalanobis scorer (from existing checkpoints)
  adaptive_adf  : ADF test → if non-stationary use Mahal, else use add(k5,k30)
  adaptive_max  : per-signal max(mahal, add) — oracle-free soft upper bound

All checkpoints are REUSED — no new model inference needed.
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
    ["pip", "install", "statsmodels", "scipy", "--quiet"],
    check=True,
)

from statsmodels.tsa.stattools import adfuller

from evaluation.evaluate import evaluate_intervals
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

# ── Checkpoint dirs ───────────────────────────────────────────────────────────
def _first_existing(*candidates):
    return next((p for p in candidates if os.path.isdir(p)), candidates[-1])

K5_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
)
K30_CKPT_DIR = _first_existing(
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_ltr_multiscale", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_ltr_multiscale", "checkpoints"),
)
MAHAL_CKPT_DIR = os.path.join(PROJECT_ROOT, "results_stl_mahalanobis", "checkpoints")
LOCAL_CKPT_DIR = os.path.join(PROJECT_ROOT, "results_fixed_preproc", "checkpoints")

print(f"k5    ckpts : {K5_CKPT_DIR}  ({len(os.listdir(K5_CKPT_DIR)) if os.path.isdir(K5_CKPT_DIR) else 'MISSING'} files)")
print(f"k30   ckpts : {K30_CKPT_DIR} ({len(os.listdir(K30_CKPT_DIR)) if os.path.isdir(K30_CKPT_DIR) else 'MISSING'} files)")
print(f"mahal ckpts : {MAHAL_CKPT_DIR} ({len(os.listdir(MAHAL_CKPT_DIR)) if os.path.isdir(MAHAL_CKPT_DIR) else 'MISSING'} files)")
print(f"local ckpts : {LOCAL_CKPT_DIR} ({len(os.listdir(LOCAL_CKPT_DIR)) if os.path.isdir(LOCAL_CKPT_DIR) else 'MISSING'} files)")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_adaptive_fusion")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV = os.path.join(DATA_DIR, "anomalies.csv")

EVT_Q_INIT = 0.90
EVT_FPR    = 0.01
ADF_PVAL_THR = 0.05  # p > 0.05 → non-stationary → use Mahal


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def add_fuse(*arrays: np.ndarray) -> np.ndarray:
    T = min(len(a) for a in arrays)
    return sum(normalize_01(a[:T]) for a in arrays)


def max_fuse(*arrays: np.ndarray) -> np.ndarray:
    T = min(len(a) for a in arrays)
    return np.maximum.reduce([normalize_01(a[:T]) for a in arrays])


def evt_threshold(scores: np.ndarray, q_init: float = EVT_Q_INIT,
                  target_fpr: float = EVT_FPR) -> float:
    u = float(np.percentile(scores, q_init * 100.0))
    exceedances = scores[scores > u] - u
    fallback = float(np.percentile(scores, (1.0 - target_fpr) * 100.0))
    if len(exceedances) < 10:
        return fallback
    try:
        c, _, scale = genpareto.fit(exceedances, floc=0)
        p_cond = min(target_fpr / max(1.0 - q_init, 1e-9), 1.0 - 1e-9)
        excess = genpareto.ppf(1.0 - p_cond, c, loc=0, scale=scale)
        thr = u + max(0.0, excess)
        return thr if u <= thr <= scores.max() else fallback
    except Exception:
        return fallback


def evt_detect(scores: np.ndarray, timestamps: np.ndarray) -> list:
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
        intervals.append([timestamps[seg_start], timestamps[len(flags) - 1]])
    return intervals


def _ivs_alpha(scores, timestamps, alpha=0.01) -> list:
    idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
    df = intervals_from_indices(idx, timestamps, scores)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]


def _eval(detected: list, gt_ivs: list) -> float:
    return evaluate_intervals(gt_ivs, detected)["F1"]


def load_pkl(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def is_nonstationary(values: np.ndarray, pval_thr: float = ADF_PVAL_THR) -> bool:
    """ADF test: returns True if signal is non-stationary (unit root present)."""
    try:
        _, pval, *_ = adfuller(values.astype(float), autolag="AIC")
        return pval > pval_thr
    except Exception:
        return False  # default to stationary if test fails


# ── Signal loading ────────────────────────────────────────────────────────────

def load_signal_values(ds: str, sig: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load raw (values, timestamps) for a signal."""
    if ds == "NAB":
        csv_path = os.path.join(DATA_DIR, "realAWSCloudwatch", f"{sig}.csv")
    elif ds == "SMAP":
        csv_path = os.path.join(DATA_DIR, "SMAP", f"{sig}.csv")
    else:
        csv_path = os.path.join(DATA_DIR, "MSL", f"{sig}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    ts_col = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    val_col = [c for c in df.columns if c not in ts_col]
    if not ts_col or not val_col:
        return None
    timestamps = pd.to_datetime(df[ts_col[0]]).view(np.int64).to_numpy()
    values = df[val_col[0]].to_numpy(dtype=float)
    return values, timestamps


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
SMAP_SIGNALS = ["D-1", "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7",
                "F-1", "F-2", "F-3", "P-1", "T-1"]
MSL_SIGNALS  = ["P-11", "T-12", "D-15", "C-1", "F-8", "F-7", "T-13",
                "D-16", "T-8", "P-14", "D-14"]

DATASETS = [("NAB", NAB_SIGNALS), ("SMAP", SMAP_SIGNALS), ("MSL", MSL_SIGNALS)]

GT = load_gt(ANOM_CSV)

# ── Main loop ─────────────────────────────────────────────────────────────────

records = []
adf_decisions = {}  # track what ADF decided per signal

for ds, signals in DATASETS:
    print(f"\n{'='*72}")
    print(f"Dataset: {ds}  ({len(signals)} signals)")
    print(f"{'='*72}")

    for sig in signals:
        gt_key = sig if ds == "NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig, []))

        # ── Load k5 scores ──
        k5_pkl = load_pkl(os.path.join(K5_CKPT_DIR, f"{ds}__{sig}__ltr.pkl"))
        if k5_pkl is None:
            print(f"  [{sig}] SKIP — k5 checkpoint missing")
            continue
        k5_scores = np.array(k5_pkl["scores"], dtype=float)
        k5_ts     = np.array(k5_pkl["timestamps"])

        # ── Load k30 scores ──
        k30_pkl = load_pkl(os.path.join(K30_CKPT_DIR, f"{ds}__{sig}__ltr_k30.pkl"))
        k30_scores = np.array(k30_pkl["scores"], dtype=float) if k30_pkl else None

        # ── Load Mahal scores ──
        mahal_pkl = load_pkl(os.path.join(MAHAL_CKPT_DIR, f"{ds}__{sig}__mahal.pkl"))
        mahal_scores = np.array(mahal_pkl["scores"], dtype=float) if mahal_pkl else None

        # ── Load local_norm k30 scores ──
        local_pkl = load_pkl(os.path.join(LOCAL_CKPT_DIR, f"{ds}__{sig}__k30_fixed.pkl"))
        local_scores = np.array(local_pkl["scores"], dtype=float) if local_pkl else None

        # ── ADF stationarity test on raw signal ──
        raw = load_signal_values(ds, sig)
        if raw is not None:
            values, _ = raw
            nonstationary = is_nonstationary(values)
        else:
            nonstationary = False
        adf_decisions[f"{ds}__{sig}"] = "mahal" if nonstationary else "ltr_add"

        # ── add(k5,k30) ──
        if k30_scores is not None:
            add_scores = add_fuse(k5_scores, k30_scores)
        else:
            add_scores = normalize_01(k5_scores)
        add_ivs = evt_detect(add_scores, k5_ts[:len(add_scores)])

        # ── Mahal ──
        mahal_ivs = None
        if mahal_scores is not None:
            mahal_ivs = evt_detect(mahal_scores, k5_ts[:len(mahal_scores)])

        # ── local_norm k30 ──
        local_ivs = None
        if local_scores is not None:
            local_ivs = evt_detect(local_scores, k5_ts[:len(local_scores)])

        # ── adaptive_adf: ADF-based selection ──
        if nonstationary and mahal_scores is not None:
            adap_adf_scores = mahal_scores
        elif k30_scores is not None:
            adap_adf_scores = add_scores
        else:
            adap_adf_scores = k5_scores
        adap_adf_ivs = evt_detect(adap_adf_scores, k5_ts[:len(adap_adf_scores)])

        # ── adaptive_adf_local: non-stationary → local_norm k30, else → add ──
        if nonstationary and local_scores is not None:
            adap_local_scores = local_scores
        elif k30_scores is not None:
            adap_local_scores = add_scores
        else:
            adap_local_scores = k5_scores
        adap_local_ivs = evt_detect(adap_local_scores, k5_ts[:len(adap_local_scores)])

        # ── adaptive_max: max(mahal, add) — tests if soft fusion helps ──
        if mahal_scores is not None and k30_scores is not None:
            adap_max_scores = max_fuse(mahal_scores, add_scores)
            adap_max_ivs = evt_detect(adap_max_scores, k5_ts[:len(adap_max_scores)])
        else:
            adap_max_scores = add_scores
            adap_max_ivs = add_ivs

        # ── Evaluate ──
        f1_base   = _eval(_ivs_alpha(k5_scores, k5_ts), gt_ivs)
        f1_add    = _eval(add_ivs, gt_ivs)
        f1_mahal  = _eval(mahal_ivs, gt_ivs) if mahal_ivs is not None else float("nan")
        f1_local  = _eval(local_ivs, gt_ivs) if local_ivs is not None else float("nan")
        f1_adap   = _eval(adap_adf_ivs, gt_ivs)
        f1_adap_l = _eval(adap_local_ivs, gt_ivs)
        f1_max    = _eval(adap_max_ivs, gt_ivs)

        decision = "mahal" if nonstationary else "add"
        print(f"\n  [{sig}]  ADF→{'NON-STAT→mahal' if nonstationary else 'STAT→add'}")
        print(f"    baseline        F1={f1_base:.4f}")
        print(f"    add(k5,k30)     F1={f1_add:.4f}")
        print(f"    Mahal(raw)      F1={f1_mahal:.4f}")
        print(f"    local_k30       F1={f1_local:.4f}")
        print(f"    adaptive_adf    F1={f1_adap:.4f}  ← ADF picks {decision}")
        print(f"    adaptive_local  F1={f1_adap_l:.4f}  ← ADF picks {'local' if nonstationary else 'add'}")
        print(f"    adaptive_max    F1={f1_max:.4f}  ← max(mahal,add)")

        records.append(dict(
            dataset=ds, signal=sig,
            adf_nonstationary=nonstationary,
            baseline_f1=f1_base,
            add_f1=f1_add,
            mahal_f1=f1_mahal,
            local_k30_f1=f1_local,
            adaptive_adf_f1=f1_adap,
            adaptive_local_f1=f1_adap_l,
            adaptive_max_f1=f1_max,
        ))

# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

print("\n\n" + "=" * 90)
print("ADF DECISIONS (non-stationary → mahal, stationary → add)")
print("=" * 90)
for ds in ["NAB", "SMAP", "MSL"]:
    sub = df[df.dataset == ds]
    nonstat = sub.adf_nonstationary.sum()
    print(f"  {ds}: {nonstat}/{len(sub)} non-stationary")
    for _, row in sub.iterrows():
        tag = "NON-STAT" if row.adf_nonstationary else "STAT    "
        print(f"    {tag}  {row.signal}")

cols = ["baseline_f1", "add_f1", "mahal_f1", "local_k30_f1",
        "adaptive_adf_f1", "adaptive_local_f1", "adaptive_max_f1"]
labels = ["baseline", "add(k5,k30)", "Mahal(raw)", "local_k30",
          "adaptive_adf", "adaptive_local", "adaptive_max"]

print("\n\n" + "=" * 90)
print("CROSS-DATASET SUMMARY")
print("=" * 90)
header = f"{'Dataset':<8}" + "".join(f"{l:>16}" for l in labels)
print(header)
print("-" * len(header))
avgs = {}
for ds in ["NAB", "SMAP", "MSL"]:
    sub = df[df.dataset == ds]
    row_str = f"{ds:<8}"
    for c in cols:
        v = sub[c].mean()
        row_str += f"{v:>16.4f}"
    print(row_str)
    avgs[ds] = {c: sub[c].mean() for c in cols}

# ALL = unweighted average across all 40 signals
all_avg = {c: df[c].mean() for c in cols}
row_str = f"{'ALL':<8}" + "".join(f"{all_avg[c]:>16.4f}" for c in cols)
print("-" * len(header))
print(row_str)

print("\n\n" + "=" * 90)
print("STUCK SIGNALS (baseline F1=0)")
print("=" * 90)
stuck = df[df.baseline_f1 == 0.0]
hdr = f"{'Signal':<24}" + "".join(f"{l:>16}" for l in labels)
print(hdr)
print("-" * len(hdr))
for _, row in stuck.iterrows():
    tag = f"{row.dataset}_{row.signal}"
    row_str = f"{tag:<24}" + "".join(f"{row[c]:>16.4f}" for c in cols)
    print(row_str)

print(f"\n\nKey metrics (ALL F1):")
for label, col in zip(labels, cols):
    print(f"  {label:<20}: {all_avg[col]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_csv = os.path.join(RESULTS_DIR, "comparison.csv")
df.to_csv(out_csv, index=False)
print(f"\nResults saved → {out_csv}")
