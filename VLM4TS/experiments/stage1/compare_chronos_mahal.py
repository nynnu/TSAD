"""Chronos backbone — original VLM4TS scoring (for report).

Uses the SAME scoring logic as the original VLM4TS paper (arXiv:2506.06836):
  memory bank → min cosine dissimilarity → top-alpha threshold

Only the backbone changes: CLIP/MAE (196 patches) → Chronos (1 window embedding).
Cached embeddings from results_chronos/embeddings/ are reused.
"""
from __future__ import annotations

import ast, os, pickle, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import genpareto

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

print(f"Project root : {PROJECT_ROOT}")

from evaluation.evaluate import evaluate_intervals
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV   = os.path.join(DATA_DIR, "anomalies.csv")
EMBED_DIR  = os.path.join(PROJECT_ROOT, "results_chronos", "embeddings")

K5_DIR = next((p for p in [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
] if os.path.isdir(p)), "")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_chronos_scoring")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Embed cache : {EMBED_DIR}  ({len(os.listdir(EMBED_DIR)) if os.path.isdir(EMBED_DIR) else 'MISSING'} files)")

ALPHA = 0.01


# ── Original VLM4TS scoring (window-level adaptation) ─────────────────────────
def memory_bank_score(embeds: np.ndarray) -> np.ndarray:
    """
    Original VLM4TS scoring adapted for window-level embeddings.

    score(t) = min_{j != t} (1 - cosine_similarity(e_t, e_j))
             = 1 - max_cosine_similarity to any other window
    """
    E = torch.tensor(embeds, dtype=torch.float32)
    E_norm = F.normalize(E, dim=-1)
    sim = E_norm @ E_norm.T
    sim.fill_diagonal_(-1.0)
    max_sim = sim.max(dim=1).values
    scores = (1.0 - max_sim).numpy()
    return scores.astype(np.float32)


# ── Thresholding ───────────────────────────────────────────────────────────────
def alpha_detect(scores, timestamps, alpha=ALPHA):
    thr = float(np.percentile(scores, (1 - alpha) * 100))
    flags = scores > thr
    if not flags.any(): return []
    ivs, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg: in_seg = True; s_ = i
        elif not f and in_seg: in_seg = False; ivs.append([timestamps[s_], timestamps[i - 1]])
    if in_seg: ivs.append([timestamps[s_], timestamps[-1]])
    return ivs

def evt_detect(scores, timestamps, q=0.90, fpr=0.01):
    if scores.max() - scores.min() < 1e-8: return []
    u = float(np.percentile(scores, q * 100))
    exc = scores[scores > u] - u
    fb = float(np.percentile(scores, (1 - fpr) * 100))
    thr = fb
    if len(exc) >= 10:
        try:
            c, _, s = genpareto.fit(exc, floc=0)
            p_c = min(fpr / max(1 - q, 1e-9), 1 - 1e-9)
            t_ = u + max(0.0, genpareto.ppf(1 - p_c, c, loc=0, scale=s))
            thr = t_ if u <= t_ <= scores.max() else fb
        except: pass
    flags = scores > thr
    if not flags.any(): return []
    ivs, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg: in_seg = True; s_ = i
        elif not f and in_seg: in_seg = False; ivs.append([timestamps[s_], timestamps[i - 1]])
    if in_seg: ivs.append([timestamps[s_], timestamps[-1]])
    return ivs


# ── Eval helpers ──────────────────────────────────────────────────────────────
def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def _eval(d, g): return evaluate_intervals(g, d)["F1"]
def _ivs_alpha(sc, ts, a=0.01):
    idx, _, _ = compute_detection_intervals(score_vector=sc, alpha=a)
    df = intervals_from_indices(idx, ts, sc)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]
def load_pkl(p): return pickle.load(open(p, "rb")) if os.path.exists(p) else None
def save_pkl(o, p): pickle.dump(o, open(p, "wb"))


# ── Datasets ──────────────────────────────────────────────────────────────────
NAB  = ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38","ec2_cpu_utilization_5f5533",
        "ec2_cpu_utilization_77c1ca","ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
        "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de","ec2_disk_write_bytes_c0d644",
        "ec2_network_in_257a54","ec2_network_in_5abac7","elb_request_count_8c0756",
        "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
        "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"]
SMAP = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","F-1","F-2","F-3","P-1","T-1"]
MSL  = ["P-11","T-12","D-15","C-1","F-8","F-7","T-13","D-16","T-8","P-14","D-14"]
DATASETS = [("NAB", NAB), ("SMAP", SMAP), ("MSL", MSL)]
GT = load_gt(ANOM_CSV)

VARIANTS = [
    ("mem_alpha", "memory bank + alpha threshold (original paper style)"),
    ("mem_evt",   "memory bank + EVT threshold"),
]


# ── Main ──────────────────────────────────────────────────────────────────────
records = []

for ds, signals in DATASETS:
    print(f"\n{'='*70}\nDataset: {ds}\n{'='*70}")

    for sig in signals:
        gt_key = sig if ds == "NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig, []))

        k5 = load_pkl(os.path.join(K5_DIR, f"{ds}__{sig}__ltr.pkl"))
        if k5 is None: print(f"  [{sig}] SKIP (no k5 baseline)"); continue
        k5_sc = np.array(k5["scores"]); k5_ts = np.array(k5["timestamps"])
        f1_base = _eval(_ivs_alpha(k5_sc, k5_ts), gt_ivs)

        ep = load_pkl(os.path.join(EMBED_DIR, f"{ds}__{sig}__chronos.pkl"))
        if ep is None:
            print(f"  [{sig}] SKIP — Chronos embed cache missing"); continue

        embeds = np.array(ep["embeddings"])   # (N, 512)
        win_ts = np.array(ep["timestamps"])
        N = len(embeds)
        print(f"\n  [{sig}]  N={N}  MAE_baseline_F1={f1_base:.4f}")

        row = dict(dataset=ds, signal=sig, mae_baseline_f1=f1_base)

        score_ckpt = os.path.join(CKPT_DIR, f"{ds}__{sig}__scores.pkl")
        sc_cached = load_pkl(score_ckpt)
        if sc_cached is not None:
            scores = np.array(sc_cached["scores"])
        else:
            scores = memory_bank_score(embeds)
            save_pkl({"scores": scores, "timestamps": win_ts}, score_ckpt)

        T = min(len(scores), len(win_ts))

        for vname, _ in VARIANTS:
            if vname == "mem_alpha":
                ivs = alpha_detect(scores[:T], win_ts[:T])
            else:
                ivs = evt_detect(scores[:T], win_ts[:T])
            f1 = _eval(ivs, gt_ivs)
            row[vname] = f1
            print(f"    {vname:<14}  F1={f1:.4f}  (Δ={f1 - f1_base:+.4f})")

        records.append(row)


# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
vcols = [v for v, _ in VARIANTS if v in df.columns]

W = 14
print("\n\n" + "="*70)
print("SUMMARY — Chronos backbone (original VLM4TS scoring)")
print("="*70)
hdr = f"{'Dataset':<8}{'MAE_base':>10}" + "".join(f"{c:>{W}}" for c in vcols)
print(hdr); print("-" * len(hdr))
for ds_ in ["NAB", "SMAP", "MSL"]:
    sub = df[df.dataset == ds_]
    print(f"{ds_:<8}{sub.mae_baseline_f1.mean():>10.4f}" +
          "".join(f"{sub[c].mean():>{W}.4f}" for c in vcols if c in sub))
print("-" * len(hdr))
print(f"{'ALL':<8}{df.mae_baseline_f1.mean():>10.4f}" +
      "".join(f"{df[c].mean():>{W}.4f}" for c in vcols if c in df))

print(f"\n[Reference — MAE backbone]")
print(f"MAE + memory bank (original):  ALL≈0.56  NAB=0.67  SMAP=0.38  MSL=0.63")
print(f"MAE + ROWA (our best):         ALL=0.63  NAB=0.74  SMAP=0.53  MSL=0.58")

df.to_csv(os.path.join(RESULTS_DIR, "comparison_chronos.csv"), index=False)
print(f"\nSaved → {os.path.join(RESULTS_DIR, 'comparison_chronos.csv')}")
