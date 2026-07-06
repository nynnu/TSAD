"""Normalizing Flow Density Estimation on MAE Embeddings.
Inspired by VAN-AD (arXiv:2603.26842).

LTR 완전 대체: temporal neighbor 없이 신호 전체 분포를 학습.

LTR:  score(e_t) = distance to k temporal neighbors      ← local, biased
NF:   score(e_t) = -log p_θ(e_t)                        ← global, exact density

장점:
  - Circular reference 없음 (긴 이상치도 탐지)
  - Multi-modal 정상 분포 포착 (Mahalanobis는 Gaussian 가정)
  - Exact likelihood, theoretically principled

캐시 재활용: results_lp_resid/embeddings/ (재인코딩 불필요, 수분 완료)
"""
from __future__ import annotations

import ast, os, pickle, subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.stats import genpareto

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

print(f"Project root : {PROJECT_ROOT}")

subprocess.run(
    ["pip", "install", "zuko", "scipy", "--quiet"], check=True,
)

import torch
import torch.nn as nn
import zuko

from evaluation.evaluate import evaluate_intervals
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV   = os.path.join(DATA_DIR, "anomalies.csv")

EMBED_DIR  = os.path.join(PROJECT_ROOT, "results_lp_resid", "embeddings")
K5_DIR     = next((p for p in [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
] if os.path.isdir(p)), "")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_nf_density")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Embed cache : {EMBED_DIR}  ({len(os.listdir(EMBED_DIR)) if os.path.isdir(EMBED_DIR) else 'MISSING'} files)")

EVT_Q_INIT = 0.90
EVT_FPR    = 0.01

# NF variants: (pca_dim, n_transforms, hidden, epochs, flow_type)
# flow_type: "nsf" = Neural Spline Flow, "maf" = Masked Autoregressive Flow (VAN-AD style)
NF_VARIANTS = [
    (32,  4, [64, 64],  300, "nsf"),   # nf_d32_t4   — default (expressive)
    (16,  4, [64, 64],  300, "nsf"),   # nf_d16_t4   — smaller
    (32,  3, [64, 64],   10, "maf"),   # maf_d32_t3  — VAN-AD style (5-10 epochs, fast)
    (64,  3, [64, 64],   10, "maf"),   # maf_d64_t3  — VAN-AD larger
]


# ── NF Model ──────────────────────────────────────────────────────────────────
def make_flow(dim: int, n_transforms: int, hidden: list,
              flow_type: str = "nsf"):
    """NSF: Neural Spline Flow (Durkan NeurIPS 2019).
       MAF: Masked Autoregressive Flow (VAN-AD style, very fast)."""
    if flow_type == "maf":
        return zuko.flows.MAF(
            features=dim, context=0,
            transforms=n_transforms, hidden_features=hidden,
        ).to(DEVICE)
    return zuko.flows.NSF(
        features=dim, context=0,
        transforms=n_transforms, hidden_features=hidden,
        randperm=True,
    ).to(DEVICE)


def fit_flow(flow, X: np.ndarray, epochs: int, lr: float = 1e-3,
             batch_size: int = 64, patience: int = 30) -> float:
    """Train NF on embeddings X. Returns final loss."""
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss, best_state, patience_cnt = float("inf"), None, 0
    N = len(X_t)

    for ep in range(epochs):
        idx = torch.randperm(N, device=DEVICE)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            batch = X_t[idx[i:i + batch_size]]
            loss = -flow().log_prob(batch).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= N
        scheduler.step()

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        flow.load_state_dict(best_state)
    return best_loss


def score_flow(flow, X: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        s = -flow().log_prob(X_t).cpu().numpy()
    return s.astype(np.float32)


# ── PCA ───────────────────────────────────────────────────────────────────────
def pca_fit_transform(X: np.ndarray, n_components: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_p = pca.fit_transform(X_s)
    return X_p, scaler, pca

def pca_transform(X, scaler, pca):
    return pca.transform(scaler.transform(X))


# ── EVT & Eval ────────────────────────────────────────────────────────────────
def evt_threshold(scores, q=EVT_Q_INIT, fpr=EVT_FPR):
    u = float(np.percentile(scores, q * 100))
    exc = scores[scores > u] - u
    fallback = float(np.percentile(scores, (1 - fpr) * 100))
    if len(exc) < 10: return fallback
    try:
        c, _, sc = genpareto.fit(exc, floc=0)
        p_cond = min(fpr / max(1 - q, 1e-9), 1 - 1e-9)
        thr = u + max(0.0, genpareto.ppf(1 - p_cond, c, loc=0, scale=sc))
        return thr if u <= thr <= scores.max() else fallback
    except: return fallback

def evt_detect(scores, timestamps):
    if scores.max() - scores.min() < 1e-8: return []
    flags = scores > evt_threshold(scores)
    if not flags.any(): return []
    intervals, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg: in_seg = True; s = i
        elif not f and in_seg: in_seg = False; intervals.append([timestamps[s], timestamps[i-1]])
    if in_seg: intervals.append([timestamps[s], timestamps[-1]])
    return intervals

def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try: gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def _eval(det, gt): return evaluate_intervals(gt, det)["F1"]
def _ivs_alpha(sc, ts, a=0.01):
    idx, _, _ = compute_detection_intervals(score_vector=sc, alpha=a)
    df = intervals_from_indices(idx, ts, sc)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]
def load_pkl(p): return pickle.load(open(p,"rb")) if os.path.exists(p) else None
def save_pkl(o, p): pickle.dump(o, open(p,"wb"))


# ── Datasets ──────────────────────────────────────────────────────────────────
NAB   = ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38","ec2_cpu_utilization_5f5533",
         "ec2_cpu_utilization_77c1ca","ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
         "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de","ec2_disk_write_bytes_c0d644",
         "ec2_network_in_257a54","ec2_network_in_5abac7","elb_request_count_8c0756",
         "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
         "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"]
SMAP  = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","F-1","F-2","F-3","P-1","T-1"]
MSL   = ["P-11","T-12","D-15","C-1","F-8","F-7","T-13","D-16","T-8","P-14","D-14"]
DATASETS = [("NAB", NAB), ("SMAP", SMAP), ("MSL", MSL)]
GT = load_gt(ANOM_CSV)


# ── Main ──────────────────────────────────────────────────────────────────────
records = []

for ds, signals in DATASETS:
    print(f"\n{'='*72}\nDataset: {ds}\n{'='*72}")

    for sig in signals:
        gt_key = sig if ds == "NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig, []))

        # baseline
        k5 = load_pkl(os.path.join(K5_DIR, f"{ds}__{sig}__ltr.pkl"))
        if k5 is None: print(f"  [{sig}] SKIP"); continue
        k5_sc = np.array(k5["scores"]); k5_ts = np.array(k5["timestamps"])
        f1_base = _eval(_ivs_alpha(k5_sc, k5_ts), gt_ivs)

        # load cached embeddings
        ep = load_pkl(os.path.join(EMBED_DIR, f"{ds}__{sig}__embeds.pkl"))
        if ep is None:
            print(f"  [{sig}] SKIP — embed cache missing (run compare_lp_residual.py first)")
            continue
        embeds = ep["embeddings"]   # (N, 768)
        win_ts = ep["timestamps"]   # (N,)

        print(f"\n  [{sig}]  N={len(embeds)}  baseline F1={f1_base:.4f}")

        row = dict(dataset=ds, signal=sig, baseline_f1=f1_base)

        for pca_d, n_tr, hid, epochs, ftype in NF_VARIANTS:
            vname = f"{'maf' if ftype=='maf' else 'nsf'}_d{pca_d}_t{n_tr}"
            ckpt  = os.path.join(CKPT_DIR, f"{ds}__{sig}__{vname}.pkl")
            cached = load_pkl(ckpt)

            if cached is not None:
                nf_scores = np.array(cached["scores"])
                print(f"    {vname:<18} [cache]", end="")
            else:
                # PCA
                X_pca, scaler, pca_model = pca_fit_transform(embeds, pca_d)
                # Flow
                lr_flow = 0.005 if ftype == "maf" else 1e-3   # VAN-AD uses 0.005
                flow = make_flow(pca_d, n_tr, hid, ftype)
                t0 = time.time()
                final_loss = fit_flow(flow, X_pca, epochs, lr=lr_flow)
                nf_scores  = score_flow(flow, X_pca)
                elapsed = time.time() - t0
                save_pkl({"scores": nf_scores, "timestamps": win_ts,
                          "pca_dim": pca_d, "n_transforms": n_tr}, ckpt)
                print(f"    {vname:<18} loss={final_loss:.3f}  {elapsed:.1f}s", end="")

            T   = min(len(nf_scores), len(win_ts))
            ivs = evt_detect(nf_scores[:T], win_ts[:T])
            f1  = _eval(ivs, gt_ivs)
            row[vname] = f1
            print(f"  F1={f1:.4f}  (Δ={f1-f1_base:+.4f})")

        records.append(row)


# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
vcols = [f"{'maf' if ft=='maf' else 'nsf'}_d{d}_t{t}" for d,t,_,_,ft in NF_VARIANTS]
vcols = [c for c in vcols if c in df.columns]

W = 14
print("\n\n" + "="*80)
print("CROSS-DATASET SUMMARY — Normalizing Flow vs Baseline")
print("="*80)
hdr = f"{'Dataset':<8}{'baseline':>10}" + "".join(f"{c:>{W}}" for c in vcols)
print(hdr); print("-"*len(hdr))
for ds_ in ["NAB","SMAP","MSL"]:
    sub = df[df.dataset==ds_]
    print(f"{ds_:<8}{sub.baseline_f1.mean():>10.4f}" +
          "".join(f"{sub[c].mean():>{W}.4f}" for c in vcols))
print("-"*len(hdr))
print(f"{'ALL':<8}{df.baseline_f1.mean():>10.4f}" +
      "".join(f"{df[c].mean():>{W}.4f}" for c in vcols))

# best NF variant
best_col = max(vcols, key=lambda c: df[c].mean())
print(f"\nBest NF variant: {best_col}  ALL={df[best_col].mean():.4f}")
print(f"Baseline:                      ALL={df.baseline_f1.mean():.4f}")
print(f"Mahalanobis (prev):            ALL≈0.5750  SMAP=0.7641")

out = os.path.join(RESULTS_DIR, "comparison.csv")
df.to_csv(out, index=False)
print(f"\nSaved → {out}")
