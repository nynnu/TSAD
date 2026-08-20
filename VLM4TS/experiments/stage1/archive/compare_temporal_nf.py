"""Temporal Conditional Normalizing Flow.
Inspired by arXiv:2603.09490 (Temporal-Conditioned NF for MVTS).

LP residual의 확률론적 일반화:
  LP:          ê_t = Ridge(e_{t-p:t-1})           deterministic, Gaussian noise assumed
  Temporal NF: p(e_t | e_{t-p:t-1}) = NF(context)  probabilistic, any distribution

점수: score_t = -log p(e_t | h_t)
         h_t = GRU([e_{t-p}, ..., e_{t-1}])   ← temporal context

장점:
  - DINOv2+LP처럼 temporal prediction 기반 → 하지만 MAE로
  - Ridge regression보다 비선형 조건부 밀도 모델링
  - LP residual이 실패한 신호도 확률론적으로 커버 가능

캐시 재활용: results_lp_resid/embeddings/ (재인코딩 불필요)
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
subprocess.run(["pip", "install", "zuko", "scipy", "--quiet"], check=True)

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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_temporal_nf")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

EVT_Q_INIT = 0.90
EVT_FPR    = 0.01
PCA_DIM    = 32

# Variants: (ar_order, gru_hidden, n_transforms, epochs)
VARIANTS = [
    (5,  32, 3, 200),   # tnf_p5_h32    — small, fast
    (10, 32, 3, 200),   # tnf_p10_h32   — medium context
    (5,  64, 4, 300),   # tnf_p5_h64    — wider
    (10, 64, 4, 300),   # tnf_p10_h64   — wider + longer context
]


# ── Temporal NF Model ─────────────────────────────────────────────────────────
class TemporalNF(nn.Module):
    """
    GRU context encoder + Conditional NSF.
    p(e_t | e_{t-p}, ..., e_{t-1}) via conditional normalizing flow.
    """
    def __init__(self, feat_dim: int, ar_order: int, gru_hidden: int,
                 n_transforms: int):
        super().__init__()
        self.ar_order  = ar_order
        self.gru = nn.GRU(feat_dim, gru_hidden, num_layers=1,
                          batch_first=True)
        self.flow = zuko.flows.NSF(
            features=feat_dim,
            context=gru_hidden,
            transforms=n_transforms,
            hidden_features=[64, 64],
        )

    def forward(self, context: torch.Tensor, target: torch.Tensor):
        """
        context: (B, ar_order, feat_dim)
        target:  (B, feat_dim)
        returns: log_prob (B,)
        """
        _, h = self.gru(context)     # h: (1, B, gru_hidden)
        h = h.squeeze(0)             # (B, gru_hidden)
        return self.flow(h).log_prob(target)

    def score(self, context: torch.Tensor, target: torch.Tensor):
        return -self.forward(context, target)


def build_sequences(X: np.ndarray, ar_order: int):
    """
    X: (N, D) → context (N-p, p, D), target (N-p, D)
    """
    N = len(X)
    ctx = np.stack([X[t - ar_order:t] for t in range(ar_order, N)], axis=0)
    tgt = X[ar_order:]
    return ctx, tgt


def fit_temporal_nf(model: TemporalNF, X_pca: np.ndarray,
                    epochs: int, lr: float = 1e-3,
                    batch_size: int = 32, patience: int = 30) -> float:
    p = model.ar_order
    ctx_np, tgt_np = build_sequences(X_pca, p)

    ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=DEVICE)
    tgt_t = torch.tensor(tgt_np, dtype=torch.float32, device=DEVICE)
    N     = len(ctx_t)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss, best_state, patience_cnt = float("inf"), None, 0

    for ep in range(epochs):
        idx = torch.randperm(N, device=DEVICE)
        ep_loss = 0.0
        for i in range(0, N, batch_size):
            b = idx[i:i + batch_size]
            loss = -model(ctx_t[b], tgt_t[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(b)
        ep_loss /= N
        sch.step()

        if ep_loss < best_loss - 1e-4:
            best_loss = ep_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_loss


def score_temporal_nf(model: TemporalNF, X_pca: np.ndarray) -> np.ndarray:
    p = model.ar_order
    N = len(X_pca)
    scores = np.zeros(N, dtype=np.float32)

    ctx_np, tgt_np = build_sequences(X_pca, p)
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=DEVICE)
    tgt_t = torch.tensor(tgt_np, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        s = model.score(ctx_t, tgt_t).cpu().numpy()

    scores[p:] = s
    scores[:p] = s.mean()
    return scores


# ── PCA ───────────────────────────────────────────────────────────────────────
def pca_fit_transform(X, k):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    k = min(k, X.shape[0] - 1, X.shape[1])
    sc = StandardScaler(); X_s = sc.fit_transform(X)
    pca = PCA(n_components=k, random_state=42); X_p = pca.fit_transform(X_s)
    return X_p, sc, pca


# ── EVT & Eval ────────────────────────────────────────────────────────────────
def evt_threshold(sc, q=EVT_Q_INIT, fpr=EVT_FPR):
    u = float(np.percentile(sc, q*100)); exc = sc[sc>u]-u
    fb = float(np.percentile(sc, (1-fpr)*100))
    if len(exc)<10: return fb
    try:
        c,_,s = genpareto.fit(exc,floc=0)
        p_c = min(fpr/max(1-q,1e-9), 1-1e-9)
        thr = u+max(0.,genpareto.ppf(1-p_c,c,loc=0,scale=s))
        return thr if u<=thr<=sc.max() else fb
    except: return fb

def evt_detect(sc, ts):
    if sc.max()-sc.min()<1e-8: return []
    flags=sc>evt_threshold(sc)
    if not flags.any(): return []
    ivs,in_seg=[],False
    for i,f in enumerate(flags):
        if f and not in_seg: in_seg=True; s_=i
        elif not f and in_seg: in_seg=False; ivs.append([ts[s_],ts[i-1]])
    if in_seg: ivs.append([ts[s_],ts[-1]])
    return ivs

def load_gt(path):
    gt={}
    with open(path,encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts=line.strip().split(",",1)
            if len(parts)==2:
                try: gt[parts[0]]=ast.literal_eval(parts[1].strip('"'))
                except: pass
    return gt

def _eval(d,g): return evaluate_intervals(g,d)["F1"]
def _ivs_alpha(sc,ts,a=0.01):
    idx,_,_=compute_detection_intervals(score_vector=sc,alpha=a)
    df=intervals_from_indices(idx,ts,sc)
    return [[r["start"],r["end"]] for _,r in df.iterrows()]
def load_pkl(p): return pickle.load(open(p,"rb")) if os.path.exists(p) else None
def save_pkl(o,p): pickle.dump(o,open(p,"wb"))


# ── Datasets ──────────────────────────────────────────────────────────────────
NAB  = ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38","ec2_cpu_utilization_5f5533",
        "ec2_cpu_utilization_77c1ca","ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
        "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de","ec2_disk_write_bytes_c0d644",
        "ec2_network_in_257a54","ec2_network_in_5abac7","elb_request_count_8c0756",
        "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
        "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"]
SMAP = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","F-1","F-2","F-3","P-1","T-1"]
MSL  = ["P-11","T-12","D-15","C-1","F-8","F-7","T-13","D-16","T-8","P-14","D-14"]
DATASETS = [("NAB",NAB),("SMAP",SMAP),("MSL",MSL)]
GT = load_gt(ANOM_CSV)


# ── Main ──────────────────────────────────────────────────────────────────────
records = []

for ds, signals in DATASETS:
    print(f"\n{'='*72}\nDataset: {ds}\n{'='*72}")

    for sig in signals:
        gt_key = sig if ds=="NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig, []))

        k5 = load_pkl(os.path.join(K5_DIR, f"{ds}__{sig}__ltr.pkl"))
        if k5 is None: print(f"  [{sig}] SKIP"); continue
        k5_sc=np.array(k5["scores"]); k5_ts=np.array(k5["timestamps"])
        f1_base = _eval(_ivs_alpha(k5_sc,k5_ts), gt_ivs)

        ep = load_pkl(os.path.join(EMBED_DIR, f"{ds}__{sig}__embeds.pkl"))
        if ep is None:
            print(f"  [{sig}] SKIP — embed cache missing"); continue
        embeds = ep["embeddings"]; win_ts = ep["timestamps"]
        X_pca, _, _ = pca_fit_transform(embeds, PCA_DIM)

        print(f"\n  [{sig}]  N={len(embeds)}  baseline F1={f1_base:.4f}")
        row = dict(dataset=ds, signal=sig, baseline_f1=f1_base)

        for p_ord, gru_h, n_tr, epochs in VARIANTS:
            vname = f"tnf_p{p_ord}_h{gru_h}"
            if len(X_pca) <= p_ord + 2:
                row[vname] = float("nan"); continue

            ckpt = os.path.join(CKPT_DIR, f"{ds}__{sig}__{vname}.pkl")
            cached = load_pkl(ckpt)

            if cached is not None:
                tnf_scores = np.array(cached["scores"])
                print(f"    {vname:<18} [cache]", end="")
            else:
                model = TemporalNF(PCA_DIM, p_ord, gru_h, n_tr).to(DEVICE)
                t0 = time.time()
                loss = fit_temporal_nf(model, X_pca, epochs)
                tnf_scores = score_temporal_nf(model, X_pca)
                elapsed = time.time() - t0
                save_pkl({"scores": tnf_scores, "timestamps": win_ts}, ckpt)
                print(f"    {vname:<18} loss={loss:.3f}  {elapsed:.1f}s", end="")

            T = min(len(tnf_scores), len(win_ts))
            ivs = evt_detect(tnf_scores[:T], win_ts[:T])
            f1  = _eval(ivs, gt_ivs)
            row[vname] = f1
            print(f"  F1={f1:.4f}  (Δ={f1-f1_base:+.4f})")

        records.append(row)


# ── Summary ───────────────────────────────────────────────────────────────────
df   = pd.DataFrame(records)
vcols = [f"tnf_p{p}_h{h}" for p,h,_,_ in VARIANTS]
vcols = [c for c in vcols if c in df.columns]

W = 16
print("\n\n"+"="*80)
print("CROSS-DATASET SUMMARY — Temporal NF vs Baseline")
print("="*80)
hdr = f"{'Dataset':<8}{'baseline':>10}"+"".join(f"{c:>{W}}" for c in vcols)
print(hdr); print("-"*len(hdr))
for ds_ in ["NAB","SMAP","MSL"]:
    sub = df[df.dataset==ds_]
    print(f"{ds_:<8}{sub.baseline_f1.mean():>10.4f}"+"".join(f"{sub[c].mean():>{W}.4f}" for c in vcols))
print("-"*len(hdr))
print(f"{'ALL':<8}{df.baseline_f1.mean():>10.4f}"+"".join(f"{df[c].mean():>{W}.4f}" for c in vcols))

best = max(vcols, key=lambda c: df[c].mean())
print(f"\nBest Temporal NF: {best}  ALL={df[best].mean():.4f}")
print(f"vs LP best SMAP:  lp_p20_sum SMAP=0.5410")
print(f"vs MAE baseline:             ALL=0.5660")

df.to_csv(os.path.join(RESULTS_DIR,"comparison.csv"), index=False)
print(f"\nSaved → {os.path.join(RESULTS_DIR,'comparison.csv')}")
