"""Patch-Level Normalizing Flow (PatchCore/PaDiM-inspired for TSAD).

핵심 아이디어
------------
기존: 윈도우 1개 = 임베딩 1개 (768-dim) → NF에 N개 데이터포인트
새로: 윈도우 1개 = **패치 196개** (각 768-dim) → NF에 N×196개 데이터포인트

장점:
  - 데이터포인트 196× 증가 → NF 학습 훨씬 안정적
  - 패치별 이상치 탐지 → 시계열 내 이상 구간의 정확한 위치 파악
  - Window-level score = mean(-log p(patch_i))  또는 top-k 패치의 max score
  - PaDiM (ICPR 2021), PatchCore (CVPR 2022) 철학의 시계열 버전

구현
----
  1. MAE 인코딩 시 patch-level 특징 추출 (196 × 768, layer-10)
     → 이미 results_lp_resid/embeddings/에 캐시됨 (forward_features는 patch tokens 반환)
     → 단, 현재 캐시는 mean-pooled (1×768). patch-level 재추출 필요.
  2. PCA: 768 → 32
  3. MAF/NSF fit on ALL patches of a signal (N×196, 32-dim)
  4. Window score = 각 패치 score의 mean (또는 top-k-mean)
"""
from __future__ import annotations

import ast, os, pickle, subprocess, sys, time
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from scipy.signal import detrend as scipy_detrend
from scipy.stats import genpareto

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

print(f"Project root : {PROJECT_ROOT}")
subprocess.run(["pip", "install", "zuko", "timm", "open-clip-torch",
                "scipy", "--quiet"], check=True)

import torch
import timm
import zuko

from evaluation.evaluate import evaluate_intervals
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
ANOM_CSV    = os.path.join(DATA_DIR, "anomalies.csv")
K5_DIR      = next((p for p in [
    os.path.join(PROJECT_ROOT, "results", "VLM4TS_results_mgmr", "checkpoints"),
    os.path.join(PROJECT_ROOT, "results_mgmr", "checkpoints"),
] if os.path.isdir(p)), "")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_patch_nf")
EMBED_DIR   = os.path.join(RESULTS_DIR, "patch_embeddings")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
for d in [RESULTS_DIR, EMBED_DIR, CKPT_DIR]: os.makedirs(d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 224; STEP_SIZE = 56
IMAGE_SIZE  = (224, 224); DPI = 100
FIG_W = FIG_H = IMAGE_SIZE[0] / DPI
EVT_Q_INIT = 0.90; EVT_FPR = 0.01
PCA_DIM = 32
PATCH_DIM = 768   # MAE ViT-B/16 hidden dim

# Score aggregation: "mean" or "topk" (mean of top-k most anomalous patches per window)
TOPK = 20   # top-20 out of 196 patches

# NF config
NF_EPOCHS = 50    # more data (196× patches) → fewer epochs needed
NF_LR     = 0.005

# ── MAE model ─────────────────────────────────────────────────────────────────
print("\n[INFO] Loading MAE model ...")
_mae = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
_mae = _mae.eval().to(DEVICE)
_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)
_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)


# ── Preprocessing & rendering ─────────────────────────────────────────────────
def global_preprocess(values):
    v = scipy_detrend(values.astype(float))
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(v)

def render_windows(values):
    imgs, starts = [], []
    t = np.arange(WINDOW_SIZE); start = 0
    while start + WINDOW_SIZE <= len(values):
        w = values[start:start+WINDOW_SIZE]
        fig, ax = plt.subplots(1,1,figsize=(FIG_W,FIG_H),dpi=DPI)
        ax.plot(t, w, '-', lw=1, color='black'); ax.set_ylim(0,1); ax.axis('off')
        fig.tight_layout(pad=0)
        buf = BytesIO(); fig.savefig(buf, format='png', dpi=DPI,
                                     bbox_inches='tight', pad_inches=0)
        plt.close(fig); buf.seek(0)
        imgs.append(np.array(PILImage.open(buf).convert('RGB').resize(IMAGE_SIZE)
                              ).transpose(2,0,1))
        starts.append(start); start += STEP_SIZE
    return np.stack(imgs,0), starts

def window_center_ts(timestamps, starts):
    c = WINDOW_SIZE // 2
    return np.array([timestamps[s+c] for s in starts])


# ── Patch embedding extraction ────────────────────────────────────────────────
def extract_patch_embeddings(imgs: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """
    imgs: (N, 3, 224, 224) uint8
    Returns: (N, 196, 768)  — all patch tokens from MAE layer-10
    """
    results = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.from_numpy(imgs[i:i+batch_size]).float().to(DEVICE)/255.0
        batch = (batch - _MEAN) / _STD
        with torch.no_grad():
            feats = _mae.forward_features(batch)   # (B, seq_len, 768)
            if feats.ndim == 3:
                # Remove CLS token if present (seq_len=197 → 196)
                if feats.shape[1] == 197:
                    feats = feats[:, 1:, :]        # (B, 196, 768)
                results.append(feats.cpu().float().numpy())
            else:
                # Already pooled — shouldn't happen but handle gracefully
                results.append(feats.unsqueeze(1).expand(-1,196,-1).cpu().float().numpy())
    return np.concatenate(results, axis=0)  # (N, 196, 768)


# ── NF ────────────────────────────────────────────────────────────────────────
def fit_nf_on_patches(patches_pca: np.ndarray, epochs: int, lr: float,
                      batch_size: int = 512) -> zuko.flows.MAF:
    """
    patches_pca: (N*196, PCA_DIM)  all patches from all windows
    Returns trained MAF flow.
    """
    dim  = patches_pca.shape[1]
    flow = zuko.flows.MAF(features=dim, context=0, transforms=3,
                          hidden_features=[64, 64]).to(DEVICE)
    X_t  = torch.tensor(patches_pca, dtype=torch.float32, device=DEVICE)
    opt  = torch.optim.Adam(flow.parameters(), lr=lr)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N    = len(X_t)

    best_loss, best_state, patience, pat_cnt = float("inf"), None, 20, 0
    for ep in range(epochs):
        idx = torch.randperm(N, device=DEVICE)
        ep_loss = 0.0
        for i in range(0, N, batch_size):
            b = X_t[idx[i:i+batch_size]]
            loss = -flow().log_prob(b).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(b)
        ep_loss /= N; sch.step()
        if ep_loss < best_loss - 1e-4:
            best_loss = ep_loss
            best_state = {k:v.clone() for k,v in flow.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience: break
    if best_state: flow.load_state_dict(best_state)
    return flow

def score_patches(flow: zuko.flows.MAF, patches_pca: np.ndarray,
                  n_windows: int, agg: str = "mean") -> np.ndarray:
    """
    patches_pca: (N*196, PCA_DIM)
    Returns: (N,) window-level anomaly scores
    """
    X_t = torch.tensor(patches_pca, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        patch_scores = -flow().log_prob(X_t).cpu().numpy()  # (N*196,)
    patch_scores = patch_scores.reshape(n_windows, -1)  # (N, 196)
    if agg == "topk":
        k = min(TOPK, patch_scores.shape[1])
        return np.sort(patch_scores, axis=1)[:, -k:].mean(axis=1)
    return patch_scores.mean(axis=1)


# ── EVT & Eval ────────────────────────────────────────────────────────────────
def evt_threshold(sc, q=EVT_Q_INIT, fpr=EVT_FPR):
    u=float(np.percentile(sc,q*100)); exc=sc[sc>u]-u
    fb=float(np.percentile(sc,(1-fpr)*100))
    if len(exc)<10: return fb
    try:
        c,_,s=genpareto.fit(exc,floc=0)
        p_c=min(fpr/max(1-q,1e-9),1-1e-9)
        thr=u+max(0.,genpareto.ppf(1-p_c,c,loc=0,scale=s))
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
            p=line.strip().split(",",1)
            if len(p)==2:
                try: gt[p[0]]=ast.literal_eval(p[1].strip('"'))
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

    csv_dir = {"NAB":"realAWSCloudwatch","SMAP":"SMAP","MSL":"MSL"}[ds]
    csv_dir = os.path.join(DATA_DIR, csv_dir)

    for sig in signals:
        gt_key = sig if ds=="NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig,[]))

        k5=load_pkl(os.path.join(K5_DIR,f"{ds}__{sig}__ltr.pkl"))
        if k5 is None: print(f"  [{sig}] SKIP"); continue
        k5_sc=np.array(k5["scores"]); k5_ts=np.array(k5["timestamps"])
        f1_base=_eval(_ivs_alpha(k5_sc,k5_ts),gt_ivs)

        print(f"\n  [{sig}]  baseline F1={f1_base:.4f}")

        # ── Load or compute patch embeddings ──────────────────────────────────
        pep = load_pkl(os.path.join(EMBED_DIR, f"{ds}__{sig}__patches.pkl"))
        if pep is not None:
            patches = pep["patches"]  # (N, 196, 768)
            win_ts  = pep["timestamps"]
            print(f"    [patches] cache  {patches.shape}")
        else:
            csv_path = os.path.join(csv_dir, f"{sig}.csv")
            if not os.path.exists(csv_path):
                print(f"    SKIP — CSV missing"); continue
            df_raw = pd.read_csv(csv_path)
            ts_col = next(c for c in df_raw.columns if "time" in c.lower())
            val_col= next(c for c in df_raw.columns if c!=ts_col)
            raw_ts = pd.to_datetime(df_raw[ts_col]).astype(np.int64).to_numpy()
            raw_v  = df_raw[val_col].to_numpy(dtype=float)
            proc_v = global_preprocess(raw_v)
            t0=time.time()
            imgs,starts=render_windows(proc_v)
            win_ts=window_center_ts(raw_ts,starts)
            print(f"    [render] {time.time()-t0:.1f}s  {imgs.shape}")
            t0=time.time()
            patches=extract_patch_embeddings(imgs)
            print(f"    [encode] {time.time()-t0:.1f}s  {patches.shape}")
            save_pkl({"patches":patches,"timestamps":win_ts},
                     os.path.join(EMBED_DIR,f"{ds}__{sig}__patches.pkl"))

        N_wins = patches.shape[0]  # number of windows
        patches_flat = patches.reshape(-1, PATCH_DIM)  # (N*196, 768)

        # ── PCA ───────────────────────────────────────────────────────────────
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        sc_ = StandardScaler()
        pca_= PCA(n_components=PCA_DIM, random_state=42)
        patches_pca = pca_.fit_transform(sc_.fit_transform(patches_flat))  # (N*196, 32)

        # ── Fit NF ────────────────────────────────────────────────────────────
        for agg_mode in ["mean", "topk"]:
            vname = f"pnf_{agg_mode}"
            ckpt  = os.path.join(CKPT_DIR, f"{ds}__{sig}__{vname}.pkl")
            cached= load_pkl(ckpt)

            if cached is not None:
                pnf_scores = np.array(cached["scores"])
                print(f"    {vname:<16} [cache]", end="")
            else:
                t0 = time.time()
                flow = fit_nf_on_patches(patches_pca, NF_EPOCHS, NF_LR)
                pnf_scores = score_patches(flow, patches_pca, N_wins, agg=agg_mode)
                elapsed = time.time()-t0
                save_pkl({"scores":pnf_scores,"timestamps":win_ts}, ckpt)
                print(f"    {vname:<16} {elapsed:.1f}s", end="")

            T=min(len(pnf_scores),len(win_ts))
            ivs=evt_detect(pnf_scores[:T],win_ts[:T])
            f1=_eval(ivs,gt_ivs)
            records.append(dict(dataset=ds,signal=sig,
                                baseline_f1=f1_base,variant=vname,f1=f1))
            print(f"  F1={f1:.4f}  (Δ={f1-f1_base:+.4f})")


# ── Summary ───────────────────────────────────────────────────────────────────
df=pd.DataFrame(records)
print("\n\n"+"="*70)
print("CROSS-DATASET SUMMARY — Patch NF vs Baseline")
print("="*70)
for vname_ in ["pnf_mean","pnf_topk"]:
    sub=df[df.variant==vname_]
    if sub.empty: continue
    print(f"\nVariant: {vname_}")
    for ds_ in ["NAB","SMAP","MSL"]:
        s2=sub[sub.dataset==ds_]
        print(f"  {ds_:<6} baseline={s2.baseline_f1.mean():.4f}  patch_nf={s2.f1.mean():.4f}"
              f"  Δ={s2.f1.mean()-s2.baseline_f1.mean():+.4f}")
    all_=sub.f1.mean(); base_=sub.baseline_f1.mean()
    print(f"  ALL    baseline={base_:.4f}  patch_nf={all_:.4f}  Δ={all_-base_:+.4f}")

df.to_csv(os.path.join(RESULTS_DIR,"comparison.csv"),index=False)
print(f"\nSaved → {os.path.join(RESULTS_DIR,'comparison.csv')}")
