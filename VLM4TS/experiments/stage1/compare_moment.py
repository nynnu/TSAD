"""MOMENT Foundation Model for Zero-Shot TSAD.
AutonLab/MOMENT-1-large (MIT, 2024) — arXiv:2402.03885

MAE와 근본적 차이
-----------------
  MAE:    이미지 → MAE(ImageNet 학습) → 임베딩
  MOMENT: 시계열 → MOMENT(시계열 데이터 학습) → 임베딩

MOMENT는 원시 시계열을 직접 입력 → 이미지 렌더링 단계 완전 제거.
시계열 패치(patch_size=8) 기반 Transformer, 컨텍스트 길이 512.
내부적으로 RevIN(Reversible Instance Normalization) 자동 적용.

실험 설계
---------
  - 윈도우 크기: 512 (MOMENT 네이티브 컨텍스트)
  - 스텝: 64
  - 전처리: global detrend+minmax (baseline 동일) OR raw
  - 임베딩: (B, 1, 512) → (B, 1024)
  - 스코링: LTR-k5, LTR-k30, Mahalanobis, NF-MAF

핵심 질문: MAE(ImageNet) vs MOMENT(시계열) 백본 교체만으로 성능이 얼마나 올라가나?
"""
from __future__ import annotations

import ast, os, pickle, subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.signal import detrend as scipy_detrend
from scipy.stats import genpareto

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_AUTO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_ENV_ROOT    = os.environ.get("VLM4TS_ROOT", "").strip()
PROJECT_ROOT = _ENV_ROOT if _ENV_ROOT and os.path.isdir(_ENV_ROOT) else _AUTO_ROOT
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

print(f"Project root : {PROJECT_ROOT}")

# momentfm 설치 — setuptools_scm이 git을 필요로 해서 Colab에서 직접 실패
# 해결책: SETUPTOOLS_SCM_PRETEND_VERSION으로 git 없이 설치
import os as _os
_os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.1.0")

for _cmd in [
    ["pip", "install", "momentfm", "--quiet"],
    ["pip", "install", "momentfm", "--no-build-isolation", "--quiet"],
    ["pip", "install", "momentfm", "--no-build-isolation",
     "--no-deps", "--quiet"],
]:
    if subprocess.run(_cmd, capture_output=True).returncode == 0:
        break
else:
    raise SystemExit(
        "\n[ERROR] momentfm 설치 실패.\n"
        "Colab 셀에서 먼저 실행:\n"
        "  import os; os.environ['SETUPTOOLS_SCM_PRETEND_VERSION']='0.1.0'\n"
        "  !pip install momentfm --no-build-isolation -q\n"
    )
subprocess.run(["pip", "install", "zuko", "scipy", "--quiet"], check=True)

import torch
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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_moment")
EMBED_DIR   = os.path.join(RESULTS_DIR, "embeddings")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
for d in [RESULTS_DIR, EMBED_DIR, CKPT_DIR]: os.makedirs(d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 512   # MOMENT's native context length
STEP_SIZE    = 64    # stride between windows
MOMENT_DIM   = 1024  # MOMENT-large hidden dim
PCA_DIM      = 64    # PCA before NF/Mahal
EVT_Q_INIT   = 0.90
EVT_FPR      = 0.01

# ── Load MOMENT ───────────────────────────────────────────────────────────────
print("\n[INFO] Loading MOMENT-1-large ...")
from momentfm import MOMENTPipeline

_moment = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "embedding"},
)
_moment.init()   # 필수: 가중치/구조 초기화
_moment.eval()
try:
    _moment = _moment.to(DEVICE)
except Exception:
    pass
print("[INFO] MOMENT loaded.")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def global_preprocess(values: np.ndarray) -> np.ndarray:
    """Global detrend + minmax [0,1] — same as MAE baseline."""
    v = scipy_detrend(values.astype(float))
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(v)


# ── Window sliding ────────────────────────────────────────────────────────────
def slide_windows(values: np.ndarray):
    """Yield (start, window_array) pairs of length WINDOW_SIZE."""
    starts = []
    start = 0
    while start + WINDOW_SIZE <= len(values):
        starts.append(start)
        start += STEP_SIZE
    return starts

def window_center_ts(timestamps: np.ndarray, starts: list) -> np.ndarray:
    c = WINDOW_SIZE // 2
    return np.array([timestamps[s + c] for s in starts])


# ── MOMENT embedding extraction ───────────────────────────────────────────────
def extract_moment_embeddings(values: np.ndarray, starts: list,
                               batch_size: int = 32) -> np.ndarray:
    """
    values: preprocessed 1D time series
    starts: list of window start indices
    Returns: (N_windows, 1024) float32
    """
    N = len(starts)
    results = []

    for i in range(0, N, batch_size):
        batch_starts = starts[i:i + batch_size]
        # Shape: (B, 1, 512) — univariate, 512 time steps
        windows = np.stack([values[s:s + WINDOW_SIZE] for s in batch_starts])
        x = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)  # (B,1,512)

        # MOMENT handles device internally; try to move to DEVICE
        try:
            x = x.to(DEVICE)
        except Exception:
            pass

        with torch.no_grad():
            B_cur = x.shape[0]
            input_mask = torch.ones(B_cur, WINDOW_SIZE,
                                    dtype=torch.long, device=x.device)

            # Approach 1: pipeline call
            out  = _moment(x_enc=x, input_mask=input_mask)
            emb  = getattr(out, "embeddings", None)

            # Approach 2: force task_name at call time
            if emb is None:
                try:
                    out  = _moment(x_enc=x, input_mask=input_mask,
                                   task_name="embedding")
                    emb  = getattr(out, "embeddings", None)
                except Exception:
                    pass

            # Approach 3: call the inner MOMENT model directly
            if emb is None:
                try:
                    inner = _moment.model
                    out2  = inner(x_enc=x, input_mask=input_mask,
                                  task_name="embedding")
                    emb   = getattr(out2, "embeddings", None)
                except Exception:
                    pass

            # Approach 4: grab any non-None tensor from the output
            if emb is None:
                for attr in ["reconstruction","forecast","logits","anomaly_scores"]:
                    val = getattr(out, attr, None)
                    if val is not None and hasattr(val, "ndim"):
                        print(f"    [MOMENT] using '{attr}' as embedding proxy")
                        emb = val
                        break

            if emb is None:
                avail = [f for f in dir(out) if not f.startswith("_")
                         and getattr(out, f, None) is not None]
                raise ValueError(
                    f"MOMENT: cannot find embeddings. Non-None fields: {avail}\n"
                    f"Try upgrading momentfm: pip install -U momentfm"
                )

            if emb.ndim == 3:
                emb = emb.mean(dim=1)     # mean-pool patches → (B, d_model)
            elif emb.ndim > 2:
                emb = emb.reshape(B_cur, -1)
        results.append(emb.cpu().float().numpy())

    return np.vstack(results)  # (N, 1024)


# ── Scoring methods ───────────────────────────────────────────────────────────

# — LTR (Local Temporal Reference) —
def ltr_score(embeds: np.ndarray, k: int) -> np.ndarray:
    """
    For each window t: cosine dissimilarity to k temporally nearest neighbors.
    Same principle as baseline LTR but on MOMENT embeddings.
    """
    N = len(embeds)
    e = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    sim = e @ e.T  # (N, N) cosine similarity matrix
    scores = np.zeros(N, dtype=np.float32)
    idx_all = np.arange(N)
    for t in range(N):
        tdist = np.abs(idx_all - t)
        tdist[t] = N + 1  # exclude self
        k_idx = np.argsort(tdist)[:k]
        scores[t] = 1.0 - sim[t, k_idx].mean()
    return scores


# — Mahalanobis —
def mahal_score(embeds: np.ndarray) -> np.ndarray:
    """Global Mahalanobis distance in PCA space."""
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    d = min(PCA_DIM, embeds.shape[0] - 1, embeds.shape[1])
    sc = StandardScaler()
    pca = PCA(n_components=d, random_state=42)
    X = pca.fit_transform(sc.fit_transform(embeds))

    try:
        cov = EmpiricalCovariance().fit(X)
        scores = cov.mahalanobis(X)          # squared Mahalanobis distances
    except Exception:
        # Fallback: L2 distance from mean
        mu = X.mean(axis=0)
        scores = np.sum((X - mu) ** 2, axis=1)
    return scores.astype(np.float32)


# — Normalizing Flow (MAF, VAN-AD style) —
def nf_score(embeds: np.ndarray, epochs: int = 10, lr: float = 0.005) -> np.ndarray:
    """MAF density estimation in PCA space. Score = -log p(e_t)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    d = min(PCA_DIM, embeds.shape[0] - 1, embeds.shape[1])
    sc = StandardScaler()
    pca = PCA(n_components=d, random_state=42)
    X = pca.fit_transform(sc.fit_transform(embeds))

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    flow = zuko.flows.MAF(features=d, context=0, transforms=3,
                          hidden_features=[64, 64]).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    N = len(X_t)
    for _ in range(epochs):
        idx = torch.randperm(N, device=DEVICE)
        for i in range(0, N, 64):
            b = X_t[idx[i:i+64]]
            loss = -flow().log_prob(b).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        scores = -flow().log_prob(X_t).cpu().numpy()
    return scores.astype(np.float32)


# ── EVT & Eval ────────────────────────────────────────────────────────────────
def evt_threshold(sc, q=EVT_Q_INIT, fpr=EVT_FPR):
    u = float(np.percentile(sc, q*100)); exc = sc[sc > u] - u
    fb = float(np.percentile(sc, (1-fpr)*100))
    if len(exc) < 10: return fb
    try:
        c, _, s = genpareto.fit(exc, floc=0)
        p_c = min(fpr / max(1-q, 1e-9), 1-1e-9)
        thr = u + max(0., genpareto.ppf(1-p_c, c, loc=0, scale=s))
        return thr if u <= thr <= sc.max() else fb
    except: return fb

def evt_detect(sc, ts):
    if sc.max() - sc.min() < 1e-8: return []
    flags = sc > evt_threshold(sc)
    if not flags.any(): return []
    ivs, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg: in_seg = True; s_ = i
        elif not f and in_seg: in_seg = False; ivs.append([ts[s_], ts[i-1]])
    if in_seg: ivs.append([ts[s_], ts[-1]])
    return ivs

def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            p = line.strip().split(",", 1)
            if len(p) == 2:
                try: gt[p[0]] = ast.literal_eval(p[1].strip('"'))
                except: pass
    return gt

def _eval(d, g): return evaluate_intervals(g, d)["F1"]
def _ivs_alpha(sc, ts, a=0.01):
    idx, _, _ = compute_detection_intervals(score_vector=sc, alpha=a)
    df = intervals_from_indices(idx, ts, sc)
    return [[r["start"], r["end"]] for _, r in df.iterrows()]
def load_pkl(p): return pickle.load(open(p,"rb")) if os.path.exists(p) else None
def save_pkl(o, p): pickle.dump(o, open(p,"wb"))


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


# ── Main ──────────────────────────────────────────────────────────────────────
records = []

for ds, signals in DATASETS:
    print(f"\n{'='*72}\nDataset: {ds}  ({len(signals)} signals)\n{'='*72}")

    csv_dir = {"NAB":"realAWSCloudwatch","SMAP":"SMAP","MSL":"MSL"}[ds]
    csv_dir = os.path.join(DATA_DIR, csv_dir)

    for sig in signals:
        gt_key = sig if ds == "NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig, []))

        # MAE baseline for comparison
        k5 = load_pkl(os.path.join(K5_DIR, f"{ds}__{sig}__ltr.pkl"))
        f1_base = float("nan")
        if k5 is not None:
            k5_sc = np.array(k5["scores"]); k5_ts = np.array(k5["timestamps"])
            f1_base = _eval(_ivs_alpha(k5_sc, k5_ts), gt_ivs)

        print(f"\n  [{sig}]  MAE-LTR F1={f1_base:.4f}")

        # ── Load raw signal ───────────────────────────────────────────────────
        csv_path = os.path.join(csv_dir, f"{sig}.csv")
        if not os.path.exists(csv_path):
            print(f"    SKIP — CSV missing"); continue

        df_raw  = pd.read_csv(csv_path)
        ts_col  = next(c for c in df_raw.columns if "time" in c.lower())
        val_col = next(c for c in df_raw.columns if c != ts_col)
        raw_ts  = pd.to_datetime(df_raw[ts_col]).astype(np.int64).to_numpy()
        raw_val = df_raw[val_col].to_numpy(dtype=float)

        # Global preprocess (same as baseline)
        proc_val = global_preprocess(raw_val)
        starts = slide_windows(proc_val)

        if len(starts) < 5:
            print(f"    SKIP — too few windows ({len(starts)})"); continue

        win_ts = window_center_ts(raw_ts, starts)

        # ── MOMENT embeddings ─────────────────────────────────────────────────
        embed_path = os.path.join(EMBED_DIR, f"{ds}__{sig}__moment.pkl")
        ep = load_pkl(embed_path)
        if ep is not None:
            embeds = ep["embeddings"]; win_ts = ep["timestamps"]
            print(f"    [embed] cache  {embeds.shape}")
        else:
            t0 = time.time()
            embeds = extract_moment_embeddings(proc_val, starts)
            elapsed = time.time() - t0
            save_pkl({"embeddings": embeds, "timestamps": win_ts}, embed_path)
            print(f"    [embed] {elapsed:.1f}s  {embeds.shape}")

        row = dict(dataset=ds, signal=sig, mae_ltr_f1=f1_base)

        # ── Scoring ───────────────────────────────────────────────────────────
        scorers = {
            "moment_ltr_k5":   lambda e: ltr_score(e, k=5),
            "moment_ltr_k30":  lambda e: ltr_score(e, k=30),
            "moment_mahal":    lambda e: mahal_score(e),
            "moment_nf_maf":   lambda e: nf_score(e, epochs=10, lr=0.005),
        }

        for vname, scorer_fn in scorers.items():
            ckpt = os.path.join(CKPT_DIR, f"{ds}__{sig}__{vname}.pkl")
            cached = load_pkl(ckpt)

            if cached is not None:
                scores = np.array(cached["scores"])
                print(f"    {vname:<22} [cache]", end="")
            else:
                t0 = time.time()
                scores = scorer_fn(embeds)
                elapsed = time.time() - t0
                save_pkl({"scores": scores, "timestamps": win_ts}, ckpt)
                print(f"    {vname:<22} {elapsed:.1f}s", end="")

            T   = min(len(scores), len(win_ts))
            ivs = evt_detect(scores[:T], win_ts[:T])
            f1  = _eval(ivs, gt_ivs)
            row[vname] = f1
            delta = f1 - f1_base if not np.isnan(f1_base) else float("nan")
            print(f"  F1={f1:.4f}  (Δ={delta:+.4f})")

        records.append(row)


# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
vcols = ["moment_ltr_k5","moment_ltr_k30","moment_mahal","moment_nf_maf"]
vcols = [c for c in vcols if c in df.columns]

W = 18
print("\n\n" + "="*90)
print("CROSS-DATASET SUMMARY — MOMENT backbone vs MAE-LTR baseline")
print("="*90)
hdr = f"{'Dataset':<8}{'mae_ltr':>10}" + "".join(f"{c:>{W}}" for c in vcols)
print(hdr); print("-"*len(hdr))
for ds_ in ["NAB","SMAP","MSL"]:
    sub = df[df.dataset == ds_]
    print(f"{ds_:<8}{sub.mae_ltr_f1.mean():>10.4f}" +
          "".join(f"{sub[c].mean():>{W}.4f}" for c in vcols))
print("-"*len(hdr))
print(f"{'ALL':<8}{df.mae_ltr_f1.mean():>10.4f}" +
      "".join(f"{df[c].mean():>{W}.4f}" for c in vcols))

# Best per dataset
print("\n\nPer-dataset winners:")
for ds_ in ["NAB","SMAP","MSL","ALL"]:
    sub = df if ds_ == "ALL" else df[df.dataset==ds_]
    best_v = max(vcols, key=lambda c: sub[c].mean())
    best_v_score = sub[best_v].mean()
    mae_score = sub.mae_ltr_f1.mean()
    diff = best_v_score - mae_score
    print(f"  {ds_:<6}: {best_v:<22} = {best_v_score:.4f}  "
          f"vs MAE-LTR={mae_score:.4f}  ({diff:+.4f})")

# Stuck signals
print("\n\nStuck signals (MAE-LTR F1=0):")
stuck = df[df.mae_ltr_f1 == 0.0]
for _, r in stuck.iterrows():
    vals = "  ".join(f"{c.split('_',2)[-1]}={r[c]:.3f}" for c in vcols)
    print(f"  {r.dataset}_{r.signal:<10}  {vals}")

print("\nKey comparison:")
print(f"  MAE + LTR (baseline):          ALL = {df.mae_ltr_f1.mean():.4f}")
print(f"  MAE + Mahal (prev best SMAP):  SMAP= 0.7641")
print(f"  Friend DINOv2+LP:              SMAP= 0.9762  ALL≈0.667")
best_overall = max(vcols, key=lambda c: df[c].mean())
print(f"  MOMENT best:                   ALL = {df[best_overall].mean():.4f}  ({best_overall})")

df.to_csv(os.path.join(RESULTS_DIR,"comparison.csv"), index=False)
print(f"\nSaved → {os.path.join(RESULTS_DIR,'comparison.csv')}")
