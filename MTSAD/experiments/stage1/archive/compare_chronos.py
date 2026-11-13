"""Chronos Foundation Model for Zero-Shot TSAD.
Amazon Chronos-T5 (2024) — arXiv:2403.07815

시계열 학습 모델 실험
---------------------
MAE(ImageNet) → Chronos(시계열 학습)로 백본 교체.
이미지 렌더링 없음. 원시 시계열 직접 입력.

두 가지 스코링 패러다임 동시 비교
----------------------------------
1. Embedding-based  : T5 encoder → 임베딩 → LTR / Mahal / NF
2. Forecast-based   : Chronos가 "다음 윈도우" 예측 → 예측 오차 = 이상치 점수
                      (Forecasting-as-Normality 패러다임)

Forecast-based 가설: 정상 구간 → Chronos가 잘 예측 → 오차 낮음
                     이상 구간 → Chronos가 못 예측 → 오차 높음

모델 선택
---------
  amazon/chronos-t5-small  : 20M 파라미터, d_model=512, 빠름
  amazon/chronos-t5-base   : 200M 파라미터, d_model=768, 추천

윈도우 설계
-----------
  Embedding scoring : window=512, step=64
  Forecast scoring  : context=448, forecast_horizon=64 (context+pred=512 step단위)
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

subprocess.run(
    ["pip", "install", "chronos-forecasting", "zuko", "scipy", "--quiet"],
    check=True,
)

import torch
import zuko
from chronos import ChronosPipeline

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_chronos")
EMBED_DIR   = os.path.join(RESULTS_DIR, "embeddings")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
for d in [RESULTS_DIR, EMBED_DIR, CKPT_DIR]: os.makedirs(d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 512    # embedding window
STEP_SIZE        = 64     # stride
FORECAST_CONTEXT = 448    # context length for forecasting
FORECAST_HORIZON = 64     # steps to predict (= STEP_SIZE)
N_SAMPLES        = 20     # Chronos forecast samples for uncertainty
PCA_DIM          = 64
EVT_Q_INIT       = 0.90
EVT_FPR          = 0.01

MODEL_NAME = "amazon/chronos-t5-small"   # fast; change to -base for better quality

# ── Load Chronos ──────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading {MODEL_NAME} ...")
_pipeline = ChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float32,
)
_model = _pipeline.model
_model.eval()
# d_model은 내부 T5 모델 config에 있음
def _get_d_model(pipeline):
    for attr in [
        lambda p: p.model.model.config.d_model,
        lambda p: p.model.config.d_model,
        lambda p: p.model.config.hidden_size,
        lambda p: p.model.encoder.config.d_model,
    ]:
        try: return attr(pipeline)
        except Exception: pass
    # fallback: chronos-t5-small=512, base=768, large=1024
    name = MODEL_NAME.lower()
    return 1024 if "large" in name else (768 if "base" in name else 512)

EMBED_DIM = _get_d_model(_pipeline)
print(f"[INFO] Chronos loaded. d_model={EMBED_DIM}")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def global_preprocess(v: np.ndarray) -> np.ndarray:
    v = scipy_detrend(v.astype(float))
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(v)


# ── Window sliding ────────────────────────────────────────────────────────────
def slide(values, win, step):
    starts = []
    s = 0
    while s + win <= len(values):
        starts.append(s)
        s += step
    return starts

def center_ts(ts, starts, win):
    c = win // 2
    return np.array([ts[s + c] for s in starts])


# ── 1) Encoder Embedding Extraction ──────────────────────────────────────────
def extract_encoder_embeddings(values: np.ndarray, starts: list,
                                batch_size: int = 32) -> np.ndarray:
    """
    Feed each 512-step window into Chronos T5 encoder.
    Returns: (N_windows, d_model) float32
    """
    results = []
    tokenizer = _pipeline.tokenizer

    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i:i + batch_size]
        windows = np.stack([values[s:s + WINDOW_SIZE] for s in batch_starts])
        ctx = torch.tensor(windows, dtype=torch.float32)         # (B, 512)

        # Chronos tokenizer: 버전에 따라 반환값 개수 다름 (2개 or 3개)
        _transform = tokenizer.context_input_transform(ctx)
        token_ids, attn_mask = _transform[0], _transform[1]
        token_ids  = token_ids.to(DEVICE)
        attn_mask  = attn_mask.to(DEVICE)

        with torch.no_grad():
            # ChronosModel wraps T5 — encoder is at _model.model.encoder
            t5_encoder = None
            for path in [
                lambda m: m.model.encoder,
                lambda m: m.encoder,
                lambda m: m.model.model.encoder,
            ]:
                try: t5_encoder = path(_model); break
                except AttributeError: pass

            if t5_encoder is not None:
                enc_out = t5_encoder(input_ids=token_ids, attention_mask=attn_mask)
                hidden  = enc_out.last_hidden_state          # (B, seq_len, d_model)
                mask_f  = attn_mask.unsqueeze(-1).float()
                pooled  = (hidden * mask_f).sum(1) / mask_f.sum(1)
            else:
                # Fallback: use pipeline's internal encode if available
                pooled = _model.encode(context=ctx.to(DEVICE))
                if pooled.ndim == 3:
                    pooled = pooled.mean(dim=1)
        results.append(pooled.cpu().float().numpy())

    return np.vstack(results)  # (N, d_model)


# ── 2) Forecast Error Scoring ────────────────────────────────────────────────
def compute_forecast_scores(values: np.ndarray, raw_ts: np.ndarray,
                             batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    For each position t (spaced by STEP_SIZE):
      - Context: values[t - FORECAST_CONTEXT : t]
      - Forecast target: values[t : t + FORECAST_HORIZON]
      - Score: median absolute error between forecast and actual

    Returns: (scores array, timestamps array)
    """
    scores_list, ts_list = [], []
    n = len(values)
    positions = []
    t = FORECAST_CONTEXT
    while t + FORECAST_HORIZON <= n:
        positions.append(t)
        t += STEP_SIZE

    for i in range(0, len(positions), batch_size):
        batch_pos = positions[i:i + batch_size]
        contexts  = [torch.tensor(values[p - FORECAST_CONTEXT:p],
                                  dtype=torch.float32) for p in batch_pos]

        # Chronos forecast — context는 위치 인수
        try:
            forecast = _pipeline.predict(
                contexts,
                prediction_length=FORECAST_HORIZON,
                num_samples=N_SAMPLES,
                limit_prediction_length=False,
            )
        except TypeError:
            forecast = _pipeline.predict(
                contexts,
                prediction_length=FORECAST_HORIZON,
                num_samples=N_SAMPLES,
            )
        fc_np = forecast.numpy() if hasattr(forecast, "numpy") else np.array(forecast)
        # Normalise to (B, n_samples, horizon)
        if fc_np.ndim == 3 and fc_np.shape[0] == N_SAMPLES:
            fc_np = fc_np.transpose(1, 0, 2)   # (n_samples,B,h) → (B,n_samples,h)
        med_forecast = np.median(fc_np, axis=1)  # (B, horizon)

        for j, p in enumerate(batch_pos):
            actual = values[p:p + FORECAST_HORIZON]
            T = min(len(actual), med_forecast.shape[1])
            err = np.abs(actual[:T] - med_forecast[j, :T]).mean()
            scores_list.append(err)
            # Timestamp = center of the forecast window
            ts_list.append(raw_ts[p + FORECAST_HORIZON // 2])

    return np.array(scores_list, dtype=np.float32), np.array(ts_list)


# ── Scoring functions ─────────────────────────────────────────────────────────
def ltr_score(embeds: np.ndarray, k: int) -> np.ndarray:
    N = len(embeds)
    e = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    sim = e @ e.T  # (N, N)
    scores = np.zeros(N, dtype=np.float32)
    idx_all = np.arange(N)
    for t in range(N):
        td = np.abs(idx_all - t); td[t] = N + 1
        k_idx = np.argsort(td)[:k]
        scores[t] = 1.0 - sim[t, k_idx].mean()
    return scores

def mahal_score(embeds: np.ndarray) -> np.ndarray:
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    d = min(PCA_DIM, embeds.shape[0] - 1, embeds.shape[1])
    X = PCA(n_components=d, random_state=42).fit_transform(
        StandardScaler().fit_transform(embeds))
    try:
        return EmpiricalCovariance().fit(X).mahalanobis(X).astype(np.float32)
    except Exception:
        mu = X.mean(0)
        return np.sum((X - mu)**2, axis=1).astype(np.float32)

def nf_score(embeds: np.ndarray, epochs: int = 10, lr: float = 0.005) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    d = min(PCA_DIM, embeds.shape[0] - 1, embeds.shape[1])
    X = PCA(n_components=d, random_state=42).fit_transform(
        StandardScaler().fit_transform(embeds))
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    flow = zuko.flows.MAF(features=d, context=0, transforms=3,
                          hidden_features=[64, 64]).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    for _ in range(epochs):
        idx = torch.randperm(len(X_t), device=DEVICE)
        for i in range(0, len(X_t), 64):
            b = X_t[idx[i:i+64]]
            loss = -flow().log_prob(b).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return -flow().log_prob(X_t).cpu().numpy().astype(np.float32)


# ── EVT & Eval ────────────────────────────────────────────────────────────────
def evt_threshold(sc, q=EVT_Q_INIT, fpr=EVT_FPR):
    u = float(np.percentile(sc, q*100)); exc = sc[sc>u]-u
    fb = float(np.percentile(sc, (1-fpr)*100))
    if len(exc) < 10: return fb
    try:
        c,_,s = genpareto.fit(exc, floc=0)
        p_c = min(fpr/max(1-q,1e-9), 1-1e-9)
        thr = u+max(0., genpareto.ppf(1-p_c, c, loc=0, scale=s))
        return thr if u<=thr<=sc.max() else fb
    except: return fb

def evt_detect(sc, ts):
    if sc.max()-sc.min() < 1e-8: return []
    flags = sc > evt_threshold(sc)
    if not flags.any(): return []
    ivs, in_seg = [], False
    for i, f in enumerate(flags):
        if f and not in_seg: in_seg=True; s_=i
        elif not f and in_seg: in_seg=False; ivs.append([ts[s_], ts[i-1]])
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
    idx,_,_ = compute_detection_intervals(score_vector=sc, alpha=a)
    df = intervals_from_indices(idx, ts, sc)
    return [[r["start"],r["end"]] for _,r in df.iterrows()]
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
DATASETS = [("NAB",NAB),("SMAP",SMAP),("MSL",MSL)]
GT = load_gt(ANOM_CSV)


# ── Main ──────────────────────────────────────────────────────────────────────
records = []

for ds, signals in DATASETS:
    print(f"\n{'='*72}\nDataset: {ds}\n{'='*72}")
    csv_dir = os.path.join(DATA_DIR,
                           {"NAB":"realAWSCloudwatch","SMAP":"SMAP","MSL":"MSL"}[ds])

    for sig in signals:
        gt_key = sig if ds=="NAB" else f"{ds}_{sig}"
        gt_ivs = GT.get(gt_key, GT.get(sig, []))

        k5 = load_pkl(os.path.join(K5_DIR, f"{ds}__{sig}__ltr.pkl"))
        f1_base = float("nan")
        if k5 is not None:
            k5_sc = np.array(k5["scores"]); k5_ts = np.array(k5["timestamps"])
            f1_base = _eval(_ivs_alpha(k5_sc, k5_ts), gt_ivs)

        print(f"\n  [{sig}]  MAE-LTR F1={f1_base:.4f}")

        csv_path = os.path.join(csv_dir, f"{sig}.csv")
        if not os.path.exists(csv_path):
            print(f"    SKIP — CSV missing"); continue

        df_raw  = pd.read_csv(csv_path)
        ts_col  = next(c for c in df_raw.columns if "time" in c.lower())
        val_col = next(c for c in df_raw.columns if c != ts_col)
        raw_ts  = pd.to_datetime(df_raw[ts_col]).astype(np.int64).to_numpy()
        raw_val = df_raw[val_col].to_numpy(dtype=float)
        proc_val = global_preprocess(raw_val)

        row = dict(dataset=ds, signal=sig, mae_ltr_f1=f1_base)

        # ── Embedding-based scoring ───────────────────────────────────────────
        emb_starts = slide(proc_val, WINDOW_SIZE, STEP_SIZE)
        if len(emb_starts) >= 5:
            win_ts = center_ts(raw_ts, emb_starts, WINDOW_SIZE)

            ep = load_pkl(os.path.join(EMBED_DIR, f"{ds}__{sig}__chronos.pkl"))
            if ep is not None:
                embeds = ep["embeddings"]; win_ts = ep["timestamps"]
                print(f"    [embed] cache  {embeds.shape}")
            else:
                t0 = time.time()
                embeds = extract_encoder_embeddings(proc_val, emb_starts)
                elapsed = time.time()-t0
                save_pkl({"embeddings":embeds,"timestamps":win_ts},
                         os.path.join(EMBED_DIR, f"{ds}__{sig}__chronos.pkl"))
                print(f"    [embed] {elapsed:.1f}s  {embeds.shape}")

            emb_scorers = {
                "chr_ltr_k5":  lambda e: ltr_score(e, k=5),
                "chr_ltr_k30": lambda e: ltr_score(e, k=30),
                "chr_mahal":   lambda e: mahal_score(e),
                "chr_nf_maf":  lambda e: nf_score(e),
            }
            for vname, fn in emb_scorers.items():
                ckpt = os.path.join(CKPT_DIR, f"{ds}__{sig}__{vname}.pkl")
                c = load_pkl(ckpt)
                if c is not None:
                    sc = np.array(c["scores"]); ts_ = np.array(c["timestamps"])
                    print(f"    {vname:<18} [cache]", end="")
                else:
                    t0 = time.time(); sc = fn(embeds); elapsed=time.time()-t0
                    ts_ = win_ts
                    save_pkl({"scores":sc,"timestamps":ts_}, ckpt)
                    print(f"    {vname:<18} {elapsed:.1f}s", end="")
                T = min(len(sc), len(ts_))
                f1 = _eval(evt_detect(sc[:T], ts_[:T]), gt_ivs)
                row[vname] = f1
                d_ = f1-f1_base if not np.isnan(f1_base) else float("nan")
                print(f"  F1={f1:.4f}  (Δ={d_:+.4f})")
        else:
            for v in ["chr_ltr_k5","chr_ltr_k30","chr_mahal","chr_nf_maf"]:
                row[v] = float("nan")

        # ── Forecast-based scoring (Forecasting-as-Normality) ─────────────────
        if len(proc_val) >= FORECAST_CONTEXT + FORECAST_HORIZON:
            fc_ckpt = os.path.join(CKPT_DIR, f"{ds}__{sig}__chr_forecast.pkl")
            fc_c = load_pkl(fc_ckpt)
            if fc_c is not None:
                fc_sc = np.array(fc_c["scores"]); fc_ts = np.array(fc_c["timestamps"])
                print(f"    {'chr_forecast':<18} [cache]", end="")
            else:
                t0 = time.time()
                fc_sc, fc_ts = compute_forecast_scores(proc_val, raw_ts)
                elapsed = time.time()-t0
                save_pkl({"scores":fc_sc,"timestamps":fc_ts}, fc_ckpt)
                print(f"    {'chr_forecast':<18} {elapsed:.1f}s", end="")
            T = min(len(fc_sc), len(fc_ts))
            f1 = _eval(evt_detect(fc_sc[:T], fc_ts[:T]), gt_ivs)
            row["chr_forecast"] = f1
            d_ = f1-f1_base if not np.isnan(f1_base) else float("nan")
            print(f"  F1={f1:.4f}  (Δ={d_:+.4f})")
        else:
            row["chr_forecast"] = float("nan")

        records.append(row)


# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
vcols = ["chr_ltr_k5","chr_ltr_k30","chr_mahal","chr_nf_maf","chr_forecast"]
vcols = [c for c in vcols if c in df.columns]

W = 16
print("\n\n"+"="*90)
print(f"CROSS-DATASET SUMMARY — Chronos ({MODEL_NAME}) vs MAE-LTR baseline")
print("="*90)
hdr = f"{'Dataset':<8}{'mae_ltr':>10}"+"".join(f"{c:>{W}}" for c in vcols)
print(hdr); print("-"*len(hdr))
for ds_ in ["NAB","SMAP","MSL"]:
    sub = df[df.dataset==ds_]
    print(f"{ds_:<8}{sub.mae_ltr_f1.mean():>10.4f}"
          +"".join(f"{sub[c].mean():>{W}.4f}" for c in vcols))
print("-"*len(hdr))
print(f"{'ALL':<8}{df.mae_ltr_f1.mean():>10.4f}"
      +"".join(f"{df[c].mean():>{W}.4f}" for c in vcols))

print("\n\nKey findings:")
print(f"  MAE+LTR baseline     : ALL={df.mae_ltr_f1.mean():.4f}")
print(f"  Current best (ROWA)  : ALL=0.6307")
print(f"  Friend DINOv2+LP     : ALL≈0.667  SMAP=0.9762")

# Forecast vs Embedding comparison for SMAP
if "chr_forecast" in df.columns:
    smap = df[df.dataset=="SMAP"]
    print(f"\n  Forecasting-as-Normality SMAP = {smap.chr_forecast.mean():.4f}")
    print(f"  Best embedding scorer  SMAP  = "
          f"{max(smap[c].mean() for c in vcols if c!='chr_forecast'):.4f}")

best = max(vcols, key=lambda c: df[c].mean())
print(f"\n  Best Chronos variant: {best}  ALL={df[best].mean():.4f}")

df.to_csv(os.path.join(RESULTS_DIR,"comparison.csv"), index=False)
print(f"\nSaved → {os.path.join(RESULTS_DIR,'comparison.csv')}")
