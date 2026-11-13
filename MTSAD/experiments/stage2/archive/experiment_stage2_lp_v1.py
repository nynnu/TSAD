"""
Stage2 with LP Stage1 (ml_L8_11_resid_topk10)
===============================================
Stage1: ml_L8_11_resid_topk10 from DINOv2 feature cache
Stage2: v52c selective verification (same logic as experiment_stage2_univar_v5_2.py)

Usage:
  python experiment_stage2_lp_v1.py
"""

import ast, base64, io, json, os, re, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin

warnings.filterwarnings("ignore")
from openai import OpenAI

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment.")
_client = OpenAI(api_key=API_KEY)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
CACHE_DIR  = BASE / "experiments/results_dinov2_ml_scoring/feature_cache"
ANOMS_CSV  = BASE / "data/anomalies.csv"
OUT_DIR    = BASE / "experiments/results_stage2_lp_v1"

DATASETS = {
    "MSL":  ["C-1","D-14","D-15","D-16","F-7","F-8","P-11","P-14","T-12","T-13","T-8"],
    "NAB":  ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38",
             "ec2_cpu_utilization_5f5533","ec2_cpu_utilization_77c1ca",
             "ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
             "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de",
             "ec2_disk_write_bytes_c0d644","ec2_network_in_257a54",
             "ec2_network_in_5abac7","elb_request_count_8c0756",
             "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
             "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"],
    "SMAP": ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"],
}

WIN       = 224
STRIDE    = 56
K         = 5
LOOSE_PCT = 90.0
MERGE_GAP = WIN // 2
MIN_IV    = 10
VLM_SLEEP = 4.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_COLORS = ["#1f77b4","#17becf","#2ca02c","#9467bd","#8c564b"]
R_COLORS = ["#d62728","#ff7f0e","#e377c2","#bcbd22","#7f7f7f"]

DOMAIN_CTX = {
    "NAB":  "AWS cloud infrastructure metric (EC2 CPU/disk/network, RDS, ELB)",
    "SMAP": "NASA spacecraft telemetry channel (SMAP satellite sensor data)",
    "MSL":  "Mars Science Laboratory rover instrument telemetry",
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_signal(ds, sig):
    if ds == "NAB":
        csv = BASE / "data/realAWSCloudwatch" / f"{sig}.csv"
    else:
        csv = BASE / "data" / ds / f"{sig}.csv"
    df = pd.read_csv(csv)
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_gt(sig, timestamps):
    anoms = pd.read_csv(ANOMS_CSV)
    row = anoms[anoms["signal"] == sig]
    if row.empty:
        return []
    events = ast.literal_eval(row.iloc[0]["events"])
    ivs = []
    for ts_s, ts_e in events:
        i_s = int(np.searchsorted(timestamps, ts_s, side="left"))
        i_e = int(np.searchsorted(timestamps, ts_e, side="right") - 1)
        ivs.append((max(0, min(i_s, len(timestamps)-1)),
                    max(0, min(i_e, len(timestamps)-1))))
    return [(s, e) for s, e in ivs if s <= e]

# ── LP Stage1 ─────────────────────────────────────────────────────────────────
def load_lp_features(ds, sig):
    pre = CACHE_DIR / f"{ds}__{sig}"
    tr_cls     = np.load(f"{pre}__train_cls_f32.npy")
    tr_patches = np.load(f"{pre}__train_patches_f16.npy").astype(np.float32)
    te_cls     = np.load(f"{pre}__test_cls_f32.npy")
    te_patches = np.load(f"{pre}__test_patches_f16.npy").astype(np.float32)
    tr_L8  = np.load(f"{pre}__train_L8_f16.npy").astype(np.float32)
    te_L8  = np.load(f"{pre}__test_L8_f16.npy").astype(np.float32)
    tr_L11 = np.load(f"{pre}__train_L11_f16.npy").astype(np.float32)
    te_L11 = np.load(f"{pre}__test_L11_f16.npy").astype(np.float32)
    return tr_cls, tr_patches, te_cls, te_patches, tr_L8, te_L8, tr_L11, te_L11

def _knn_dist(query_flat, bank_flat, k=K):
    q = torch.tensor(query_flat, dtype=torch.float32, device=DEVICE)
    b = torch.tensor(bank_flat,  dtype=torch.float32, device=DEVICE)
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    result = []
    PBATCH = 256
    for i in range(0, len(q), PBATCH):
        d = 1.0 - (q[i:i+PBATCH] @ b.T)
        result.append(torch.topk(d, k, dim=1, largest=False).values.mean(1).cpu().numpy())
    return np.concatenate(result)

def _residual(patches, cls):
    cls_n = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    cls_exp = cls_n[:, None, :]
    dot = (patches * cls_exp).sum(axis=-1, keepdims=True)
    return patches - dot * cls_exp

def compute_ml_topk10(tr_cls, tr_p, te_cls, te_p, tr_L8, te_L8, tr_L11, te_L11):
    tr_ml = np.concatenate([tr_L8, tr_L11], axis=-1)
    te_ml = np.concatenate([te_L8, te_L11], axis=-1)
    D_ml = tr_ml.shape[-1]
    tr_cls_ml = np.concatenate([tr_cls, tr_cls], axis=-1)
    te_cls_ml = np.concatenate([te_cls, te_cls], axis=-1)
    tr_r = _residual(tr_ml, tr_cls_ml)
    te_r = _residual(te_ml, te_cls_ml)
    N_te, P, _ = te_r.shape
    dist = _knn_dist(te_r.reshape(-1, D_ml), tr_r.reshape(-1, D_ml))
    knn = dist.reshape(N_te, P)
    k10 = max(1, int(P * 0.10))
    return np.sort(knn, axis=1)[:, -k10:].mean(axis=1)  # (N_te,)

def stage1_lp(win_scores, split, T):
    """LP candidates in FULL SIGNAL coordinates."""
    T_test = T - split
    thr = float(np.percentile(win_scores, LOOSE_PCT))
    binary = np.zeros(T_test, dtype=int)
    for i, s in enumerate(range(0, T_test - WIN + 1, STRIDE)):
        if win_scores[i] >= thr:
            binary[s:min(s+WIN, T_test)] = 1
    raw = []
    in_seg, ss = False, 0
    for i, v in enumerate(binary):
        if v and not in_seg: ss, in_seg = i, True
        elif not v and in_seg: raw.append((ss, i-1)); in_seg = False
    if in_seg: raw.append((ss, T_test-1))
    merged = []
    for iv in raw:
        if merged and iv[0]-merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(list(iv))
    # Convert to full signal coordinates (+split)
    return [(s+split, e+split) for s, e in merged if e-s+1 >= MIN_IV]

def build_score_ts(win_scores, split, T):
    """Build full-signal score time series (zeros for train, LP scores for test)."""
    T_test = T - split
    score_ts = np.zeros(T)
    count_ts = np.zeros(T)
    for i, s in enumerate(range(0, T_test - WIN + 1, STRIDE)):
        fs = split + s
        fe = min(fs + WIN, T)
        score_ts[fs:fe] += win_scores[i]
        count_ts[fs:fe] += 1
    with np.errstate(invalid='ignore'):
        return np.where(count_ts > 0, score_ts / count_ts, 0.0)

# ── Interval utils (same as v5.2) ─────────────────────────────────────────────
def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

def get_intervals(binary):
    ivs, in_seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg: s, in_seg = i, True
        elif not v and in_seg: ivs.append((s, i-1)); in_seg = False
    if in_seg: ivs.append((s, len(binary)-1))
    return ivs

def interval_f1(gt_ivs, pred_ivs):
    if not gt_ivs: return 0., 0., 0.
    TP_p = sum(1 for d in pred_ivs if any(_ov(d,g) for g in gt_ivs))
    TP_g = sum(1 for g in gt_ivs  if any(_ov(g,d) for d in pred_ivs))
    FP   = sum(1 for d in pred_ivs if not any(_ov(d,g) for g in gt_ivs))
    FN   = sum(1 for g in gt_ivs  if not any(_ov(g,d) for d in pred_ivs))
    p = TP_p/(TP_p+FP) if (TP_p+FP) else 0.
    r = TP_g/(TP_g+FN) if (TP_g+FN) else 0.
    return (2*p*r/(p+r) if p+r else 0.), p, r

def is_tp(cand, gt_ivs):
    return any(_ov(cand, g) for g in gt_ivs)

# ── Score helpers ─────────────────────────────────────────────────────────────
def candidate_peak_score(score_ts_raw, cand):
    s0, e0 = cand
    L = max(e0-s0+1, 1)
    smooth = uniform_filter1d(score_ts_raw, size=max(5, L//4))
    seg = smooth[s0:e0+1]
    return float(np.max(seg)) if len(seg) else 0.

def generate_options(score_ts_raw, cand, T):
    """Boundary options based on score time series (accepts prebuilt score_ts)."""
    s0, e0 = cand
    L = max(e0-s0+1, 1)
    margin = max(3*L, 100)
    smooth = uniform_filter1d(score_ts_raw, size=max(5, L//4))

    inner = smooth[s0:e0+1]
    tau_h = float(np.percentile(inner, 50)) if len(inner) else float(smooth[s0])
    tau_l = float(np.percentile(inner, 25)) if len(inner) else float(smooth[s0]*0.5)

    left = [("original", s0)]
    for t in range(s0, max(0, s0-margin)-1, -1):
        if smooth[t] < tau_h: left.append(("high_cross", max(0, t))); break
    for t in range(s0, max(0, s0-margin)-1, -1):
        if smooth[t] < tau_l: left.append(("low_cross", max(0, t+1))); break
    lr = smooth[max(0, s0-margin):s0+1]
    if len(lr) > 3:
        lmins = argrelmin(lr, order=max(1, len(lr)//10))[0]
        if len(lmins):
            left.append(("local_min", int(np.clip(max(0,s0-margin)+lmins[-1], 0, s0))))
    deriv = np.diff(smooth)
    dl = deriv[max(0,s0-margin):s0]
    if len(dl):
        idx = int(np.argmax(dl))
        left.append(("deriv_rise", int(np.clip(max(0,s0-margin)+idx, 0, s0))))

    right = [("original", e0)]
    for t in range(e0, min(T, e0+margin+1)):
        if smooth[t] < tau_h: right.append(("high_cross", min(T-1, t))); break
    for t in range(e0, min(T, e0+margin+1)):
        if smooth[t] < tau_l: right.append(("low_cross", min(T-1, t-1))); break
    rr = smooth[e0:min(T, e0+margin+1)]
    if len(rr) > 3:
        rmins = argrelmin(rr, order=max(1, len(rr)//10))[0]
        if len(rmins):
            right.append(("local_min", int(np.clip(e0+rmins[0], e0, T-1))))
    dr = deriv[e0:min(T-1, e0+margin)]
    if len(dr):
        idx = int(np.argmin(dr))
        right.append(("deriv_fall", int(np.clip(e0+idx, e0, T-1))))

    seen_l, left_dedup = set(), []
    for nm, t in left:
        t = int(np.clip(t, 0, s0))
        if t not in seen_l: seen_l.add(t); left_dedup.append((nm, t))
    seen_r, right_dedup = set(), []
    for nm, t in right:
        t = int(np.clip(t, e0, T-1))
        if t not in seen_r: seen_r.add(t); right_dedup.append((nm, t))

    return left_dedup, right_dedup, smooth

def oracle_select(left_opts, right_opts, cand, others, gt_ivs):
    best_f1, best_li, best_ri, best_iv = -1., 0, 0, cand
    for li, (_, l) in enumerate(left_opts):
        for ri, (_, r) in enumerate(right_opts):
            if l > r: continue
            iv = (l, r)
            f1, _, _ = interval_f1(gt_ivs, list(others)+[iv])
            if f1 > best_f1:
                best_f1, best_li, best_ri, best_iv = f1, li, ri, iv
    return best_li, best_ri, best_iv, best_f1

def make_summary(vals, smooth, cand, T):
    s0, e0 = cand; L = e0-s0+1
    margin = max(3*L, 100)
    pre  = vals[max(0,s0-margin):max(0,s0)]
    ins  = vals[s0:e0+1]
    post = vals[min(T,e0+1):min(T,e0+1+margin)]
    sc   = smooth[s0:e0+1]
    def ss(arr): return (float(np.mean(arr)), float(np.std(arr))) if len(arr) else (0.,0.)
    pm,ps = ss(pre); im,ist = ss(ins); pom,pos = ss(post)
    pk = float(sc.max()) if len(sc) else 0.
    mn = float(sc.mean()) if len(sc) else 0.
    pct = float(np.mean(smooth <= pk)*100)
    return {"interval":[s0,e0],"length":int(L),"peak_score":round(pk,4),"mean_score":round(mn,4),
            "score_pct":round(pct,1),"pre_mean":round(pm,4),"pre_std":round(ps,4),
            "inside_mean":round(im,4),"inside_std":round(ist,4),
            "post_mean":round(pom,4),"post_std":round(pos,4)}

# ── Visualization ─────────────────────────────────────────────────────────────
def _img_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _draw_opts(ax, left_opts, right_opts, ymin, ymax):
    for li,(nm,t) in enumerate(left_opts):
        c=L_COLORS[li%len(L_COLORS)]; ls="-" if li==0 else "--"
        ax.axvline(t,color=c,ls=ls,lw=1.8 if li==0 else 1.0,alpha=0.85)
        ax.text(t,ymax-0.06*(ymax-ymin)*(li+1),f"L{li}",color=c,fontsize=6,ha="center",fontweight="bold")
    for ri,(nm,t) in enumerate(right_opts):
        c=R_COLORS[ri%len(R_COLORS)]; ls="-" if ri==0 else "--"
        ax.axvline(t,color=c,ls=ls,lw=1.8 if ri==0 else 1.0,alpha=0.85)
        ax.text(t,ymin+0.06*(ymax-ymin)*(ri+1),f"R{ri}",color=c,fontsize=6,ha="center",fontweight="bold")

def make_images_with_opts(vals, smooth, cand, left_opts, right_opts, cid, T):
    s0, e0 = cand; L = max(e0-s0+1,1); margin = max(3*L,150)
    zs, ze = max(0, s0-margin), min(T-1, e0+margin)
    def yr(arr,p=0.08):
        mn,mx=float(arr.min()),float(arr.max()); pad=(mx-mn)*p or 0.1
        return mn-pad,mx+pad
    fig,ax=plt.subplots(figsize=(12,3))
    ax.plot(vals,color="#333",lw=0.5,alpha=0.8); ax.axvspan(s0,e0,color="salmon",alpha=0.4)
    ym,yM=yr(vals); _draw_opts(ax,left_opts,right_opts,ym,yM)
    ax.set_xlim(0,T-1); ax.set_ylim(ym,yM)
    patches=([mpatches.Patch(color=L_COLORS[i],label=f"L{i}:{nm}") for i,(nm,_) in enumerate(left_opts)]+
             [mpatches.Patch(color=R_COLORS[i],label=f"R{i}:{nm}") for i,(nm,_) in enumerate(right_opts)])
    ax.legend(handles=patches,fontsize=5,ncol=4,loc="upper right")
    ax.set_title(f"Panel 1 — Global (T={T}) | Cand#{cid} [{s0},{e0}]",fontsize=8)
    img1=_img_b64(fig)
    fig,ax=plt.subplots(figsize=(10,3.5))
    ax.plot(np.arange(zs,ze+1),vals[zs:ze+1],color="#333",lw=1.1)
    ax.axvspan(s0,e0,color="salmon",alpha=0.35); ym,yM=yr(vals[zs:ze+1])
    _draw_opts(ax,left_opts,right_opts,ym,yM); ax.set_xlim(zs,ze); ax.set_ylim(ym,yM)
    opt_ticks=sorted(set([t for _,t in left_opts]+[t for _,t in right_opts]))
    ax.set_xticks(opt_ticks); ax.tick_params(axis='x',labelsize=6,rotation=45)
    ax.set_title(f"Panel 2 — Local Zoom [{zs},{ze}]",fontsize=8); img2=_img_b64(fig)
    fig,ax=plt.subplots(figsize=(10,2.5))
    ax.plot(np.arange(zs,ze+1),smooth[zs:ze+1],color="#e67e22",lw=1.2)
    ax.axvspan(s0,e0,color="salmon",alpha=0.2); ym,yM=yr(smooth[zs:ze+1])
    _draw_opts(ax,left_opts,right_opts,ym,yM); ax.set_xlim(zs,ze)
    ax.set_xticks(opt_ticks); ax.tick_params(axis='x',labelsize=6,rotation=45)
    ax.set_title(f"Panel 3 — LP Score Curve (local)",fontsize=8); img3=_img_b64(fig)
    return img1,img2,img3

def make_images_verify(vals, cand, cid, T):
    s0,e0=cand; L=max(e0-s0+1,1); margin=max(3*L,150)
    zs,ze=max(0,s0-margin),min(T-1,e0+margin)
    def yr(arr,p=0.08):
        mn,mx=float(arr.min()),float(arr.max()); pad=(mx-mn)*p or 0.1; return mn-pad,mx+pad
    fig,ax=plt.subplots(figsize=(12,3))
    ax.plot(vals,color="#333",lw=0.5,alpha=0.8)
    ax.axvspan(s0,e0,color="salmon",alpha=0.4,label=f"Cand#{cid}[{s0},{e0}]")
    ym,yM=yr(vals); ax.set_xlim(0,T-1); ax.set_ylim(ym,yM)
    ax.legend(fontsize=7,loc="upper right")
    ax.set_title(f"Panel 1 — Global (T={T}) | Cand#{cid} [{s0},{e0}]",fontsize=8); img1=_img_b64(fig)
    fig,ax=plt.subplots(figsize=(10,3.5))
    ax.plot(np.arange(zs,ze+1),vals[zs:ze+1],color="#333",lw=1.1)
    ax.axvspan(s0,e0,color="salmon",alpha=0.35,label=f"Cand#{cid}")
    ym,yM=yr(vals[zs:ze+1]); ax.set_xlim(zs,ze); ax.set_ylim(ym,yM)
    ax.axvline(s0,color="blue",ls="--",lw=1.2,label=f"start={s0}")
    ax.axvline(e0,color="red",ls="--",lw=1.2,label=f"end={e0}")
    ax.legend(fontsize=6); ax.set_title(f"Panel 2 — Local Zoom [{zs},{ze}]",fontsize=8); img2=_img_b64(fig)
    return img1,img2

# ── VLM calls (identical to v5.2) ─────────────────────────────────────────────
def _parse_json(raw, keys):
    raw = re.sub(r"```(?:json)?","",raw).strip().strip("`").strip()
    try: return json.loads(raw)
    except: pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    result = {}
    for k in keys:
        m2 = re.search(rf'"{k}"\s*:\s*("[\w_]+"|\d+)', raw)
        if m2:
            v = m2.group(1).strip('"')
            result[k] = int(v) if v.isdigit() else v
    return result if result else None

def _vlm_call(system, content, tries=4):
    for attempt in range(tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":system},
                          {"role":"user","content":content}],
                temperature=0.0, max_tokens=80,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc).lower()
            wait = (attempt+1)*30 if ("rate_limit" in err or "429" in err) else 5
            time.sleep(wait)
    return None

def _img_content(b64):
    return {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}","detail":"high"}}

SYS_BOUNDARY = (
    "You are a time-series anomaly detection assistant. "
    "The candidate interval is already confirmed as anomalous. "
    "Your task is ONLY to select the best left and right temporal boundary "
    "from the pre-computed options. Do not invent coordinates outside the list."
)

def prompt_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx):
    s0,e0=cand
    ll="\n".join(f"  L{i}: {nm} = index {t}" for i,(nm,t) in enumerate(left_opts))
    rl="\n".join(f"  R{i}: {nm} = index {t}" for i,(nm,t) in enumerate(right_opts))
    nu="\n".join(f"  {k}: {v}" for k,v in summary.items())
    return (f"Domain: {ds_ctx}\nCandidate #{cid}: [{s0}, {e0}]\n\n"
            f"=== LEFT BOUNDARY OPTIONS ===\n{ll}\n\n=== RIGHT BOUNDARY OPTIONS ===\n{rl}\n\n"
            f"=== NUMERICAL SUMMARY ===\n{nu}\n\n"
            f"The candidate is anomalous. Select which boundary options best capture its true extent.\n"
            f"L0 and R0 are the Stage-1 boundaries. Other options may extend the boundary.\n"
            f"Images show: [1] global series [2] local zoom [3] LP score curve.\n"
            f"Boundary options are labeled L0-L{len(left_opts)-1} (blue) and R0-R{len(right_opts)-1} (red).\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{\"left_option\": 0, \"right_option\": 0}}\n'
            f"Integers in range [0,{len(left_opts)-1}] and [0,{len(right_opts)-1}].")

def call_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx, imgs):
    text = prompt_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx)
    content = [{"type":"text","text":text}]+[_img_content(b) for b in imgs]
    raw = _vlm_call(SYS_BOUNDARY, content)
    if raw is None: return None
    out = _parse_json(raw, ["left_option","right_option"])
    if out is None: return None
    return {"left_option":  int(np.clip(int(out.get("left_option",0)),  0, len(left_opts)-1)),
            "right_option": int(np.clip(int(out.get("right_option",0)), 0, len(right_opts)-1))}

SYS_VERIFY = (
    "You are a false-positive rejection filter for a time-series anomaly detector. "
    "The candidate was proposed by a high-recall visual detector. "
    "Your role is ONLY to reject clear false positives. "
    "When in doubt, choose 'uncertain'. Do not be aggressive in discarding."
)

def prompt_verify(cid, cand, summary, ds_ctx):
    s0,e0=cand
    nu="\n".join(f"  {k}: {v}" for k,v in summary.items())
    return (f"Domain: {ds_ctx}\nCandidate #{cid}: [{s0}, {e0}]\n\n"
            f"=== NUMERICAL SUMMARY ===\n{nu}\n\n"
            f"Images show: [1] global series with candidate highlighted (red), "
            f"[2] local zoom around the candidate.\n\n"
            f"=== YOUR TASK ===\n"
            f"Decide whether this candidate is:\n"
            f"  keep     — shows a genuine anomaly (spike, level shift, trend change)\n"
            f"  discard  — clearly a normal fluctuation consistent with the rest of the series\n"
            f"  uncertain — ambiguous, domain-specific, or cannot be confidently rejected\n\n"
            f"IMPORTANT: discard ONLY if the evidence strongly shows normal behavior. "
            f"If the candidate shows ANY unusual pattern, unusual magnitude, or domain-specific signal, "
            f"choose 'keep' or 'uncertain'.\n\n"
            f"Return ONLY valid JSON:\n{{\"decision\": \"keep\"}}\n"
            f'Valid values: "keep", "discard", "uncertain"')

def call_verify(cid, cand, summary, ds_ctx, imgs):
    text = prompt_verify(cid, cand, summary, ds_ctx)
    content = [{"type":"text","text":text}]+[_img_content(b) for b in imgs]
    raw = _vlm_call(SYS_VERIFY, content)
    if raw is None: return "uncertain"
    out = _parse_json(raw, ["decision"])
    if out is None: return "uncertain"
    dec = str(out.get("decision","uncertain")).lower().strip().strip('"')
    return dec if dec in ("keep","discard","uncertain") else "uncertain"

# ── Per-signal runner ─────────────────────────────────────────────────────────
def run_signal(ds, sig):
    timestamps, vals = load_signal(ds, sig)
    T = len(timestamps)
    split = T // 2
    gt_ivs = load_gt(sig, timestamps)
    ds_ctx = DOMAIN_CTX.get(ds, ds)

    # LP Stage1
    tr_cls, tr_p, te_cls, te_p, tr_L8, te_L8, tr_L11, te_L11 = load_lp_features(ds, sig)
    win_scores = compute_ml_topk10(tr_cls, tr_p, te_cls, te_p, tr_L8, te_L8, tr_L11, te_L11)
    candidates = stage1_lp(win_scores, split, T)
    score_ts = build_score_ts(win_scores, split, T)

    f1_s1, p_s1, r_s1 = interval_f1(gt_ivs, candidates)
    print(f"  Stage1(LP): {len(candidates)} candidates, F1={f1_s1:.4f}", flush=True)

    if not candidates:
        return {"ds":ds,"sig":sig,"T":T,"n_gt":len(gt_ivs),"n_s1":0,
                "f1_s1":0.,"f1_out":0.,"f1_oracle":0.,"oracle_gap":0.,
                "gap_recovery":float("nan"),"n_kept":0,"n_api":0,"per_candidate":[]}

    peak_scores = [candidate_peak_score(score_ts, c) for c in candidates]
    top1_idx = int(np.argmax(peak_scores))
    n_cands = len(candidates)
    print(f"  n={n_cands} top1=cid{top1_idx}(peak={peak_scores[top1_idx]:.4f})", flush=True)

    per_cand, kept_ivs, api_calls = [], [], 0

    for cid, cand in enumerate(candidates):
        s0, e0 = cand
        others = [c for j,c in enumerate(candidates) if j!=cid]
        left_opts, right_opts, smooth_c = generate_options(score_ts, cand, T)
        summary = make_summary(vals, smooth_c, cand, T)
        oracle_li, oracle_ri, oracle_iv, _ = oracle_select(left_opts, right_opts, cand, others, gt_ivs)

        # v52c: n<=2 → Mode A; else top1 → Mode A, rest → Mode C
        cand_mode = "A" if (n_cands <= 2 or cid == top1_idx) else "C"

        pc = {"cid":cid,"orig":list(cand),
              "left_opts":[[nm,int(t)] for nm,t in left_opts],
              "right_opts":[[nm,int(t)] for nm,t in right_opts],
              "n_left":len(left_opts),"n_right":len(right_opts),
              "oracle_li":oracle_li,"oracle_ri":oracle_ri,"oracle_iv":list(oracle_iv),
              "is_tp":is_tp(cand,gt_ivs),"decision":"keep","final_iv":list(cand),
              "vlm_li":0,"vlm_ri":0,"match_oracle_l":False,"match_oracle_r":False,
              "cand_mode":cand_mode,"is_top1":cid==top1_idx,"peak_score":round(peak_scores[cid],4)}

        if cand_mode == "A":
            imgs = make_images_with_opts(vals, smooth_c, cand, left_opts, right_opts, cid, T)
            out = call_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx, imgs)
            api_calls += 1
            li, ri = (out["left_option"], out["right_option"]) if out else (0, 0)
            final_iv = (left_opts[li][1], right_opts[ri][1])
            pc.update({"decision":"keep","vlm_li":li,"vlm_ri":ri,"final_iv":list(final_iv),
                       "match_oracle_l":li==oracle_li,"match_oracle_r":ri==oracle_ri})
            kept_ivs.append(final_iv)
        else:
            imgs_v = make_images_verify(vals, cand, cid, T)
            dec = call_verify(cid, cand, summary, ds_ctx, imgs_v)
            api_calls += 1
            if dec == "discard":
                pc.update({"decision":"discard","final_iv":list(cand)})
            else:
                imgs_b = make_images_with_opts(vals, smooth_c, cand, left_opts, right_opts, cid, T)
                out = call_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx, imgs_b)
                api_calls += 1
                li, ri = (out["left_option"], out["right_option"]) if out else (0, 0)
                final_iv = (left_opts[li][1], right_opts[ri][1])
                pc.update({"decision":dec,"vlm_li":li,"vlm_ri":ri,"final_iv":list(final_iv),
                           "match_oracle_l":li==oracle_li,"match_oracle_r":ri==oracle_ri})
                kept_ivs.append(final_iv)

        per_cand.append(pc)
        status = pc["decision"].upper()[:4]
        print(f"    C{cid}[{cand_mode}][{s0},{e0}] {status} "
              f"L{pc['vlm_li']} R{pc['vlm_ri']} oracle=L{oracle_li}R{oracle_ri} "
              f"tp={pc['is_tp']} pk={pc['peak_score']:.3f}", flush=True)

    f1_out, p_out, r_out = interval_f1(gt_ivs, kept_ivs)
    oracle_ivs = [pc["oracle_iv"] for pc in per_cand]
    f1_oracle, _, _ = interval_f1(gt_ivs, oracle_ivs)
    gap = f1_oracle - f1_s1
    recovery = (f1_out - f1_s1) / gap if abs(gap) > 1e-4 else float("nan")

    n_prot  = sum(1 for pc in per_cand if pc["cand_mode"]=="A")
    n_verif = sum(1 for pc in per_cand if pc["cand_mode"]=="C")
    n_disc  = sum(1 for pc in per_cand if pc["cand_mode"]=="C" and pc["decision"]=="discard")

    print(f"  OUT: F1={f1_out:.4f} (S1={f1_s1:.4f} Oracle={f1_oracle:.4f} "
          f"Rec={recovery:.1%}) kept={len(kept_ivs)}/{n_cands} "
          f"prot={n_prot} verif={n_verif}(disc={n_disc}) api={api_calls}", flush=True)

    return {"ds":ds,"sig":sig,"T":T,"n_gt":len(gt_ivs),"n_s1":n_cands,
            "f1_s1":f1_s1,"p_s1":p_s1,"r_s1":r_s1,
            "f1_out":f1_out,"p_out":p_out,"r_out":r_out,
            "f1_oracle":f1_oracle,"oracle_gap":gap,"gap_recovery":recovery,
            "n_kept":len(kept_ivs),"n_api":api_calls,
            "n_prot":n_prot,"n_verif":n_verif,"n_verif_disc":n_disc,
            "s1_ivs":[list(c) for c in candidates],
            "out_ivs":[list(iv) for iv in kept_ivs],
            "oracle_ivs":oracle_ivs,"per_candidate":per_cand}

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL = OUT_DIR / "partial_results.jsonl"
    LOG     = OUT_DIR / "run.log"

    log_f = open(LOG, "a", encoding="utf-8", errors="replace")
    _print_orig = print
    import builtins
    def _tee(*a, **kw):
        _print_orig(*a, **kw)
        kw.pop("file", None)
        _print_orig(*a, file=log_f, **kw); log_f.flush()
    builtins.print = _tee

    done = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line); done[f"{r['ds']}__{r['sig']}"] = r
            except: pass
    print(f"Stage2 LP v1 (ml_L8_11_resid_topk10 → v52c)  done={len(done)}", flush=True)
    print(f"Device: {DEVICE}  Cache: {CACHE_DIR}", flush=True)
    print("NOTE: 3 NAB signals (825cc2, 257a54, iio) have no GT in test portion — F1 capped at 0", flush=True)

    all_results = list(done.values())
    for ds, sigs in DATASETS.items():
        for sig in sigs:
            key = f"{ds}__{sig}"
            if key in done:
                print(f"  [SKIP] {ds}/{sig}", flush=True); continue
            print(f"\n  [{ds}] {sig}", flush=True)
            try:
                r = run_signal(ds, sig)
                all_results.append(r)
                with open(PARTIAL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
            except Exception as exc:
                import traceback
                print(f"  [ERROR] {exc}", flush=True); traceback.print_exc()

    # Summary
    ok = [r for r in all_results if r.get("n_gt", 0) > 0]
    print(f"\n{'='*80}", flush=True)
    print("Stage2 LP v1 — RESULTS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    for ds in ["NAB","SMAP","MSL"]:
        rs = [r for r in ok if r["ds"]==ds]
        if not rs: continue
        avg_s1  = sum(r["f1_s1"]  for r in rs)/len(rs)
        avg_out = sum(r["f1_out"] for r in rs)/len(rs)
        avg_or  = sum(r["f1_oracle"] for r in rs)/len(rs)
        print(f"  {ds}: S1={avg_s1:.4f}  OUT={avg_out:.4f}  Oracle={avg_or:.4f}", flush=True)

    all_s1  = sum(r["f1_s1"]  for r in ok)/len(ok)
    all_out = sum(r["f1_out"] for r in ok)/len(ok)
    all_or  = sum(r["f1_oracle"] for r in ok)/len(ok)
    lp_s1_ref = 0.6553
    # Use per-experiment LP oracle (not LTR oracle 0.6817)
    gap_rec = (all_out - lp_s1_ref) / (all_or - lp_s1_ref) if abs(all_or - lp_s1_ref) > 1e-4 else float("nan")

    print(f"\n  ALL ({len(ok)} signals):", flush=True)
    print(f"    LP Stage1 F1 = {all_s1:.4f}  (ref: {lp_s1_ref:.4f})", flush=True)
    print(f"    Output F1    = {all_out:.4f}  ({all_out-lp_s1_ref:+.4f} vs LP Stage1)", flush=True)
    print(f"    Oracle F1    = {all_or:.4f}", flush=True)
    print(f"    Gap recovery = {gap_rec:.1%}", flush=True)
    print(f"    Reference: LTR Stage1=0.6174  v52c=0.6790", flush=True)
    print(f"    Total API    = {sum(r.get('n_api',0) for r in ok)}", flush=True)
    log_f.close()
    print(f"\nSaved → {OUT_DIR}", flush=True)
