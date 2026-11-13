"""
Stage2 Univariate v5.2 — Selective Verification
=================================================
Selective verification: VLM verification applied only to candidates
deemed safe to reject. High-confidence candidates are protected.

Sub-modes:
  v52a  — per-signal:     n_cands <= 2 → Mode A (boundary-only); else → Mode C (two-call)
  v52b  — per-candidate:  top-1 by DINOv2 peak → Mode A;  others → Mode C
  v52c  — combined:       n<=2 → all A;  else top-1→A, others→C   [RECOMMENDED]

v5.2-d (oracle mode selector) is computed offline from v5.1 results — no new API calls.

Usage:
  python experiment_stage2_univar_v5_2.py --mode v52a
  python experiment_stage2_univar_v5_2.py --mode v52b
  python experiment_stage2_univar_v5_2.py --mode v52c
"""

import argparse, ast, base64, io, json, os, pickle, re, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin

warnings.filterwarnings("ignore")
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment.")
_client = OpenAI(api_key=API_KEY)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
DINO_DIR  = BASE / "results/VLM4TS_results_dino_ltr/checkpoints"
ANOMS_CSV = BASE / "data/anomalies.csv"

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
LOOSE_PCT = 10.0
MERGE_GAP = WIN // 2
MIN_IV    = 10
VLM_SLEEP = 4.0

L_COLORS = ["#1f77b4","#17becf","#2ca02c","#9467bd","#8c564b"]
R_COLORS = ["#d62728","#ff7f0e","#e377c2","#bcbd22","#7f7f7f"]

# ── Data loading ──────────────────────────────────────────────────────────────
def _sig_path(ds, sig):
    if ds == "NAB":
        return BASE / "data/realAWSCloudwatch" / f"{sig}.csv"
    return BASE / "data" / ds / f"{sig}.csv"

def load_signal(ds, sig):
    df = pd.read_csv(_sig_path(ds, sig))
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_dino(ds, sig):
    with open(DINO_DIR / f"{ds}__{sig}__dino_k5.pkl", "rb") as f:
        return pickle.load(f)["scores"]

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

# ── Interval utils ────────────────────────────────────────────────────────────
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

# ── Stage 1 ───────────────────────────────────────────────────────────────────
def stage1(scores):
    T = len(scores)
    all_ws = np.array([scores[s:s+WIN].mean()
                       for s in range(0, T-WIN+1, STRIDE)])
    thr = float(np.percentile(all_ws, 100 - LOOSE_PCT))
    binary = np.zeros(T, dtype=int)
    for i, s in enumerate(range(0, T-WIN+1, STRIDE)):
        if all_ws[i] >= thr: binary[s:s+WIN] = 1
    raw = get_intervals(binary)
    merged = []
    for iv in raw:
        if merged and iv[0]-merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], iv[1])
        else: merged.append(list(iv))
    return [(s,e) for s,e in merged if e-s+1 >= MIN_IV], all_ws

# ── Score computation ─────────────────────────────────────────────────────────
def compute_base_smooth(all_ws, T):
    """Build averaged score time series (no smoothing)."""
    score_ts = np.zeros(T)
    count_ts = np.zeros(T)
    for i, s in enumerate(range(0, T-WIN+1, STRIDE)):
        if i < len(all_ws):
            score_ts[s:s+WIN] += all_ws[i]
            count_ts[s:s+WIN] += 1
    with np.errstate(invalid='ignore'):
        return np.where(count_ts > 0, score_ts/count_ts, 0.)

def candidate_peak_score(score_ts_raw, cand):
    """Peak DINOv2 score inside a candidate interval (using L-adaptive smooth)."""
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    smooth = uniform_filter1d(score_ts_raw, size=max(5, L//4))
    seg = smooth[s0:e0+1]
    return float(np.max(seg)) if len(seg) else 0.

# ── Boundary options (L-adaptive, matches oracle_boundary_analysis.py) ────────
def generate_options(all_ws, cand, T):
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    margin = max(3*L, 100)

    score_ts = np.zeros(T)
    count_ts = np.zeros(T)
    for i, s in enumerate(range(0, T-WIN+1, STRIDE)):
        if i < len(all_ws):
            score_ts[s:s+WIN] += all_ws[i]
            count_ts[s:s+WIN] += 1
    with np.errstate(invalid='ignore'):
        score_ts = np.where(count_ts > 0, score_ts/count_ts, 0.)
    smooth = uniform_filter1d(score_ts, size=max(5, L//4))

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

# ── Oracle ────────────────────────────────────────────────────────────────────
def oracle_select(left_opts, right_opts, cand, others, gt_ivs):
    best_f1, best_li, best_ri, best_iv = -1., 0, 0, cand
    for li, (_, l) in enumerate(left_opts):
        for ri, (_, r) in enumerate(right_opts):
            if l > r: continue
            iv = (l, r)
            f1, _, _ = interval_f1(gt_ivs, list(others) + [iv])
            if f1 > best_f1:
                best_f1, best_li, best_ri, best_iv = f1, li, ri, iv
    return best_li, best_ri, best_iv, best_f1

def is_tp(cand, gt_ivs):
    return any(_ov(cand, g) for g in gt_ivs)

# ── Visualization ─────────────────────────────────────────────────────────────
def _img_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _draw_opts(ax, left_opts, right_opts, ymin, ymax):
    for li, (nm, t) in enumerate(left_opts):
        c = L_COLORS[li % len(L_COLORS)]; ls = "-" if li==0 else "--"
        ax.axvline(t, color=c, ls=ls, lw=1.8 if li==0 else 1.0, alpha=0.85)
        ax.text(t, ymax-0.06*(ymax-ymin)*(li+1), f"L{li}", color=c,
                fontsize=6, ha="center", fontweight="bold")
    for ri, (nm, t) in enumerate(right_opts):
        c = R_COLORS[ri % len(R_COLORS)]; ls = "-" if ri==0 else "--"
        ax.axvline(t, color=c, ls=ls, lw=1.8 if ri==0 else 1.0, alpha=0.85)
        ax.text(t, ymin+0.06*(ymax-ymin)*(ri+1), f"R{ri}", color=c,
                fontsize=6, ha="center", fontweight="bold")

def make_images_with_opts(vals, smooth, cand, left_opts, right_opts, cid, T):
    s0, e0 = cand
    L = max(e0-s0+1, 1)
    margin = max(3*L, 150)
    zs, ze = max(0, s0-margin), min(T-1, e0+margin)

    def yrange(arr, pad_frac=0.08):
        mn, mx = float(arr.min()), float(arr.max())
        pad = (mx-mn)*pad_frac or 0.1
        return mn-pad, mx+pad

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(vals, color="#333", lw=0.5, alpha=0.8)
    ax.axvspan(s0, e0, color="salmon", alpha=0.4)
    ym, yM = yrange(vals)
    _draw_opts(ax, left_opts, right_opts, ym, yM)
    ax.set_xlim(0, T-1); ax.set_ylim(ym, yM)
    patches = ([mpatches.Patch(color=L_COLORS[i], label=f"L{i}:{nm}")
                for i,(nm,_) in enumerate(left_opts)] +
               [mpatches.Patch(color=R_COLORS[i], label=f"R{i}:{nm}")
                for i,(nm,_) in enumerate(right_opts)])
    ax.legend(handles=patches, fontsize=5, ncol=4, loc="upper right")
    ax.set_title(f"Panel 1 — Global (T={T}) | Cand#{cid} [{s0},{e0}]", fontsize=8)
    img1 = _img_b64(fig)

    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(np.arange(zs,ze+1), vals[zs:ze+1], color="#333", lw=1.1)
    ax.axvspan(s0, e0, color="salmon", alpha=0.35)
    ym, yM = yrange(vals[zs:ze+1])
    _draw_opts(ax, left_opts, right_opts, ym, yM)
    ax.set_xlim(zs, ze); ax.set_ylim(ym, yM)
    opt_ticks = sorted(set([t for _,t in left_opts]+[t for _,t in right_opts]))
    ax.set_xticks(opt_ticks); ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.set_title(f"Panel 2 — Local Zoom [{zs},{ze}]", fontsize=8)
    img2 = _img_b64(fig)

    fig, ax = plt.subplots(figsize=(10,2.5))
    ax.plot(np.arange(zs,ze+1), smooth[zs:ze+1], color="#e67e22", lw=1.2)
    ax.axvspan(s0, e0, color="salmon", alpha=0.2)
    ym, yM = yrange(smooth[zs:ze+1])
    _draw_opts(ax, left_opts, right_opts, ym, yM)
    ax.set_xlim(zs, ze)
    ax.set_xticks(opt_ticks); ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.set_title(f"Panel 3 — DINOv2 Score Curve (local)", fontsize=8)
    img3 = _img_b64(fig)

    return img1, img2, img3

def make_images_verify(vals, cand, cid, T):
    s0, e0 = cand
    L = max(e0-s0+1, 1)
    margin = max(3*L, 150)
    zs, ze = max(0, s0-margin), min(T-1, e0+margin)

    def yrange(arr, pad_frac=0.08):
        mn, mx = float(arr.min()), float(arr.max())
        pad = (mx-mn)*pad_frac or 0.1
        return mn-pad, mx+pad

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(vals, color="#333", lw=0.5, alpha=0.8)
    ax.axvspan(s0, e0, color="salmon", alpha=0.4, label=f"Cand#{cid}[{s0},{e0}]")
    ym, yM = yrange(vals)
    ax.set_xlim(0, T-1); ax.set_ylim(ym, yM)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Panel 1 — Global (T={T}) | Cand#{cid} [{s0},{e0}]", fontsize=8)
    img1 = _img_b64(fig)

    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(np.arange(zs,ze+1), vals[zs:ze+1], color="#333", lw=1.1)
    ax.axvspan(s0, e0, color="salmon", alpha=0.35, label=f"Cand#{cid}")
    ym, yM = yrange(vals[zs:ze+1])
    ax.set_xlim(zs, ze); ax.set_ylim(ym, yM)
    ax.axvline(s0, color="blue", ls="--", lw=1.2, label=f"start={s0}")
    ax.axvline(e0, color="red",  ls="--", lw=1.2, label=f"end={e0}")
    ax.legend(fontsize=6)
    ax.set_title(f"Panel 2 — Local Zoom [{zs},{ze}]", fontsize=8)
    img2 = _img_b64(fig)

    return img1, img2

def make_summary(vals, smooth, cand, T):
    s0, e0 = cand
    L = e0-s0+1
    margin = max(3*L, 100)
    pre  = vals[max(0,s0-margin):max(0,s0)]
    ins  = vals[s0:e0+1]
    post = vals[min(T,e0+1):min(T,e0+1+margin)]
    sc   = smooth[s0:e0+1]

    def ss(arr): return (float(np.mean(arr)), float(np.std(arr))) if len(arr) else (0.,0.)
    pm, ps = ss(pre); im, ist = ss(ins); pom, pos = ss(post)
    pk = float(sc.max()) if len(sc) else 0.
    mn = float(sc.mean()) if len(sc) else 0.
    pct = float(np.mean(smooth <= pk)*100)

    return {"interval": [s0,e0], "length": int(L),
            "peak_score": round(pk,4), "mean_score": round(mn,4),
            "score_pct": round(pct,1),
            "pre_mean": round(pm,4), "pre_std": round(ps,4),
            "inside_mean": round(im,4), "inside_std": round(ist,4),
            "post_mean": round(pom,4), "post_std": round(pos,4)}

# ── VLM calls ─────────────────────────────────────────────────────────────────
def _parse_json(raw, keys):
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try: return json.loads(raw)
    except Exception: pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    result = {}
    for k in keys:
        pat = rf'"{k}"\s*:\s*("[\w_]+"|\d+)'
        m2 = re.search(pat, raw)
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

# ── VLM boundary selector ─────────────────────────────────────────────────────
SYS_BOUNDARY = (
    "You are a time-series anomaly detection assistant. "
    "The candidate interval is already confirmed as anomalous. "
    "Your task is ONLY to select the best left and right temporal boundary "
    "from the pre-computed options. Do not invent coordinates outside the list."
)

def prompt_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx):
    s0, e0 = cand
    ll = "\n".join(f"  L{i}: {nm} = index {t}" for i,(nm,t) in enumerate(left_opts))
    rl = "\n".join(f"  R{i}: {nm} = index {t}" for i,(nm,t) in enumerate(right_opts))
    nu = "\n".join(f"  {k}: {v}" for k,v in summary.items())
    return (
        f"Domain: {ds_ctx}\nCandidate #{cid}: [{s0}, {e0}]\n\n"
        f"=== LEFT BOUNDARY OPTIONS ===\n{ll}\n\n"
        f"=== RIGHT BOUNDARY OPTIONS ===\n{rl}\n\n"
        f"=== NUMERICAL SUMMARY ===\n{nu}\n\n"
        f"The candidate is anomalous. Select which boundary options best capture its true extent.\n"
        f"L0 and R0 are the Stage-1 boundaries. Other options may extend the boundary.\n"
        f"Images show: [1] global series [2] local zoom [3] DINOv2 score curve.\n"
        f"Boundary options are labeled L0-L{len(left_opts)-1} (blue) and R0-R{len(right_opts)-1} (red).\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{\"left_option\": 0, \"right_option\": 0}}\n'
        f"Integers in range [0,{len(left_opts)-1}] and [0,{len(right_opts)-1}]."
    )

def call_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx, imgs):
    text = prompt_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx)
    content = [{"type":"text","text":text}] + [_img_content(b) for b in imgs]
    raw = _vlm_call(SYS_BOUNDARY, content)
    if raw is None: return None
    out = _parse_json(raw, ["left_option","right_option"])
    if out is None: return None
    n_l, n_r = len(left_opts), len(right_opts)
    return {
        "left_option":  int(np.clip(int(out.get("left_option",0)),  0, n_l-1)),
        "right_option": int(np.clip(int(out.get("right_option",0)), 0, n_r-1)),
    }

# ── VLM conservative verifier ─────────────────────────────────────────────────
SYS_VERIFY = (
    "You are a false-positive rejection filter for a time-series anomaly detector. "
    "The candidate was proposed by a high-recall visual detector. "
    "Your role is ONLY to reject clear false positives. "
    "When in doubt, choose 'uncertain'. Do not be aggressive in discarding."
)

def prompt_verify(cid, cand, summary, ds_ctx):
    s0, e0 = cand
    nu = "\n".join(f"  {k}: {v}" for k,v in summary.items())
    return (
        f"Domain: {ds_ctx}\nCandidate #{cid}: [{s0}, {e0}]\n\n"
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
        f"Return ONLY valid JSON:\n"
        f'{{\"decision\": \"keep\"}}\n'
        f'Valid values: "keep", "discard", "uncertain"'
    )

def call_verify(cid, cand, summary, ds_ctx, imgs):
    text = prompt_verify(cid, cand, summary, ds_ctx)
    content = [{"type":"text","text":text}] + [_img_content(b) for b in imgs]
    raw = _vlm_call(SYS_VERIFY, content)
    if raw is None: return "uncertain"
    out = _parse_json(raw, ["decision"])
    if out is None: return "uncertain"
    dec = str(out.get("decision","uncertain")).lower().strip().strip('"')
    if dec not in ("keep","discard","uncertain"): return "uncertain"
    return dec

# ── Verification metrics ──────────────────────────────────────────────────────
def compute_verif_metrics(per_cand, gt_ivs, candidates):
    tp_cands = [i for i,c in enumerate(candidates) if is_tp(c, gt_ivs)]
    fp_cands = [i for i,c in enumerate(candidates) if not is_tp(c, gt_ivs)]
    if not per_cand: return {}

    decisions = {pc["cid"]: pc.get("decision","keep") for pc in per_cand}
    kept = {cid for cid,d in decisions.items() if d != "discard"}

    tp_kept = sum(1 for i in tp_cands if i in kept)
    fp_disc = sum(1 for i in fp_cands if i not in kept)
    tp_disc = sum(1 for i in tp_cands if i not in kept)

    return {
        "tp_retention":  tp_kept/len(tp_cands) if tp_cands else float("nan"),
        "fp_rejection":  fp_disc/len(fp_cands) if fp_cands else float("nan"),
        "false_discard": tp_disc/len(tp_cands) if tp_cands else float("nan"),
        "n_tp": len(tp_cands), "n_fp": len(fp_cands),
    }

# ── Per-candidate mode decision (v5.2 logic) ──────────────────────────────────
def select_cand_mode(cid, n_cands, top1_idx, sub_mode):
    """
    Returns 'A' (boundary-only, protected) or 'C' (verify+boundary) for candidate cid.
    sub_mode: 'v52a' | 'v52b' | 'v52c'
    """
    if sub_mode == "v52a":
        # per-signal: if n_cands <= 2, all protected; else all through verification
        return "A" if n_cands <= 2 else "C"
    elif sub_mode == "v52b":
        # per-candidate: top-1 by DINOv2 peak score is protected
        return "A" if cid == top1_idx else "C"
    elif sub_mode == "v52c":
        # combined: n<=2 → all protected; else top-1 protected, others verified
        if n_cands <= 2:
            return "A"
        return "A" if cid == top1_idx else "C"
    return "C"  # fallback

# ── Signal runner ─────────────────────────────────────────────────────────────
DOMAIN_CTX = {
    "NAB":  "AWS cloud infrastructure metric (EC2 CPU/disk/network, RDS, ELB)",
    "SMAP": "NASA spacecraft telemetry channel (SMAP satellite sensor data)",
    "MSL":  "Mars Science Laboratory rover instrument telemetry",
}

def run_signal(ds, sig, sub_mode):
    timestamps, vals = load_signal(ds, sig)
    T = len(timestamps)
    gt_ivs = load_gt(sig, timestamps)
    all_ws = load_dino(ds, sig)
    candidates, all_ws = stage1(all_ws)

    f1_s1, p_s1, r_s1 = interval_f1(gt_ivs, candidates)
    ds_ctx = DOMAIN_CTX.get(ds, ds)
    print(f"  Stage1: {len(candidates)} candidates, F1={f1_s1:.4f}", flush=True)

    if not candidates:
        return {"ds":ds, "sig":sig, "T":T, "n_gt":len(gt_ivs), "n_s1":0,
                "f1_s1":0., "f1_out":0., "f1_oracle":0., "oracle_gap":0.,
                "gap_recovery":float("nan"), "n_kept":0, "n_api":0,
                "mode":sub_mode, "per_candidate":[]}

    # Compute DINOv2 peak score for each candidate (for top-1 selection)
    score_ts_raw = compute_base_smooth(all_ws, T)
    peak_scores = [candidate_peak_score(score_ts_raw, c) for c in candidates]
    top1_idx = int(np.argmax(peak_scores))
    n_cands = len(candidates)

    print(f"  n={n_cands}, top1=cid{top1_idx}(peak={peak_scores[top1_idx]:.4f}), "
          f"sub_mode={sub_mode}", flush=True)

    per_cand = []
    kept_ivs = []
    api_calls = 0

    for cid, cand in enumerate(candidates):
        s0, e0 = cand
        others = [c for j,c in enumerate(candidates) if j!=cid]
        left_opts, right_opts, smooth_c = generate_options(all_ws, cand, T)
        summary = make_summary(vals, smooth_c, cand, T)
        oracle_li, oracle_ri, oracle_iv, _ = oracle_select(
            left_opts, right_opts, cand, others, gt_ivs)

        cand_mode = select_cand_mode(cid, n_cands, top1_idx, sub_mode)

        pc = {
            "cid": cid, "orig": list(cand),
            "left_opts": [[nm,int(t)] for nm,t in left_opts],
            "right_opts": [[nm,int(t)] for nm,t in right_opts],
            "n_left": len(left_opts), "n_right": len(right_opts),
            "oracle_li": oracle_li, "oracle_ri": oracle_ri,
            "oracle_iv": list(oracle_iv),
            "is_tp": is_tp(cand, gt_ivs),
            "decision": "keep", "final_iv": list(cand),
            "vlm_li": 0, "vlm_ri": 0,
            "match_oracle_l": False, "match_oracle_r": False,
            "cand_mode": cand_mode,          # 'A' or 'C'
            "is_top1": cid == top1_idx,
            "peak_score": round(peak_scores[cid], 4),
        }

        if cand_mode == "A":
            # Protected: boundary selection only, no verification
            imgs = make_images_with_opts(vals, smooth_c, cand, left_opts, right_opts, cid, T)
            out = call_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx, imgs)
            api_calls += 1
            li, ri = (out["left_option"], out["right_option"]) if out else (0, 0)
            final_iv = (left_opts[li][1], right_opts[ri][1])
            pc.update({"decision":"keep", "vlm_li":li, "vlm_ri":ri,
                       "final_iv":list(final_iv),
                       "match_oracle_l": li==oracle_li,
                       "match_oracle_r": ri==oracle_ri})
            kept_ivs.append(final_iv)

        else:
            # Mode C: verify first, then boundary for survivors
            imgs_v = make_images_verify(vals, cand, cid, T)
            dec = call_verify(cid, cand, summary, ds_ctx, imgs_v)
            api_calls += 1

            if dec == "discard":
                pc.update({"decision": "discard", "final_iv": list(cand)})
            else:
                imgs_b = make_images_with_opts(vals, smooth_c, cand,
                                               left_opts, right_opts, cid, T)
                out = call_boundary(cid, cand, left_opts, right_opts, summary, ds_ctx, imgs_b)
                api_calls += 1
                li, ri = (out["left_option"], out["right_option"]) if out else (0, 0)
                final_iv = (left_opts[li][1], right_opts[ri][1])
                pc.update({"decision": dec, "vlm_li":li, "vlm_ri":ri,
                           "final_iv": list(final_iv),
                           "match_oracle_l": li==oracle_li,
                           "match_oracle_r": ri==oracle_ri})
                kept_ivs.append(final_iv)

        per_cand.append(pc)
        prot_tag = "[PROT]" if cand_mode == "A" else "[VERIF]"
        status = pc["decision"].upper()[:4]
        print(f"    C{cid}{prot_tag}[{s0},{e0}] {status} "
              f"L{pc['vlm_li']}/{left_opts[pc['vlm_li']][1] if pc['vlm_li']<len(left_opts) else '?'} "
              f"R{pc['vlm_ri']}/{right_opts[pc['vlm_ri']][1] if pc['vlm_ri']<len(right_opts) else '?'} "
              f"oracle=L{oracle_li}R{oracle_ri} tp={pc['is_tp']} pk={pc['peak_score']:.3f}",
              flush=True)

    f1_out, p_out, r_out = interval_f1(gt_ivs, kept_ivs)
    oracle_all = [pc["oracle_iv"] for pc in per_cand]
    f1_oracle, _, _ = interval_f1(gt_ivs, oracle_all)

    gap = f1_oracle - f1_s1
    recovery = (f1_out - f1_s1) / gap if abs(gap) > 1e-4 else float("nan")

    n = len(per_cand)
    n_la = sum(1 for pc in per_cand if pc["match_oracle_l"])
    n_ra = sum(1 for pc in per_cand if pc["match_oracle_r"])

    vm = compute_verif_metrics(per_cand, gt_ivs, candidates)

    # Protected vs verified breakdown
    n_prot   = sum(1 for pc in per_cand if pc["cand_mode"] == "A")
    n_verif  = sum(1 for pc in per_cand if pc["cand_mode"] == "C")
    n_verif_disc = sum(1 for pc in per_cand if pc["cand_mode"]=="C" and pc["decision"]=="discard")

    print(f"  OUT: F1={f1_out:.4f} (S1={f1_s1:.4f} Oracle={f1_oracle:.4f} "
          f"Rec={recovery:.1%}) kept={len(kept_ivs)}/{n} "
          f"prot={n_prot} verif={n_verif}(disc={n_verif_disc}) "
          f"tp_ret={vm.get('tp_retention',float('nan')):.0%} "
          f"fp_rej={vm.get('fp_rejection',float('nan')):.0%} "
          f"false_disc={vm.get('false_discard',float('nan')):.0%}", flush=True)

    return {
        "ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
        "n_s1": len(candidates), "mode": sub_mode,
        "f1_s1": f1_s1, "p_s1": p_s1, "r_s1": r_s1,
        "f1_out": f1_out, "p_out": p_out, "r_out": r_out,
        "f1_oracle": f1_oracle, "oracle_gap": gap, "gap_recovery": recovery,
        "n_kept": len(kept_ivs), "n_api": api_calls,
        "l_option_acc": n_la/n if n else 0.,
        "r_option_acc": n_ra/n if n else 0.,
        "n_prot": n_prot, "n_verif": n_verif, "n_verif_disc": n_verif_disc,
        **vm,
        "s1_ivs": [list(c) for c in candidates],
        "out_ivs": [list(iv) for iv in kept_ivs],
        "oracle_ivs": oracle_all,
        "per_candidate": per_cand,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v52a","v52b","v52c"], required=True,
                        help="v52a=count-guard  v52b=top1-guard  v52c=combined[recommended]")
    args = parser.parse_args()
    MODE = args.mode

    OUT_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments") / f"results_{MODE}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL = OUT_DIR / "partial_results.jsonl"
    LOG     = OUT_DIR / "run.log"

    import functools
    _print_orig = print
    log_f = open(LOG, "a", encoding="utf-8", errors="replace")
    def _tee(*a, **kw):
        _print_orig(*a, **kw)
        kw.pop("file", None)
        _print_orig(*a, file=log_f, **kw)
        log_f.flush()
    import builtins
    builtins.print = _tee

    done = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[f"{r['ds']}__{r['sig']}"] = r
            except: pass
    print(f"Mode={MODE}  Checkpoint: {len(done)} done", flush=True)

    all_results = list(done.values())

    for ds, sigs in DATASETS.items():
        for sig in sigs:
            key = f"{ds}__{sig}"
            if key in done:
                print(f"  [SKIP] {ds}/{sig}", flush=True); continue
            print(f"\n  [{ds}] {sig}", flush=True)
            try:
                r = run_signal(ds, sig, MODE)
                all_results.append(r)
                with open(PARTIAL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
            except Exception as exc:
                import traceback
                print(f"  [ERROR] {exc}", flush=True)
                traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    ok = [r for r in all_results if r.get("n_s1",0) > 0]
    if not ok:
        print("No results."); raise SystemExit

    print(f"\n{'='*90}", flush=True)
    print(f"v5.2 Mode={MODE}  RESULTS", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  {'Signal':<46} {'S1':>6} {'OUT':>6} {'Oracle':>7} {'Recov':>7} "
          f"{'Kept':>5} {'Prot':>5} {'TPreten':>8} {'FPrej':>6} {'FDisc':>6}", flush=True)
    print(f"  {'-'*90}", flush=True)

    ds_totals = {}
    for ds in ["NAB","SMAP","MSL"]:
        print(f"\n  [{ds}]", flush=True)
        ds_results = [r for r in ok if r["ds"]==ds]
        for r in ds_results:
            rec = r.get("gap_recovery")
            rec_str = f"{rec:.1%}" if (rec==rec and rec is not None) else "N/A"
            tp_r = r.get("tp_retention"); tp_str = f"{tp_r:.0%}" if tp_r==tp_r and tp_r is not None else "N/A"
            fp_r = r.get("fp_rejection"); fp_str = f"{fp_r:.0%}" if fp_r==fp_r and fp_r is not None else "N/A"
            fd_r = r.get("false_discard");fd_str = f"{fd_r:.0%}" if fd_r==fd_r and fd_r is not None else "N/A"
            print(f"  {r['sig']:<46} {r['f1_s1']:>6.4f} {r['f1_out']:>6.4f} {r['f1_oracle']:>7.4f} "
                  f"{rec_str:>7} {r['n_kept']:>2}/{r['n_s1']:<2} "
                  f"{r.get('n_prot',0):>4} {tp_str:>8} {fp_str:>6} {fd_str:>6}", flush=True)

        f1s    = [r["f1_s1"]  for r in ds_results]
        f1outs = [r["f1_out"] for r in ds_results]
        f1ors  = [r["f1_oracle"] for r in ds_results]
        avg_s1  = sum(f1s)/len(f1s)
        avg_out = sum(f1outs)/len(f1outs)
        avg_or  = sum(f1ors)/len(f1ors)
        print(f"  AVG{' '*43} {avg_s1:.4f} {avg_out:.4f} {avg_or:.4f}", flush=True)
        ds_totals[ds] = (avg_s1, avg_out, avg_or, len(ds_results))

    # ALL
    s1_all = 0.6174
    oracle_bd = 0.6817
    all_outs = [r["f1_out"] for r in ok]
    all_s1s  = [r["f1_s1"]  for r in ok]
    all_ors  = [r["f1_oracle"] for r in ok]
    f1_all   = sum(all_outs)/len(all_outs)
    or_all   = sum(all_ors)/len(all_ors)
    gap_rec  = (f1_all - s1_all) / (oracle_bd - s1_all)

    total_api = sum(r.get("n_api",0) for r in ok)
    tp_rets = [r["tp_retention"] for r in ok if r.get("tp_retention")==r.get("tp_retention") and r.get("tp_retention") is not None]
    fp_rejs = [r["fp_rejection"] for r in ok if r.get("fp_rejection")==r.get("fp_rejection") and r.get("fp_rejection") is not None]
    fd_rs   = [r["false_discard"] for r in ok if r.get("false_discard")==r.get("false_discard") and r.get("false_discard") is not None]

    print(f"\n  ALL ({len(ok)} signals)  Mode={MODE}", flush=True)
    print(f"    Stage1 F1   = {s1_all:.4f}", flush=True)
    print(f"    Output F1   = {f1_all:.4f}  ({f1_all-s1_all:+.4f} vs Stage1)", flush=True)
    print(f"    Oracle F1   = {oracle_bd:.4f}", flush=True)
    print(f"    Oracle gap  = {oracle_bd-s1_all:+.4f}", flush=True)
    print(f"    Gap recover = {gap_rec:.1%}", flush=True)
    print(f"    Reference: Stage1={s1_all:.4f}  v4=0.6526  v5=0.5883  v51C=0.6408", flush=True)
    print(f"    Total API calls   = {total_api}", flush=True)
    if tp_rets: print(f"    Avg TP retention  = {sum(tp_rets)/len(tp_rets):.1%}", flush=True)
    if fp_rejs: print(f"    Avg FP rejection  = {sum(fp_rejs)/len(fp_rejs):.1%}", flush=True)
    if fd_rs:   print(f"    Avg false discard = {sum(fd_rs)/len(fd_rs):.1%}", flush=True)

    log_f.close()
    print(f"\nSaved -> {OUT_DIR}", flush=True)
