"""
Stage2 Univariate v5 — Constrained Temporal Grounding
======================================================
핵심 변경: VLM이 free-form interval을 생성하지 않는다.
  - decision:      keep(1) / discard(0)
  - left_option:   L0-L4 중 선택
  - right_option:  R0-R4 중 선택

oracle_boundary_analysis.py에서 확인된 oracle gap (+0.0643)을
VLM이 실제로 얼마나 회복하는지 측정하는 실험.

v5.0 scope (strictly):
  - candidate one-call
  - decision + left_option + right_option only
  - 3 panels: global / local zoom / score curve
  - numerical summary
  - NO merge/split/secondary/anomaly_type/explanation
"""

import ast, base64, io, json, os, pickle, re, time, warnings
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
OUT_DIR   = BASE / "experiments/results_stage2_univar_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL   = OUT_DIR / "partial_results.jsonl"

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

# option colors: left=blues, right=reds
L_COLORS = ["#1f77b4", "#17becf", "#2ca02c", "#9467bd", "#8c564b"]
R_COLORS = ["#d62728", "#ff7f0e", "#e377c2", "#bcbd22", "#7f7f7f"]

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

def interval_iou(a, b):
    inter = max(0, min(a[1],b[1]) - max(a[0],b[0]) + 1)
    union = max(a[1],b[1]) - min(a[0],b[0]) + 1
    return inter/union if union > 0 else 0.

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
    candidates = [(s,e) for s,e in merged if e-s+1 >= MIN_IV]
    return candidates, all_ws

def score_to_ts(all_ws, T):
    """window-level scores → per-timestep (smooth)"""
    score_ts = np.zeros(T)
    count_ts = np.zeros(T)
    for i, s in enumerate(range(0, T-WIN+1, STRIDE)):
        if i < len(all_ws):
            score_ts[s:s+WIN] += all_ws[i]
            count_ts[s:s+WIN] += 1
    with np.errstate(invalid='ignore'):
        score_ts = np.where(count_ts > 0, score_ts/count_ts, 0.)
    return uniform_filter1d(score_ts, size=7)

# ── Boundary option generator ─────────────────────────────────────────────────
def generate_options(smooth, all_ws, cand, T):
    """
    Left L0-L4, Right R0-R4 boundary candidates.
    Returns: left_opts, right_opts as lists of (name, timestep)
    """
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    margin = max(3*L, 100)

    inner = smooth[s0:e0+1]
    tau_h = float(np.percentile(inner, 50)) if len(inner) else smooth[s0]
    tau_l = float(np.percentile(inner, 25)) if len(inner) else smooth[s0]*0.5

    # ── Left options ──────────────────────────────────────────────────────────
    left = [("original", s0)]

    # L1: threshold high crossing going left
    for t in range(s0, max(0, s0-margin)-1, -1):
        if smooth[t] < tau_h: left.append(("high_cross", max(0, t))); break

    # L2: threshold low crossing going left
    for t in range(s0, max(0, s0-margin)-1, -1):
        if smooth[t] < tau_l: left.append(("low_cross", max(0, t+1))); break

    # L3: local minimum before peak
    lr = smooth[max(0, s0-margin):s0+1]
    if len(lr) > 3:
        lmins = argrelmin(lr, order=max(1, len(lr)//10))[0]
        if len(lmins):
            left.append(("local_min", int(np.clip(max(0,s0-margin)+lmins[-1], 0, s0))))

    # L4: max derivative (rising edge) going left
    deriv = np.diff(smooth)
    dl = deriv[max(0,s0-margin):s0]
    if len(dl):
        idx = int(np.argmax(dl))
        left.append(("deriv_rise", int(np.clip(max(0,s0-margin)+idx, 0, s0))))

    # ── Right options ─────────────────────────────────────────────────────────
    right = [("original", e0)]

    # R1: threshold high crossing going right
    for t in range(e0, min(T, e0+margin+1)):
        if smooth[t] < tau_h: right.append(("high_cross", min(T-1, t))); break

    # R2: threshold low crossing going right
    for t in range(e0, min(T, e0+margin+1)):
        if smooth[t] < tau_l: right.append(("low_cross", min(T-1, t-1))); break

    # R3: local minimum after peak
    rr = smooth[e0:min(T, e0+margin+1)]
    if len(rr) > 3:
        rmins = argrelmin(rr, order=max(1, len(rr)//10))[0]
        if len(rmins):
            right.append(("local_min", int(np.clip(e0+rmins[0], e0, T-1))))

    # R4: max falling derivative going right
    dr = deriv[e0:min(T-1, e0+margin)]
    if len(dr):
        idx = int(np.argmin(dr))  # most negative = sharpest drop
        right.append(("deriv_fall", int(np.clip(e0+idx, e0, T-1))))

    # deduplicate, keep order, clip
    seen_l, left_dedup = set(), []
    for nm, t in left:
        t = int(np.clip(t, 0, s0))
        if t not in seen_l: seen_l.add(t); left_dedup.append((nm, t))

    seen_r, right_dedup = set(), []
    for nm, t in right:
        t = int(np.clip(t, e0, T-1))
        if t not in seen_r: seen_r.add(t); right_dedup.append((nm, t))

    return left_dedup, right_dedup

# ── Oracle selection (for oracle recovery metric) ──────────────────────────────
def oracle_select(left_opts, right_opts, cand, others, gt_ivs):
    """Returns (left_idx, right_idx, oracle_interval, oracle_f1)"""
    best_f1, best_li, best_ri, best_iv = -1., 0, 0, cand
    for li, (_, l) in enumerate(left_opts):
        for ri, (_, r) in enumerate(right_opts):
            if l > r: continue
            iv = (l, r)
            f1, _, _ = interval_f1(gt_ivs, list(others) + [iv])
            if f1 > best_f1:
                best_f1, best_li, best_ri, best_iv = f1, li, ri, iv
    return best_li, best_ri, best_iv, best_f1

# ── Visualization ─────────────────────────────────────────────────────────────
def _img_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _draw_options(ax, left_opts, right_opts, ymin, ymax):
    """Draw vertical lines for all boundary options."""
    for li, (nm, t) in enumerate(left_opts):
        c = L_COLORS[li % len(L_COLORS)]
        ls = "-" if li == 0 else "--"
        lw = 1.8 if li == 0 else 1.0
        ax.axvline(t, color=c, ls=ls, lw=lw, alpha=0.85)
        ax.text(t, ymax - 0.05*(ymax-ymin)*(li+1),
                f"L{li}", color=c, fontsize=6, ha="center", fontweight="bold")
    for ri, (nm, t) in enumerate(right_opts):
        c = R_COLORS[ri % len(R_COLORS)]
        ls = "-" if ri == 0 else "--"
        lw = 1.8 if ri == 0 else 1.0
        ax.axvline(t, color=c, ls=ls, lw=lw, alpha=0.85)
        ax.text(t, ymin + 0.05*(ymax-ymin)*(ri+1),
                f"R{ri}", color=c, fontsize=6, ha="center", fontweight="bold")

def make_evidence_images(vals, smooth, cand, left_opts, right_opts, cand_id, T):
    """
    Returns 3 base64 images:
      1. Global plot with candidate + all options
      2. Local zoom (3L context each side)
      3. Score curve (local) with options
    """
    s0, e0 = cand
    L = max(e0 - s0 + 1, 1)
    margin = max(3*L, 150)
    zs = max(0, s0 - margin)
    ze = min(T-1, e0 + margin)

    ymin_g, ymax_g = float(vals.min()), float(vals.max())
    pad = (ymax_g - ymin_g) * 0.08 or 0.1
    ymin_g -= pad; ymax_g += pad

    ymin_z, ymax_z = float(vals[zs:ze+1].min()), float(vals[zs:ze+1].max())
    pad_z = (ymax_z - ymin_z) * 0.08 or 0.1
    ymin_z -= pad_z; ymax_z += pad_z

    # ── Panel 1: Global ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
    ax.plot(vals, color="#333333", lw=0.5, alpha=0.8)
    ax.axvspan(s0, e0, color="salmon", alpha=0.4, label=f"Cand#{cand_id}[{s0},{e0}]")
    _draw_options(ax, left_opts, right_opts, ymin_g, ymax_g)
    ax.set_xlim(0, T-1); ax.set_ylim(ymin_g, ymax_g)
    ax.set_title(f"Panel 1 — Global (T={T})  |  Candidate #{cand_id} [{s0},{e0}]", fontsize=8)
    # legend for option names
    patches = ([mpatches.Patch(color=L_COLORS[i], label=f"L{i}: {nm}")
                for i, (nm,_) in enumerate(left_opts)] +
               [mpatches.Patch(color=R_COLORS[i], label=f"R{i}: {nm}")
                for i, (nm,_) in enumerate(right_opts)])
    ax.legend(handles=patches, fontsize=5, ncol=4, loc="upper right")
    img_global = _img_b64(fig)

    # ── Panel 2: Local zoom ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    x_zoom = np.arange(zs, ze+1)
    ax.plot(x_zoom, vals[zs:ze+1], color="#333333", lw=1.1)
    ax.axvspan(s0, e0, color="salmon", alpha=0.35, label=f"Candidate [{s0},{e0}]")
    _draw_options(ax, left_opts, right_opts, ymin_z, ymax_z)
    ax.set_xlim(zs, ze); ax.set_ylim(ymin_z, ymax_z)
    ax.set_title(f"Panel 2 — Local Zoom [{zs},{ze}]  (margin={margin})", fontsize=8)
    ax.set_xlabel("time index"); ax.set_ylabel("value")
    # x-axis ticks at option positions
    opt_ticks = sorted(set([t for _,t in left_opts] + [t for _,t in right_opts]))
    ax.set_xticks(opt_ticks)
    ax.tick_params(axis='x', labelsize=6, rotation=45)
    img_zoom = _img_b64(fig)

    # ── Panel 3: Score curve ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=100)
    x_sc = np.arange(zs, ze+1)
    ax.plot(x_sc, smooth[zs:ze+1], color="#e67e22", lw=1.2, label="DINOv2 score")
    ax.axvspan(s0, e0, color="salmon", alpha=0.25)
    _draw_options(ax, left_opts, right_opts,
                  float(smooth[zs:ze+1].min()), float(smooth[zs:ze+1].max()))
    ax.set_xlim(zs, ze)
    ax.set_title(f"Panel 3 — DINOv2 Score Curve (local)", fontsize=8)
    ax.set_xlabel("time index"); ax.set_ylabel("anomaly score")
    ax.set_xticks(opt_ticks)
    ax.tick_params(axis='x', labelsize=6, rotation=45)
    img_score = _img_b64(fig)

    return img_global, img_zoom, img_score

# ── Numerical summary ─────────────────────────────────────────────────────────
def make_numerical_summary(vals, smooth, all_ws, cand, T):
    s0, e0 = cand
    L = e0 - s0 + 1
    margin = max(3*L, 100)

    pre_s, pre_e = max(0, s0-margin), max(0, s0-1)
    post_s, post_e = min(T-1, e0+1), min(T-1, e0+margin)

    def safe_stats(arr):
        if len(arr) == 0: return 0., 0.
        return float(np.mean(arr)), float(np.std(arr))

    pre_m, pre_s_ = safe_stats(vals[pre_s:pre_e+1])
    in_m, in_s_   = safe_stats(vals[s0:e0+1])
    po_m, po_s_   = safe_stats(vals[post_s:post_e+1])

    sc = smooth[s0:e0+1]
    peak_sc   = float(sc.max()) if len(sc) else 0.
    mean_sc   = float(sc.mean()) if len(sc) else 0.
    pct_sc    = float(np.mean(smooth <= peak_sc) * 100)

    return {
        "candidate_id": None,  # filled later
        "interval": [s0, e0],
        "length": int(L),
        "peak_dino_score": round(peak_sc, 4),
        "mean_dino_score": round(mean_sc, 4),
        "score_percentile": round(pct_sc, 1),
        "pre_mean": round(pre_m, 4),
        "pre_std":  round(pre_s_, 4),
        "inside_mean": round(in_m, 4),
        "inside_std":  round(in_s_, 4),
        "post_mean": round(po_m, 4),
        "post_std":  round(po_s_, 4),
    }

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a time-series anomaly detection expert. "
    "You will verify a candidate anomaly interval and select the most accurate temporal boundaries "
    "from a pre-computed list of options. "
    "Do NOT invent new coordinates. Only select from the provided options."
)

def build_user_prompt(cand_id, cand, left_opts, right_opts, summary, ds_ctx):
    s0, e0 = cand
    left_lines  = "\n".join(f"  L{i}: {nm} = index {t}"
                            for i, (nm, t) in enumerate(left_opts))
    right_lines = "\n".join(f"  R{i}: {nm} = index {t}"
                            for i, (nm, t) in enumerate(right_opts))
    num = "\n".join(f"  {k}: {v}" for k, v in summary.items() if k != "candidate_id")

    return f"""Domain: {ds_ctx}
Candidate #{cand_id}: initial interval [{s0}, {e0}]

=== LEFT BOUNDARY OPTIONS (choose one) ===
{left_lines}

=== RIGHT BOUNDARY OPTIONS (choose one) ===
{right_lines}

=== NUMERICAL SUMMARY ===
{num}

=== YOUR TASK ===
1. Decide: should this candidate be kept (1) or discarded (0)?
   - Discard if it is clearly a normal fluctuation or noise.
   - Keep if it shows a genuine anomaly (spike, level shift, trend change, etc.).

2. If kept, select the best left and right boundary index from the options above.
   - L0 and R0 are the original Stage-1 boundaries.
   - Other options may better capture the true anomaly extent.
   - If you are unsure, choose L0 and R0 (original).

You are shown three images:
  - Panel 1: Full time series with candidate highlighted (red). Boundary options are labeled L0-L4 (blue) and R0-R4 (red).
  - Panel 2: Local zoom around the candidate with boundary options labeled.
  - Panel 3: DINOv2 anomaly score curve with boundary options labeled.

Return ONLY valid JSON with exactly this schema:
{{
  "decision": 0,
  "left_option": 0,
  "right_option": 0
}}
decision=1 means keep, decision=0 means discard.
left_option and right_option are integers (0 to {len(left_opts)-1} and 0 to {len(right_opts)-1} respectively).
"""

# ── VLM call ──────────────────────────────────────────────────────────────────
def call_vlm(cand_id, cand, left_opts, right_opts, summary, ds_ctx,
             img_global, img_zoom, img_score, tries=4):
    user_text = build_user_prompt(cand_id, cand, left_opts, right_opts, summary, ds_ctx)
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_global}", "detail": "high"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_zoom}",   "detail": "high"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_score}",  "detail": "low"}},
    ]
    for attempt in range(tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": content},
                ],
                temperature=0.0, max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                return json.loads(raw)
            except Exception:
                m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except: pass
            # fallback: regex extract fields
            dec = re.search(r'"decision"\s*:\s*([01])', raw)
            lo  = re.search(r'"left_option"\s*:\s*(\d+)', raw)
            ro  = re.search(r'"right_option"\s*:\s*(\d+)', raw)
            if dec:
                return {
                    "decision": int(dec.group(1)),
                    "left_option":  int(lo.group(1)) if lo else 0,
                    "right_option": int(ro.group(1)) if ro else 0,
                }
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep((attempt+1)*30)
            else:
                time.sleep(5)
    return None  # all retries failed

# ── Per-signal runner ─────────────────────────────────────────────────────────
DOMAIN_CTX = {
    "NAB":  "AWS cloud infrastructure metric (EC2 CPU/disk/network, RDS, ELB)",
    "SMAP": "NASA spacecraft telemetry channel (SMAP satellite sensor data)",
    "MSL":  "Mars Science Laboratory rover instrument telemetry",
}

def run_signal(ds, sig):
    timestamps, vals = load_signal(ds, sig)
    T = len(timestamps)
    gt_ivs = load_gt(sig, timestamps)
    scores  = load_dino(ds, sig)
    candidates, all_ws = stage1(scores)
    smooth = score_to_ts(all_ws, T)

    f1_s1, p_s1, r_s1 = interval_f1(gt_ivs, candidates)
    ds_ctx = DOMAIN_CTX.get(ds, ds)

    print(f"  Stage1: {len(candidates)} candidates, F1={f1_s1:.4f}", flush=True)

    if not candidates:
        return {"ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
                "n_s1": 0, "f1_s1": 0., "f1_v5": 0.,
                "n_kept": 0, "n_api": 0,
                "oracle_f1": 0., "oracle_gap": 0., "gap_recovery": 0.,
                "s1_ivs": [], "v5_ivs": [], "oracle_ivs": [],
                "per_candidate": []}

    per_cand = []
    kept_ivs = []
    api_calls = 0

    for cid, cand in enumerate(candidates):
        s0, e0 = cand
        others = [c for j, c in enumerate(candidates) if j != cid]

        left_opts, right_opts = generate_options(smooth, all_ws, cand, T)

        # oracle for this candidate
        oracle_li, oracle_ri, oracle_iv, _ = oracle_select(
            left_opts, right_opts, cand, others, gt_ivs)

        # numerical summary
        summary = make_numerical_summary(vals, smooth, all_ws, cand, T)
        summary["candidate_id"] = cid

        # evidence images
        img_g, img_z, img_s = make_evidence_images(
            vals, smooth, cand, left_opts, right_opts, cid, T)

        # VLM call
        out = call_vlm(cid, cand, left_opts, right_opts, summary, ds_ctx,
                       img_g, img_z, img_s)
        api_calls += 1

        if out is None:
            # fallback: keep original
            decision = 1
            li, ri = 0, 0
            parse_ok = False
        else:
            decision = int(out.get("decision", 1))
            li = int(out.get("left_option", 0))
            ri = int(out.get("right_option", 0))
            # clamp to valid range
            li = max(0, min(li, len(left_opts)-1))
            ri = max(0, min(ri, len(right_opts)-1))
            parse_ok = True

        if decision == 1:
            chosen_l = left_opts[li][1]
            chosen_r = right_opts[ri][1]
            final_iv = (min(chosen_l, chosen_r), max(chosen_l, chosen_r))
            kept_ivs.append(final_iv)
        else:
            final_iv = None

        # gt match info for this candidate
        gt_match = [g for g in gt_ivs if _ov(cand, g)]
        is_tp = len(gt_match) > 0
        oracle_match = oracle_iv == cand  # did oracle change boundary?

        per_cand.append({
            "cid": cid, "orig": list(cand),
            "left_opts": [[nm, int(t)] for nm,t in left_opts],
            "right_opts": [[nm, int(t)] for nm,t in right_opts],
            "n_left": len(left_opts), "n_right": len(right_opts),
            "oracle_li": oracle_li, "oracle_ri": oracle_ri,
            "oracle_iv": list(oracle_iv),
            "vlm_decision": decision, "vlm_li": li, "vlm_ri": ri,
            "final_iv": list(final_iv) if final_iv else None,
            "is_tp_s1": is_tp, "parse_ok": parse_ok,
            "vlm_matches_oracle_l": li == oracle_li,
            "vlm_matches_oracle_r": ri == oracle_ri,
        })

        status = "KEEP" if decision == 1 else "DISC"
        print(f"    C{cid}[{s0},{e0}] → {status} "
              f"L{li}({left_opts[li][1]}) R{ri}({right_opts[ri][1]}) "
              f"oracle=L{oracle_li}R{oracle_ri}", flush=True)

    # metrics
    f1_v5, p_v5, r_v5 = interval_f1(gt_ivs, kept_ivs)

    # oracle boundary F1 (all kept with oracle boundaries)
    oracle_all = [pc["oracle_iv"] for pc in per_cand
                  if pc["vlm_decision"] == 1 or True]  # oracle keeps all
    # proper oracle: oracle selection with all candidates (run once)
    oracle_kept = [pc["oracle_iv"] for pc in per_cand]
    f1_oracle, _, _ = interval_f1(gt_ivs, oracle_kept)

    gap = f1_oracle - f1_s1
    recovery = (f1_v5 - f1_s1) / gap if abs(gap) > 1e-4 else float("nan")

    # VLM option accuracy
    n_l_correct = sum(1 for pc in per_cand if pc["vlm_matches_oracle_l"])
    n_r_correct = sum(1 for pc in per_cand if pc["vlm_matches_oracle_r"])
    n = len(per_cand)

    print(f"  v5: F1={f1_v5:.4f}  (S1={f1_s1:.4f}, Oracle={f1_oracle:.4f}, "
          f"Recovery={recovery:.1%})  kept={len(kept_ivs)}/{n}  "
          f"L-acc={n_l_correct/n:.1%} R-acc={n_r_correct/n:.1%}", flush=True)

    return {
        "ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
        "n_s1": len(candidates),
        "f1_s1": f1_s1, "p_s1": p_s1, "r_s1": r_s1,
        "f1_v5": f1_v5, "p_v5": p_v5, "r_v5": r_v5,
        "f1_oracle": f1_oracle,
        "oracle_gap": gap, "gap_recovery": recovery,
        "n_kept": len(kept_ivs), "n_api": api_calls,
        "l_option_acc": n_l_correct/n if n else 0.,
        "r_option_acc": n_r_correct/n if n else 0.,
        "s1_ivs": [list(c) for c in candidates],
        "v5_ivs": [list(iv) for iv in kept_ivs],
        "oracle_ivs": oracle_kept,
        "per_candidate": per_cand,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # load checkpoint
    done = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[f"{r['ds']}__{r['sig']}"] = r
            except: pass
    print(f"Checkpoint: {len(done)} done", flush=True)

    all_results = list(done.values())

    for ds, sigs in DATASETS.items():
        for sig in sigs:
            key = f"{ds}__{sig}"
            if key in done:
                print(f"  [SKIP] {ds}/{sig}", flush=True)
                continue
            print(f"\n  [{ds}] {sig}", flush=True)
            try:
                r = run_signal(ds, sig)
                all_results.append(r)
                with open(PARTIAL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
            except Exception as exc:
                print(f"  [ERROR] {exc}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok = [r for r in all_results if r.get("n_s1", 0) > 0]
    if not ok:
        print("No results."); raise SystemExit

    print("\n" + "="*85, flush=True)
    print("v5 CONSTRAINED TEMPORAL GROUNDING — RESULTS", flush=True)
    print("="*85, flush=True)

    for ds in ["NAB", "SMAP", "MSL"]:
        rows = [r for r in ok if r["ds"] == ds]
        if not rows: continue
        print(f"\n  {ds} ({len(rows)} signals)", flush=True)
        print(f"  {'Signal':<46} {'S1':>6} {'v5':>6} {'Oracle':>7} {'Recov':>7} {'Kept':>5} {'L-acc':>6} {'R-acc':>6}", flush=True)
        print(f"  {'-'*80}", flush=True)
        for r in rows:
            rec = f"{r['gap_recovery']:.1%}" if not (isinstance(r['gap_recovery'], float) and np.isnan(r['gap_recovery'])) else "  N/A"
            print(f"  {r['sig']:<46} {r['f1_s1']:>6.4f} {r['f1_v5']:>6.4f} "
                  f"{r['f1_oracle']:>7.4f} {rec:>7} "
                  f"{r['n_kept']:>2}/{r['n_s1']:<2} "
                  f"{r['l_option_acc']:>6.1%} {r['r_option_acc']:>6.1%}", flush=True)
        avg_s1 = np.mean([r["f1_s1"] for r in rows])
        avg_v5 = np.mean([r["f1_v5"] for r in rows])
        avg_or = np.mean([r["f1_oracle"] for r in rows])
        avg_la = np.mean([r["l_option_acc"] for r in rows])
        avg_ra = np.mean([r["r_option_acc"] for r in rows])
        print(f"  {'AVG':<46} {avg_s1:>6.4f} {avg_v5:>6.4f} {avg_or:>7.4f} {'':>7} {'':>5} {avg_la:>6.1%} {avg_ra:>6.1%}", flush=True)

    all_s1 = np.mean([r["f1_s1"] for r in ok])
    all_v5 = np.mean([r["f1_v5"] for r in ok])
    all_or = np.mean([r["f1_oracle"] for r in ok])
    gap    = all_or - all_s1
    rec    = (all_v5 - all_s1) / gap if abs(gap) > 1e-4 else float("nan")

    print(f"\n  ALL ({len(ok)} signals):", flush=True)
    print(f"    Stage1 F1          = {all_s1:.4f}", flush=True)
    print(f"    v5 F1              = {all_v5:.4f}  (+{all_v5-all_s1:+.4f} over Stage1)", flush=True)
    print(f"    Oracle boundary F1 = {all_or:.4f}", flush=True)
    print(f"    Oracle gap         = {gap:+.4f}", flush=True)
    print(f"    Gap recovery rate  = {rec:.1%}", flush=True)
    print(f"    v4 (reference)     = 0.6526", flush=True)

    total_api = sum(r.get("n_api", 0) for r in ok)
    print(f"    Total API calls    = {total_api}", flush=True)

    pd.DataFrame([{k: v for k,v in r.items() if k != "per_candidate"}
                  for r in all_results]).to_csv(OUT_DIR/"summary.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}", flush=True)
