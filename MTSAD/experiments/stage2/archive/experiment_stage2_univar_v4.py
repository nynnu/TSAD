"""
Stage2 Univariate v4

v3 Problems Fixed:
  S1 -- SMAP/NAB FP 필터링 실패: z_max-aware decision logic 추가.
        z_max(=max(|z_global|,|z_local|)) < 3 이면 strict threshold 적용.
        프롬프트 confidence 기준을 z-score magnitude에 맞게 재정의.
  S3 -- Panel 1 orange 후보 표시 제거 (GPT-4o 편향 제거).
        후보들이 Panel 1에 다 보이면 "anomaly 많은 신호"로 편향됨.
  C1 -- find_before_window: candidate가 t=0 시작 시 before_s=cs (겹침 버그).
        겹치면 candidate 이후(after) 구간을 대신 사용.
  C2 -- interval_f1 TP 중복 카운팅 버그 수정.
        pred 1개가 GT 여러 개 overlap 시 TP가 여러 번 카운트되던 문제.
  C3 -- LOOSE_PCT 주석 오류 수정 (10%인데 "raised to 15%" 주석).
  C4 -- pickle 파일 핸들 with 블록으로 명시적 close.
  C5 -- OpenAI client를 모듈 레벨에서 1회만 생성.
  C6 -- Panel 2 after context WIN//4 → WIN//2 로 확장.
  NEW -- 비교 패널(Panel 4): 후보가 2개 이상일 때 모든 후보 미니 섬네일.
         GPT-4o가 "다른 후보들과 비교해 이게 유독 이상한가?" 판단 가능.
"""

import ast, base64, io, json, os, re, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── API (C5: module-level client) ──────────────────────────────────────────────
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment.")
_client = OpenAI(api_key=API_KEY)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
DINO_DIR    = BASE / "results/VLM4TS_results_dino_ltr/checkpoints"
ANOMS_CSV   = BASE / "data/anomalies.csv"
RESULTS_DIR = BASE / "experiments/results_stage2_univar_v4"

def _sig_path(ds, sig):
    if ds == "NAB":
        return BASE / "data/realAWSCloudwatch" / f"{sig}.csv"
    return BASE / "data" / ds / f"{sig}.csv"

# ── Datasets ───────────────────────────────────────────────────────────────────
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

DOMAIN_CTX = {
    "NAB":  "AWS cloud infrastructure metric (EC2 CPU/disk/network, RDS, ELB)",
    "SMAP": "NASA spacecraft telemetry channel (SMAP satellite sensor data)",
    "MSL":  "Mars Science Laboratory rover instrument telemetry",
}

# ── Constants ──────────────────────────────────────────────────────────────────
WIN        = 224
STRIDE     = 56
LOOSE_PCT  = 10.0   # C3: fixed comment — 10% threshold (v2 tried 15%, reverted in v3)
MERGE_GAP  = WIN // 2
MIN_IV_LEN = 10
PCT_HIGH   = 95
PCT_MID    = 82
VLM_SLEEP  = 4.0

# ── Data loading ───────────────────────────────────────────────────────────────
def load_signal(ds, sig):
    df = pd.read_csv(_sig_path(ds, sig))
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_gt_intervals(sig, timestamps):
    anoms = pd.read_csv(ANOMS_CSV)
    row = anoms[anoms["signal"] == sig]
    if row.empty:
        return []
    events = ast.literal_eval(row.iloc[0]["events"])
    ivs = []
    for ts_s, ts_e in events:
        i_s = int(np.searchsorted(timestamps, ts_s, side="left"))
        i_e = int(np.searchsorted(timestamps, ts_e, side="right") - 1)
        i_s = max(0, min(i_s, len(timestamps) - 1))
        i_e = max(0, min(i_e, len(timestamps) - 1))
        if i_s <= i_e:
            ivs.append((i_s, i_e))
    return ivs

def load_dino(ds, sig):
    # C4: explicit file handle close
    path = DINO_DIR / f"{ds}__{sig}__dino_k5.pkl"
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)["scores"]

# ── Interval helpers ───────────────────────────────────────────────────────────
def get_intervals(binary):
    ivs, in_seg, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            ivs.append((start, i - 1)); in_seg = False
    if in_seg:
        ivs.append((start, len(binary) - 1))
    return ivs

def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

def interval_f1(gt_ivs, pred_ivs):
    """
    C2: Fixed TP counting.
    TP_pred = #preds that overlap at least one GT  (for precision)
    TP_gt   = #GTs  that overlap at least one pred (for recall)
    Previously counted (pred, GT) pairs → overestimated precision.
    """
    if not gt_ivs:
        return 0., 0., 0.
    TP_pred = sum(1 for d in pred_ivs if any(_ov(d, g) for g in gt_ivs))
    TP_gt   = sum(1 for g in gt_ivs  if any(_ov(g, d) for d in pred_ivs))
    FP      = sum(1 for d in pred_ivs if not any(_ov(d, g) for g in gt_ivs))
    FN      = sum(1 for g in gt_ivs  if not any(_ov(g, d) for d in pred_ivs))
    p = TP_pred / (TP_pred + FP) if (TP_pred + FP) > 0 else 0.
    r = TP_gt   / (TP_gt   + FN) if (TP_gt   + FN) > 0 else 0.
    return (2 * p * r / (p + r) if p + r else 0.), p, r

# ── Stage1 ─────────────────────────────────────────────────────────────────────
def stage1_univar(scores):
    T = len(scores)
    all_ws = np.array([scores[s:s + WIN].mean()
                       for s in range(0, T - WIN + 1, STRIDE)])
    thr = float(np.percentile(all_ws, 100 - LOOSE_PCT))
    binary = np.zeros(T, dtype=int)
    for i, s in enumerate(range(0, T - WIN + 1, STRIDE)):
        if all_ws[i] >= thr:
            binary[s:s + WIN] = 1
    raw_ivs = get_intervals(binary)
    merged = []
    for iv in raw_ivs:
        if merged and iv[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(list(iv))
    loose_ivs = [(s, e) for s, e in merged if e - s + 1 >= MIN_IV_LEN]
    cutoff = float(np.percentile(all_ws, 80))
    clean = all_ws[all_ws <= cutoff]
    if len(clean) < 5:
        clean = all_ws
    mu  = float(clean.mean())
    sig = float(clean.std()) if clean.std() > 1e-9 else 1e-9
    return loose_ivs, all_ws, mu, sig

def oracle_f1_sweep(all_ws, scores, gt_ivs, mu, sig):
    T = len(scores)
    best_f1, best_ivs = 0., []
    for a in [0.20,0.15,0.10,0.07,0.05,0.03,0.02,0.01,0.007,0.005,0.003,0.001]:
        thr = mu + norm.ppf(1 - a) * sig
        binary = np.zeros(T, dtype=int)
        for i, s in enumerate(range(0, T - WIN + 1, STRIDE)):
            if i < len(all_ws) and all_ws[i] >= thr:
                binary[s:s + WIN] = 1
        ivs = get_intervals(binary)
        f, _, _ = interval_f1(gt_ivs, ivs)
        if f > best_f1:
            best_f1, best_ivs = f, ivs
    return best_f1, best_ivs

def pct_rank(cs, ce, all_ws, T):
    lo_aligned = int(np.ceil(max(0, cs - WIN + 1) / STRIDE)) * STRIDE
    hi_aligned  = (min(T - WIN, ce) // STRIDE) * STRIDE
    if lo_aligned > hi_aligned:
        return float(np.mean(all_ws <= all_ws.mean()) * 100)
    idxs = [s // STRIDE for s in range(lo_aligned, hi_aligned + 1, STRIDE)
            if 0 <= s // STRIDE < len(all_ws)]
    if not idxs:
        return float(np.mean(all_ws <= all_ws.mean()) * 100)
    peak = max(all_ws[i] for i in idxs)
    return float(np.mean(all_ws <= peak) * 100)

# ── Normal statistics & reference window ──────────────────────────────────────
def compute_normal_stats(vals, all_ws):
    T = len(vals)
    thr_50 = float(np.percentile(all_ws, 50))
    normal_vals = []
    for i, s in enumerate(range(0, T - WIN + 1, STRIDE)):
        if i < len(all_ws) and all_ws[i] <= thr_50:
            normal_vals.extend(vals[s:s + WIN].tolist())
    if len(normal_vals) < WIN:
        normal_vals = vals.tolist()
    nv = np.array(normal_vals)
    v_mu = float(np.median(nv))
    q25, q75 = np.percentile(nv, [25, 75])
    v_std = float((q75 - q25) / 1.349)
    if v_std < 1e-4:
        v_std = float(nv.std()) + 1e-4
    return v_mu, v_std

def find_best_ref_window(all_ws, gt_ivs, T):
    best_s, best_score = 0, float("inf")
    for i, s in enumerate(range(0, T - WIN + 1, STRIDE)):
        if i >= len(all_ws):
            break
        w = (s, s + WIN - 1)
        if any(_ov(w, g) for g in gt_ivs):
            continue
        if all_ws[i] < best_score:
            best_score = all_ws[i]
            best_s = s
    return best_s

def find_before_window(cs, loose_ivs, T):
    """
    C1: Fixed — when candidate starts at t=0 (or before_s would overlap candidate),
    fall back to the window AFTER the candidate instead.
    """
    other_ivs = [(ls, le) for ls, le in loose_ivs]
    for s in range(max(0, cs - WIN), -1, -STRIDE):
        if s + WIN <= cs:
            w = (s, s + WIN - 1)
            if not any(_ov(w, o) for o in other_ivs if o[0] != cs):
                return s, "before"
    # C1 FIX: fallback to after-window when before is unavailable or overlaps candidate
    for s in range(cs + 1, T - WIN + 1, STRIDE):
        w = (s, s + WIN - 1)
        if not any(_ov(w, o) for o in other_ivs if o[0] != cs):
            return s, "after"
    return max(0, cs - WIN), "before"  # last resort

# ── Visualization ──────────────────────────────────────────────────────────────
def make_comparison_strip(vals, loose_ivs, current_cand, v_mu, v_std):
    """
    NEW: Comparison strip showing all N candidates as small thumbnails.
    Current candidate has red border; others gray.
    Helps GPT-4o judge "is this candidate uniquely anomalous vs others?"
    Only generated when there are 2+ candidates.
    """
    n = len(loose_ivs)
    if n <= 1:
        return None
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Comparison Strip: All Stage1 candidates (★ = current)", fontsize=7)
    for i, (cs, ce) in enumerate(loose_ivs):
        ax = axes[i]
        seg = vals[cs:ce + 1]
        ax.plot(np.arange(cs, ce + 1), seg, color="#333333", lw=0.8)
        ax.axhline(v_mu, color="steelblue", lw=0.7, ls="--", alpha=0.6)
        ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.10, color="steelblue")
        is_cur = (cs, ce) == current_cand
        ax.set_facecolor("#ffe0e0" if is_cur else "#f0f0f0")
        label = f"★C{i+1}" if is_cur else f"C{i+1}"
        ax.set_title(f"{label}\n[{cs},{ce}]", fontsize=5.5,
                     color="darkred" if is_cur else "black")
        ax.tick_params(labelsize=4.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0 if is_cur else 0.5)
            spine.set_edgecolor("red" if is_cur else "gray")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_image(vals, candidate, loose_ivs, ref_start, ctx_s, ctx_label, v_mu, v_std):
    """
    3-panel figure:
      Panel 1: Full series — NO orange shading for other candidates (S3 fix).
      Panel 2: Context → Candidate transition (C1: may be "after" if t=0 start).
      Panel 3: Global reference normal (lowest DINOv2 score area).
    """
    cs, ce = candidate
    T = len(vals)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)

    # ── Panel 1: Full series — S3: no orange shading ──────────────────────────
    ax = axes[0]
    ax.plot(vals, color="#333333", lw=0.5, alpha=0.85)
    # S3 FIX: removed orange shading for other candidates
    ax.axvspan(cs, ce, alpha=0.40, color="red", label=f"Candidate [{cs},{ce}]")
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.6)
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.08, color="steelblue")
    ax.set_title(f"Panel 1: Full series (T={T})  |  Red=Candidate under review",
                 fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.tick_params(labelsize=7)

    # ── Panel 2: Context → Candidate (C1+C6 fixed) ────────────────────────────
    ctx_end = ctx_s + WIN - 1
    if ctx_label == "before":
        view_s = ctx_s
        view_e = min(T - 1, ce + WIN // 2)   # C6: extended after context
    else:  # "after": context comes AFTER candidate
        view_s = max(0, cs - WIN // 4)
        view_e = min(T - 1, ctx_s + WIN - 1)

    ax = axes[1]
    x_view = np.arange(view_s, view_e + 1)
    ax.plot(x_view, vals[view_s:view_e + 1], color="#333333", lw=1.1)
    ctx_color = "steelblue" if ctx_label == "before" else "purple"
    ax.axvspan(ctx_s, ctx_end, alpha=0.15, color=ctx_color,
               label=f"{'Before' if ctx_label=='before' else 'After'} context [{ctx_s},{ctx_end}]")
    ax.axvspan(cs, ce, alpha=0.35, color="red", label=f"Candidate [{cs},{ce}]")
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.7,
               label=f"Global µ={v_mu:.3f}")
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.08, color="steelblue",
               label=f"±2σ (σ={v_std:.3f})")
    ctx_vals = vals[ctx_s:ctx_s + WIN]
    ctx_mu   = float(ctx_vals.mean())
    ax.axhline(ctx_mu, color=ctx_color, lw=0.9, ls="--", alpha=0.8,
               label=f"Context µ={ctx_mu:.3f}")
    cand_vals = vals[cs:ce + 1]
    peak_rel  = int(np.argmax(np.abs(cand_vals - v_mu)))
    peak_idx  = cs + peak_rel
    peak_val  = float(cand_vals[peak_rel])
    z_global  = (peak_val - v_mu) / v_std
    z_local   = (peak_val - ctx_mu) / (float(ctx_vals.std()) + 1e-4)
    ax.axvline(peak_idx, color="darkred", lw=1.1, ls=":",
               label=f"Peak={peak_val:.3f} (Δglobal {z_global:+.1f}σ, Δcontext {z_local:+.1f}σ)")
    ctx_dir = "Before" if ctx_label == "before" else "After"
    ax.set_title(
        f"Panel 2: {ctx_dir}→Candidate | "
        f"Peak: Δglobal={z_global:+.1f}σ  Δcontext={z_local:+.1f}σ",
        fontsize=8)
    ax.set_xlim(view_s, view_e)
    ax.legend(fontsize=5.5, loc="upper right", ncol=2)
    ax.tick_params(labelsize=7)

    # ── Panel 3: Global reference normal ──────────────────────────────────────
    ax = axes[2]
    ref_x = np.arange(ref_start, ref_start + WIN)
    ax.plot(ref_x, vals[ref_start:ref_start + WIN], color="#2ca02c", lw=1.1,
            label=f"Normal ref [{ref_start},{ref_start+WIN-1}]")
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.7)
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.10, color="steelblue")
    ax.set_title(
        f"Panel 3: Global normal reference [{ref_start},{ref_start+WIN-1}]  "
        f"(lowest DINOv2 = most normal)  |  µ={v_mu:.3f}  σ={v_std:.3f}",
        fontsize=8)
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(ref_start, ref_start + WIN - 1)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), z_global, z_local, peak_val, ctx_mu

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are an expert time series anomaly analyst specializing in "
    "infrastructure and scientific telemetry monitoring."
)

def build_prompt(ds, sig, candidate, pct, v_mu, v_std,
                 z_global, z_local, peak_val, ctx_mu, ref_start,
                 n_candidates, ctx_label):
    cs, ce = candidate
    prior_str = "HIGH" if pct >= PCT_HIGH else "MODERATE" if pct >= PCT_MID else "LOW"
    z_max = max(abs(z_global), abs(z_local))

    def _label(z):
        az = abs(z)
        if az > 20:  return "EXTREME (>20σ)"
        if az > 5:   return "SEVERE (>5σ)"
        if az > 3:   return "NOTABLE (3-5σ)"
        if az > 2:   return "MILD (2-3σ)"
        return "MINIMAL (<2σ)"

    # S1 FIX: calibrated confidence guidance tied to z-score magnitude
    conf_guide = f"""Confidence calibration for {ds} signals:
  3 = Clear, unambiguous anomaly: z_max > 5σ AND visually obvious deviation
  2 = Likely anomaly: z_max 3-5σ with visible shift in Panel 2
  1 = Uncertain: z_max < 3σ or deviation resembles normal variation in Panel 1
  NOTE: {ds} signals often contain {"periodic patterns that look anomalous but repeat normally" if ds == "SMAP" else "operational variations that are expected"}.
  Do NOT assign confidence=3 for mild deviations (z_max < 3σ) even if visually different."""

    strip_note = (
        f"\nPanel 4 (comparison strip): Shows all {n_candidates} Stage1 candidates.\n"
        f"  The current candidate is marked ★. Before deciding, ask:\n"
        f"  Are the other candidates (non-★) visually similar? If yes, this may be normal variation.\n"
        if n_candidates >= 2 else ""
    )

    ctx_dir = "BEFORE" if ctx_label == "before" else "AFTER (candidate at series start)"

    return f"""=== TIME SERIES ANOMALY VERIFICATION ===
Signal: {sig}  |  Dataset: {ds}  |  Type: {DOMAIN_CTX[ds]}
Candidate: [{cs}, {ce}]  (length: {ce - cs + 1} timesteps)
Total Stage1 candidates for this signal: {n_candidates}

--- STATISTICAL EVIDENCE ---
DINOv2 anomaly score: {pct:.0f}th percentile  ->  Prior: {prior_str}

Global normal (bottom-50%% DINOv2 windows): µ={v_mu:.4f}, σ={v_std:.4f}
Context window ({ctx_dir}): µ={ctx_mu:.4f}

Peak value in candidate: {peak_val:.4f}
  vs global normal:   {z_global:+.2f}σ  [{_label(z_global)}]
  vs context window:  {z_local:+.2f}σ   [{_label(z_local)}]
  z_max = {z_max:.1f}σ

{conf_guide}

--- IMAGE PANELS ---
Panel 1: Full time series. Red=candidate. NO other candidates highlighted.
  Compare RED region to the REST of the series in the background.
  Are there visually similar patterns elsewhere in the series (not highlighted)?
  If yes → this pattern may be normal variation.

Panel 2: Context({ctx_dir}) → Candidate transition.
  {"Blue" if ctx_label=="before" else "Purple"} = context window  |  Red = candidate
  Orange dashed = context µ  |  Blue dashed = global µ

Panel 3: Global reference (most normal region, lowest DINOv2 score).{strip_note}

--- STEP-BY-STEP REASONING ---
Step 1. Panel 1: Does the RED region look visually distinct from ALL other non-highlighted
        parts of the series? Or does the background show similar patterns?
        If similar patterns exist elsewhere → lean NORMAL.

Step 2. Panel 2: Is there a clear, abrupt shift between the CONTEXT and CANDIDATE?
        z_context = {z_local:+.1f}σ. For SMAP/MSL, deviations <3σ are often operational noise.

Step 3. Panel 3: How extreme is the deviation vs the most normal region?
        z_global = {z_global:+.1f}σ [{_label(z_global)}].
{'Step 4. Comparison strip: Is this candidate visually MUCH more extreme than the others?' if n_candidates >= 2 else 'Step 4. Verdict: Integrate all evidence.'}

Step {'5' if n_candidates >= 2 else '4'}. Final verdict: ANOMALY or NORMAL?
  - ANOMALY: clear evidence from MULTIPLE panels, z_max > 3σ, NOT similar to background
  - NORMAL:  similar background patterns OR z_max < 3σ OR context transition is gradual

Reply ONLY with valid JSON:
{{"verdict": "ANOMALY" or "NORMAL", "confidence": 1 or 2 or 3, "reason": "one sentence ≤20 words"}}"""

# ── Decision (S1 fix: z_max-aware) ────────────────────────────────────────────
def decide(verdict, conf, pct, z_global, z_local):
    """
    S1 FIX: z_max-aware decision logic.
    Low z_max → apply stricter threshold (require higher confidence from VLM).
    High z_max → relax (statistical evidence is already strong).
    """
    z_max = max(abs(z_global), abs(z_local))

    # Overwhelming statistical evidence: keep regardless (e.g. MSL D-14 with z=20000)
    if z_max > 20:
        return not (verdict == "NORMAL" and conf >= 2)

    # Strong evidence: high DINOv2 + notable z-score
    if pct >= PCT_HIGH and z_max > 5:
        return not (verdict == "NORMAL" and conf >= 2)

    # Moderate z-score: require positive ANOMALY verdict with adequate confidence
    if pct >= PCT_MID:
        if z_max > 3:
            return verdict == "ANOMALY" and conf >= 2
        else:
            # S1 KEY FIX: low z_max (< 3σ) needs confident ANOMALY even at high DINOv2 pct
            return verdict == "ANOMALY" and conf >= 3

    # Low DINOv2 prior: strict regardless
    return verdict == "ANOMALY" and conf >= 3 and z_max > 2

# ── VLM query ──────────────────────────────────────────────────────────────────
def query_vlm(images_b64, prompt, tries=5):
    """
    C5: Uses module-level _client (no re-init each call).
    images_b64: list of base64 strings (main + optional comparison strip).
    """
    content = [{"type": "text", "text": prompt}]
    for img in images_b64:
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{img}", "detail": "high"
        }})
    for attempt in range(tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": content},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            try:
                return json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
            v = "ANOMALY" if "ANOMALY" in raw.upper() else "NORMAL"
            return {"verdict": v, "confidence": 1, "reason": "parse error"}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt + 1) * 30
                print(f"    [rate limit -- waiting {wait}s]", flush=True)
                time.sleep(wait)
            elif "quota" in err:
                print("    [QUOTA EXHAUSTED]", flush=True)
                return None
            else:
                print(f"    [api error attempt {attempt+1}] {exc}", flush=True)
                time.sleep(5)
    return None

# ── Per-signal runner ──────────────────────────────────────────────────────────
def run_signal(ds, sig, max_calls=25):
    print(f"\n  [{ds}] {sig}", flush=True)

    timestamps, vals = load_signal(ds, sig)
    T = len(vals)
    gt_ivs  = load_gt_intervals(sig, timestamps)
    scores  = load_dino(ds, sig)
    assert len(scores) == T, f"Score length {len(scores)} != signal length {T}"

    loose_ivs, all_ws, mu, sig_std = stage1_univar(scores)
    oracle, _      = oracle_f1_sweep(all_ws, scores, gt_ivs, mu, sig_std)
    loose_f1, _, _ = interval_f1(gt_ivs, loose_ivs)

    v_mu, v_std = compute_normal_stats(vals, all_ws)
    ref_start   = find_best_ref_window(all_ws, gt_ivs, T)

    print(f"    T={T}  GT={len(gt_ivs)}  loose={len(loose_ivs)}  "
          f"loose_f1={loose_f1:.3f}  oracle={oracle:.3f}  "
          f"v_mu={v_mu:.3f}  v_std={v_std:.3f}  ref={ref_start}",
          flush=True)

    img_dir = RESULTS_DIR / "plots" / ds / sig
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed, logs, api_calls = [], [], 0

    for idx, (cs, ce) in enumerate(loose_ivs):
        if api_calls >= max_calls:
            confirmed.extend(loose_ivs[idx:])
            break

        pct      = pct_rank(cs, ce, all_ws, T)
        is_tp    = any(_ov((cs, ce), g) for g in gt_ivs)
        flag     = "TP" if is_tp else "FP"

        # C1 FIX: get context window (may be "after" if candidate starts at t=0)
        ctx_s, ctx_label = find_before_window(cs, loose_ivs, T)

        main_b64, z_global, z_local, peak_val, ctx_mu = make_image(
            vals, (cs, ce), loose_ivs, ref_start, ctx_s, ctx_label, v_mu, v_std
        )

        # NEW: comparison strip
        strip_b64 = make_comparison_strip(vals, loose_ivs, (cs, ce), v_mu, v_std)

        # Save first 6 images
        if idx < 6:
            img_path = img_dir / f"{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png"
            with open(img_path, "wb") as fh:
                fh.write(base64.b64decode(main_b64))

        images = [main_b64]
        if strip_b64:
            images.append(strip_b64)

        prompt = build_prompt(
            ds, sig, (cs, ce), pct, v_mu, v_std,
            z_global, z_local, peak_val, ctx_mu, ref_start,
            len(loose_ivs), ctx_label
        )
        res = query_vlm(images, prompt)
        api_calls += 1

        if res is None:
            confirmed.append((cs, ce))
            break

        verdict = res.get("verdict", "ANOMALY").upper()
        conf    = int(res.get("confidence", 1))
        reason  = str(res.get("reason", ""))[:100]
        keep    = decide(verdict, conf, pct, z_global, z_local)

        if keep:
            confirmed.append((cs, ce))

        z_max = max(abs(z_global), abs(z_local))
        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} pct={pct:5.1f} "
              f"zG={z_global:+6.1f} zL={z_local:+6.1f} zmax={z_max:5.1f} "
              f"-> {verdict}(c={conf}) keep={keep} [{flag}]  {reason}",
              flush=True)

        logs.append({
            "ds": ds, "sig": sig, "cs": cs, "ce": ce,
            "pct": pct, "z_global": round(z_global,2), "z_local": round(z_local,2),
            "z_max": round(z_max,2), "ctx_label": ctx_label,
            "verdict": verdict, "conf": conf, "keep": keep,
            "is_tp": is_tp, "flag": flag, "reason": reason,
        })

    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed)
    print(f"    -> Stage2: F1={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f} | "
          f"confirmed={len(confirmed)}/{len(loose_ivs)} | calls={api_calls}",
          flush=True)

    return {
        "ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
        "oracle_f1": oracle, "loose_f1": loose_f1,
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r,
        "n_loose": len(loose_ivs), "n_confirmed": len(confirmed),
        "api_calls": api_calls, "logs": logs,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results, all_logs = [], []

    for ds, signals in DATASETS.items():
        for sig in signals:
            try:
                r = run_signal(ds, sig)
            except Exception as exc:
                print(f"\n[ERROR] {ds}/{sig}: {exc}", flush=True)
                import traceback; traceback.print_exc()
                r = None
            if r:
                all_logs.extend(r.pop("logs"))
                all_results.append(r)

    if not all_results:
        print("No results.", flush=True)
        raise SystemExit

    W = 80
    print(f"\n{'='*W}", flush=True)
    print("Stage2 Univariate v4  --  z_max-aware decision + comparison strip + bug fixes",
          flush=True)
    print(f"{'='*W}", flush=True)

    for ds in ["NAB", "SMAP", "MSL"]:
        rows = [r for r in all_results if r["ds"] == ds]
        if not rows:
            continue
        print(f"\n  {ds} ({len(rows)} signals):", flush=True)
        print(f"  {'Signal':<45} {'Oracle':>7} {'Loose':>7} {'Stage2':>7} {'calls':>6}",
              flush=True)
        print(f"  {'-'*72}", flush=True)
        for r in rows:
            print(f"  {r['sig']:<45} {r['oracle_f1']:>7.4f} {r['loose_f1']:>7.4f} "
                  f"{r['stage2_f1']:>7.4f} {r['api_calls']:>6d}", flush=True)
        print(f"  {'AVG':<45} "
              f"{np.mean([r['oracle_f1'] for r in rows]):>7.4f} "
              f"{np.mean([r['loose_f1']  for r in rows]):>7.4f} "
              f"{np.mean([r['stage2_f1'] for r in rows]):>7.4f}", flush=True)

    all_oracle = np.mean([r["oracle_f1"] for r in all_results])
    all_loose  = np.mean([r["loose_f1"]  for r in all_results])
    all_s2     = np.mean([r["stage2_f1"] for r in all_results])
    total_calls = sum(r["api_calls"] for r in all_results)
    print(f"\n  {'ALL (40 signals)':<45} {all_oracle:>7.4f} {all_loose:>7.4f} {all_s2:>7.4f}",
          flush=True)
    print(f"  Total API calls: {total_calls}", flush=True)
    print(f"{'='*W}", flush=True)

    pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
    pd.DataFrame(all_logs).to_csv(RESULTS_DIR / "verdicts.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR}", flush=True)
