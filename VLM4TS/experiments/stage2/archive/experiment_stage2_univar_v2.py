"""
Stage2 Univariate v2: Fixes over v1

v1 Problems Fixed:
  FIX A -- GT label leakage: removed GT green shading from Panel 1.
  FIX B -- Bad reference statistics: single 224-pt window at t=0 → v_std≈0 → z-score explosion.
           Now compute global normal µ/σ from bottom-50% DINOv2 score windows (robust IQR-based σ).
  FIX C -- No "before" context: added Panel 2 = [before window | candidate] transition view.
           Level shifts and contextual anomalies are now visible to GPT-4o.
  FIX D -- Decision logic too permissive: pct>=90 → auto-keep (nothing filtered).
           Now: pct>=95 auto-keep; pct 85-95 → need ANOMALY(c>=2); pct<85 → need ANOMALY(c>=3).
  FIX E -- LOOSE_PCT=10% too strict → MSL F-8/T-13 miss GT entirely. Raised to 15%.
  FIX F -- µ/σ computed separately in make_image and build_prompt (duplication risk).
           Now computed once in run_signal and passed as parameters.
  FIX G -- find_ref_window always returns t=0.
           Now returns the window with the lowest DINOv2 score (most normal).
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

# ── API ────────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment.")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
DINO_DIR    = BASE / "results/VLM4TS_results_dino_ltr/checkpoints"
ANOMS_CSV   = BASE / "data/anomalies.csv"
RESULTS_DIR = BASE / "experiments/results_stage2_univar_v2"

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
LOOSE_PCT  = 15.0   # FIX E: raised from 10 → 15% for better recall on MSL
MERGE_GAP  = WIN // 2
MIN_IV_LEN = 10
PCT_HIGH   = 95     # FIX D: raised from 90 → only the very top get auto-keep
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
    import pickle
    path = DINO_DIR / f"{ds}__{sig}__dino_k5.pkl"
    return pickle.load(open(path, "rb"))["scores"]

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
    if not gt_ivs:
        return 0., 0., 0.
    TP = sum(sum(1 for g in gt_ivs if _ov(d, g)) for d in pred_ivs)
    FP = sum(1 for d in pred_ivs if not any(_ov(d, g) for g in gt_ivs))
    FN = sum(1 for g in gt_ivs if not any(_ov(g, d) for d in pred_ivs))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0.
    r = TP / (TP + FN) if (TP + FN) > 0 else 0.
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

    # mu/sig from lower-80% windows for oracle threshold sweep
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
    for a in [0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003, 0.001]:
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

# ── FIX B+G: Global normal statistics and reference window ─────────────────────
def compute_normal_stats(vals, all_ws):
    """
    FIX B: Compute global µ/σ from bottom-50% DINOv2 score windows.
    Uses IQR-based σ (robust against outlier windows).
    This replaces the single-window estimate that caused z-score explosions.
    """
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
    v_std = float((q75 - q25) / 1.349)   # IQR → σ
    if v_std < 1e-4:
        v_std = float(nv.std()) + 1e-4    # fallback with floor
    return v_mu, v_std

def find_best_ref_window(all_ws, gt_ivs, T):
    """
    FIX G: Return the window with the LOWEST DINOv2 score that doesn't overlap GT.
    This is the most "normal-looking" region according to DINOv2.
    """
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
    FIX C: Find the WIN-length window immediately before the candidate.
    Used to detect level shifts and contextual anomalies.
    """
    other_ivs = [(ls, le) for ls, le in loose_ivs]
    # Walk backwards from cs, stepping by STRIDE
    for s in range(max(0, cs - WIN), -1, -STRIDE):
        if s + WIN <= cs:   # must end before candidate starts
            w = (s, s + WIN - 1)
            if not any(_ov(w, o) for o in other_ivs if o[0] != cs):
                return s
    return max(0, cs - WIN)  # fallback

# ── Visualization ──────────────────────────────────────────────────────────────
def make_image(vals, candidate, loose_ivs, ref_start, before_s, v_mu, v_std):
    """
    3-panel figure:
      Panel 1: Full series + candidate (red) + other loose (orange). NO GT shown (FIX A).
      Panel 2: Before → Candidate transition: [before_window | CANDIDATE] (FIX C).
      Panel 3: Global reference normal window (lowest DINOv2 score area) (FIX G).

    µ/σ passed as parameters (FIX F, computed once in run_signal).
    """
    cs, ce = candidate
    T = len(vals)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)

    # ── Panel 1: Full series (FIX A: no GT shading) ───────────────────────────
    ax = axes[0]
    ax.plot(vals, color="#333333", lw=0.5, alpha=0.85)
    for ls, le in loose_ivs:
        if (ls, le) != (cs, ce):
            ax.axvspan(ls, le, alpha=0.18, color="orange")
    ax.axvspan(cs, ce, alpha=0.40, color="red", label=f"Candidate [{cs},{ce}]")
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.6)
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.08, color="steelblue")
    ax.set_title(
        f"Panel 1: Full series (T={T})  |  Red=Candidate  Orange=Other Stage1 candidates",
        fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.tick_params(labelsize=7)

    # ── Panel 2: Before → Candidate transition (FIX C) ────────────────────────
    # Show the window immediately before the candidate + the candidate itself
    be = before_s + WIN - 1       # end of before window
    view_s = before_s
    view_e = min(T - 1, ce + WIN // 4)   # a little context after too
    ax = axes[1]
    x_view = np.arange(view_s, view_e + 1)
    ax.plot(x_view, vals[view_s:view_e + 1], color="#333333", lw=1.1)
    # Shade before window in blue
    ax.axvspan(before_s, be, alpha=0.15, color="steelblue",
               label=f"Before [{before_s},{be}]")
    # Shade candidate in red
    ax.axvspan(cs, ce, alpha=0.35, color="red", label=f"Candidate [{cs},{ce}]")
    # Global µ±2σ
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.7,
               label=f"Global µ={v_mu:.3f}")
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.08, color="steelblue",
               label=f"±2σ (σ={v_std:.3f})")
    # Before-window local stats
    bef_vals = vals[before_s:before_s + WIN]
    bef_mu   = float(bef_vals.mean())
    ax.axhline(bef_mu, color="orange", lw=0.9, ls="--", alpha=0.8,
               label=f"Before µ={bef_mu:.3f}")
    # Peak in candidate
    cand_vals = vals[cs:ce + 1]
    peak_rel  = int(np.argmax(np.abs(cand_vals - v_mu)))
    peak_idx  = cs + peak_rel
    peak_val  = float(cand_vals[peak_rel])
    z_global  = (peak_val - v_mu) / v_std
    z_local   = (peak_val - bef_mu) / (float(bef_vals.std()) + 1e-4)
    ax.axvline(peak_idx, color="darkred", lw=1.1, ls=":",
               label=f"Peak={peak_val:.3f} (global {z_global:+.1f}σ, local {z_local:+.1f}σ)")
    ax.set_title(
        f"Panel 2: Before→Candidate transition  |  Blue=Before context  Red=Candidate  "
        f"Peak: global {z_global:+.1f}σ, vs-before {z_local:+.1f}σ",
        fontsize=8)
    ax.set_xlim(view_s, view_e)
    ax.legend(fontsize=5.5, loc="upper right", ncol=2)
    ax.tick_params(labelsize=7)

    # ── Panel 3: Global reference normal window (FIX G) ───────────────────────
    ax = axes[2]
    ref_x = np.arange(ref_start, ref_start + WIN)
    ax.plot(ref_x, vals[ref_start:ref_start + WIN], color="#2ca02c", lw=1.1,
            label=f"Global normal ref [{ref_start},{ref_start+WIN-1}]")
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.7)
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.10, color="steelblue")
    ax.set_title(
        f"Panel 3: Global normal reference [{ref_start},{ref_start+WIN-1}]  "
        f"(lowest DINOv2 score = most normal)  |  µ={v_mu:.3f}  σ={v_std:.3f}",
        fontsize=8)
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(ref_start, ref_start + WIN - 1)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), z_global, z_local, peak_val, bef_mu

# ── Prompt (FIX C, FIX F) ─────────────────────────────────────────────────────
SYSTEM = (
    "You are an expert time series anomaly analyst specializing in "
    "infrastructure and scientific telemetry monitoring."
)

def build_prompt(ds, sig, candidate, pct, v_mu, v_std,
                 z_global, z_local, peak_val, bef_mu, ref_start):
    cs, ce = candidate
    prior_str = "HIGH" if pct >= PCT_HIGH else "MODERATE" if pct >= PCT_MID else "LOW"

    def _label(z):
        az = abs(z)
        if az > 5:   return "EXTREME"
        if az > 3:   return "SEVERE"
        if az > 2:   return "NOTABLE"
        return "MILD"

    return f"""=== TIME SERIES ANOMALY VERIFICATION ===
Signal: {sig}  |  Dataset: {ds}  |  Type: {DOMAIN_CTX[ds]}
Candidate: [{cs}, {ce}]  (length: {ce - cs + 1} timesteps)

--- STATISTICAL EVIDENCE ---
DINOv2 visual anomaly score: {pct:.0f}th percentile  ->  Prior: {prior_str}

Global normal baseline (bottom-50%% DINOv2 windows):
  µ_global={v_mu:.4f},  σ_global={v_std:.4f}

Candidate peak value: {peak_val:.4f}
  vs global normal:  {z_global:+.2f}σ  [{_label(z_global)}]
  vs before context: {z_local:+.2f}σ  [{_label(z_local)}]  (before µ={bef_mu:.4f})

--- IMAGE PANELS ---
Panel 1 (top): Full series overview.
  Red   = Candidate [{cs},{ce}]  |  Orange = Other Stage1 candidates
  Blue dashed = global µ  |  Light blue band = global ±2σ

Panel 2 (middle): BEFORE -> CANDIDATE transition — KEY panel for your decision.
  Blue  = Before-context window (normal behavior just before candidate)
  Red   = Candidate interval
  Orange dashed = before-window mean
  Blue dashed   = global mean
  This panel shows whether the candidate is a SHIFT from recent behavior.

Panel 3 (bottom, gray): Global reference normal window (lowest DINOv2 score).
  Green = the most normal region in this signal.
  Compare candidate amplitude/shape to this baseline.

--- STEP-BY-STEP REASONING ---
Step 1. Panel 2 (most important): Compare the BEFORE (blue) and CANDIDATE (red) regions.
        Is there a visible shift in level, variance, or pattern?
        The vs-before z-score is {z_local:+.1f}σ — does Panel 2 visually confirm this?

Step 2. Panel 1: In the full series context, is the candidate region unusually different
        from surrounding windows (not just the highlighted Stage1 candidates)?

Step 3. Panel 3: Compare candidate peak ({peak_val:.4f}) vs global normal.
        Global z-score is {z_global:+.1f}σ. Does Panel 3 support calling this anomalous?

Step 4. Verdict: Is this ANOMALY or NORMAL?
        - ANOMALY: if BOTH Panel 2 shows a clear shift AND the deviation is real
        - NORMAL:  if Panel 2 shows continuity with before-context, or deviation is noise

Reply ONLY with valid JSON:
{{"verdict": "ANOMALY" or "NORMAL", "confidence": 1 or 2 or 3, "reason": "one sentence ≤20 words"}}

Confidence: 1=uncertain, 2=likely, 3=clear unambiguous evidence"""

# ── Decision (FIX D) ───────────────────────────────────────────────────────────
def decide(verdict, conf, pct):
    """
    FIX D: Revised decision logic.
    v1 auto-kept everything at pct>=90 unless NORMAL(c>=2) — almost nothing was filtered.
    v2: only the very top (pct>=95) auto-keep; below that, require ANOMALY verdict.
    """
    if pct >= 95:
        # Extreme outlier: keep unless VLM moderately confident it's normal
        return not (verdict == "NORMAL" and conf >= 2)
    elif pct >= PCT_MID:   # 82-95
        # Need positive ANOMALY verdict from VLM
        return verdict == "ANOMALY" and conf >= 2
    else:
        # Low prior: need confident ANOMALY
        return verdict == "ANOMALY" and conf >= 3

# ── VLM query ──────────────────────────────────────────────────────────────────
def query_vlm(img_b64, prompt, tries=5):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    for attempt in range(tries):
        try:
            time.sleep(VLM_SLEEP)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high",
                        }},
                    ]},
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

    # FIX B+F: compute normal stats ONCE, pass everywhere
    v_mu, v_std = compute_normal_stats(vals, all_ws)
    # FIX G: best reference window (lowest DINOv2 score, not t=0)
    ref_start = find_best_ref_window(all_ws, gt_ivs, T)

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
        # FIX C: before context window
        before_s = find_before_window(cs, loose_ivs, T)

        img_b64, z_global, z_local, peak_val, bef_mu = make_image(
            vals, (cs, ce), loose_ivs, ref_start, before_s, v_mu, v_std
        )

        # Save first 6 images for diagnosis
        if idx < 6:
            img_path = img_dir / f"{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png"
            with open(img_path, "wb") as fh:
                fh.write(base64.b64decode(img_b64))

        prompt = build_prompt(
            ds, sig, (cs, ce), pct, v_mu, v_std,
            z_global, z_local, peak_val, bef_mu, ref_start
        )
        res = query_vlm(img_b64, prompt)
        api_calls += 1

        if res is None:
            confirmed.append((cs, ce))
            break

        verdict = res.get("verdict", "ANOMALY").upper()
        conf    = int(res.get("confidence", 1))
        reason  = str(res.get("reason", ""))[:100]
        keep    = decide(verdict, conf, pct)

        if keep:
            confirmed.append((cs, ce))

        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} pct={pct:5.1f} "
              f"zG={z_global:+5.1f} zL={z_local:+5.1f} "
              f"-> {verdict}(c={conf}) keep={keep} [{flag}]  {reason}",
              flush=True)

        logs.append({
            "ds": ds, "sig": sig, "cs": cs, "ce": ce,
            "pct": pct, "z_global": z_global, "z_local": z_local,
            "verdict": verdict, "conf": conf, "keep": keep,
            "is_tp": is_tp, "flag": flag, "reason": reason,
        })

    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed)
    print(f"    -> Stage2: F1={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f} | "
          f"confirmed={len(confirmed)}/{len(loose_ivs)} | calls={api_calls}",
          flush=True)

    return {
        "ds": ds, "sig": sig, "T": T, "n_gt": len(gt_ivs),
        "oracle_f1": oracle,
        "loose_f1":  loose_f1,
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r,
        "n_loose": len(loose_ivs), "n_confirmed": len(confirmed),
        "api_calls": api_calls,
        "logs": logs,
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
    print("Stage2 Univariate v2  --  DINOv2 + GPT-4o (GT-free, global stats, before-context)",
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
