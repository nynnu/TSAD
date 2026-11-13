"""
Stage2 Univariate v1: DINOv2 Stage1 + GPT-4o Chain-of-Thought Verification
Datasets: NAB (16) + SMAP (13) + MSL (11) = 40 univariate signals

Design:
- Stage1: DINOv2 dino_k5 window scores, top-10% percentile → loose candidate intervals
- Stage2: 3-panel visualization + chain-of-thought prompt → GPT-4o ANOMALY/NORMAL
- Metric: Interval F1 (VLM4TS standard: window-overlap TP/FP/FN)
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
BASE       = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
DINO_DIR   = BASE / "results/VLM4TS_results_dino_ltr/checkpoints"
ANOMS_CSV  = BASE / "data/anomalies.csv"
RESULTS_DIR = BASE / "experiments/results_stage2_univar_v1"

# ── Signal paths (NAB lives in realAWSCloudwatch/) ────────────────────────────
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
LOOSE_PCT  = 10.0   # top 10% window scores → loose candidates
MERGE_GAP  = WIN // 2
MIN_IV_LEN = 10
PCT_HIGH   = 90
PCT_MID    = 75
VLM_SLEEP  = 4.0

# ── Data loading ───────────────────────────────────────────────────────────────
def load_signal(ds, sig):
    df = pd.read_csv(_sig_path(ds, sig))
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_gt_intervals(sig, timestamps):
    """Timestamp-based GT events → index-based intervals."""
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
    """
    Compute sliding window scores, threshold at top LOOSE_PCT%, merge into intervals.
    Returns: loose_ivs, all_ws (per-window), mu, sigma (from lower-80% of all_ws)
    """
    T = len(scores)
    all_ws = np.array([scores[s:s + WIN].mean()
                       for s in range(0, T - WIN + 1, STRIDE)])

    thr = float(np.percentile(all_ws, 100 - LOOSE_PCT))

    # Mark windows above threshold → timestep binary → merge
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

    # Normal baseline: exclude top-20% windows (anomalous) before computing mu/sig
    cutoff = float(np.percentile(all_ws, 80))
    clean = all_ws[all_ws <= cutoff]
    if len(clean) < 5:
        clean = all_ws
    mu  = float(clean.mean())
    sig = float(clean.std()) if clean.std() > 1e-9 else 1e-9

    return loose_ivs, all_ws, mu, sig

def oracle_f1_sweep(all_ws, scores, gt_ivs, mu, sig):
    """Best interval F1 over alpha threshold sweep (upper bound)."""
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
    """Percentile rank of candidate's peak window score among all_ws."""
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

# ── Reference normal window ────────────────────────────────────────────────────
def find_ref_window(gt_ivs, T):
    """First WIN-length window not overlapping any GT interval."""
    for s in range(0, T - WIN + 1, STRIDE):
        w = (s, s + WIN - 1)
        if not any(_ov(w, g) for g in gt_ivs):
            return s
    return 0

# ── Visualization ──────────────────────────────────────────────────────────────
def make_image(vals, candidate, gt_ivs, loose_ivs, ref_start):
    """
    3-panel figure:
      Panel 1 (top):    Full series + candidate (red) + other loose (orange) + µ±2σ
      Panel 2 (middle): Zoomed candidate + context + µ±2σ + peak marker
      Panel 3 (bottom): Reference normal window (green, gray bg)
    """
    cs, ce = candidate
    T = len(vals)

    # Reference normal statistics (from the reference window)
    ref_vals = vals[ref_start:ref_start + WIN]
    v_mu  = float(ref_vals.mean())
    v_std = float(ref_vals.std()) if ref_vals.std() > 1e-9 else 1e-9

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)

    # ── Panel 1: Full series ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(vals, color="#333333", lw=0.5, alpha=0.85)
    # Other loose candidates (orange, behind red)
    for ls, le in loose_ivs:
        if (ls, le) != (cs, ce):
            ax.axvspan(ls, le, alpha=0.18, color="orange")
    # GT intervals (light green)
    for gs, ge in gt_ivs:
        ax.axvspan(gs, ge, alpha=0.15, color="green")
    # Candidate (red)
    ax.axvspan(cs, ce, alpha=0.40, color="red", label=f"Candidate [{cs},{ce}]")
    # µ±2σ bands
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.7)
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.08, color="steelblue")
    ax.set_title(f"Panel 1: Full series (T={T})  |  Red=Candidate  Orange=Other candidates  Green=GT",
                 fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.tick_params(labelsize=7)

    # ── Panel 2: Zoomed candidate + context ───────────────────────────────────
    ctx = max(WIN // 2, (ce - cs + 1))
    z_s = max(0, cs - ctx)
    z_e = min(T - 1, ce + ctx)
    ax = axes[1]
    x_zoom = np.arange(z_s, z_e + 1)
    ax.plot(x_zoom, vals[z_s:z_e + 1], color="#333333", lw=1.1)
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.7, label=f"µ={v_mu:.3f}")
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.10, color="steelblue",
               label=f"±2σ (σ={v_std:.3f})")
    ax.axvspan(cs, ce, alpha=0.35, color="red", label="Candidate")
    # Peak value marker
    cand_vals = vals[cs:ce + 1]
    peak_rel  = int(np.argmax(np.abs(cand_vals - v_mu)))
    peak_idx  = cs + peak_rel
    peak_val  = float(cand_vals[peak_rel])
    z_score   = (peak_val - v_mu) / v_std
    ax.axvline(peak_idx, color="darkred", lw=1.1, ls=":",
               label=f"Peak={peak_val:.3f} ({z_score:+.1f}σ)")
    ax.set_title(f"Panel 2: Zoomed [{z_s},{z_e}]  |  Dashed=µ, Band=±2σ  |  "
                 f"Peak z-score={z_score:+.1f}σ", fontsize=8)
    ax.set_xlim(z_s, z_e)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.tick_params(labelsize=7)

    # ── Panel 3: Reference normal ──────────────────────────────────────────────
    ax = axes[2]
    ref_x = np.arange(ref_start, ref_start + WIN)
    ax.plot(ref_x, vals[ref_start:ref_start + WIN], color="#2ca02c", lw=1.1,
            label=f"Normal ref [{ref_start},{ref_start+WIN-1}]")
    ax.axhline(v_mu, color="steelblue", lw=0.9, ls="--", alpha=0.7)
    ax.axhspan(v_mu - 2 * v_std, v_mu + 2 * v_std, alpha=0.10, color="steelblue")
    ax.set_title(f"Panel 3: Reference normal window [{ref_start},{ref_start+WIN-1}]  "
                 f"|  µ={v_mu:.3f}  σ={v_std:.3f}", fontsize=8)
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(ref_start, ref_start + WIN - 1)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are an expert time series anomaly analyst specializing in "
    "infrastructure and scientific telemetry monitoring."
)

def build_prompt(ds, sig, candidate, vals, pct, ref_start):
    cs, ce = candidate
    ref_vals = vals[ref_start:ref_start + WIN]
    v_mu  = float(ref_vals.mean())
    v_std = float(ref_vals.std()) if ref_vals.std() > 1e-9 else 1e-9

    cand_vals = vals[cs:ce + 1]
    peak_val  = float(cand_vals[np.argmax(np.abs(cand_vals - v_mu))])
    z_score   = (peak_val - v_mu) / v_std
    prior_str = "HIGH" if pct >= PCT_HIGH else "MODERATE" if pct >= PCT_MID else "LOW"

    deviation_label = (
        "SEVERE (>4σ)"   if abs(z_score) > 4 else
        "NOTABLE (2-4σ)" if abs(z_score) > 2 else
        "MILD (<2σ)"
    )

    return f"""=== TIME SERIES ANOMALY VERIFICATION ===
Signal: {sig}  |  Dataset: {ds}  |  Type: {DOMAIN_CTX[ds]}
Candidate interval: [{cs}, {ce}]  (length: {ce - cs + 1} timesteps)

--- STATISTICAL EVIDENCE ---
DINOv2 anomaly score: {pct:.0f}th percentile  →  Prior: {prior_str}
  (higher percentile = more visually anomalous compared to all other windows)

Reference normal window (Panel 3):  µ={v_mu:.4f},  σ={v_std:.4f}
Peak value in candidate:            {peak_val:.4f}
Peak z-score:                       {z_score:+.1f}σ  [{deviation_label}]

--- IMAGE PANELS ---
Panel 1 (top):    Full time series.
  • Red shading   = Candidate interval [{cs},{ce}] under review
  • Orange shading = Other Stage1 candidates (context)
  • Green shading  = Ground-truth anomaly intervals (for reference only)
  • Blue dashed line = reference µ;  light blue band = ±2σ range

Panel 2 (middle): Zoomed view of candidate + surrounding context.
  • Red region = candidate.  Dotted vertical = peak value.
  • Blue band = ±2σ range based on the normal reference window.

Panel 3 (bottom, gray bg): Confirmed normal reference window [{ref_start},{ref_start + WIN - 1}].
  • Green line = normal signal behavior.  Use this as your baseline.

--- STEP-BY-STEP REASONING ---
Step 1. Panel 1: Is the red candidate region visually distinct from the surrounding signal?
Step 2. Panel 2: Does the peak value breach the ±2σ band? By how much?
Step 3. Panel 3: Compare the candidate's shape/amplitude to the normal reference. What differs?
Step 4. Verdict: considering all evidence above, is this candidate ANOMALY or NORMAL?

Reply ONLY with valid JSON (no markdown, no extra text):
{{"verdict": "ANOMALY" or "NORMAL", "confidence": 1 or 2 or 3, "reason": "one sentence max 20 words"}}

Confidence scale: 1=uncertain, 2=likely, 3=clear evidence"""

# ── Decision ───────────────────────────────────────────────────────────────────
def decide(verdict, conf, pct):
    if pct >= PCT_HIGH:
        # High prior: keep as ANOMALY unless VLM is at least moderately confident it's normal
        return not (verdict == "NORMAL" and conf >= 2)
    elif pct >= PCT_MID:
        return verdict == "ANOMALY" and conf >= 2
    else:
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
                print(f"    [rate limit — waiting {wait}s]", flush=True)
                time.sleep(wait)
            elif "quota" in err:
                print("    [QUOTA EXHAUSTED]", flush=True)
                return None
            else:
                print(f"    [api error attempt {attempt+1}] {exc}", flush=True)
                time.sleep(5)
    return None

# ── Per-signal runner ──────────────────────────────────────────────────────────
def run_signal(ds, sig, max_calls=20):
    print(f"\n  [{ds}] {sig}", flush=True)

    timestamps, vals = load_signal(ds, sig)
    T = len(vals)
    gt_ivs  = load_gt_intervals(sig, timestamps)
    scores  = load_dino(ds, sig)
    assert len(scores) == T, f"Score length {len(scores)} != signal length {T}"

    loose_ivs, all_ws, mu, sig_std = stage1_univar(scores)
    oracle, oracle_ivs = oracle_f1_sweep(all_ws, scores, gt_ivs, mu, sig_std)
    loose_f1, _, _     = interval_f1(gt_ivs, loose_ivs)
    ref_start          = find_ref_window(gt_ivs, T)

    print(f"    T={T}  GT={len(gt_ivs)}  loose={len(loose_ivs)}  "
          f"loose_f1={loose_f1:.3f}  oracle={oracle:.3f}  ref_win={ref_start}",
          flush=True)

    img_dir = RESULTS_DIR / "plots" / ds / sig
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed, logs, api_calls = [], [], 0

    for idx, (cs, ce) in enumerate(loose_ivs):
        if api_calls >= max_calls:
            # Exceeded call budget: carry remaining candidates through unverified
            confirmed.extend(loose_ivs[idx:])
            break

        pct   = pct_rank(cs, ce, all_ws, T)
        is_tp = any(_ov((cs, ce), g) for g in gt_ivs)
        flag  = "TP" if is_tp else "FP"

        img_b64 = make_image(vals, (cs, ce), gt_ivs, loose_ivs, ref_start)

        # Save first 6 images for diagnosis
        if idx < 6:
            img_path = img_dir / f"{idx:02d}_{cs}_{ce}_{flag}_p{pct:.0f}.png"
            with open(img_path, "wb") as fh:
                fh.write(base64.b64decode(img_b64))

        prompt = build_prompt(ds, sig, (cs, ce), vals, pct, ref_start)
        res    = query_vlm(img_b64, prompt)
        api_calls += 1

        if res is None:
            # API failure: keep candidate and stop
            confirmed.append((cs, ce))
            break

        verdict = res.get("verdict", "ANOMALY").upper()
        conf    = int(res.get("confidence", 1))
        reason  = str(res.get("reason", ""))[:100]
        keep    = decide(verdict, conf, pct)

        if keep:
            confirmed.append((cs, ce))

        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} pct={pct:5.1f} "
              f"-> {verdict}(c={conf}) keep={keep} [{flag}]  {reason}",
              flush=True)

        logs.append({
            "ds": ds, "sig": sig, "cs": cs, "ce": ce,
            "pct": pct, "verdict": verdict, "conf": conf,
            "keep": keep, "is_tp": is_tp, "flag": flag, "reason": reason,
        })

    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed)
    print(f"    → Stage2: F1={s2_f1:.4f} P={s2_p:.2f} R={s2_r:.2f} | "
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

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 80
    print(f"\n{'='*W}", flush=True)
    print("Stage2 Univariate v1  --  DINOv2 Stage1 + GPT-4o Chain-of-Thought", flush=True)
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
        ds_oracle = np.mean([r["oracle_f1"] for r in rows])
        ds_loose  = np.mean([r["loose_f1"]  for r in rows])
        ds_s2     = np.mean([r["stage2_f1"] for r in rows])
        print(f"  {'AVG':<45} {ds_oracle:>7.4f} {ds_loose:>7.4f} {ds_s2:>7.4f}", flush=True)

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
    print(f"\nSaved → {RESULTS_DIR}", flush=True)
