"""
VLM4TS Official — wrapper experiment script.

Runs the official VLM4TS pipeline (ZLHe0/VLM4TS) on our 40 signals
(NAB realAWSCloudwatch / SMAP / MSL) and evaluates with both:
  (A) their original evaluate_intervals (TP=pair counting)
  (B) our C2-fixed interval_f1 (TP_pred/TP_gt separated)

Differences vs. our v4:
  Stage1 : ViT-B-16 CLIP (theirs) vs. DINOv2 dino_k5 (ours)
  Stage2 : one full-series image → GPT-4o outputs interval_index list (theirs)
           vs. per-candidate 3-panel → binary ANOMALY/NORMAL (ours)
  API    : responses.create() was SDK 2.x only; adapted to chat.completions for SDK 1.x
"""

import ast, base64, io, json, os, re, sys, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OFFICIAL_SRC = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS_official\src")
ANOMS_CSV    = BASE / "data/anomalies.csv"
RESULTS_DIR  = BASE / "experiments/results_vlm4ts_official"

# Add official src to path so we can import their modules
sys.path.insert(0, str(OFFICIAL_SRC))

# ── OpenAI (SDK 1.x compatible) ───────────────────────────────────────────────
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY in environment.")
_client = OpenAI(api_key=API_KEY)

# ── Datasets (same as our experiments) ────────────────────────────────────────
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

def _sig_path(ds, sig):
    if ds == "NAB":
        return BASE / "data/realAWSCloudwatch" / f"{sig}.csv"
    return BASE / "data" / ds / f"{sig}.csv"

# ── Data loading ───────────────────────────────────────────────────────────────
def load_signal(ds, sig):
    df = pd.read_csv(_sig_path(ds, sig))
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_gt_intervals(sig, timestamps):
    anoms = pd.read_csv(ANOMS_CSV)
    row   = anoms[anoms["signal"] == sig]
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

# ── Interval helpers ───────────────────────────────────────────────────────────
def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

def interval_f1_official(gt_ivs, pred_ivs):
    """Their formula: TP counts (pred, GT) pairs (same as v3 bug baseline)."""
    if not gt_ivs:
        return 0., 0., 0.
    TP = sum(sum(1 for g in gt_ivs if _ov(d, g)) for d in pred_ivs)
    FP = sum(1 for d in pred_ivs if not any(_ov(d, g) for g in gt_ivs))
    FN = sum(1 for g in gt_ivs if not any(_ov(g, d) for d in pred_ivs))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0.
    r = TP / (TP + FN) if (TP + FN) > 0 else 0.
    return (2*p*r/(p+r) if p+r else 0.), p, r

def interval_f1_fixed(gt_ivs, pred_ivs):
    """Our C2-fixed formula: TP_pred / TP_gt separated."""
    if not gt_ivs:
        return 0., 0., 0.
    TP_pred = sum(1 for d in pred_ivs if any(_ov(d, g) for g in gt_ivs))
    TP_gt   = sum(1 for g in gt_ivs  if any(_ov(g, d) for d in pred_ivs))
    FP      = sum(1 for d in pred_ivs if not any(_ov(d, g) for g in gt_ivs))
    FN      = sum(1 for g in gt_ivs  if not any(_ov(g, d) for d in pred_ivs))
    p = TP_pred / (TP_pred + FP) if (TP_pred + FP) > 0 else 0.
    r = TP_gt   / (TP_gt   + FN) if (TP_gt   + FN) > 0 else 0.
    return (2*p*r/(p+r) if p+r else 0.), p, r

# ── Stage1: ViT4TS ─────────────────────────────────────────────────────────────
def run_vit4ts_stage1(timestamps, vals, alpha=0.01):
    """
    Runs the official ViT4TS detector from their codebase.
    Returns list of (start_idx, end_idx) tuples.
    """
    from models.vit4ts import ViT4TS

    df = pd.DataFrame({"timestamp": timestamps, "value": vals})
    detector = ViT4TS(alpha=alpha, verbose=False)
    result = detector.detect(df)
    if result.empty:
        return []
    ivs = []
    for _, row in result.iterrows():
        s = int(np.searchsorted(timestamps, row["start"], side="left"))
        e = int(np.searchsorted(timestamps, row["end"],   side="right") - 1)
        s = max(0, min(s, len(timestamps) - 1))
        e = max(0, min(e, len(timestamps) - 1))
        if s <= e:
            ivs.append((s, e))
    return ivs

# ── Stage2: VLM4TS (SDK 1.x adapted) ─────────────────────────────────────────
VLM_PROMPT = """You are an expert in both time-series analysis and multimodal (vision + language) reasoning.  You will be shown:

1. **A plot of raw time-series data**
   - X-axis: time step index
   - Y-axis: signal value over time

2. **Preliminary "vision-based" anomaly windows**
   - A list of intervals detected by a coarse, purely visual model
   - These may include false positives (locally odd but globally normal) and false negatives (statistically or contextually anomalous but visually subtle)

Your goal is to **integrate both sources**—the visual plot and the preliminary windows—and produce a **refined, final anomaly detection** for the entire series. Specifically:
- **Eliminate** any preliminary windows that look anomalous in isolation but are consistent with the overall trend.
- **Add** any intervals that the visual model missed but which break temporal continuity or exhibit clear statistical irregularities (spikes, level shifts, abrupt changes).

**Response format**
Reply **only** with a JSON object containing these fields:

1. `"interval_index"`:
   An array of `[start, end]` pairs (inclusive indices) for each detected anomaly.
   If there are no anomalies, return [].

2. `"confidence"`:
   A parallel array of integers (one per interval) on a 1-3 scale:
   - 1 = Low confidence (~50-70%)
   - 2 = Medium confidence (~70-95%)
   - 3 = High confidence (>95%)
   If no anomalies, return [].

3. `"abnormal_description"`:
   A single paragraph (less than 100 words) summarizing why these intervals are anomalous.

**Important**
- Estimate interval boundaries using the tick marks on the x-axis as precisely as possible.
- The very first segment may appear atypical due to slicing; do not flag it without clear anomaly evidence.
- Do not include any extra keys or commentary—only the JSON object above.
"""

def make_full_series_image(vals):
    """Generate full time series plot (identical to their _generate_full_plot)."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=dpi)
    ax.plot(np.arange(len(vals)), vals, color="black", lw=0.6, label="Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series")
    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def run_vlm4ts_stage2(timestamps, vals, stage1_ivs, sleep=4.0, tries=5):
    """
    Runs their Stage2 VLM verification.
    Adapted from their vlm4ts.py to use SDK 1.x chat.completions.create()
    instead of SDK 2.x responses.create().
    """
    img_b64     = make_full_series_image(vals)
    prompt_text = VLM_PROMPT + f"\nVision-based model detected intervals (indices): {stage1_ivs}"

    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{img_b64}", "detail": "high"
        }},
    ]

    for attempt in range(tries):
        try:
            time.sleep(sleep)
            resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=500,
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
            return {"interval_index": stage1_ivs, "confidence": [], "abnormal_description": "parse error"}
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
def run_signal(ds, sig, alpha=0.01):
    print(f"\n  [{ds}] {sig}", flush=True)

    timestamps, vals = load_signal(ds, sig)
    gt_ivs           = load_gt_intervals(sig, timestamps)

    # Stage1: ViT4TS
    try:
        s1_ivs = run_vit4ts_stage1(timestamps, vals, alpha=alpha)
    except Exception as exc:
        print(f"    [Stage1 ERROR] {exc}", flush=True)
        return None

    s1_f1_off, _, _ = interval_f1_official(gt_ivs, s1_ivs)
    s1_f1_fix, _, _ = interval_f1_fixed(gt_ivs, s1_ivs)
    print(f"    Stage1: {len(s1_ivs)} intervals | F1_official={s1_f1_off:.4f} F1_fixed={s1_f1_fix:.4f}", flush=True)
    print(f"    Stage1 intervals: {s1_ivs}", flush=True)

    # Stage2: VLM4TS
    res = run_vlm4ts_stage2(timestamps, vals, s1_ivs)
    if res is None:
        confirmed = s1_ivs
    else:
        raw_ivs = res.get("interval_index", [])
        confirmed = []
        for iv in raw_ivs:
            if isinstance(iv, (list, tuple)) and len(iv) == 2:
                s, e = int(iv[0]), int(iv[1])
                s = max(0, min(s, len(timestamps) - 1))
                e = max(0, min(e, len(timestamps) - 1))
                if s <= e:
                    confirmed.append((s, e))
        print(f"    VLM output: {confirmed}", flush=True)

    s2_f1_off, s2_p_off, s2_r_off = interval_f1_official(gt_ivs, confirmed)
    s2_f1_fix, s2_p_fix, s2_r_fix = interval_f1_fixed(gt_ivs, confirmed)

    print(f"    Stage2: {len(confirmed)} intervals | "
          f"F1_official={s2_f1_off:.4f} (P={s2_p_off:.2f} R={s2_r_off:.2f}) | "
          f"F1_fixed={s2_f1_fix:.4f}", flush=True)

    return {
        "ds": ds, "sig": sig, "T": len(timestamps), "n_gt": len(gt_ivs),
        "stage1_f1_official": s1_f1_off,
        "stage1_f1_fixed":    s1_f1_fix,
        "stage2_f1_official": s2_f1_off,
        "stage2_f1_fixed":    s2_f1_fix,
        "n_stage1": len(s1_ivs), "n_stage2": len(confirmed),
        "description": res.get("abnormal_description","") if res else "",
    }

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for ds, signals in DATASETS.items():
        for sig in signals:
            try:
                r = run_signal(ds, sig)
            except Exception as exc:
                print(f"\n[ERROR] {ds}/{sig}: {exc}", flush=True)
                import traceback; traceback.print_exc()
                r = None
            if r:
                all_results.append(r)

    if not all_results:
        print("No results.", flush=True)
        raise SystemExit

    W = 90
    print(f"\n{'='*W}", flush=True)
    print("VLM4TS Official — ViT-B-16 Stage1 + GPT-4o full-series Stage2", flush=True)
    print(f"{'='*W}", flush=True)
    print(f"{'':2} {'Signal':<45} {'S1-off':>7} {'S1-fix':>7} {'S2-off':>7} {'S2-fix':>7}", flush=True)
    print(f"{'='*W}", flush=True)

    for ds in ["NAB", "SMAP", "MSL"]:
        rows = [r for r in all_results if r["ds"] == ds]
        if not rows:
            continue
        print(f"\n  {ds} ({len(rows)} signals):", flush=True)
        for r in rows:
            print(f"  {r['sig']:<45} "
                  f"{r['stage1_f1_official']:>7.4f} {r['stage1_f1_fixed']:>7.4f} "
                  f"{r['stage2_f1_official']:>7.4f} {r['stage2_f1_fixed']:>7.4f}", flush=True)
        print(f"  {'AVG':<45} "
              f"{np.mean([r['stage1_f1_official'] for r in rows]):>7.4f} "
              f"{np.mean([r['stage1_f1_fixed']    for r in rows]):>7.4f} "
              f"{np.mean([r['stage2_f1_official'] for r in rows]):>7.4f} "
              f"{np.mean([r['stage2_f1_fixed']    for r in rows]):>7.4f}", flush=True)

    # Overall
    all_s2_off = np.mean([r["stage2_f1_official"] for r in all_results])
    all_s2_fix = np.mean([r["stage2_f1_fixed"]    for r in all_results])
    all_s1_off = np.mean([r["stage1_f1_official"] for r in all_results])
    all_s1_fix = np.mean([r["stage1_f1_fixed"]    for r in all_results])

    print(f"\n  {'ALL (40 signals)':<45} "
          f"{all_s1_off:>7.4f} {all_s1_fix:>7.4f} {all_s2_off:>7.4f} {all_s2_fix:>7.4f}", flush=True)
    print(f"\n  Columns: S1-off=Stage1 F1 (their formula), S1-fix=Stage1 F1 (our fixed)")
    print(f"           S2-off=Stage2 F1 (their formula), S2-fix=Stage2 F1 (our fixed)")
    print(f"{'='*W}", flush=True)

    # ── Comparison summary ─────────────────────────────────────────────────────
    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │ Method comparison (fixed F1 formula, our 40 signals)      │")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │ VLM4TS official Stage1 (ViT-B-16)     : {all_s1_fix:.4f}         │")
    print(f"  │ VLM4TS official Stage2 (full-series)  : {all_s2_fix:.4f}         │")
    print(f"  │                                                          │")
    print(f"  │ Our Stage1 (DINOv2 dino_k5)           : 0.6174         │")
    print(f"  │ Our Stage2 v4 (per-cand. z_max-aware) : 0.6526         │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    pd.DataFrame(all_results).to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR}", flush=True)
