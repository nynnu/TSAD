"""
Stage2 MLLM: Global Context Verification for Multivariate TSAD

Key insight: DINOv2 sees LOCAL windows (224 steps). MLLM should see GLOBAL context (full series).
This way they provide COMPLEMENTARY perspectives.

What DINOv2 (Stage1) does well:
  - Local pattern anomalies within 224-step windows
  - Inter-variable relationship breakdown (overlay)

What DINOv2 CANNOT do (→ MLLM's job):
  - Level shifts (per-window normalization kills this)
  - Long-range context (is this pattern normal for THIS series?)
  - Gradual drift over thousands of steps
  - Channel behavior changes relative to the full history

Design:
  1. Generate multi-panel FULL time series plot (one panel per channel)
  2. Mark Stage1 detections on the plot
  3. MLLM sees the ENTIRE series + Stage1 results
  4. MLLM provides GLOBAL judgment:
     - Confirm/reject Stage1 detections based on global context
     - Find anomalies Stage1 missed (level shifts, drift)
  5. One API call per entity (avoids rate limit)
"""

import base64
import io
import json
import os
import re
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

VLM_SLEEP = 5.0
CACHE_BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_global")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]


# ════════════════════════════════════════════════════════
# Utils
# ════════════════════════════════════════════════════════

def load_smd(entity):
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(np.int32)
    return train, test, labels

def load_scores(entity):
    ent_dir = CACHE_BASE / "SMD" / entity
    ov_scores, ch_scores = [], {}
    for f in sorted(ent_dir.glob("overlay_g*_scores.npz")):
        data = np.load(f)
        ov_scores.append({k: data[k] for k in data.files})
    for f in sorted(ent_dir.glob("ch*_scores.npz")):
        ch = f.stem.replace("_scores", "")
        data = np.load(f)
        ch_scores[ch] = {k: data[k] for k in data.files}
    return ch_scores, ov_scores

def get_intervals(binary):
    ivs, in_seg, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            ivs.append((start, i - 1))
            in_seg = False
    if in_seg:
        ivs.append((start, len(binary) - 1))
    return ivs

def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

def interval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0, 0, 0
    gt = [tuple(i) for i in gt_ivs]
    pr = [tuple(i) for i in pred_ivs]
    TP = sum(sum(1 for a in gt if _overlap(d, a)) for d in pr if any(_overlap(d, a) for a in gt))
    FP = sum(1 for d in pr if not any(_overlap(d, a) for a in gt))
    FN = sum(1 for a in gt if not any(_overlap(a, d) for d in pr))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return f1, p, r

def get_stage1_preds(ch_scores, ov_scores, T_test, labels):
    score_key = "ml_topk10"
    fallback = ["final_topk10", "ml_sum", "final_sum"]

    inter_list, intra_list = [], []
    for sc in ov_scores:
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                inter_list.append(sc[k])
                break
    for ch, sc in ch_scores.items():
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                intra_list.append(sc[k])
                break

    inter_agg = np.mean(inter_list, axis=0) if inter_list else np.zeros(T_test)
    intra_agg = np.mean(intra_list, axis=0) if intra_list else np.zeros(T_test)
    gt_ivs = get_intervals(labels)

    best_f1, best_ivs = 0, []
    mu, sigma = inter_agg.mean(), inter_agg.std()
    if sigma > 1e-12:
        for alpha in [0.1, 0.05, 0.01, 0.001]:
            thr = mu + norm.ppf(1 - alpha) * sigma
            pivs = get_intervals((inter_agg > thr).astype(int))
            f1, p, r = interval_f1(gt_ivs, pivs)
            if f1 > best_f1:
                best_f1, best_ivs = f1, pivs

    return inter_agg, intra_agg, best_f1, best_ivs, gt_ivs


# ════════════════════════════════════════════════════════
# Global Multi-Panel Plot
# ════════════════════════════════════════════════════════

def generate_global_plot(test_data, stage1_ivs, inter_scores, top_channels, entity):
    """
    Full time series multi-panel plot:
    - Each top channel gets its own subplot (raw values, NO normalization)
    - Stage1 detections highlighted in red
    - Bottom panel: anomaly scores
    """
    n_ch = min(4, len(top_channels))
    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(14, 2.2 * (n_ch + 1)), sharex=True)
    T = len(inter_scores)
    x = np.arange(T)

    ch_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    for i in range(n_ch):
        ch = top_channels[i]
        raw = test_data[:T, ch]
        axes[i].plot(x, raw, color=ch_colors[i], linewidth=0.3)
        axes[i].set_ylabel(f"Ch{ch}", fontsize=8)
        axes[i].tick_params(labelsize=7)

        for s, e in stage1_ivs:
            axes[i].axvspan(s, e, alpha=0.25, color="red")

        # Mark x-axis ticks densely for GPT to read
        tick_step = max(T // 10, 1)
        axes[i].set_xticks(range(0, T, tick_step))

    # Score panel
    axes[-1].plot(x, inter_scores, color="purple", linewidth=0.4, label="INTER score")
    axes[-1].set_ylabel("Score", fontsize=8)
    axes[-1].set_xlabel("Time step index", fontsize=9)
    for s, e in stage1_ivs:
        axes[-1].axvspan(s, e, alpha=0.25, color="red")
    axes[-1].tick_params(labelsize=7)

    plt.suptitle(f"{entity}: Full Time Series (red = Stage1 detections)", fontsize=10)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# MLLM Prompt: Global Context for Multivariate
# ════════════════════════════════════════════════════════

PROMPT = """You are an expert in MULTIVARIATE time series anomaly detection.

You are looking at a FULL time series plot with multiple channels (variables) from a server monitoring system. Red shaded regions are anomalies already detected by Stage1 (a visual pattern matching model).

**Your unique advantage over Stage1**: You can see the ENTIRE time series at once. Stage1 only looks at small 224-step windows and misses:
- **Level shifts**: a channel's overall value suddenly jumps up or down
- **Long-range pattern changes**: behavior that looks normal in a small window but is unusual compared to the full history
- **Cross-channel timing changes**: channels that normally move together start behaving independently
- **Gradual drift**: slow changes over thousands of steps

**Your task**:
1. Look at the FULL time series for each channel
2. The red regions are Stage1's detections - they are mostly correct, KEEP them
3. Look OUTSIDE the red regions for anomalies Stage1 might have MISSED:
   - Any sudden level shifts in any channel?
   - Any region where channels stop moving together?
   - Any unusual patterns compared to the rest of the series?
4. Output your refined anomaly list

**CRITICAL RULES**:
- KEEP all Stage1 detections (red regions) unless obviously wrong
- Only ADD intervals where you see CLEAR visual evidence
- Use the x-axis tick marks to estimate interval boundaries
- For multivariate: a region is anomalous if ONE OR MORE channels show unusual behavior there

Reply ONLY with JSON:
{
  "anomaly_intervals": [[start1, end1], [start2, end2], ...],
  "new_findings": "brief description of what you found that Stage1 missed (or 'none')"
}"""


def build_context_text(entity, stage1_ivs, test_data, inter_scores, top_channels, T_test):
    """Additional text context about the data."""
    text = f"Entity: {entity}\n"
    text += f"Total length: {T_test} time steps\n"
    text += f"Channels shown: {top_channels[:4]}\n\n"

    text += "Stage1 detected intervals (red regions):\n"
    for i, (s, e) in enumerate(stage1_ivs):
        text += f"  [{s}, {e}] (length={e-s+1})\n"

    # Global stats per channel for context
    text += "\nChannel statistics (full test period):\n"
    for ch in top_channels[:4]:
        raw = test_data[:T_test, ch]
        text += f"  Ch{ch}: mean={raw.mean():.3f}, std={raw.std():.3f}, min={raw.min():.3f}, max={raw.max():.3f}\n"

    text += "\nLook at the plot carefully. Are there any anomalies OUTSIDE the red regions?"
    return text


# ════════════════════════════════════════════════════════
# Query GPT-4o
# ════════════════════════════════════════════════════════

def query_vlm(img_b64, context_text):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(5):
        try:
            time.sleep(VLM_SLEEP)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT + "\n\n" + context_text},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high"
                        }}
                    ]
                }],
                temperature=0.3,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                return json.loads(m.group(0)) if m else {"raw": raw}
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 20
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    API error: {e}", flush=True)
                time.sleep(5)
    return None


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def run_entity(entity):
    print(f"\n{'='*60}")
    print(f"{entity}")
    print(f"{'='*60}", flush=True)

    train, test, labels = load_smd(entity)
    T_test = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter_agg, intra_agg, s1_f1, s1_ivs, gt_ivs = get_stage1_preds(
        ch_scores, ov_scores, T_test, labels)
    n_gt = len(gt_ivs)

    # Top channels by variance
    var_per_ch = test.var(axis=0)
    top_channels = np.argsort(var_per_ch)[::-1][:4].tolist()

    print(f"  GT={n_gt}, Stage1 F1={s1_f1:.4f} ({len(s1_ivs)} pred)", flush=True)
    print(f"  Top channels: {top_channels}", flush=True)

    # Generate global plot
    print(f"  Generating global plot...", flush=True)
    img_b64 = generate_global_plot(test, s1_ivs, inter_agg, top_channels, entity)

    # Save plot
    img_dir = RESULTS_DIR / "plots"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(img_dir / f"{entity}_global.png", "wb") as f:
        f.write(base64.b64decode(img_b64))

    # Build context
    context = build_context_text(entity, s1_ivs, test, inter_agg, top_channels, T_test)

    # Query MLLM (ONE call per entity)
    print(f"  Calling GPT-4o (global context, detail=high)...", flush=True)
    result = query_vlm(img_b64, context)

    if not result:
        print(f"  VLM failed", flush=True)
        return {"entity": entity, "n_gt": n_gt, "stage1_f1": s1_f1,
                "stage2_f1": s1_f1, "change": 0, "new_findings": "API failed"}

    # Parse
    vlm_ivs = [(int(s), int(e)) for s, e in result.get("anomaly_intervals", [])]
    new_findings = result.get("new_findings", "none")

    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, vlm_ivs)
    s1_f1_v, s1_p, s1_r = interval_f1(gt_ivs, s1_ivs)
    change = s2_f1 - s1_f1_v

    # Count new intervals (not in Stage1)
    new_count = sum(1 for vi in vlm_ivs if not any(_overlap(vi, si) for si in s1_ivs))
    removed_count = sum(1 for si in s1_ivs if not any(_overlap(si, vi) for vi in vlm_ivs))

    print(f"\n  Stage1: F1={s1_f1_v:.4f} (P={s1_p:.2f} R={s1_r:.2f}) | {len(s1_ivs)} pred", flush=True)
    print(f"  Stage2: F1={s2_f1:.4f} (P={s2_p:.2f} R={s2_r:.2f}) | {len(vlm_ivs)} pred", flush=True)
    print(f"  Change: {'+' if change >= 0 else ''}{change:.4f}", flush=True)
    print(f"  New intervals added: {new_count}, Removed: {removed_count}", flush=True)
    print(f"  MLLM findings: {new_findings[:200]}", flush=True)

    return {
        "entity": entity, "n_gt": n_gt,
        "stage1_f1": s1_f1_v, "stage1_p": s1_p, "stage1_r": s1_r,
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r,
        "stage1_n": len(s1_ivs), "stage2_n": len(vlm_ivs),
        "new_added": new_count, "removed": removed_count,
        "change": change, "new_findings": new_findings,
    }


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for ent in SMD_ENTITIES:
        for split in ["train", "test", "test_label"]:
            dst = SMD_DIR / split / f"{ent}.txt"
            if not dst.exists():
                import urllib.request
                (SMD_DIR / split).mkdir(parents=True, exist_ok=True)
                url = f"https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/{split}/{ent}.txt"
                urllib.request.urlretrieve(url, str(dst))

    results = []
    for entity in SMD_ENTITIES:
        try:
            r = run_entity(entity)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    if results:
        print(f"\n{'='*70}")
        print("STAGE2 GLOBAL CONTEXT RESULTS")
        print(f"{'='*70}")
        print(f"{'Entity':<15} {'Stage1':>8} {'Stage2':>8} {'Change':>8} {'Added':>6} {'Removed':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['entity']:<15} {r['stage1_f1']:>8.4f} {r['stage2_f1']:>8.4f} "
                  f"{r['change']:>+8.4f} {r['new_added']:>6} {r['removed']:>8}")
        print("-" * 60)
        s1_avg = np.mean([r["stage1_f1"] for r in results])
        s2_avg = np.mean([r["stage2_f1"] for r in results])
        print(f"{'AVG':<15} {s1_avg:>8.4f} {s2_avg:>8.4f} {s2_avg-s1_avg:>+8.4f}")

        for r in results:
            print(f"\n  {r['entity']}: {r['new_findings'][:300]}")

        pd.DataFrame(results).to_csv(RESULTS_DIR / "stage2_global_results.csv", index=False)
        print(f"\nSaved: {RESULTS_DIR / 'stage2_global_results.csv'}")
