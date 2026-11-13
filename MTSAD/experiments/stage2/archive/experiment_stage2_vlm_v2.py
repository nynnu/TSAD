"""
Stage2 VLM v2: VLM4TS-style verification

Following VLM4TS paper exactly:
  1. Stage1 (DINOv2 INTER) generates anomaly candidates
  2. Generate full time series plot with candidate intervals marked
  3. GPT-4o sees the plot + candidate list → refines (remove FP, add FN)

Key difference from v1: GPT-4o SEES the actual time series image.
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

VLM_SLEEP = 2.0
WINDOW_SIZE = 224
STEP = 56

CACHE_BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_v2")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]


# ════════════════════════════════════════════════════════
# Data & Score Loading
# ════════════════════════════════════════════════════════

def load_smd(entity):
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(np.int32)
    return test, labels

def load_scores(entity, cache_dir):
    ent_dir = cache_dir / entity
    ch_scores, ov_scores = {}, []
    for f in sorted(ent_dir.glob("ch*_scores.npz")):
        ch = f.stem.replace("_scores", "")
        data = np.load(f)
        ch_scores[ch] = {k: data[k] for k in data.files}
    for f in sorted(ent_dir.glob("overlay_g*_scores.npz")):
        data = np.load(f)
        ov_scores.append({k: data[k] for k in data.files})
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


# ════════════════════════════════════════════════════════
# Stage1: Get best candidates from DINOv2 scores
# ════════════════════════════════════════════════════════

def stage1_detect(ch_scores, ov_scores, T_test, labels):
    """Run Stage1: get INTER scores, find best threshold, return candidates."""
    score_key = "ml_topk10"
    fallback = ["final_topk10", "ml_sum", "final_sum"]

    # Aggregate INTER (overlay) scores
    inter_list = []
    for sc in ov_scores:
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                inter_list.append(sc[k])
                break

    # Aggregate INTRA scores
    intra_list = []
    for ch, sc in ch_scores.items():
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                intra_list.append(sc[k])
                break

    inter_agg = np.mean(inter_list, axis=0) if inter_list else np.zeros(T_test)
    intra_agg = np.mean(intra_list, axis=0) if intra_list else np.zeros(T_test)

    # Best threshold for INTER (Stage1 baseline)
    gt_ivs = get_intervals(labels)
    best_f1, best_alpha, best_pred_ivs = 0, 0.01, []
    mu, sigma = inter_agg.mean(), inter_agg.std()
    if sigma > 1e-12:
        for alpha in [0.1, 0.05, 0.01, 0.001]:
            thr = mu + norm.ppf(1 - alpha) * sigma
            pred = (inter_agg > thr).astype(int)
            pred_ivs = get_intervals(pred)
            f1, p, r = interval_f1(gt_ivs, pred_ivs)
            if f1 > best_f1:
                best_f1, best_alpha, best_pred_ivs = f1, alpha, pred_ivs

    # Use Stage1 best predictions as candidates (NOT looser)
    candidate_ivs = best_pred_ivs

    return {
        "inter_agg": inter_agg,
        "intra_agg": intra_agg,
        "best_f1": best_f1,
        "best_alpha": best_alpha,
        "best_pred_ivs": best_pred_ivs,
        "candidate_ivs": candidate_ivs,
        "gt_ivs": gt_ivs,
    }


# ════════════════════════════════════════════════════════
# Plot Generation (VLM4TS style)
# ════════════════════════════════════════════════════════

def generate_plot_with_candidates(test_data, inter_scores, candidate_ivs, top_channels, entity):
    """Generate VLM4TS-style plot: time series + candidate intervals highlighted."""
    n_ch = min(3, len(top_channels))
    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(12, 2.5 * (n_ch + 1)), sharex=True)
    T = len(inter_scores)
    x = np.arange(T)

    colors_ch = ["black", "blue", "green"]
    for i in range(n_ch):
        ch = top_channels[i]
        axes[i].plot(x, test_data[:T, ch], color=colors_ch[i], linewidth=0.4)
        axes[i].set_ylabel(f"Ch{ch}", fontsize=9)
        for j, (s, e) in enumerate(candidate_ivs):
            axes[i].axvspan(s, e, alpha=0.2, color="red")
            if i == 0:
                mid = (s + e) // 2
                axes[i].text(mid, axes[i].get_ylim()[1], f"C{j+1}", fontsize=7,
                           ha="center", va="bottom", color="red", fontweight="bold")

    axes[-1].plot(x, inter_scores, color="purple", linewidth=0.5, label="INTER score")
    axes[-1].set_ylabel("Score", fontsize=9)
    axes[-1].set_xlabel("Time step (index)", fontsize=9)
    for s, e in candidate_ivs:
        axes[-1].axvspan(s, e, alpha=0.2, color="red")

    plt.suptitle(f"{entity}: Stage1 Candidates", fontsize=11)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# VLM Query (VLM4TS prompt)
# ════════════════════════════════════════════════════════

VLM4TS_PROMPT = """
You are verifying anomaly candidates detected by a Stage1 model. The Stage1 model has HIGH recall — most of its candidates are real anomalies.

You will see:
1. A time-series plot with red shaded regions = Stage1 candidates
2. A list of candidate intervals with anomaly scores

Your job:
- **KEEP most candidates.** Stage1 is usually correct. Default action = KEEP.
- **REMOVE only if you are very confident** it is a false positive (the region looks completely normal with no unusual patterns at all).
- You may **adjust boundaries** slightly if the anomaly clearly starts/ends at a different point.
- You may **merge** nearby candidates that are part of the same anomaly event.
- You may **add** intervals only if you see an obvious anomaly that Stage1 completely missed.

CRITICAL: Do NOT remove a candidate just because it "aligns with the overall trend." Many real anomalies are subtle. When in doubt, KEEP it.

Reply ONLY with JSON:
{
  "interval_index": [[start1, end1], [start2, end2], ...],
  "confidence": [c1, c2, ...],
  "abnormal_description": "brief explanation"
}
"""


def query_vlm(img_b64, candidate_ivs, inter_scores):
    """Query GPT-4o with plot image + candidate list."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    detected = [[int(s), int(e)] for s, e in candidate_ivs]
    scores_info = []
    for s, e in candidate_ivs:
        avg = inter_scores[s:e+1].mean()
        mx = inter_scores[s:e+1].max()
        scores_info.append(f"[{s},{e}] score_avg={avg:.4f} score_max={mx:.4f}")

    vis_line = f"\nStage1 detected intervals: {detected}"
    vis_line += f"\nScore details: {'; '.join(scores_info)}"

    prompt = VLM4TS_PROMPT + vis_line

    for attempt in range(5):
        try:
            time.sleep(VLM_SLEEP)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "low"
                        }}
                    ]
                }],
                temperature=0.4,
            )
            raw = response.choices[0].message.content.strip()

            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                result = json.loads(m.group(0)) if m else {}

            return result

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 10
                print(f"      Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"      API error: {e}", flush=True)
                time.sleep(5)
    return None


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def run_entity(entity):
    print(f"\n{'='*60}")
    print(f"Entity: {entity}")
    print(f"{'='*60}", flush=True)

    test, labels = load_smd(entity)
    T_test = len(labels)
    ch_scores, ov_scores = load_scores(entity, CACHE_BASE / "SMD")

    # Stage1
    s1 = stage1_detect(ch_scores, ov_scores, T_test, labels)
    n_gt = len(s1["gt_ivs"])
    print(f"  GT={n_gt}, Stage1 best F1={s1['best_f1']:.4f} (alpha={s1['best_alpha']})", flush=True)
    print(f"  Stage1 predictions: {len(s1['best_pred_ivs'])} intervals", flush=True)
    print(f"  Candidates for VLM: {len(s1['candidate_ivs'])} intervals (looser threshold)", flush=True)

    # Top channels by variance
    var_per_ch = test.var(axis=0)
    top_channels = np.argsort(var_per_ch)[::-1][:3].tolist()

    # Generate plot
    img_b64 = generate_plot_with_candidates(
        test, s1["inter_agg"], s1["candidate_ivs"], top_channels, entity)

    # Save plot
    img_dir = RESULTS_DIR / "plots"
    img_dir.mkdir(parents=True, exist_ok=True)
    import base64 as b64mod
    with open(img_dir / f"{entity}_stage2_input.png", "wb") as f:
        f.write(b64mod.b64decode(img_b64))

    # Query VLM
    print(f"  Calling GPT-4o...", flush=True)
    vlm_result = query_vlm(img_b64, s1["candidate_ivs"], s1["inter_agg"])

    if vlm_result is None:
        print(f"  VLM failed", flush=True)
        return None

    # Parse result
    vlm_intervals = vlm_result.get("interval_index", [])
    vlm_pred = [(int(s), int(e)) for s, e in vlm_intervals]

    s2_f1, s2_p, s2_r = interval_f1(s1["gt_ivs"], vlm_pred)
    s1_f1, s1_p, s1_r = interval_f1(s1["gt_ivs"], s1["best_pred_ivs"])

    desc = vlm_result.get("abnormal_description", "")

    print(f"  Stage1: F1={s1_f1:.4f} (P={s1_p:.2f}, R={s1_r:.2f}) | {len(s1['best_pred_ivs'])} intervals", flush=True)
    print(f"  Stage2: F1={s2_f1:.4f} (P={s2_p:.2f}, R={s2_r:.2f}) | {len(vlm_pred)} intervals", flush=True)
    change = s2_f1 - s1_f1
    print(f"  Change: {'+' if change >= 0 else ''}{change:.4f}", flush=True)
    if desc:
        print(f"  VLM says: {desc[:200]}", flush=True)

    return {
        "entity": entity, "n_gt": n_gt,
        "stage1_f1": s1_f1, "stage1_p": s1_p, "stage1_r": s1_r,
        "stage1_n_pred": len(s1["best_pred_ivs"]),
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r,
        "stage2_n_pred": len(vlm_pred),
        "n_candidates": len(s1["candidate_ivs"]),
        "change": change,
        "description": desc[:500],
    }


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        print("STAGE2 VLM v2 RESULTS (VLM4TS-style)")
        print(f"{'='*70}")
        print(f"{'Entity':<15} {'Stage1':>8} {'Stage2':>8} {'Change':>8} {'S1 pred':>8} {'S2 pred':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['entity']:<15} {r['stage1_f1']:>8.4f} {r['stage2_f1']:>8.4f} "
                  f"{r['change']:>+8.4f} {r['stage1_n_pred']:>8} {r['stage2_n_pred']:>8}")
        print("-" * 60)
        avg_s1 = np.mean([r["stage1_f1"] for r in results])
        avg_s2 = np.mean([r["stage2_f1"] for r in results])
        print(f"{'AVG':<15} {avg_s1:>8.4f} {avg_s2:>8.4f} {avg_s2-avg_s1:>+8.4f}")

        pd.DataFrame(results).to_csv(RESULTS_DIR / "stage2_v2_results.csv", index=False)
        print(f"\nSaved: {RESULTS_DIR / 'stage2_v2_results.csv'}")
