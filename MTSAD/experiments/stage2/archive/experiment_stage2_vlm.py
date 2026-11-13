"""
Stage2 VLM (GPT-4o): DINOv2 Stage1 scores → GPT-4o verification

Methods:
  B1: Text only (Stage1 anomaly intervals + scores as text)
  B2: Text + Image (line plot image + Stage1 scores)

Uses cached scores from colab_multivariate_v2.py results.
"""

import ast
import io
import os
import json
import time
import base64
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from PIL import Image

warnings.filterwarnings("ignore")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

WINDOW_SIZE = 224
STEP = 56
VLM_SLEEP = 3.0
MAX_RETRIES = 5

CACHE_BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_vlm")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]


# ════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════

def load_smd(entity):
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(np.int32)
    return test, labels


def load_cached_scores(entity, cache_dir):
    ent_dir = cache_dir / entity
    ch_scores = {}
    for f in sorted(ent_dir.glob("ch*_scores.npz")):
        ch = f.stem.replace("_scores", "")
        data = np.load(f)
        ch_scores[ch] = {k: data[k] for k in data.files}

    ov_scores = []
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
# Stage1 Score → Anomaly Candidates
# ════════════════════════════════════════════════════════

def get_anomaly_candidates(scores_ts, alpha=0.01, include_low=True):
    """Convert ts-level scores to anomaly candidate intervals with severity."""
    mu, sigma = scores_ts.mean(), scores_ts.std()
    if sigma < 1e-12:
        return []

    thresholds = {
        "HIGH": mu + norm.ppf(1 - 0.001) * sigma,
        "MEDIUM": mu + norm.ppf(1 - 0.01) * sigma,
        "LOW": mu + norm.ppf(1 - 0.1) * sigma,
    }

    candidates = []

    for level, thr in [("HIGH", thresholds["HIGH"]), ("MEDIUM", thresholds["MEDIUM"])]:
        binary = (scores_ts > thr).astype(int)
        ivs = get_intervals(binary)
        for s, e in ivs:
            already = any(c["start"] <= s and c["end"] >= e and c["severity"] != level for c in candidates)
            if not already:
                avg_score = scores_ts[s:e+1].mean()
                candidates.append({"start": int(s), "end": int(e), "severity": level,
                                  "score": float(avg_score), "length": int(e - s + 1)})

    if include_low:
        binary_low = (scores_ts > thresholds["LOW"]).astype(int)
        low_ivs = get_intervals(binary_low)
        for s, e in low_ivs:
            already = any(c["start"] <= s and c["end"] >= e for c in candidates)
            if not already:
                avg_score = scores_ts[s:e+1].mean()
                candidates.append({"start": int(s), "end": int(e), "severity": "LOW",
                                  "score": float(avg_score), "length": int(e - s + 1)})

    candidates.sort(key=lambda x: -x["score"])
    return candidates[:30]


# ════════════════════════════════════════════════════════
# GPT-4o API Call
# ════════════════════════════════════════════════════════

def call_gpt4o(messages, max_retries=MAX_RETRIES):
    """Call GPT-4o with retry logic."""
    try:
        from openai import OpenAI
    except ImportError:
        os.system("pip install openai")
        from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 10
                print(f"      Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"      API error: {e}", flush=True)
                time.sleep(5)
    return None


def encode_image_base64(img):
    """PIL Image → base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ════════════════════════════════════════════════════════
# Generate Line Plot for Stage2
# ════════════════════════════════════════════════════════

def generate_lineplot(test_data, channel_idx, start=None, end=None):
    """Generate a line plot image of a single channel."""
    if start is not None and end is not None:
        data = test_data[start:end, channel_idx]
        title = f"Channel {channel_idx} (t={start}-{end})"
    else:
        data = test_data[:, channel_idx]
        title = f"Channel {channel_idx} (full)"

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(data, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def generate_overview_plot(test_data, scores_ts, candidates, top_channels, n_show=3):
    """Generate overview plot showing top channels + score + candidates."""
    fig, axes = plt.subplots(n_show + 1, 1, figsize=(12, 2.5 * (n_show + 1)), sharex=True)

    T = len(scores_ts)
    x = range(T)

    for i, ch in enumerate(top_channels[:n_show]):
        axes[i].plot(x, test_data[:T, ch], color="black", linewidth=0.4)
        axes[i].set_ylabel(f"Ch{ch}", fontsize=9)
        for c in candidates:
            if c["severity"] == "HIGH":
                axes[i].axvspan(c["start"], c["end"], alpha=0.3, color="red")
            elif c["severity"] == "MEDIUM":
                axes[i].axvspan(c["start"], c["end"], alpha=0.15, color="orange")

    axes[-1].plot(x, scores_ts, color="blue", linewidth=0.5)
    axes[-1].set_ylabel("Anomaly Score", fontsize=9)
    axes[-1].set_xlabel("Time step")
    for c in candidates:
        color = "red" if c["severity"] == "HIGH" else "orange"
        axes[-1].axvspan(c["start"], c["end"], alpha=0.2, color=color)

    plt.suptitle("DINOv2 Stage1 Overview", fontsize=11)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ════════════════════════════════════════════════════════
# Stage2 Prompts
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert time series anomaly detector verifying Stage1 results.

You will receive:
1. CURRENT PREDICTIONS: intervals that Stage1 already flagged as anomalous (these are our best guess)
2. NEAR-MISS CANDIDATES: intervals that Stage1 was uncertain about (scored high but below threshold)
3. Score statistics from both intra-variate (per-channel) and inter-variate (cross-channel) analysis

Your task is to REFINE the current predictions:
- KEEP predictions that look correct (most should be kept)
- REMOVE obvious false positives (if score is barely above threshold and isolated)
- ADD near-miss candidates that are likely real anomalies (especially if both INTRA and INTER flag them)
- MERGE nearby intervals (<50 steps apart) that belong to the same event

Important: The current predictions are already good. Only make changes you are confident about.

Output ONLY valid JSON: {"anomaly_intervals": [[start, end], ...]}"""


def build_threshold_sweep_text(scores_ts, T_test):
    """Build text showing candidates at multiple threshold levels."""
    mu, sigma = scores_ts.mean(), scores_ts.std()
    if sigma < 1e-12:
        return "No meaningful score variation detected.\n"

    text = ""
    for alpha, label in [(0.001, "Very Strict (top 0.1%)"),
                         (0.01, "Strict (top 1%)"),
                         (0.05, "Moderate (top 5%)"),
                         (0.1, "Loose (top 10%)")]:
        thr = mu + norm.ppf(1 - alpha) * sigma
        binary = (scores_ts > thr).astype(int)
        ivs = get_intervals(binary)
        if ivs:
            text += f"\n  [{label}] threshold={thr:.4f}, {len(ivs)} intervals:\n"
            for s, e in ivs[:15]:
                avg_sc = scores_ts[s:e+1].mean()
                max_sc = scores_ts[s:e+1].max()
                text += f"    [{s}, {e}] len={e-s+1} avg_score={avg_sc:.4f} max_score={max_sc:.4f}\n"
            if len(ivs) > 15:
                text += f"    ... and {len(ivs)-15} more intervals\n"
        else:
            text += f"\n  [{label}] threshold={thr:.4f}, 0 intervals\n"
    return text


def get_best_threshold_predictions(scores_ts):
    """Find best alpha via F1-proxy (score spread) and return predicted intervals."""
    mu, sigma = scores_ts.mean(), scores_ts.std()
    if sigma < 1e-12:
        return [], 0.01

    best_alpha = 0.01
    best_n = 0
    for alpha in [0.1, 0.01, 0.001]:
        thr = mu + norm.ppf(1 - alpha) * sigma
        binary = (scores_ts > thr).astype(int)
        ivs = get_intervals(binary)
        if len(ivs) > best_n and len(ivs) <= 30:
            best_n = len(ivs)
            best_alpha = alpha

    thr = mu + norm.ppf(1 - best_alpha) * sigma
    binary = (scores_ts > thr).astype(int)
    return get_intervals(binary), best_alpha


def build_b1_prompt(entity, candidates, intra_scores, inter_scores, T_test):
    """B1: Give Stage1 best predictions + near-miss candidates for refinement."""

    # Get best predictions from INTER (our strongest signal)
    inter_preds, inter_alpha = get_best_threshold_predictions(inter_scores)
    intra_preds, intra_alpha = get_best_threshold_predictions(intra_scores)

    # Combined score predictions
    combined = 0.3 * (intra_scores / (intra_scores.max() + 1e-8)) + 0.7 * (inter_scores / (inter_scores.max() + 1e-8))
    combined_preds, combined_alpha = get_best_threshold_predictions(combined)

    text = f"Entity: {entity}\nTotal length: {T_test} time steps\n\n"

    # Current best predictions
    text += "=== CURRENT PREDICTIONS (Stage1 best) ===\n"
    text += f"Based on INTER-variate analysis (alpha={inter_alpha}):\n"
    for i, (s, e) in enumerate(inter_preds[:20]):
        avg_sc = inter_scores[s:e+1].mean()
        max_sc = inter_scores[s:e+1].max()
        # Check if INTRA also flags this region
        intra_avg = intra_scores[s:e+1].mean()
        intra_thr = intra_scores.mean() + norm.ppf(1 - 0.1) * intra_scores.std()
        intra_flag = "YES" if intra_avg > intra_thr else "no"
        text += (f"  Pred {i+1}: [{s}, {e}] len={e-s+1} "
                f"inter_score={avg_sc:.4f} intra_confirms={intra_flag}\n")

    # Near-miss candidates (looser threshold, not in current predictions)
    mu_i, sig_i = inter_scores.mean(), inter_scores.std()
    loose_thr = mu_i + norm.ppf(1 - 0.15) * sig_i  # even looser
    loose_binary = (inter_scores > loose_thr).astype(int)
    loose_ivs = get_intervals(loose_binary)

    near_misses = []
    for s, e in loose_ivs:
        already = any(abs(s - ps) < 50 or abs(e - pe) < 50 for ps, pe in inter_preds)
        if not already:
            near_misses.append((s, e))

    if near_misses:
        text += f"\n=== NEAR-MISS CANDIDATES (below threshold but suspicious) ===\n"
        for i, (s, e) in enumerate(near_misses[:10]):
            avg_sc = inter_scores[s:e+1].mean()
            intra_avg = intra_scores[s:e+1].mean()
            intra_flag = "YES" if intra_avg > intra_thr else "no"
            text += (f"  Near-miss {i+1}: [{s}, {e}] len={e-s+1} "
                    f"inter_score={avg_sc:.4f} intra_confirms={intra_flag}\n")

    # Also show INTRA-only detections (INTER missed but INTRA caught)
    intra_only = []
    for s, e in intra_preds:
        already = any(abs(s - ps) < 100 for ps, pe in inter_preds)
        if not already:
            intra_only.append((s, e))

    if intra_only:
        text += f"\n=== INTRA-ONLY DETECTIONS (not flagged by INTER) ===\n"
        for i, (s, e) in enumerate(intra_only[:10]):
            avg_sc = intra_scores[s:e+1].mean()
            text += f"  Intra-only {i+1}: [{s}, {e}] len={e-s+1} intra_score={avg_sc:.4f}\n"

    text += "\nRefine the CURRENT PREDICTIONS: keep most, remove false positives, add deserving near-misses."
    text += "\nOutput ONLY JSON: {\"anomaly_intervals\": [[start, end], ...]}"
    return text


def build_b2_prompt(entity, candidates, intra_scores, inter_scores, T_test):
    """B2: Same text as B1 (image is sent separately)."""
    return build_b1_prompt(entity, candidates, intra_scores, inter_scores, T_test)


# ════════════════════════════════════════════════════════
# Parse GPT-4o Response
# ════════════════════════════════════════════════════════

def parse_response(response_text):
    """Extract anomaly intervals from GPT-4o response."""
    if not response_text:
        return []
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response_text[start:end])
            intervals = data.get("anomaly_intervals", [])
            return [(int(s), int(e)) for s, e in intervals]
    except:
        pass

    import re
    pairs = re.findall(r'\[(\d+),\s*(\d+)\]', response_text)
    return [(int(s), int(e)) for s, e in pairs]


# ════════════════════════════════════════════════════════
# Run Stage2
# ════════════════════════════════════════════════════════

def run_stage2(entity, test_data, labels, ch_scores, ov_scores, method="B1"):
    T_test = len(labels)
    n_gt = len(get_intervals(labels))
    if n_gt == 0:
        return None

    # Get aggregate scores
    score_key = "ml_topk10"
    fallback = ["final_topk10", "ml_sum", "final_sum"]

    intra_list = []
    top_channels = []
    for ch, sc in ch_scores.items():
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                intra_list.append(sc[k])
                top_channels.append(int(ch.replace("ch", "")))
                break

    inter_list = []
    for sc in ov_scores:
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                inter_list.append(sc[k])
                break

    if not intra_list or not inter_list:
        print(f"    No scores for {entity}", flush=True)
        return None

    intra_agg = np.mean(intra_list, axis=0)
    inter_agg = np.mean(inter_list, axis=0)

    # Use INTER for candidates (it's better)
    combined = 0.3 * (intra_agg / (intra_agg.max() + 1e-8)) + 0.7 * (inter_agg / (inter_agg.max() + 1e-8))
    candidates = get_anomaly_candidates(combined)

    print(f"    Stage1 candidates: {len(candidates)} (HIGH={sum(1 for c in candidates if c['severity']=='HIGH')}, "
          f"MED={sum(1 for c in candidates if c['severity']=='MEDIUM')})", flush=True)

    # Stage1 baseline: best threshold sweep on INTER (our best Stage1)
    gt_ivs = get_intervals(labels)
    s1_f1 = 0
    s1_p, s1_r = 0, 0
    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = inter_agg.mean(), inter_agg.std()
        if sigma < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred_binary = (inter_agg > thr).astype(int)
        pred_ivs = get_intervals(pred_binary)
        f1, p, r = interval_f1(gt_ivs, pred_ivs)
        if f1 > s1_f1:
            s1_f1, s1_p, s1_r = f1, p, r

    # Build prompt
    text = build_b1_prompt(entity, candidates, intra_agg, inter_agg, T_test)

    if method == "B1":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    elif method == "B2":
        img = generate_overview_plot(test_data, combined, candidates, top_channels[:3])
        img_b64 = encode_image_base64(img)

        # Save image for reference
        img_dir = RESULTS_DIR / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(img_dir / f"{entity}_{method}_overview.png")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "low"
                }}
            ]}
        ]

    # Call GPT-4o
    print(f"    Calling GPT-4o ({method})...", flush=True)
    response = call_gpt4o(messages)
    time.sleep(VLM_SLEEP)

    if not response:
        print(f"    GPT-4o failed", flush=True)
        return None

    # Parse response
    pred_intervals = parse_response(response)
    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, pred_intervals)

    print(f"    Stage1: F1={s1_f1:.4f} (P={s1_p:.2f}, R={s1_r:.2f})", flush=True)
    print(f"    Stage2 {method}: F1={s2_f1:.4f} (P={s2_p:.2f}, R={s2_r:.2f}) "
          f"pred={len(pred_intervals)} intervals", flush=True)

    return {
        "entity": entity,
        "n_gt": n_gt,
        "method": method,
        "stage1_f1": s1_f1,
        "stage1_p": s1_p,
        "stage1_r": s1_r,
        "stage2_f1": s2_f1,
        "stage2_p": s2_p,
        "stage2_r": s2_r,
        "n_candidates": len(candidates),
        "n_pred": len(pred_intervals),
        "response": response[:500],
    }


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"API Key set: {bool(OPENAI_API_KEY)}", flush=True)

    all_results = []

    for entity in SMD_ENTITIES:
        print(f"\n{'='*60}")
        print(f"Entity: {entity}")
        print(f"{'='*60}", flush=True)

        try:
            test, labels = load_smd(entity)
            ch_scores, ov_scores = load_cached_scores(entity, CACHE_BASE / "SMD")

            for method in ["B1", "B2"]:
                r = run_stage2(entity, test, labels, ch_scores, ov_scores, method=method)
                if r:
                    all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("STAGE2 VLM RESULTS")
        print(f"{'='*70}")
        print(f"{'Entity':<15} {'Method':<6} {'Stage1 F1':>10} {'Stage2 F1':>10} {'Change':>8}")
        print("-" * 55)

        for r in all_results:
            change = r["stage2_f1"] - r["stage1_f1"]
            marker = "+" if change > 0 else ""
            print(f"{r['entity']:<15} {r['method']:<6} {r['stage1_f1']:>10.4f} {r['stage2_f1']:>10.4f} {marker}{change:>7.4f}")

        # Averages by method
        print("-" * 55)
        for method in ["B1", "B2"]:
            mrs = [r for r in all_results if r["method"] == method]
            if mrs:
                s1_avg = np.mean([r["stage1_f1"] for r in mrs])
                s2_avg = np.mean([r["stage2_f1"] for r in mrs])
                print(f"{'AVG':<15} {method:<6} {s1_avg:>10.4f} {s2_avg:>10.4f} {s2_avg-s1_avg:>+8.4f}")

        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_DIR / "stage2_vlm_results.csv", index=False)
        print(f"\nSaved: {RESULTS_DIR / 'stage2_vlm_results.csv'}")
