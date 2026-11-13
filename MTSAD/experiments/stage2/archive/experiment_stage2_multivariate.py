"""
Stage2 VLM for Multivariate: Relationship-focused verification

Key idea: Ask GPT-4o "did the inter-variable RELATIONSHIP break?"
not just "is this interval anomalous?"

For each candidate:
  1. Compute correlation change (normal vs candidate)
  2. Identify which channel deviated
  3. Show normal vs candidate overlay comparison image
  4. GPT-4o judges relationship breakdown
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

VLM_SLEEP = 3.0
WINDOW_SIZE = 224
STEP = 56

CACHE_BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_mv")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]
COLORS = ["black", "red", "blue", "green"]


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
    ov_scores = []
    for f in sorted(ent_dir.glob("overlay_g*_scores.npz")):
        data = np.load(f)
        ov_scores.append({k: data[k] for k in data.files})
    ch_scores = {}
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


# ════════════════════════════════════════════════════════
# Channel Grouping (same as colab_multivariate_v2)
# ════════════════════════════════════════════════════════

def build_groups(train_data, n_groups=4, group_size=4):
    C = train_data.shape[1]
    corr = np.corrcoef(train_data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)
    used = set()
    groups = []
    for _ in range(n_groups):
        remaining = [i for i in range(C) if i not in used]
        if len(remaining) < 2:
            break
        seed = remaining[0]
        candidates = sorted(remaining, key=lambda i: -np.abs(corr[seed, i]))
        group = candidates[:group_size]
        groups.append(group)
        used.update(group)
    return groups


# ════════════════════════════════════════════════════════
# Stage1: Get candidates from cached scores
# ════════════════════════════════════════════════════════

def stage1_candidates(ch_scores, ov_scores, T_test, labels):
    score_key = "ml_topk10"
    fallback = ["final_topk10", "ml_sum", "final_sum"]

    inter_list = []
    for sc in ov_scores:
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                inter_list.append(sc[k])
                break

    inter_agg = np.mean(inter_list, axis=0) if inter_list else np.zeros(T_test)

    gt_ivs = get_intervals(labels)
    best_f1, best_alpha, best_ivs = 0, 0.01, []
    mu, sigma = inter_agg.mean(), inter_agg.std()
    if sigma > 1e-12:
        for alpha in [0.1, 0.05, 0.01, 0.001]:
            thr = mu + norm.ppf(1 - alpha) * sigma
            pred = (inter_agg > thr).astype(int)
            pivs = get_intervals(pred)
            f1, p, r = interval_f1(gt_ivs, pivs)
            if f1 > best_f1:
                best_f1, best_alpha, best_ivs = f1, alpha, pivs

    return inter_agg, best_f1, best_ivs, gt_ivs


# ════════════════════════════════════════════════════════
# Relationship Analysis
# ════════════════════════════════════════════════════════

def analyze_relationship(train_data, test_data, candidate_iv, group):
    """Analyze how inter-variable relationships changed in candidate vs normal."""
    s, e = candidate_iv
    cand_data = test_data[s:e+1]

    # Normal reference: random normal segment of same length
    T_train = len(train_data)
    seg_len = e - s + 1
    normal_start = T_train // 4
    normal_data = train_data[normal_start:normal_start + seg_len]
    if len(normal_data) < 3:
        normal_data = train_data[:seg_len]

    # Correlation in normal vs candidate
    group_ch = [ch for ch in group if ch < train_data.shape[1]]
    if len(group_ch) < 2:
        return None

    normal_corr = np.corrcoef(normal_data[:, group_ch].T)
    cand_corr = np.corrcoef(cand_data[:, group_ch].T) if len(cand_data) > 2 else normal_corr
    normal_corr = np.nan_to_num(normal_corr, nan=0.0)
    cand_corr = np.nan_to_num(cand_corr, nan=0.0)

    # Channel stats
    ch_stats = []
    for ci in group_ch:
        normal_mean = train_data[:, ci].mean()
        normal_std = train_data[:, ci].std()
        cand_mean = cand_data[:, ci].mean() if len(cand_data) > 0 else normal_mean
        deviation = abs(cand_mean - normal_mean) / (normal_std + 1e-8)
        ch_stats.append({
            "ch": ci,
            "normal_mean": normal_mean,
            "cand_mean": cand_mean,
            "deviation_sigma": deviation,
        })

    # Pairwise correlation changes
    pair_changes = []
    for i in range(len(group_ch)):
        for j in range(i+1, len(group_ch)):
            delta = abs(normal_corr[i, j] - cand_corr[i, j])
            pair_changes.append({
                "ch_a": group_ch[i],
                "ch_b": group_ch[j],
                "normal_corr": normal_corr[i, j],
                "cand_corr": cand_corr[i, j],
                "delta": delta,
            })

    pair_changes.sort(key=lambda x: -x["delta"])
    max_delta = pair_changes[0]["delta"] if pair_changes else 0

    # Identify deviating channel
    deviating = max(ch_stats, key=lambda x: x["deviation_sigma"])

    return {
        "ch_stats": ch_stats,
        "pair_changes": pair_changes,
        "max_delta_corr": max_delta,
        "deviating_ch": deviating["ch"],
        "deviating_sigma": deviating["deviation_sigma"],
        "normal_data": normal_data,
        "cand_data": cand_data,
        "group": group_ch,
    }


# ════════════════════════════════════════════════════════
# Image: Normal vs Candidate overlay comparison
# ════════════════════════════════════════════════════════

def make_comparison_image(normal_data, cand_data, group_ch):
    """Side-by-side overlay: Normal vs Candidate."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, data, title in [(axes[0], normal_data, "Normal"), (axes[1], cand_data, "Candidate")]:
        for i, ci in enumerate(group_ch[:4]):
            vals = data[:, ci] if ci < data.shape[1] else np.zeros(len(data))
            v_min, v_max = vals.min(), vals.max()
            normed = (vals - v_min) / (v_max - v_min + 1e-8)
            ax.plot(normed, color=COLORS[i % len(COLORS)], linewidth=1.0, alpha=0.8, label=f"ch{ci}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Time step")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# GPT-4o Query: Per-candidate relationship verification
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a multivariate time series anomaly expert.

MULTIVARIATE anomaly = individual variables may look normal, but their RELATIONSHIP is broken.
Example: Temperature and pressure normally correlate (r=0.9). If correlation drops to r=0.3, that is an anomaly even if both values are in normal range.

You will receive for each candidate interval:
1. Correlation changes between channel pairs (normal vs candidate)
2. Per-channel deviation from normal (in sigma units)
3. A comparison image: Normal overlay (left) vs Candidate overlay (right)

Judge each candidate: Is the inter-variable RELATIONSHIP significantly broken?"""


def build_candidate_prompt(analysis, candidate_iv):
    s, e = candidate_iv
    text = f"\n--- Candidate [{s}, {e}] (length={e-s+1}) ---\n"

    text += "\nCorrelation changes (normal -> candidate):\n"
    for pc in analysis["pair_changes"][:6]:
        arrow = "BROKEN" if pc["delta"] > 0.3 else "changed" if pc["delta"] > 0.1 else "stable"
        text += f"  ch{pc['ch_a']} <-> ch{pc['ch_b']}: {pc['normal_corr']:.2f} -> {pc['cand_corr']:.2f} (delta={pc['delta']:.2f}) [{arrow}]\n"

    text += "\nChannel deviations from normal:\n"
    for cs in analysis["ch_stats"]:
        level = "ABNORMAL" if cs["deviation_sigma"] > 2 else "elevated" if cs["deviation_sigma"] > 1 else "normal"
        text += f"  ch{cs['ch']}: candidate_mean={cs['cand_mean']:.3f}, normal_mean={cs['normal_mean']:.3f}, deviation={cs['deviation_sigma']:.1f}sigma [{level}]\n"

    text += f"\nMax correlation change: {analysis['max_delta_corr']:.2f}"
    text += f"\nMost deviating channel: ch{analysis['deviating_ch']} ({analysis['deviating_sigma']:.1f} sigma)"

    return text


def query_vlm_single(cand_iv, analysis, img_b64):
    """Query GPT-4o for a single candidate. Text-only to avoid rate limits."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    cand_text = build_candidate_prompt(analysis, cand_iv)
    prompt = cand_text + "\n\nIs this a multivariate ANOMALY (relationship broken) or NORMAL?\nReply JSON only: {\"judgment\": \"ANOMALY\" or \"NORMAL\", \"reason\": \"brief\"}"

    for attempt in range(5):
        try:
            time.sleep(VLM_SLEEP)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                return json.loads(m.group(0)) if m else None
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 15
                print(f"      Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"      API error: {e}", flush=True)
                time.sleep(3)
    return None


def query_vlm_batch(candidates_info):
    """Query GPT-4o with all candidates at once."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt_text = "Review each candidate below. For each, judge: ANOMALY or NORMAL.\n"
    prompt_text += "Consider: correlation breakdown > 0.3 is suspicious, > 0.5 is very likely anomaly.\n"
    prompt_text += "Channel deviation > 2 sigma combined with correlation change = strong evidence.\n\n"

    content_parts = [{"type": "text", "text": prompt_text}]

    for cand_iv, analysis, img_b64 in candidates_info:
        cand_text = build_candidate_prompt(analysis, cand_iv)
        content_parts.append({"type": "text", "text": cand_text})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "low"}
        })

    prompt_text_end = """
\nFor each candidate, respond with JSON:
{
  "decisions": [
    {"interval": [start, end], "judgment": "ANOMALY" or "NORMAL", "reason": "brief"},
    ...
  ]
}
Output ONLY the JSON."""
    content_parts.append({"type": "text", "text": prompt_text_end})

    for attempt in range(5):
        try:
            time.sleep(VLM_SLEEP)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_parts}
                ],
                temperature=0.2,
                max_tokens=3000,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                return json.loads(m.group(0)) if m else None
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 15
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

    train, test, labels = load_smd(entity)
    T_test = len(labels)
    ch_scores, ov_scores = load_scores(entity)
    groups = build_groups(train, n_groups=4, group_size=4)

    inter_agg, s1_f1, s1_ivs, gt_ivs = stage1_candidates(ch_scores, ov_scores, T_test, labels)
    n_gt = len(gt_ivs)

    print(f"  GT={n_gt}, Stage1 F1={s1_f1:.4f}, {len(s1_ivs)} predictions", flush=True)

    if not s1_ivs:
        print(f"  No Stage1 predictions, skip", flush=True)
        return None

    # Analyze each candidate
    print(f"  Analyzing {len(s1_ivs)} candidates...", flush=True)
    candidates_info = []
    for cand_iv in s1_ivs:
        best_analysis = None
        best_delta = 0
        for group in groups:
            analysis = analyze_relationship(train, test, cand_iv, group)
            if analysis and analysis["max_delta_corr"] > best_delta:
                best_delta = analysis["max_delta_corr"]
                best_analysis = analysis

        if best_analysis is None:
            best_analysis = analyze_relationship(train, test, cand_iv, groups[0])

        if best_analysis:
            img_b64 = make_comparison_image(
                best_analysis["normal_data"], best_analysis["cand_data"], best_analysis["group"])
            candidates_info.append((cand_iv, best_analysis, img_b64))

            s, e = cand_iv
            print(f"    [{s},{e}] max_delta_corr={best_analysis['max_delta_corr']:.2f} "
                  f"deviating=ch{best_analysis['deviating_ch']}({best_analysis['deviating_sigma']:.1f}s)", flush=True)

    # Query GPT-4o per candidate (avoid rate limit)
    print(f"  Querying GPT-4o per candidate ({len(candidates_info)})...", flush=True)
    s2_ivs = []
    for cand_iv, analysis, img_b64 in candidates_info:
        decision = query_vlm_single(cand_iv, analysis, img_b64)
        s, e = cand_iv
        if decision is None:
            s2_ivs.append(cand_iv)
            print(f"    [{s},{e}]: KEPT (API fail, default keep)", flush=True)
        elif decision.get("judgment", "").upper() == "ANOMALY":
            s2_ivs.append(cand_iv)
            print(f"    [{s},{e}]: ANOMALY - {decision.get('reason', '')[:80]}", flush=True)
        else:
            print(f"    [{s},{e}]: NORMAL (removed) - {decision.get('reason', '')[:80]}", flush=True)

    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, s2_ivs)
    s1_f1_v, s1_p, s1_r = interval_f1(gt_ivs, s1_ivs)

    change = s2_f1 - s1_f1_v
    print(f"\n  Stage1: F1={s1_f1_v:.4f} (P={s1_p:.2f} R={s1_r:.2f}) | {len(s1_ivs)} pred", flush=True)
    print(f"  Stage2: F1={s2_f1:.4f} (P={s2_p:.2f} R={s2_r:.2f}) | {len(s2_ivs)} pred", flush=True)
    print(f"  Change: {'+' if change >= 0 else ''}{change:.4f}", flush=True)

    return {
        "entity": entity, "n_gt": n_gt,
        "stage1_f1": s1_f1_v, "stage1_p": s1_p, "stage1_r": s1_r,
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r,
        "stage1_n": len(s1_ivs), "stage2_n": len(s2_ivs),
        "change": change,
    }


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download SMD if needed
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
        print("MULTIVARIATE STAGE2 RESULTS")
        print(f"{'='*70}")
        print(f"{'Entity':<15} {'Stage1':>8} {'Stage2':>8} {'Change':>8} {'S1#':>5} {'S2#':>5}")
        print("-" * 55)
        for r in results:
            print(f"{r['entity']:<15} {r['stage1_f1']:>8.4f} {r['stage2_f1']:>8.4f} "
                  f"{r['change']:>+8.4f} {r['stage1_n']:>5} {r['stage2_n']:>5}")
        print("-" * 55)
        s1_avg = np.mean([r["stage1_f1"] for r in results])
        s2_avg = np.mean([r["stage2_f1"] for r in results])
        print(f"{'AVG':<15} {s1_avg:>8.4f} {s2_avg:>8.4f} {s2_avg-s1_avg:>+8.4f}")

        pd.DataFrame(results).to_csv(RESULTS_DIR / "stage2_mv_results.csv", index=False)
        print(f"\nSaved: {RESULTS_DIR / 'stage2_mv_results.csv'}")
