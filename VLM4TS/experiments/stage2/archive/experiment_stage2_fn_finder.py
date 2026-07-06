"""
Stage2 FN Finder: Find anomalies that Stage1 MISSED.

Stage1 (DINOv2 INTER) has high precision but may miss:
  - Subtle relationship changes below threshold
  - Level shifts (killed by per-window normalization)
  - Anomalies in channels not covered by overlay groups

Strategy:
  1. Identify "near-miss" regions (Stage1 score elevated but below threshold)
  2. For each near-miss, compute relationship stats and show to GPT-4o
  3. GPT-4o decides: "is this a missed anomaly?"
  4. Final = Stage1 predictions + GPT-4o confirmed near-misses
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
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_fn")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]
COLORS = ["black", "red", "blue", "green"]


# ════════════════════════════════════════════════════════
# Utils (same as before)
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
# Stage1 + Near-miss detection
# ════════════════════════════════════════════════════════

def get_stage1_and_nearmiss(ch_scores, ov_scores, T_test, labels):
    score_key = "ml_topk10"
    fallback = ["final_topk10", "ml_sum", "final_sum"]

    inter_list = []
    for sc in ov_scores:
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                inter_list.append(sc[k])
                break

    intra_list = []
    for ch, sc in ch_scores.items():
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                intra_list.append(sc[k])
                break

    inter_agg = np.mean(inter_list, axis=0) if inter_list else np.zeros(T_test)
    intra_agg = np.mean(intra_list, axis=0) if intra_list else np.zeros(T_test)

    gt_ivs = get_intervals(labels)

    # Best Stage1 threshold
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

    # Near-misses: regions with elevated score but NOT in Stage1 predictions
    # Use a looser threshold
    loose_thr = mu + norm.ppf(1 - 0.2) * sigma  # top 20%
    loose_binary = (inter_agg > loose_thr).astype(int)
    loose_ivs = get_intervals(loose_binary)

    near_misses = []
    for s, e in loose_ivs:
        already = any(abs(s - ps) < 100 or _overlap((s, e), (ps, pe)) for ps, pe in best_ivs)
        if not already and (e - s) > 5:  # at least 5 steps
            avg_score = inter_agg[s:e+1].mean()
            near_misses.append((s, e, avg_score))

    near_misses.sort(key=lambda x: -x[2])
    near_misses = near_misses[:10]

    return {
        "inter_agg": inter_agg,
        "intra_agg": intra_agg,
        "best_f1": best_f1,
        "best_ivs": best_ivs,
        "near_misses": near_misses,
        "gt_ivs": gt_ivs,
    }


# ════════════════════════════════════════════════════════
# Relationship analysis for near-misses
# ════════════════════════════════════════════════════════

def analyze_nearmiss(train_data, test_data, s, e, C):
    """Analyze inter-variable relationships in near-miss region."""
    cand = test_data[s:e+1]
    seg_len = e - s + 1

    # Normal reference from train
    normal = train_data[len(train_data)//4 : len(train_data)//4 + seg_len]
    if len(normal) < 3:
        normal = train_data[:seg_len]

    # Pick top variance channels
    var_per_ch = test_data.var(axis=0)
    top_ch = np.argsort(var_per_ch)[::-1][:6].tolist()

    # Correlation changes
    if len(cand) < 3 or len(normal) < 3:
        return None

    normal_corr = np.corrcoef(normal[:, top_ch].T)
    cand_corr = np.corrcoef(cand[:, top_ch].T)
    normal_corr = np.nan_to_num(normal_corr, nan=0.0)
    cand_corr = np.nan_to_num(cand_corr, nan=0.0)

    # Find biggest correlation change
    pair_changes = []
    for i in range(len(top_ch)):
        for j in range(i+1, len(top_ch)):
            delta = abs(normal_corr[i, j] - cand_corr[i, j])
            pair_changes.append({
                "ch_a": top_ch[i], "ch_b": top_ch[j],
                "normal": normal_corr[i, j], "cand": cand_corr[i, j],
                "delta": delta,
            })
    pair_changes.sort(key=lambda x: -x["delta"])

    # Channel stats
    ch_stats = []
    for ci in top_ch:
        n_mean = train_data[:, ci].mean()
        n_std = train_data[:, ci].std() + 1e-8
        c_mean = cand[:, ci].mean()
        ch_stats.append({
            "ch": ci,
            "normal_mean": n_mean,
            "cand_mean": c_mean,
            "dev_sigma": abs(c_mean - n_mean) / n_std,
        })

    return {
        "pair_changes": pair_changes[:6],
        "ch_stats": ch_stats,
        "max_delta": pair_changes[0]["delta"] if pair_changes else 0,
        "top_ch": top_ch[:4],
    }


# ════════════════════════════════════════════════════════
# GPT-4o query for near-miss verification
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a multivariate time series anomaly expert.

Stage1 already detected some anomalies. Now you are checking NEAR-MISS regions that Stage1 was uncertain about.

For each near-miss, you will see:
- Its anomaly score (elevated but below Stage1's threshold)
- Correlation changes between channel pairs
- Channel deviations from normal

Your job: decide if each near-miss is a REAL anomaly that Stage1 missed.

Guidelines:
- If correlation change (delta) > 0.3 AND channel deviation > 1.5 sigma: likely ANOMALY
- If only one of the above: UNCERTAIN, lean toward ANOMALY
- If both are small (delta < 0.15, dev < 1 sigma): NORMAL
- When in doubt, say ANOMALY (better to catch than miss)"""


def query_nearmiss(s, e, score, analysis):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    text = f"Near-miss region [{s}, {e}] (length={e-s+1}, score={score:.4f})\n"
    text += "\nCorrelation changes (normal -> candidate):\n"
    for pc in analysis["pair_changes"][:4]:
        label = "BROKEN" if pc["delta"] > 0.3 else "changed" if pc["delta"] > 0.15 else "stable"
        text += f"  ch{pc['ch_a']} <-> ch{pc['ch_b']}: {pc['normal']:.2f} -> {pc['cand']:.2f} (delta={pc['delta']:.2f}) [{label}]\n"

    text += "\nChannel deviations:\n"
    for cs in analysis["ch_stats"][:4]:
        level = "ABNORMAL" if cs["dev_sigma"] > 2 else "elevated" if cs["dev_sigma"] > 1 else "normal"
        text += f"  ch{cs['ch']}: {cs['cand_mean']:.3f} vs normal {cs['normal_mean']:.3f} ({cs['dev_sigma']:.1f} sigma) [{level}]\n"

    text += f"\nMax corr change: {analysis['max_delta']:.2f}"
    text += "\n\nIs this a MISSED ANOMALY? Reply JSON: {\"judgment\": \"ANOMALY\" or \"NORMAL\", \"reason\": \"brief\"}"

    for attempt in range(5):
        try:
            time.sleep(VLM_SLEEP)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            try:
                return json.loads(raw)
            except:
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                return json.loads(m.group(0)) if m else None
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 15
                print(f"      Rate limited, {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"      Error: {e}", flush=True)
                time.sleep(3)
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

    s1 = get_stage1_and_nearmiss(ch_scores, ov_scores, T_test, labels)
    n_gt = len(s1["gt_ivs"])

    print(f"  GT={n_gt}, Stage1 F1={s1['best_f1']:.4f} ({len(s1['best_ivs'])} pred)", flush=True)
    print(f"  Near-misses found: {len(s1['near_misses'])}", flush=True)

    if not s1["near_misses"]:
        print(f"  No near-misses → Stage2 = Stage1", flush=True)
        return {
            "entity": entity, "n_gt": n_gt,
            "stage1_f1": s1["best_f1"], "stage2_f1": s1["best_f1"],
            "near_misses_found": 0, "near_misses_added": 0, "change": 0,
        }

    # Analyze and query each near-miss
    added = []
    for s, e, score in s1["near_misses"]:
        analysis = analyze_nearmiss(train, test, s, e, test.shape[1])
        if analysis is None:
            continue

        print(f"    Near-miss [{s},{e}] score={score:.4f} max_delta={analysis['max_delta']:.2f}", end="", flush=True)

        decision = query_nearmiss(s, e, score, analysis)
        if decision and decision.get("judgment", "").upper() == "ANOMALY":
            added.append((s, e))
            print(f" → ADDED ({decision.get('reason', '')[:60]})", flush=True)
        elif decision:
            print(f" → skipped ({decision.get('reason', '')[:60]})", flush=True)
        else:
            print(f" → API fail", flush=True)

    # Final predictions = Stage1 + added near-misses
    final_ivs = list(s1["best_ivs"]) + added
    s2_f1, s2_p, s2_r = interval_f1(s1["gt_ivs"], final_ivs)
    s1_f1, s1_p, s1_r = interval_f1(s1["gt_ivs"], s1["best_ivs"])

    change = s2_f1 - s1_f1
    print(f"\n  Stage1: F1={s1_f1:.4f} (P={s1_p:.2f} R={s1_r:.2f}) | {len(s1['best_ivs'])} pred", flush=True)
    print(f"  Stage2: F1={s2_f1:.4f} (P={s2_p:.2f} R={s2_r:.2f}) | {len(final_ivs)} pred (+{len(added)} added)", flush=True)
    print(f"  Change: {'+' if change >= 0 else ''}{change:.4f}", flush=True)

    return {
        "entity": entity, "n_gt": n_gt,
        "stage1_f1": s1_f1, "stage1_p": s1_p, "stage1_r": s1_r,
        "stage2_f1": s2_f1, "stage2_p": s2_p, "stage2_r": s2_r,
        "stage1_n": len(s1["best_ivs"]), "stage2_n": len(final_ivs),
        "near_misses_found": len(s1["near_misses"]),
        "near_misses_added": len(added),
        "change": change,
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
        print("STAGE2 FN FINDER RESULTS")
        print(f"{'='*70}")
        print(f"{'Entity':<15} {'Stage1':>8} {'Stage2':>8} {'Change':>8} {'NM found':>9} {'NM added':>9}")
        print("-" * 62)
        for r in results:
            print(f"{r['entity']:<15} {r['stage1_f1']:>8.4f} {r['stage2_f1']:>8.4f} "
                  f"{r['change']:>+8.4f} {r['near_misses_found']:>9} {r['near_misses_added']:>9}")
        print("-" * 62)
        s1_avg = np.mean([r["stage1_f1"] for r in results])
        s2_avg = np.mean([r["stage2_f1"] for r in results])
        print(f"{'AVG':<15} {s1_avg:>8.4f} {s2_avg:>8.4f} {s2_avg-s1_avg:>+8.4f}")

        pd.DataFrame(results).to_csv(RESULTS_DIR / "stage2_fn_results.csv", index=False)
        print(f"\nSaved: {RESULTS_DIR / 'stage2_fn_results.csv'}")
