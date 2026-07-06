"""
Stage2 MLLM: Calibrated Comparison Verification for Multivariate TSAD

Root cause of all previous Stage2 failures:
  GPT-4o has NO calibration — it doesn't know what "normal" looks like.
  Without a reference, it either over-detects or does nothing.

This approach:
  Stage1: LOOSE fixed threshold (alpha=0.1) → high recall, many FP candidates
  Stage2: Per-candidate COMPARISON:
    LEFT image  = random normal window from TRAIN data (this is what normal looks like)
    RIGHT image = candidate window from TEST data
    Question    = "Does right show anomalous behavior compared to left?"

  GPT-4o can now do RELATIVE comparison instead of ABSOLUTE judgment.
  This gives it explicit calibration — it knows what normal looks like.

Expected outcome:
  Stage1 (loose): high recall, low precision
  Stage2 (calibrated): prune FPs while keeping TPs → F1 improvement

Usage:
  Set OPENAI_API_KEY environment variable before running:
    $env:OPENAI_API_KEY = "sk-proj-..."
  Then run:
    python experiment_stage2_calibrated.py
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

# Load API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable not set.\n"
        "Run: $env:OPENAI_API_KEY = 'sk-proj-...'"
    )

VLM_SLEEP   = 3.0
LOOSE_ALPHA = 0.1          # fixed loose threshold for Stage1 → more candidates
WIN         = 224          # window size (same as DINOv2)
CACHE_BASE  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR     = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_calibrated")

SMD_ENTITIES  = ["machine-1-1", "machine-1-2", "machine-1-5"]
N_NORMAL_REFS = 3   # sample N normal windows for the reference
TOP_K_CH      = 4   # top channels to visualize


# ════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════

def load_smd(entity):
    test   = np.loadtxt(SMD_DIR / "test"       / f"{entity}.txt", delimiter=",")
    train  = np.loadtxt(SMD_DIR / "train"      / f"{entity}.txt", delimiter=",")
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


# ════════════════════════════════════════════════════════
# Interval / F1 Utils
# ════════════════════════════════════════════════════════

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
        return 0.0, 0.0, 0.0
    gt = [tuple(i) for i in gt_ivs]
    pr = [tuple(i) for i in pred_ivs]
    TP = sum(1 for d in pr if any(_overlap(d, a) for a in gt))
    FP = sum(1 for d in pr if not any(_overlap(d, a) for a in gt))
    FN = sum(1 for a in gt if not any(_overlap(a, d) for d in pr))
    p  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


# ════════════════════════════════════════════════════════
# Stage1: LOOSE threshold (alpha=0.1)
# ════════════════════════════════════════════════════════

def get_stage1_loose(ch_scores, ov_scores, T_test, labels):
    """
    Stage1 with FIXED loose threshold (alpha=0.1).
    Purpose: high recall, many candidates including FPs — Stage2 will prune.
    Returns both loose intervals AND oracle-best intervals for comparison.
    """
    score_key = "ml_topk10"
    fallback  = ["final_topk10", "ml_sum", "final_sum"]

    inter_list = []
    for sc in ov_scores:
        for k in [score_key] + fallback:
            if k in sc and len(sc[k]) == T_test:
                inter_list.append(sc[k])
                break

    inter_agg = np.mean(inter_list, axis=0) if inter_list else np.zeros(T_test)
    gt_ivs    = get_intervals(labels)

    mu, sigma = inter_agg.mean(), inter_agg.std()
    if sigma < 1e-12:
        return inter_agg, [], gt_ivs, 0.0, []

    # Fixed loose threshold
    thr       = mu + norm.ppf(1 - LOOSE_ALPHA) * sigma
    loose_ivs = get_intervals((inter_agg > thr).astype(int))

    # Oracle best for comparison
    best_f1, best_ivs = 0.0, []
    for alpha in [0.1, 0.05, 0.01, 0.001]:
        t2   = mu + norm.ppf(1 - alpha) * sigma
        pivs = get_intervals((inter_agg > t2).astype(int))
        f1, _, _ = interval_f1(gt_ivs, pivs)
        if f1 > best_f1:
            best_f1, best_ivs = f1, pivs

    return inter_agg, loose_ivs, gt_ivs, best_f1, best_ivs


# ════════════════════════════════════════════════════════
# Top Channels
# ════════════════════════════════════════════════════════

def get_top_channels(test, k=TOP_K_CH):
    var = test.var(axis=0)
    return np.argsort(var)[::-1][:k].tolist()


# ════════════════════════════════════════════════════════
# Comparison Image: [Normal Ref | Candidate]
# ════════════════════════════════════════════════════════

def _normalize_01(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _pick_normal_starts(T_train, win=WIN, n=N_NORMAL_REFS, rng_seed=42):
    rng   = np.random.default_rng(rng_seed)
    starts, tries = [], 0
    while len(starts) < n and tries < 300:
        s = int(rng.integers(0, max(1, T_train - win)))
        if all(abs(s - ss) >= win for ss in starts):
            starts.append(s)
        tries += 1
    return starts


def generate_comparison_image(train, test, candidate_iv, top_chs, win=WIN):
    """
    Side-by-side comparison:
      LEFT  = Normal reference from train (N windows, each normalized, overlaid)
      RIGHT = Candidate window from test (normalized, overlaid)

    Channels overlaid on same axis (normalized 0-1) to highlight inter-channel
    relationships — same visualization as DINOv2 INTER overlay scoring.
    """
    s, e = candidate_iv

    # Candidate window
    cand_win = test[s : s + win, :]  # always WIN steps

    # Normal reference windows from train
    ref_starts = _pick_normal_starts(len(train), win=win, n=N_NORMAL_REFS)
    ref_wins   = [train[rs : rs + win, :] for rs in ref_starts]

    n_ch   = len(top_chs)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b"][:n_ch]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── LEFT: Normal Reference ──
    ax = axes[0]
    for ref_w in ref_wins:
        for i, ch in enumerate(top_chs):
            seg = _normalize_01(ref_w[:, ch])
            ax.plot(seg, color=colors[i], alpha=0.3, linewidth=0.6)
    for i, ch in enumerate(top_chs):
        seg = _normalize_01(ref_wins[0][:, ch])
        ax.plot(seg, color=colors[i], linewidth=1.4, label=f"Ch{ch}")
    ax.set_title("NORMAL reference (train)", fontsize=10, color="darkgreen", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time steps", fontsize=8)
    ax.set_ylabel("Normalized value [0,1]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")

    # ── RIGHT: Candidate ──
    ax = axes[1]
    for i, ch in enumerate(top_chs):
        seg = _normalize_01(cand_win[:, ch])
        ax.plot(seg, color=colors[i], linewidth=1.4, label=f"Ch{ch}")
    ax.set_title(f"CANDIDATE [{s},{e}] (test)", fontsize=10, color="darkred", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time steps", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")

    plt.suptitle(f"Normal vs Candidate | Channels: {top_chs}", fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert anomaly detector for multivariate server monitoring time series.

You will be shown a side-by-side comparison image:
- LEFT panel  = NORMAL behavior sampled from training data (guaranteed normal operation)
- RIGHT panel = CANDIDATE interval from test data to evaluate

Each panel shows multiple server metric channels (CPU, memory, disk I/O, network, etc.)
all normalized to [0,1] and overlaid on the same axis.
This overlay highlights INTER-CHANNEL RELATIONSHIPS.

In NORMAL operation (left panel), channels show:
- Stable, predictable oscillation patterns
- Consistent relative positions between channels
- Smooth co-variation: channels that track each other continue to do so

ANOMALOUS behavior (right should differ from left):
- Channel relationships break down (lines that co-vary in left diverge in right)
- Sudden isolated spikes/drops in one channel while others stay stable
- Chaotic, tangled crossing patterns not present in the left panel
- Structural change in how channels move together

Your ONLY job: compare RIGHT to LEFT visually and decide if RIGHT is anomalous."""


def build_candidate_prompt(s, e, entity, top_chs, score_mean):
    return f"""Entity: {entity} | Candidate window: [{s}, {e}]
Channels: {top_chs} | Stage1 anomaly score: {score_mean:.4f}

Look carefully at both panels:
1. Do the channel RELATIONSHIPS look different? (lines that co-vary in left — do they still co-vary in right?)
2. Are there sudden isolated changes in right that don't appear in left?
3. Does the overall pattern structure look different?

Be STRICT: only say ANOMALY if the right panel is clearly, visually different from the left.
If right looks like it could be a normal sample (similar to left), say NORMAL.

Reply with JSON only:
{{
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "reason": "one-sentence visual observation"
}}"""


# ════════════════════════════════════════════════════════
# Query GPT-4o
# ════════════════════════════════════════════════════════

def query_vlm_single(img_b64, text_prompt, attempt_max=5):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(attempt_max):
        try:
            time.sleep(VLM_SLEEP)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",      "text": text_prompt},
                            {"type": "image_url", "image_url": {
                                "url":    f"data:image/png;base64,{img_b64}",
                                "detail": "high"
                            }}
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
                verdict = "ANOMALY" if "ANOMALY" in raw.upper() else "NORMAL"
                return {"verdict": verdict, "confidence": "LOW", "reason": raw[:120]}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt + 1) * 30
                print(f"      Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            elif "insufficient_quota" in err:
                print(f"      Quota exhausted!", flush=True)
                return None
            else:
                print(f"      API error: {exc}", flush=True)
                time.sleep(5)
    return None


# ════════════════════════════════════════════════════════
# Main per-entity
# ════════════════════════════════════════════════════════

def run_entity(entity):
    print(f"\n{'='*60}")
    print(f"{entity}")
    print(f"{'='*60}", flush=True)

    train, test, labels = load_smd(entity)
    T_test = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter_agg, loose_ivs, gt_ivs, oracle_f1, oracle_ivs = get_stage1_loose(
        ch_scores, ov_scores, T_test, labels)

    n_gt      = len(gt_ivs)
    loose_f1, loose_p, loose_r = interval_f1(gt_ivs, loose_ivs)

    print(f"  GT={n_gt} | Stage1 oracle F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose  F1={loose_f1:.4f} (P={loose_p:.2f} R={loose_r:.2f}) | {len(loose_ivs)} candidates", flush=True)

    top_chs = get_top_channels(test)
    print(f"  Top channels: {top_chs}", flush=True)

    # Save comparison plots
    img_dir = RESULTS_DIR / "comparison_plots" / entity
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed_ivs = []
    candidate_log = []

    print(f"  Querying GPT-4o per candidate ({len(loose_ivs)} total)...", flush=True)

    for idx, (s, e) in enumerate(loose_ivs):
        score_mean = float(inter_agg[s : e + 1].mean())

        img_b64 = generate_comparison_image(train, test, (s, e), top_chs)

        # Save plot for inspection
        with open(img_dir / f"cand_{idx:03d}_{s}_{e}.png", "wb") as fout:
            fout.write(base64.b64decode(img_b64))

        prompt = build_candidate_prompt(s, e, entity, top_chs, score_mean)
        result = query_vlm_single(img_b64, prompt)

        if result is None:
            verdict, confidence, reason = "ANOMALY", "LOW", "API failed, kept by default"
        else:
            verdict    = result.get("verdict",    "ANOMALY").upper()
            confidence = result.get("confidence", "LOW").upper()
            reason     = result.get("reason",     "")[:120]

        if verdict == "ANOMALY":
            confirmed_ivs.append((s, e))

        is_tp = any(_overlap((s, e), g) for g in gt_ivs)
        flag  = "TP" if is_tp else "FP"

        print(f"    [{s:6d},{e:6d}] score={score_mean:.4f} → {verdict} ({confidence}) [{flag}]", flush=True)
        print(f"      {reason}", flush=True)

        candidate_log.append({
            "entity": entity, "start": s, "end": e,
            "score": score_mean, "verdict": verdict,
            "confidence": confidence, "reason": reason,
            "is_tp": is_tp,
        })

    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed_ivs)

    print(f"\n  ── Final ──", flush=True)
    print(f"  Stage1 oracle : F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose  : F1={loose_f1:.4f} ({len(loose_ivs)} pred)", flush=True)
    print(f"  Stage2 calib  : F1={s2_f1:.4f} (P={s2_p:.2f} R={s2_r:.2f}) | {len(confirmed_ivs)} confirmed", flush=True)
    print(f"  vs oracle: {s2_f1 - oracle_f1:+.4f}  |  vs loose: {s2_f1 - loose_f1:+.4f}", flush=True)

    return {
        "entity":            entity,
        "n_gt":              n_gt,
        "oracle_f1":         oracle_f1,
        "loose_f1":          loose_f1,
        "loose_n":           len(loose_ivs),
        "stage2_f1":         s2_f1,
        "stage2_p":          s2_p,
        "stage2_r":          s2_r,
        "stage2_n":          len(confirmed_ivs),
        "removed":           len(loose_ivs) - len(confirmed_ivs),
        "change_vs_oracle":  s2_f1 - oracle_f1,
        "change_vs_loose":   s2_f1 - loose_f1,
        "candidate_log":     candidate_log,
    }


# ════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results, all_logs = [], []

    for entity in SMD_ENTITIES:
        try:
            r = run_entity(entity)
            if r:
                all_logs.extend(r.pop("candidate_log"))
                results.append(r)
        except Exception as ex:
            print(f"  ERROR in {entity}: {ex}", flush=True)
            import traceback; traceback.print_exc()

    if results:
        print(f"\n{'='*75}")
        print("STAGE2 CALIBRATED COMPARISON RESULTS")
        print(f"{'='*75}")
        print(f"{'Entity':<15} {'S1 Oracle':>10} {'S1 Loose':>10} {'S2 Calib':>10} {'ΔOracle':>9} {'ΔLoose':>8}  Kept/Total")
        print("-" * 75)
        for r in results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>10.4f} {r['loose_f1']:>10.4f} "
                  f"{r['stage2_f1']:>10.4f} {r['change_vs_oracle']:>+9.4f} "
                  f"{r['change_vs_loose']:>+8.4f}  {r['stage2_n']}/{r['loose_n']}")
        print("-" * 75)
        o_avg  = np.mean([r["oracle_f1"]  for r in results])
        l_avg  = np.mean([r["loose_f1"]   for r in results])
        s2_avg = np.mean([r["stage2_f1"]  for r in results])
        print(f"{'AVG':<15} {o_avg:>10.4f} {l_avg:>10.4f} {s2_avg:>10.4f} "
              f"{s2_avg - o_avg:>+9.4f} {s2_avg - l_avg:>+8.4f}")

        pd.DataFrame(results).to_csv(
            RESULTS_DIR / "stage2_calibrated_results.csv", index=False)
        pd.DataFrame(all_logs).to_csv(
            RESULTS_DIR / "candidate_verdicts.csv", index=False)

        print(f"\nSaved:")
        print(f"  {RESULTS_DIR / 'stage2_calibrated_results.csv'}")
        print(f"  {RESULTS_DIR / 'candidate_verdicts.csv'}")
        print(f"  Comparison plots: {RESULTS_DIR / 'comparison_plots'}/")
