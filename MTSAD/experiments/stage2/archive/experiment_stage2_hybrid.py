"""
Stage2 MLLM: Hybrid Approach — Global Context + Zoomed Candidate View

이전 실패 원인:
  - Global only: 28,000 steps 중 55-step 이상 → 3px → GPT 못 봄
  - Per-window comparison: 전역 context 없어서 교정 불가

이번 접근:
  각 후보마다 2개 이미지를 GPT-4o에게 동시에 제공:
    Image 1 (Global): 전체 시계열 multi-panel, 모든 후보 위치 표시
    Image 2 (Zoomed): 해당 후보 ± context 구간의 확대 뷰
  → GPT-4o가 "전역 패턴"과 "로컬 디테일" 동시에 판단 가능

  Stage1: alpha=0.3 (loose) → FP 포함 많은 후보
  Stage2: per-candidate GPT-4o 판정 + confidence 필터링
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
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY 환경변수를 설정하세요.")

VLM_SLEEP    = 4.0
LOOSE_ALPHA  = 0.3          # FP 생성용 느슨한 threshold
ZOOM_CONTEXT = 1500         # 후보 앞뒤로 볼 context 길이 (steps)
TOP_K_CH     = 5            # 채널 수
CACHE_BASE   = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_hybrid")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]


# ════════════════════════════════════════════════════════
# Data / Score Loading
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
# Interval / F1
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
# Stage1: LOOSE alpha=0.3
# ════════════════════════════════════════════════════════

def get_stage1_loose(ch_scores, ov_scores, T_test, labels):
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

    thr_loose = mu + norm.ppf(1 - LOOSE_ALPHA) * sigma
    loose_ivs = get_intervals((inter_agg > thr_loose).astype(int))

    best_f1, best_ivs = 0.0, []
    for alpha in [0.1, 0.05, 0.01, 0.001]:
        t   = mu + norm.ppf(1 - alpha) * sigma
        ivs = get_intervals((inter_agg > t).astype(int))
        f1, _, _ = interval_f1(gt_ivs, ivs)
        if f1 > best_f1:
            best_f1, best_ivs = f1, ivs

    return inter_agg, loose_ivs, gt_ivs, best_f1, best_ivs


def get_top_channels(test, k=TOP_K_CH):
    return np.argsort(test.var(axis=0))[::-1][:k].tolist()


# ════════════════════════════════════════════════════════
# Image 1: Global Context (full series, all candidates marked)
# ════════════════════════════════════════════════════════

def generate_global_image(test, all_candidates, current_iv, inter_scores, top_chs, entity, T):
    """
    전체 시계열 stacked subplot.
    - 모든 Stage1 후보: 연한 빨간 음영
    - 현재 평가 중인 후보: 진한 빨간 음영 + 화살표
    """
    n_ch  = len(top_chs)
    n_rows = n_ch + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 1.8 * n_rows), sharex=True)

    ch_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b"][:n_ch]
    x = np.arange(T)
    tick_step = max(T // 10, 1)

    cs, ce = current_iv

    for i, ch in enumerate(top_chs):
        ax = axes[i]
        ax.plot(x, test[:T, ch], color=ch_colors[i], linewidth=0.35, alpha=0.85)
        ax.set_ylabel(f"Ch{ch}", fontsize=6, rotation=0, labelpad=20)
        ax.tick_params(labelsize=5)
        ax.set_xticks(range(0, T, tick_step))
        # 모든 후보 (연한)
        for s, e in all_candidates:
            if (s, e) != (cs, ce):
                ax.axvspan(s, e, alpha=0.12, color="red")
        # 현재 후보 (진한)
        ax.axvspan(cs, ce, alpha=0.40, color="red")

    ax = axes[-1]
    ax.plot(x, inter_scores[:T], color="purple", linewidth=0.45)
    for s, e in all_candidates:
        if (s, e) != (cs, ce):
            ax.axvspan(s, e, alpha=0.12, color="red")
    ax.axvspan(cs, ce, alpha=0.40, color="red")
    ax.set_ylabel("Score", fontsize=6)
    ax.set_xlabel("Time step", fontsize=8)
    ax.tick_params(labelsize=5)
    ax.set_xticks(range(0, T, tick_step))

    plt.suptitle(f"{entity} | Full series | Evaluating [{cs},{ce}] (dark red)",
                 fontsize=8, y=1.002)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# Image 2: Zoomed Candidate View
# ════════════════════════════════════════════════════════

def generate_zoom_image(test, current_iv, inter_scores, top_chs, T):
    """
    후보 구간 ± ZOOM_CONTEXT 확대 뷰.
    - 후보 전후 normal 구간과 함께 표시
    - 후보 구간: 빨간 음영
    - 각 채널 개별 subplot (y값 그대로, 절대 스케일)
    """
    cs, ce   = current_iv
    z_start  = max(0, cs - ZOOM_CONTEXT)
    z_end    = min(T, ce + ZOOM_CONTEXT)
    x        = np.arange(z_start, z_end)

    n_ch  = len(top_chs)
    n_rows = n_ch + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.8 * n_rows), sharex=True)

    ch_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b"][:n_ch]

    for i, ch in enumerate(top_chs):
        ax   = axes[i]
        vals = test[z_start:z_end, ch]
        ax.plot(x, vals, color=ch_colors[i], linewidth=0.7)
        ax.axvspan(cs, ce, alpha=0.30, color="red", label="Candidate" if i == 0 else "")
        ax.set_ylabel(f"Ch{ch}", fontsize=7, rotation=0, labelpad=22)
        ax.tick_params(labelsize=6)
        # 후보 구간 경계선
        ax.axvline(cs, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axvline(ce, color="red", linewidth=0.8, linestyle="--", alpha=0.7)

    ax = axes[-1]
    ax.plot(x, inter_scores[z_start:z_end], color="purple", linewidth=0.6)
    ax.axvspan(cs, ce, alpha=0.30, color="red")
    ax.axvline(cs, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(ce, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylabel("Score", fontsize=7)
    ax.set_xlabel(f"Time step (zoomed: {z_start}-{z_end})", fontsize=8)
    ax.tick_params(labelsize=6)

    cand_len = ce - cs + 1
    plt.suptitle(
        f"ZOOMED VIEW: Candidate [{cs},{ce}] (len={cand_len}) | Context +-{ZOOM_CONTEXT} steps",
        fontsize=8, y=1.002
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# Prompt
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert anomaly detector for multivariate server monitoring time series.
You will be shown two images to evaluate a single candidate anomaly interval."""


def build_hybrid_prompt(entity, cs, ce, top_chs, T, all_candidates, inter_score_mean, inter_agg, gt_ivs_ref=None):
    cand_len = ce - cs + 1
    n_total  = len(all_candidates)
    cand_idx = next((i for i, iv in enumerate(all_candidates) if iv == (cs, ce)), -1)

    # Context scores around candidate
    z_start = max(0, cs - ZOOM_CONTEXT)
    z_end   = min(T, ce + ZOOM_CONTEXT)
    before_score = float(inter_agg[z_start:cs].mean()) if cs > z_start else 0.0
    after_score  = float(inter_agg[ce:z_end].mean())   if z_end > ce  else 0.0

    return f"""You are evaluating ONE candidate anomaly interval for entity "{entity}".

IMAGE 1 (Global Context):
- Full time series for all channels (length={T})
- Dark red shading = current candidate [{cs}, {ce}]
- Light red shading = other Stage1 candidates (not evaluated now)
- Bottom panel = anomaly score

IMAGE 2 (Zoomed View):
- Magnified view of the candidate and surrounding +-{ZOOM_CONTEXT} steps
- Red shading = the candidate interval [{cs}, {ce}] (length={cand_len})
- You can see the ACTUAL channel values before, during, and after the candidate

Candidate details:
  Interval: [{cs}, {ce}]  |  Length: {cand_len} steps
  Stage1 anomaly score (during): {inter_score_mean:.4f}
  Stage1 score (surrounding context): before={before_score:.4f}, after={after_score:.4f}
  Channels shown: {top_chs}
  Position: candidate {cand_idx+1} of {n_total} total

EVALUATION CRITERIA:
1. Global consistency (Image 1): Does this interval STAND OUT from the rest of the series?
   - Level shift: channel values jump to a different baseline
   - Structural break: pattern changes abruptly
   - If it looks like normal fluctuation → NORMAL

2. Local detail (Image 2): What exactly happens INSIDE the candidate vs. the surrounding context?
   - Do channels show sudden spikes / drops / divergence?
   - Compare the red-shaded region to the gray regions before and after
   - If the behavior inside the red region looks similar to outside → NORMAL

3. Multi-channel: True anomalies usually affect MULTIPLE channels simultaneously.

Reply ONLY with JSON:
{{
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": 1, 2, or 3,
  "reason": "one sentence: what specific visual evidence supports your verdict"
}}

Confidence:
  1 = Ambiguous, subtle, not clearly different from normal
  2 = Moderately clear local irregularity
  3 = Strong, unambiguous evidence of anomaly

Be honest: if you cannot see a clear difference in Image 2, say NORMAL with confidence 1 or 2."""


# ════════════════════════════════════════════════════════
# Query GPT-4o (2 images per candidate)
# ════════════════════════════════════════════════════════

def query_vlm_hybrid(img_global_b64, img_zoom_b64, prompt, attempt_max=5):
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
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url":    f"data:image/png;base64,{img_global_b64}",
                                "detail": "low"      # global: low detail (overview)
                            }},
                            {"type": "image_url", "image_url": {
                                "url":    f"data:image/png;base64,{img_zoom_b64}",
                                "detail": "high"     # zoom: high detail (key image)
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
                return {"verdict": verdict, "confidence": 1, "reason": raw[:120]}

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
    print(f"\n{'='*65}")
    print(f"{entity}")
    print(f"{'='*65}", flush=True)

    train, test, labels = load_smd(entity)
    T = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter_agg, loose_ivs, gt_ivs, oracle_f1, oracle_ivs = get_stage1_loose(
        ch_scores, ov_scores, T, labels)

    n_gt = len(gt_ivs)
    loose_f1, loose_p, loose_r = interval_f1(gt_ivs, loose_ivs)

    print(f"  GT={n_gt}", flush=True)
    print(f"  Stage1 oracle:  F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose:   F1={loose_f1:.4f} (P={loose_p:.2f} R={loose_r:.2f}) "
          f"| {len(loose_ivs)} candidates", flush=True)

    top_chs = get_top_channels(test)
    print(f"  Top channels: {top_chs}", flush=True)

    img_dir = RESULTS_DIR / "plots" / entity
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed_ivs = []
    candidate_log = []

    print(f"  Querying GPT-4o per candidate (hybrid, {len(loose_ivs)} total)...", flush=True)

    for idx, (cs, ce) in enumerate(loose_ivs):
        score_mean = float(inter_agg[cs:ce+1].mean())
        is_tp      = any(_overlap((cs, ce), g) for g in gt_ivs)
        flag       = "TP" if is_tp else "FP"

        # 두 이미지 생성
        img_global = generate_global_image(
            test, loose_ivs, (cs, ce), inter_agg, top_chs, entity, T)
        img_zoom   = generate_zoom_image(
            test, (cs, ce), inter_agg, top_chs, T)

        # 저장 (처음 5개만)
        if idx < 5:
            for name, b64 in [("global", img_global), ("zoom", img_zoom)]:
                with open(img_dir / f"cand_{idx:02d}_{cs}_{ce}_{name}.png", "wb") as f:
                    f.write(base64.b64decode(b64))

        # 프롬프트
        prompt = build_hybrid_prompt(
            entity, cs, ce, top_chs, T, loose_ivs, score_mean, inter_agg)

        # API 호출
        result = query_vlm_hybrid(img_global, img_zoom, prompt)

        if result is None:
            verdict, conf, reason = "ANOMALY", 1, "API failed, kept"
        else:
            verdict = result.get("verdict",    "ANOMALY").upper()
            conf    = int(result.get("confidence", 1))
            reason  = result.get("reason",     "")[:120]

        # confidence >= 2 만 확정 (논문 방식: confidence=1 제거)
        if verdict == "ANOMALY" and conf >= 2:
            confirmed_ivs.append((cs, ce))

        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} score={score_mean:.4f} "
              f"-> {verdict}(c={conf}) [{flag}]", flush=True)
        print(f"      {reason}", flush=True)

        candidate_log.append({
            "entity": entity, "start": cs, "end": ce,
            "length": ce - cs + 1,
            "score": score_mean, "verdict": verdict, "confidence": conf,
            "reason": reason, "is_tp": is_tp,
        })

    # Evaluate
    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed_ivs)
    removed = sum(1 for iv in loose_ivs if iv not in confirmed_ivs)
    added   = sum(1 for vi in confirmed_ivs if not any(_overlap(vi, iv) for iv in loose_ivs))

    print(f"\n  -- Final --", flush=True)
    print(f"  Stage1 oracle : F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose  : F1={loose_f1:.4f} ({len(loose_ivs)} pred)", flush=True)
    print(f"  Stage2 hybrid : F1={s2_f1:.4f} (P={s2_p:.2f} R={s2_r:.2f}) "
          f"| {len(confirmed_ivs)} confirmed", flush=True)
    print(f"  vs oracle: {s2_f1 - oracle_f1:+.4f}  |  vs loose: {s2_f1 - loose_f1:+.4f}", flush=True)
    print(f"  Removed: {removed}, Added: {added}", flush=True)

    return {
        "entity":            entity,
        "n_gt":              n_gt,
        "oracle_f1":         oracle_f1,
        "oracle_n":          len(oracle_ivs),
        "loose_f1":          loose_f1,
        "loose_p":           loose_p,
        "loose_r":           loose_r,
        "loose_n":           len(loose_ivs),
        "stage2_f1":         s2_f1,
        "stage2_p":          s2_p,
        "stage2_r":          s2_r,
        "stage2_n":          len(confirmed_ivs),
        "removed":           removed,
        "added":             added,
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
        print("HYBRID STAGE2 RESULTS (Global + Zoom)")
        print(f"{'='*75}")
        print(f"{'Entity':<15} {'S1 Oracle':>10} {'S1 Loose':>10} "
              f"{'S2 Hybrid':>10} {'dOracle':>9} {'dLoose':>8}  n")
        print("-" * 75)
        for r in results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>10.4f} {r['loose_f1']:>10.4f} "
                  f"{r['stage2_f1']:>10.4f} {r['change_vs_oracle']:>+9.4f} "
                  f"{r['change_vs_loose']:>+8.4f}  "
                  f"{r['stage2_n']}/{r['loose_n']}")
        print("-" * 75)
        o_avg  = np.mean([r["oracle_f1"]  for r in results])
        l_avg  = np.mean([r["loose_f1"]   for r in results])
        s2_avg = np.mean([r["stage2_f1"]  for r in results])
        print(f"{'AVG':<15} {o_avg:>10.4f} {l_avg:>10.4f} {s2_avg:>10.4f} "
              f"{s2_avg - o_avg:>+9.4f} {s2_avg - l_avg:>+8.4f}")

        pd.DataFrame(results).to_csv(
            RESULTS_DIR / "stage2_hybrid_results.csv", index=False)
        pd.DataFrame(all_logs).to_csv(
            RESULTS_DIR / "candidate_verdicts.csv", index=False)

        print(f"\nSaved: {RESULTS_DIR / 'stage2_hybrid_results.csv'}")
        print(f"Plots: {RESULTS_DIR / 'plots/'}")
