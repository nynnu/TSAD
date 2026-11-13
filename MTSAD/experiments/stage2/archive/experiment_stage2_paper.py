"""
Stage2 MLLM: VLM4TS 논문 구현 (다변량 확장)

논문이 실제로 하는 것:
  Stage1: alpha=0.1 고정 (loose, high-recall) → FP 많은 후보들
  Stage2:
    - Visual: 전체 시계열 stacked subplot (채널별 multi-panel, full length)
    - Text: Stage1 후보 목록 (채널별 요약 포함)
    - Prompt: "eliminate consistent (FP) + add missed + confidence 1-3"
    - Filter: confidence=1 제거

이전 실패 원인:
  - global: Oracle Stage1 사용 → FP 없어서 VLM이 추가만 함
  - calibrated: per-window 비교 → 전역 context 없어 모든 것을 ANOMALY 판정
  - 이번: 논문 원본 프롬프트 + loose Stage1 → 논문 방식 그대로
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
    raise EnvironmentError(
        "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
        "실행 전: export OPENAI_API_KEY='sk-proj-...'"
    )

VLM_SLEEP   = 5.0
LOOSE_ALPHA = 0.3          # 더 느슨한 threshold → FP 생성 (VLM이 제거할 대상)
TOP_K_CH    = 6            # 시각화할 채널 수 (다변량)
CACHE_BASE  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR     = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_paper_a03")

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
# Stage1: LOOSE alpha=0.1 (논문과 동일)
# ════════════════════════════════════════════════════════

def get_stage1_loose(ch_scores, ov_scores, T_test, labels):
    """
    alpha=0.1 고정 → 논문의 "high-recall screening" 재현.
    Oracle(best F1) 탐색은 비교용으로만 계산.
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

    # Loose (논문 방식)
    thr_loose = mu + norm.ppf(1 - LOOSE_ALPHA) * sigma
    loose_ivs = get_intervals((inter_agg > thr_loose).astype(int))

    # Oracle best (비교용)
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
# Visual: 전체 시계열 Stacked Subplot (논문 Figure 6)
# ════════════════════════════════════════════════════════

def generate_full_series_plot(test, stage1_ivs, inter_scores, top_chs, entity, T):
    """
    논문의 Figure 6 하단: stacked-subplot visualization of full multivariate series.
    - 채널별 subplot (raw values, GLOBAL y-scale 고정)
    - Stage1 후보 구간 빨간 음영
    - 하단 anomaly score 패널
    - x-axis tick marks (GPT가 구간 경계 읽을 수 있도록)
    """
    n_ch = len(top_chs)
    n_rows = n_ch + 1  # 채널 + score 패널
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.0 * n_rows), sharex=True)

    ch_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"][:n_ch]
    x = np.arange(T)

    tick_step = max(T // 12, 1)
    tick_positions = list(range(0, T, tick_step))

    for i, ch in enumerate(top_chs):
        ax = axes[i]
        raw = test[:T, ch]
        ax.plot(x, raw, color=ch_colors[i], linewidth=0.4, alpha=0.9)
        ax.set_ylabel(f"Ch{ch}", fontsize=7, rotation=0, labelpad=22)
        ax.tick_params(labelsize=6)
        ax.set_xticks(tick_positions)
        # Stage1 candidate 구간 표시
        for s, e in stage1_ivs:
            ax.axvspan(s, e, alpha=0.20, color="red")

    # Anomaly score 패널 (마지막)
    ax = axes[-1]
    ax.plot(x, inter_scores[:T], color="purple", linewidth=0.5, label="Stage1 score")
    for s, e in stage1_ivs:
        ax.axvspan(s, e, alpha=0.20, color="red")
    ax.set_ylabel("Score", fontsize=7)
    ax.set_xlabel("Time step index", fontsize=9)
    ax.tick_params(labelsize=6)
    ax.set_xticks(tick_positions)
    ax.legend(fontsize=6, loc="upper right")

    plt.suptitle(
        f"{entity} — Full Multivariate Series  |  Red = Stage1 candidates (alpha=0.1)",
        fontsize=9, y=1.005
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# Prompt (논문 Appendix 원본 구조 + 다변량 확장)
# ════════════════════════════════════════════════════════

VLM_SYSTEM = """You are an expert in both time-series analysis and multimodal (vision + language) reasoning."""


def build_prompt(entity, stage1_ivs, top_chs, T, inter_agg):
    """
    논문 Appendix 프롬프트 구조 그대로 + 다변량 맞게 확장.
    핵심:
      - Eliminate: 전체 트렌드와 일치하는 후보 (FP) 제거
      - Add: 놓친 이상 구간 추가
      - Confidence 1-3 (1=low → 최종 필터링으로 제거)
    """

    # Stage1 후보 텍스트 목록
    candidate_text = ""
    for i, (s, e) in enumerate(stage1_ivs):
        score_mean = float(inter_agg[s:e+1].mean())
        candidate_text += f"  Candidate {i+1}: [{s}, {e}]  (score={score_mean:.4f})\n"

    prompt = f"""You will be shown:
1. A stacked multi-panel plot of raw multivariate time-series data
   - X-axis: time step index (0 to {T-1})
   - Y-axis: signal values for each channel
   - Each subplot is one server metric channel: {top_chs}
   - Bottom panel: anomaly score from Stage1 visual screening
2. Preliminary anomaly candidates detected by Stage1 (red shaded regions):

{candidate_text}
These candidates were detected by a local visual pattern-matching model (may include false positives).

Your goal is to integrate both sources — the visual plot and the preliminary candidates — and produce a refined, final anomaly detection for the ENTIRE series. Specifically:

• ELIMINATE any preliminary candidates that appear anomalous in isolation but are CONSISTENT WITH THE OVERALL TREND of the series (false positives). These are regions that look like the rest of the normal series when viewed in global context.

• ADD any intervals that Stage1 missed but which clearly BREAK TEMPORAL CONTINUITY or exhibit clear statistical irregularities (sudden spikes, level shifts, abrupt structural changes) visible in the channels.

• For multivariate: a true anomaly typically affects MULTIPLE CHANNELS simultaneously or shows clear cross-channel relationship breakdown. Single-channel noise is less likely to be a true anomaly.

Reply ONLY with a JSON object containing these fields:
{{
  "interval_index": [[start1, end1], [start2, end2], ...],
  "confidence": [c1, c2, ...],
  "abnormal_description": "brief paragraph (under 100 words) summarizing why these intervals are anomalous"
}}

Confidence scale:
  1 = Low: ambiguous or very subtle deviation — consistent with normal fluctuation
  2 = Medium: clear local irregularity but moderate global uncertainty
  3 = High: strong statistical or contextual evidence of anomaly

Important:
- Estimate interval boundaries using the x-axis tick marks as precisely as possible
- Do NOT include extra keys or commentary — only the JSON object above
- If Stage1 candidates look globally consistent (normal), assign confidence=1 or remove them
- Be conservative: prefer fewer high-confidence detections over many uncertain ones"""

    return prompt


# ════════════════════════════════════════════════════════
# Query GPT-4o (1 call per entity)
# ════════════════════════════════════════════════════════

def query_vlm(img_b64, prompt, attempt_max=5):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(attempt_max):
        try:
            time.sleep(VLM_SLEEP)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": VLM_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url":    f"data:image/png;base64,{img_b64}",
                                "detail": "high"
                            }}
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                return json.loads(m.group(0)) if m else {"raw": raw}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt + 1) * 30
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            elif "insufficient_quota" in err:
                print(f"    Quota exhausted!", flush=True)
                return None
            else:
                print(f"    API error: {exc}", flush=True)
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
    T = len(labels)
    ch_scores, ov_scores = load_scores(entity)

    inter_agg, loose_ivs, gt_ivs, oracle_f1, oracle_ivs = get_stage1_loose(
        ch_scores, ov_scores, T, labels)

    n_gt = len(gt_ivs)
    loose_f1, loose_p, loose_r = interval_f1(gt_ivs, loose_ivs)

    print(f"  GT={n_gt}", flush=True)
    print(f"  Stage1 oracle: F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose (alpha=0.1): F1={loose_f1:.4f} "
          f"(P={loose_p:.2f} R={loose_r:.2f}) | {len(loose_ivs)} candidates", flush=True)

    top_chs = get_top_channels(test)
    print(f"  Top channels: {top_chs}", flush=True)

    # 전체 시계열 stacked plot 생성
    print(f"  Generating full multivariate series plot...", flush=True)
    img_b64 = generate_full_series_plot(test, loose_ivs, inter_agg, top_chs, entity, T)

    # 이미지 저장
    img_dir = RESULTS_DIR / "plots"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(img_dir / f"{entity}_fullseries.png", "wb") as f:
        f.write(base64.b64decode(img_b64))

    # 프롬프트 빌드 (논문 원본 구조)
    prompt = build_prompt(entity, loose_ivs, top_chs, T, inter_agg)

    # GPT-4o 1 call per entity
    print(f"  Calling GPT-4o (full series + {len(loose_ivs)} candidates)...", flush=True)
    result = query_vlm(img_b64, prompt)

    if not result:
        print(f"  VLM call failed", flush=True)
        return None

    # Parse result
    raw_ivs    = result.get("interval_index", [])
    confidences = result.get("confidence", [])
    description = result.get("abnormal_description", "")

    if not raw_ivs:
        print(f"  VLM returned empty intervals", flush=True)
        vlm_ivs_all = []
        vlm_ivs_filtered = []
    else:
        vlm_ivs_all = [(int(s), int(e)) for s, e in raw_ivs]
        # confidence=1 필터링 (논문의 핵심 필터)
        if confidences and len(confidences) == len(vlm_ivs_all):
            vlm_ivs_filtered = [iv for iv, c in zip(vlm_ivs_all, confidences) if c >= 2]
            removed_low_conf = len(vlm_ivs_all) - len(vlm_ivs_filtered)
        else:
            vlm_ivs_filtered = vlm_ivs_all
            removed_low_conf = 0

    # Evaluate
    s2_all_f1, s2_all_p, s2_all_r = interval_f1(gt_ivs, vlm_ivs_all)
    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, vlm_ivs_filtered)

    # FP 제거 / 추가 카운트 (loose 기준)
    removed = sum(1 for iv in loose_ivs if not any(_overlap(iv, vi) for vi in vlm_ivs_filtered))
    added   = sum(1 for vi in vlm_ivs_filtered if not any(_overlap(vi, iv) for iv in loose_ivs))

    print(f"\n  ── Results ──", flush=True)
    print(f"  Stage1 oracle : F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose  : F1={loose_f1:.4f} ({len(loose_ivs)} pred)", flush=True)
    print(f"  Stage2 raw    : F1={s2_all_f1:.4f} ({len(vlm_ivs_all)} pred, before conf filter)", flush=True)
    print(f"  Stage2 final  : F1={s2_f1:.4f} (P={s2_p:.2f} R={s2_r:.2f}) | {len(vlm_ivs_filtered)} pred", flush=True)
    print(f"  vs oracle: {s2_f1 - oracle_f1:+.4f}  |  vs loose: {s2_f1 - loose_f1:+.4f}", flush=True)
    print(f"  Removed from loose: {removed}, Added new: {added}", flush=True)
    if confidences:
        print(f"  Confidences: {confidences}", flush=True)
    print(f"  Description: {description[:200]}", flush=True)

    return {
        "entity":           entity,
        "n_gt":             n_gt,
        "oracle_f1":        oracle_f1,
        "oracle_n":         len(oracle_ivs),
        "loose_f1":         loose_f1,
        "loose_p":          loose_p,
        "loose_r":          loose_r,
        "loose_n":          len(loose_ivs),
        "stage2_raw_f1":    s2_all_f1,
        "stage2_raw_n":     len(vlm_ivs_all),
        "stage2_f1":        s2_f1,
        "stage2_p":         s2_p,
        "stage2_r":         s2_r,
        "stage2_n":         len(vlm_ivs_filtered),
        "removed_from_loose": removed,
        "added_new":        added,
        "change_vs_oracle": s2_f1 - oracle_f1,
        "change_vs_loose":  s2_f1 - loose_f1,
        "confidences":      str(confidences),
        "description":      description[:300],
    }


# ════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for entity in SMD_ENTITIES:
        try:
            r = run_entity(entity)
            if r:
                results.append(r)
        except Exception as ex:
            print(f"  ERROR in {entity}: {ex}", flush=True)
            import traceback; traceback.print_exc()

    if results:
        print(f"\n{'='*80}")
        print("VLM4TS PAPER METHOD -- MULTIVARIATE RESULTS")
        print(f"{'='*80}")
        print(f"{'Entity':<15} {'S1 Oracle':>10} {'S1 Loose':>10} {'S2 Raw':>8} {'S2 Final':>10} {'ΔOracle':>9} {'ΔLoose':>8}  n")
        print("-" * 80)
        for r in results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>10.4f} {r['loose_f1']:>10.4f} "
                  f"{r['stage2_raw_f1']:>8.4f} {r['stage2_f1']:>10.4f} "
                  f"{r['change_vs_oracle']:>+9.4f} {r['change_vs_loose']:>+8.4f}  "
                  f"{r['stage2_n']}/{r['loose_n']}")
        print("-" * 80)
        o_avg  = np.mean([r["oracle_f1"]     for r in results])
        l_avg  = np.mean([r["loose_f1"]      for r in results])
        s2_avg = np.mean([r["stage2_f1"]     for r in results])
        print(f"{'AVG':<15} {o_avg:>10.4f} {l_avg:>10.4f} {'':>8} {s2_avg:>10.4f} "
              f"{s2_avg - o_avg:>+9.4f} {s2_avg - l_avg:>+8.4f}")

        pd.DataFrame(results).to_csv(
            RESULTS_DIR / "stage2_paper_results.csv", index=False)
        print(f"\nSaved: {RESULTS_DIR / 'stage2_paper_results.csv'}")
        print(f"Plots: {RESULTS_DIR / 'plots'}/")
