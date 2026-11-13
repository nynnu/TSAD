"""
Stage2 MLLM: Test-Derived Normal Prototype + DINOv2-Style Visualization

핵심 혁신 3가지:

  1) Global Normalization (DINOv2와 동일)
     - y축을 전체 test 시계열 global min/max로 고정
     - level shift가 절대 수치로 시각화됨 (윈도우별 정규화 시 사라지는 정보 보존)

  2) Test-Derived Normal Prototype (분포 이동 문제 해결)
     - Stage1 inter score 최저 N개 test 윈도우 = 이 기계의 실제 정상 상태
     - train 데이터 불필요 → train/test 분포 이동 문제 완전 해결
     - 동일 test 기간 → 계절성/드리프트 자동 반영

  3) DINOv2-Style Overlay Comparison
     - 같은 채널, 같은 global normalization으로 두 overlay 생성
     - 왼쪽: 정상 prototype (lowest-score test windows)
     - 오른쪽: 후보 윈도우
     - GPT-4o가 Stage1이 "다르다"고 판단한 바로 그 시각적 차이를 확인

이전 실패 원인과 해결:
  - Train reference 실패 → Test-derived prototype으로 대체
  - 윈도우별 정규화 실패 → Global normalization
  - Per-candidate 방식 유지 (전역 context 없어서 실패한 게 아님)
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
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY 환경변수를 설정하세요.")

VLM_SLEEP       = 4.0
LOOSE_ALPHA     = 0.3          # loose threshold → FP 포함 후보
WIN             = 224          # DINOv2 window size (stride=56)
STRIDE          = 56
N_NORMAL_REFS   = 3            # 정상 prototype 윈도우 수
TOP_K_CH        = 5            # overlay에 보여줄 채널 수
LOCAL_RADIUS    = 4000         # 후보 ±이 범위 내에서 정상 참조 탐색 (시간적 locality)
CACHE_BASE      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR         = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR     = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_localref")

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
# Stage1: Loose threshold
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
# Innovation 1: Test-Derived Normal Prototype Selection
# ════════════════════════════════════════════════════════

def get_local_normal_starts(inter_agg, candidate_iv, loose_ivs, T,
                             win=WIN, stride=STRIDE, n=N_NORMAL_REFS,
                             radius=LOCAL_RADIUS):
    """
    후보 구간 근처 (±radius steps)에서 가장 점수 낮은 정상 윈도우 선택.

    핵심: 시간적으로 가까운 정상 참조 → intra-test concept drift 해결.
    - 같은 시간대 = 같은 "정상 상태" (레벨, 계절성 동일)
    - Stage1 loose 후보와 겹치는 윈도우는 제외
    """
    cs, ce   = candidate_iv
    t_start  = max(0, cs - radius)
    t_end    = min(T - win, ce + radius)

    # 후보 구간에서 모든 슬라이딩 윈도우 수집
    window_scores = []
    for start in range(t_start, t_end, stride):
        # 현재 후보와 겹치면 제외
        if _overlap((start, start + win - 1), (cs, ce)):
            continue
        # 다른 Stage1 후보와 겹치면 제외 (정상 윈도우만)
        if any(_overlap((start, start + win - 1), (s, e))
               for s, e in loose_ivs if (s, e) != (cs, ce)):
            continue
        score = float(inter_agg[start:start+win].mean())
        window_scores.append((score, start))

    if not window_scores:
        # fallback: 후보 구간 외 전체에서 탐색
        for start in range(0, T - win, stride):
            if _overlap((start, start+win-1), (cs, ce)):
                continue
            score = float(inter_agg[start:start+win].mean())
            window_scores.append((score, start))

    window_scores.sort(key=lambda x: x[0])

    selected_starts = []
    for score, start in window_scores:
        if len(selected_starts) >= n:
            break
        if all(abs(start - s) >= win for s in selected_starts):
            selected_starts.append(start)

    scores_selected = [inter_agg[s:s+win].mean() for s in selected_starts]
    return selected_starts, scores_selected


# ════════════════════════════════════════════════════════
# Innovation 2: Global-Normalized DINOv2-Style Overlay
# ════════════════════════════════════════════════════════

def compute_global_norm_params(test, top_chs):
    """
    전체 test 시계열의 global min/max per channel.
    DINOv2와 동일: [mint xt, maxt xt] for every image.
    """
    ch_min = {ch: test[:, ch].min() for ch in top_chs}
    ch_max = {ch: test[:, ch].max() for ch in top_chs}
    return ch_min, ch_max


def _global_normalize(vals, ch_min, ch_max):
    """Global min/max로 0-1 정규화 (DINOv2 방식)."""
    lo, hi = ch_min, ch_max
    if hi - lo < 1e-9:
        return np.zeros_like(vals)
    return (vals - lo) / (hi - lo)


def render_overlay_window(data, start, end, top_chs, ch_min, ch_max, win=WIN):
    """
    단일 윈도우의 DINOv2-style overlay 이미지.
    - Global normalization (전체 시계열 기준)
    - 모든 채널 같은 축에 겹침
    - 배경 없이 라인만
    """
    seg_len = min(end - start, win)
    x       = np.arange(seg_len)
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b"][:len(top_chs)]

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    for i, ch in enumerate(top_chs):
        raw  = data[start:start + seg_len, ch]
        norm = _global_normalize(raw, ch_min[ch], ch_max[ch])
        ax.plot(x, norm, color=colors[i], linewidth=1.0,
                label=f"Ch{ch}", alpha=0.85)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, seg_len - 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.5)
    ax.set_title(f"t=[{start},{end}]", fontsize=7)
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_comparison_image_v2(test, normal_starts, candidate_iv, top_chs,
                                  ch_min, ch_max, inter_agg, win=WIN):
    """
    DINOv2-Style Comparison Image:
      - 왼쪽 열: N개 정상 prototype overlay (test-derived, global norm)
      - 오른쪽 열: 후보 윈도우 overlay (global norm)
    """
    cs, ce = candidate_iv
    n_refs  = len(normal_starts)

    # Figure layout: 1 row × (n_refs + 1) cols
    # 또는 2 rows × ceil((n_refs+1)/2) cols - 가독성을 위해 2행 레이아웃
    n_cols = min(n_refs + 1, 4)
    n_rows = max(1, (n_refs + 1 + n_cols - 1) // n_cols)

    # 심플하게: 1행 N+1열
    total = n_refs + 1
    fig, axes = plt.subplots(1, total, figsize=(3.5 * total, 3.5))
    if total == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(top_chs)]
    seg_x  = np.arange(win)

    # ── Normal prototypes (왼쪽) ──
    for i, ns in enumerate(normal_starts):
        ax   = axes[i]
        nscore = float(inter_agg[ns:ns+win].mean())
        for j, ch in enumerate(top_chs):
            raw  = test[ns:ns+win, ch]
            norm = _global_normalize(raw, ch_min[ch], ch_max[ch])
            ax.plot(seg_x[:len(norm)], norm, color=colors[j],
                    linewidth=0.9, alpha=0.85)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=6)
        ax.set_title(f"NORMAL ref {i+1}\nt=[{ns},{ns+win}]\nscore={nscore:.4f}",
                     fontsize=7, color="darkgreen")
        ax.set_facecolor("#f0fff0")  # 연한 녹색 배경 = 정상
        for sp in ax.spines.values():
            sp.set_edgecolor("green")
            sp.set_linewidth(1.5)

    # ── Candidate (오른쪽 마지막 열) ──
    ax   = axes[-1]
    cscore = float(inter_agg[cs:ce+1].mean())
    clen = min(ce - cs + 1, win)
    cx   = np.arange(clen)
    for j, ch in enumerate(top_chs):
        raw  = test[cs:cs+clen, ch]
        norm = _global_normalize(raw, ch_min[ch], ch_max[ch])
        ax.plot(cx, norm, color=colors[j], linewidth=0.9,
                label=f"Ch{ch}", alpha=0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=6)
    ax.set_title(f"CANDIDATE\nt=[{cs},{ce}]\nscore={cscore:.4f}",
                 fontsize=7, color="darkred")
    ax.set_facecolor("#fff0f0")  # 연한 빨간 배경 = 후보
    for sp in ax.spines.values():
        sp.set_edgecolor("red")
        sp.set_linewidth(1.5)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.5)

    plt.suptitle(
        f"DINOv2-Style Overlay Comparison | Channels: {top_chs} | Global norm [0,1]",
        fontsize=8, y=1.04
    )
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert anomaly detector specializing in multivariate server monitoring time series.
You understand how inter-channel correlation patterns indicate system health."""


def build_testref_prompt(entity, cs, ce, top_chs, n_refs, cand_score, normal_scores,
                          ch_min, ch_max):
    score_mean_normal = np.mean(normal_scores)
    score_ratio = cand_score / score_mean_normal if score_mean_normal > 0 else 1.0

    # Explain what global normalization means
    ch_range_info = ""
    for ch in top_chs[:3]:
        lo, hi = ch_min[ch], ch_max[ch]
        ch_range_info += f"  Ch{ch}: raw range [{lo:.3f}, {hi:.3f}]\n"

    return f"""Entity: {entity} | Evaluating candidate interval [{cs}, {ce}]

You are shown {n_refs + 1} overlay images, all using IDENTICAL global normalization:
- Each channel is normalized to [0,1] using the SAME global min/max across the ENTIRE test series
  {ch_range_info.strip()}
- This means: if a channel shifts to a higher value globally, it appears HIGHER on the plot
- If channels diverge or their relative positions change → structural anomaly

LEFT {n_refs} images = NORMAL REFERENCE
  These are the {n_refs} windows with the LOWEST anomaly scores in the entire test series.
  They represent NORMAL operating behavior of this machine.
  Average score: {score_mean_normal:.4f}

RIGHT image = CANDIDATE [{cs}, {ce}]
  Stage1 visual screening flagged this window as potentially anomalous.
  Score: {cand_score:.4f} (= {score_ratio:.1f}x the normal reference score)

YOUR TASK:
1. Look at the relative positions and shapes of the colored lines (each color = one channel)
2. In the NORMAL references: note how the channels relate to each other
   - Which channels run high? Which run low? Which track each other?
3. In the CANDIDATE: does the inter-channel relationship CHANGE compared to the references?
   - Do channels that normally track each other suddenly diverge?
   - Does a channel shift to a position it never occupies in normal operation?
   - Does the overall pattern structure look fundamentally different?

IMPORTANT: Because we use global normalization, a channel at y=0.8 in normal AND candidate
means it's at the SAME absolute value. A difference in position means a TRUE value change.

Be decisive. If you see a clear structural difference → ANOMALY.
If the candidate looks like it could be from the normal references → NORMAL.

Reply ONLY with JSON:
{{
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": 1, 2, or 3,
  "key_difference": "what specific visual difference (or lack thereof) led to your verdict"
}}

Confidence: 1=ambiguous, 2=moderate evidence, 3=strong clear evidence"""


# ════════════════════════════════════════════════════════
# Query GPT-4o
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
                    {"role": "system", "content": SYSTEM_PROMPT},
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
                temperature=0.1,
                max_tokens=350,
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
                return {"verdict": verdict, "confidence": 1,
                        "key_difference": raw[:120]}

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
    print(f"  Stage1 oracle : F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose  : F1={loose_f1:.4f} (P={loose_p:.2f} R={loose_r:.2f})"
          f" | {len(loose_ivs)} candidates", flush=True)

    top_chs = get_top_channels(test)
    print(f"  Top channels  : {top_chs}", flush=True)

    # Global normalization params (DINOv2 방식)
    ch_min, ch_max = compute_global_norm_params(test, top_chs)

    img_dir = RESULTS_DIR / "plots" / entity
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed_ivs = []
    candidate_log = []

    print(f"  Querying GPT-4o per candidate ({len(loose_ivs)} total)...", flush=True)

    for idx, (cs, ce) in enumerate(loose_ivs):
        cand_score = float(inter_agg[cs:ce+1].mean())
        is_tp      = any(_overlap((cs, ce), g) for g in gt_ivs)
        flag       = "TP" if is_tp else "FP"

        # ★ 로컬 정상 참조 선택 (후보 근처 시간대)
        normal_starts, normal_scores = get_local_normal_starts(
            inter_agg, (cs, ce), loose_ivs, T)

        # DINOv2-style comparison image 생성
        img_b64 = generate_comparison_image_v2(
            test, normal_starts, (cs, ce), top_chs,
            ch_min, ch_max, inter_agg)

        # 이미지 저장 (처음 8개)
        if idx < 8:
            with open(img_dir / f"cand_{idx:02d}_{cs}_{ce}_{flag}.png", "wb") as f:
                f.write(base64.b64decode(img_b64))

        # 프롬프트 구성
        prompt = build_testref_prompt(
            entity, cs, ce, top_chs, len(normal_starts),
            cand_score, normal_scores, ch_min, ch_max)

        # API 호출
        result = query_vlm(img_b64, prompt)

        if result is None:
            verdict, conf, key_diff = "ANOMALY", 1, "API failed"
        else:
            verdict   = result.get("verdict",        "ANOMALY").upper()
            conf      = int(result.get("confidence", 1))
            key_diff  = result.get("key_difference", result.get("reason", ""))[:130]

        # confidence >= 2 AND ANOMALY → 확정
        if verdict == "ANOMALY" and conf >= 2:
            confirmed_ivs.append((cs, ce))

        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} sc={cand_score:.4f} "
              f"-> {verdict}(c={conf}) [{flag}]", flush=True)
        print(f"      {key_diff}", flush=True)

        candidate_log.append({
            "entity": entity, "start": cs, "end": ce,
            "length": ce - cs + 1,
            "score": cand_score, "verdict": verdict, "confidence": conf,
            "key_difference": key_diff, "is_tp": is_tp,
        })

    # Evaluate
    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed_ivs)
    removed = len([iv for iv in loose_ivs if iv not in confirmed_ivs])
    added   = len([vi for vi in confirmed_ivs
                   if not any(_overlap(vi, iv) for iv in loose_ivs)])

    print(f"\n  -- Final --", flush=True)
    print(f"  Stage1 oracle : F1={oracle_f1:.4f} ({len(oracle_ivs)} pred)", flush=True)
    print(f"  Stage1 loose  : F1={loose_f1:.4f} ({len(loose_ivs)} pred)", flush=True)
    print(f"  Stage2 testref: F1={s2_f1:.4f} (P={s2_p:.2f} R={s2_r:.2f})"
          f" | {len(confirmed_ivs)} confirmed", flush=True)
    print(f"  vs oracle: {s2_f1 - oracle_f1:+.4f}  |  vs loose: {s2_f1 - loose_f1:+.4f}",
          flush=True)
    print(f"  Removed: {removed}, Added new: {added}", flush=True)

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
        print("TEST-REF STAGE2 RESULTS (DINOv2-style + Test Normal Prototype)")
        print(f"{'='*75}")
        print(f"{'Entity':<15} {'S1 Oracle':>10} {'S1 Loose':>10} "
              f"{'S2 TestRef':>11} {'dOracle':>9} {'dLoose':>8}  n")
        print("-" * 75)
        for r in results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>10.4f} {r['loose_f1']:>10.4f} "
                  f"{r['stage2_f1']:>11.4f} {r['change_vs_oracle']:>+9.4f} "
                  f"{r['change_vs_loose']:>+8.4f}  "
                  f"{r['stage2_n']}/{r['loose_n']}")
        print("-" * 75)
        o_avg  = np.mean([r["oracle_f1"]  for r in results])
        l_avg  = np.mean([r["loose_f1"]   for r in results])
        s2_avg = np.mean([r["stage2_f1"]  for r in results])
        print(f"{'AVG':<15} {o_avg:>10.4f} {l_avg:>10.4f} "
              f"{s2_avg:>11.4f} {s2_avg - o_avg:>+9.4f} {s2_avg - l_avg:>+8.4f}")

        pd.DataFrame(results).to_csv(
            RESULTS_DIR / "stage2_testref_results.csv", index=False)
        pd.DataFrame(all_logs).to_csv(
            RESULTS_DIR / "candidate_verdicts.csv", index=False)

        print(f"\nSaved: {RESULTS_DIR / 'stage2_testref_results.csv'}")
        print(f"Plots (first 8 per entity): {RESULTS_DIR / 'plots/'}")
