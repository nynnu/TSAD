"""
Stage2 MLLM: Consensus-Channel Before/After Comparison

[설계 원칙]
이전 실패의 두 핵심 원인을 동시에 해결:

  문제 1 - 잘못된 채널 선택:
    기존: 전체 시계열에서 분산 높은 채널 top-K
    개선: 이 후보 구간에서 per-channel INTRA score 높은 채널
         = 이 구간에서 실제로 이상 신호가 있는 채널

  문제 2 - 부적절한 정상 기준:
    기존: train reference (분포 이동) / 전역 low-score window (intra-test drift)
    개선: 후보 직전·직후의 비후보 윈도우 (Before / After)
         = 같은 시간대의 자연스러운 정상 맥락

[Stage1] Inter-overlay 유지 (per-channel oracle 보다 강력)
[Stage2] 후보 구간에서 가장 이상한 채널 K개 + Before/After 비교

[시각화] 3-패널: [Before | Candidate | After]
  - Global normalization (전체 test 기준 y축 고정 = level shift 보임)
  - 합의 채널만 표시 (노이즈 감소, GPT-4o 집중도 향상)
  - Before/After 없으면 local min-score window로 fallback
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

# ─────────────────────────────────────────────
# Environment / paths
# ─────────────────────────────────────────────
def _load_env(env_path=None):
    """
    .env 파일에서 KEY=VALUE 형태로 환경변수 로드.
    env_path 미지정 시 스크립트 디렉토리 및 부모 디렉토리 탐색.
    """
    search_paths = []
    if env_path:
        search_paths.append(Path(env_path))
    else:
        here = Path(__file__).resolve().parent
        search_paths = [here / ".env", here.parent / ".env"]
    for p in search_paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return

_load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY 환경변수가 없습니다. "
        ".env 파일(OPENAI_API_KEY=sk-...) 또는 환경변수를 설정하세요."
    )

CACHE_BASE   = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")
SMD_DIR      = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
RESULTS_DIR  = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_stage2_perchannel")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-5"]

# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
WIN             = 224    # DINOv2 window size (stride = 56)
STRIDE          = 56
LOOSE_ALPHA     = 0.3    # Stage1 inter-overlay loose threshold
TOP_K_DISPLAY   = 4      # 시각화에 표시할 채널 수
BEFORE_K_RANGE  = 6      # before 탐색 범위 = 6 × WIN steps
AFTER_K_RANGE   = 6      # after  탐색 범위
CONF_THRESHOLD  = 2      # confidence >= 이 값 + ANOMALY → 확정
VLM_SLEEP       = 4.0    # API 호출 간격(s)
SCORE_KEY_ORDER = ["ml_topk10", "final_topk10", "ml_sum", "final_sum"]


# ══════════════════════════════════════════════
# Data / Score Loading
# ══════════════════════════════════════════════

def load_smd(entity: str):
    """SMD 데이터셋 로드: train, test, test_label."""
    test   = np.loadtxt(SMD_DIR / "test"       / f"{entity}.txt", delimiter=",")
    train  = np.loadtxt(SMD_DIR / "train"      / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt",
                        delimiter=",").astype(np.int32)
    return train, test, labels


def _best_score_array(scores_dict: dict, T: int):
    """score key fallback 순서로 T와 길이가 맞는 배열 반환. 없으면 None."""
    for k in SCORE_KEY_ORDER:
        if k in scores_dict and scores_dict[k].shape[0] == T:
            return scores_dict[k].copy()
    return None


def load_scores(entity: str):
    """
    반환값:
      ch_scores : dict[int, np.ndarray]  채널 인덱스 → INTRA score 배열 (length T)
      ov_scores : list[np.ndarray]        각 overlay group의 INTER score 배열
    """
    ent_dir = CACHE_BASE / "SMD" / entity
    ch_scores, ov_scores = {}, []

    for f in sorted(ent_dir.glob("ch*_scores.npz")):
        ch_idx = int(f.stem.replace("ch", "").replace("_scores", ""))
        data = np.load(f)
        score_dict = {k: data[k] for k in data.files}
        ch_scores[ch_idx] = score_dict   # 원시 dict 그대로 보존

    for f in sorted(ent_dir.glob("overlay_g*_scores.npz")):
        data = np.load(f)
        ov_scores.append({k: data[k] for k in data.files})

    return ch_scores, ov_scores


# ══════════════════════════════════════════════
# Interval / F1 utilities
# ══════════════════════════════════════════════

def get_intervals(binary: np.ndarray):
    """1D binary array → [(start, end), ...] interval list."""
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


def _overlap(a, b) -> bool:
    """두 (start, end) interval이 겹치는지 확인."""
    return not (a[1] < b[0] or b[1] < a[0])


def interval_f1(gt_ivs, pred_ivs):
    """Interval-level F1 (overlap-based TP/FP/FN)."""
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


# ══════════════════════════════════════════════
# Stage1: Inter-overlay loose threshold
# ══════════════════════════════════════════════

def get_inter_agg(ov_scores: list, T: int) -> np.ndarray:
    """모든 overlay group 점수의 평균 → 단일 inter_agg 배열."""
    arrays = []
    for sc in ov_scores:
        arr = _best_score_array(sc, T)
        if arr is not None:
            arrays.append(arr)
    return np.mean(arrays, axis=0) if arrays else np.zeros(T)


def get_stage1_results(ov_scores: list, T: int, labels: np.ndarray):
    """
    반환값:
      inter_agg   : np.ndarray (T,)  - inter-overlay 점수
      loose_ivs   : list[(s,e)]      - LOOSE_ALPHA 기준 후보 구간
      gt_ivs      : list[(s,e)]      - ground truth 구간
      oracle_f1   : float            - 최적 alpha에서의 F1
      oracle_ivs  : list[(s,e)]      - oracle 예측 구간
    """
    inter_agg = get_inter_agg(ov_scores, T)
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


# ══════════════════════════════════════════════
# Consensus channel selection
# ══════════════════════════════════════════════

def get_consensus_channels(ch_scores: dict, candidate_iv, T: int,
                            n: int = TOP_K_DISPLAY, test: np.ndarray = None):
    """
    후보 구간 [s, e]에서 per-channel INTRA score가 가장 높은 채널 n개 선택.

    score가 없는 채널의 경우 test 시계열 분산으로 fallback.
    항상 정확히 n개 반환 (부족하면 분산 높은 채널로 보충).
    """
    cs, ce = candidate_iv
    ch_window_scores = {}

    for ch_idx, score_dict in ch_scores.items():
        arr = _best_score_array(score_dict, T)
        if arr is not None:
            ch_window_scores[ch_idx] = float(arr[cs:ce + 1].mean())

    # 점수 내림차순 정렬
    sorted_by_score = sorted(ch_window_scores.items(),
                             key=lambda x: x[1], reverse=True)
    selected = [ch for ch, _ in sorted_by_score[:n]]

    # 부족하면 test 분산 높은 채널로 보충
    if len(selected) < n and test is not None:
        n_ch = test.shape[1]
        all_chs = list(range(n_ch))
        var_sorted = sorted(all_chs, key=lambda c: test[:, c].var(), reverse=True)
        for ch in var_sorted:
            if ch not in selected:
                selected.append(ch)
            if len(selected) >= n:
                break

    return selected[:n], {ch: ch_window_scores.get(ch, 0.0) for ch in selected[:n]}


# ══════════════════════════════════════════════
# Global normalization (DINOv2 방식)
# ══════════════════════════════════════════════

def compute_global_norm(test: np.ndarray, channels: list):
    """
    전체 test 시계열 기준 채널별 global min/max.
    DINOv2와 동일: y축을 전체 기간 기준으로 고정 → level shift가 절대 수치로 보임.
    """
    ch_min = {ch: float(test[:, ch].min()) for ch in channels}
    ch_max = {ch: float(test[:, ch].max()) for ch in channels}
    return ch_min, ch_max


def _normalize(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Global min/max로 0-1 정규화. 범위가 너무 작으면 0.5 상수로."""
    if hi - lo < 1e-9:
        return np.full_like(vals, 0.5, dtype=float)
    return (vals.astype(float) - lo) / (hi - lo)


# ══════════════════════════════════════════════
# Before / After window selection
# ══════════════════════════════════════════════

def _is_clear_window(start: int, end: int, exclude_ivs: list) -> bool:
    """[start, end]가 모든 exclude_ivs와 겹치지 않으면 True."""
    return all(not _overlap((start, end), iv) for iv in exclude_ivs)


def find_before_window(candidate_iv, loose_ivs: list, T: int,
                       inter_agg: np.ndarray = None):
    """
    후보 직전에서 시작해 BEFORE_K_RANGE * WIN 범위 안에서
    loose_ivs와 겹치지 않는 첫 번째 유효 윈도우를 반환.

    - 탐색 방향: candidate start에서 뒤로 (stride = WIN // 2)
    - 여러 후보가 있으면 inter_agg score 가장 낮은 것 선택
    - 없으면 None 반환
    """
    cs, _ = candidate_iv
    other_ivs = [iv for iv in loose_ivs if iv != candidate_iv]

    candidates = []
    step = WIN // 2
    for offset in range(step, BEFORE_K_RANGE * WIN + step, step):
        start = cs - offset - WIN + 1
        end   = start + WIN - 1
        if start < 0:
            break
        if _is_clear_window(start, end, other_ivs):
            sc = float(inter_agg[start:end + 1].mean()) if inter_agg is not None else 0.0
            candidates.append((sc, start))

    if not candidates:
        return None
    # inter_agg 점수가 가장 낮은 것(= 가장 정상적인 것) 선택
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_after_window(candidate_iv, loose_ivs: list, T: int,
                      inter_agg: np.ndarray = None):
    """
    후보 직후에서 시작해 AFTER_K_RANGE * WIN 범위 안에서 유효 윈도우 반환.
    없으면 None.
    """
    _, ce = candidate_iv
    other_ivs = [iv for iv in loose_ivs if iv != candidate_iv]

    candidates = []
    step = WIN // 2
    for offset in range(step, AFTER_K_RANGE * WIN + step, step):
        start = ce + offset
        end   = start + WIN - 1
        if end >= T:
            break
        if _is_clear_window(start, end, other_ivs):
            sc = float(inter_agg[start:end + 1].mean()) if inter_agg is not None else 0.0
            candidates.append((sc, start))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_local_fallback(candidate_iv, loose_ivs: list, inter_agg: np.ndarray,
                        T: int, n: int = 2):
    """
    Before/After 모두 실패 시: candidate 근처에서 점수 낮은 n개 윈도우 탐색.
    """
    cs, ce = candidate_iv
    other_ivs = [iv for iv in loose_ivs if iv != candidate_iv]
    radius = 5000

    window_scores = []
    t_start = max(0, cs - radius)
    t_end   = min(T - WIN, ce + radius)
    for start in range(t_start, t_end, STRIDE):
        end = start + WIN - 1
        if _is_clear_window(start, end, other_ivs) and not _overlap((start, end), (cs, ce)):
            sc = float(inter_agg[start:end + 1].mean())
            window_scores.append((sc, start))

    window_scores.sort(key=lambda x: x[0])
    selected = []
    for _, start in window_scores:
        if all(abs(start - s) >= WIN for s in selected):
            selected.append(start)
        if len(selected) >= n:
            break
    return selected


# ══════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════

PANEL_COLORS = {
    "before":    ("#2196F3", "#e3f2fd", "blue"),    # 파란색 = before
    "candidate": ("#f44336", "#fff3e0", "red"),     # 빨간색 = candidate
    "after":     ("#4CAF50", "#e8f5e9", "green"),   # 초록색 = after
    "fallback":  ("#9C27B0", "#f3e5f5", "purple"),  # 보라색 = fallback
}
LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b"]


def _draw_panel(ax, data: np.ndarray, channels: list, ch_min: dict, ch_max: dict,
                start: int, label: str, panel_type: str, score: float):
    """단일 패널 렌더링 (global normalization)."""
    _, face_color, edge_color = PANEL_COLORS[panel_type]
    seg_len = min(len(data) - start, WIN)
    x       = np.arange(seg_len)

    for i, ch in enumerate(channels):
        raw  = data[start:start + seg_len, ch]
        norm = _normalize(raw, ch_min[ch], ch_max[ch])
        ax.plot(x, norm, color=LINE_COLORS[i % len(LINE_COLORS)],
                linewidth=1.0, alpha=0.9, label=f"Ch{ch}")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, seg_len - 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=6)
    ax.set_title(f"{label}\nt=[{start},{start+seg_len-1}]\nscore={score:.4f}",
                 fontsize=7, color=edge_color, fontweight="bold")
    ax.set_facecolor(face_color)
    for sp in ax.spines.values():
        sp.set_edgecolor(edge_color)
        sp.set_linewidth(1.8)
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.5, ncol=2)


def generate_comparison_image(
        test: np.ndarray,
        candidate_iv,
        before_start,           # int or None
        after_start,            # int or None
        fallback_starts: list,  # list[int], used if both before/after are None
        channels: list,
        ch_min: dict,
        ch_max: dict,
        inter_agg: np.ndarray,
) -> str:
    """
    비교 이미지 생성.
    패널 구성 (가능한 것만):
      before_start  → 왼쪽 BEFORE 패널 (파란색)
      candidate_iv  → 중앙 CANDIDATE 패널 (빨간색)
      after_start   → 오른쪽 AFTER 패널 (초록색)
      fallback_starts → BEFORE/AFTER 둘 다 없을 때 fallback 패널들 (보라색)
    반환: base64-encoded PNG string
    """
    cs, ce = candidate_iv

    # 패널 구성
    panels = []  # list of (data_start, label, panel_type, is_candidate)
    if before_start is not None:
        panels.append((before_start, "BEFORE (normal)", "before", False))
    panels.append((cs, f"CANDIDATE [{cs},{ce}]", "candidate", True))
    if after_start is not None:
        panels.append((after_start, "AFTER (normal)", "after", False))
    elif before_start is None:
        # 둘 다 없으면 fallback
        for i, fs in enumerate(fallback_starts):
            panels.append((fs, f"LOCAL REF {i+1}", "fallback", False))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.8 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (start, label, ptype, is_cand) in zip(axes, panels):
        if is_cand:
            score = float(inter_agg[cs:ce + 1].mean())
            seg_data = test
        else:
            score = float(inter_agg[start:start + WIN].mean())
            seg_data = test
        _draw_panel(ax, seg_data, channels, ch_min, ch_max,
                    start, label, ptype, score)

    ref_info = ""
    if before_start is not None and after_start is not None:
        ref_info = "Before & After available"
    elif before_start is not None:
        ref_info = "Before only (no After)"
    elif after_start is not None:
        ref_info = "After only (no Before)"
    else:
        ref_info = "Fallback local ref"

    plt.suptitle(
        f"Channels: {channels} | Global norm [0,1] | {ref_info}",
        fontsize=8, y=1.03
    )
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ══════════════════════════════════════════════
# Prompt Construction
# ══════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert anomaly detector specializing in multivariate server monitoring "
    "time series. You analyze inter-channel correlation patterns to determine system health."
)


def build_prompt(entity: str, candidate_iv, channels: list,
                 ch_scores_in_window: dict,
                 before_start, after_start, fallback_starts: list,
                 inter_cand_score: float, inter_normal_scores: list,
                 n_ch_total: int) -> str:
    """
    GPT-4o 프롬프트 구성.

    핵심 구조:
    1. 이 채널들이 왜 선택됐는지 설명 (per-channel 점수)
    2. Global normalization 의미 설명
    3. Before/After reference 설명
    4. 비교 태스크 명시
    """
    cs, ce = candidate_iv

    # 채널 점수 설명
    ch_score_lines = "\n".join(
        f"  Ch{ch}: window anomaly score = {ch_scores_in_window.get(ch, 0.0):.4f}"
        for ch in channels
    )

    # reference 타입 설명
    if before_start is not None and after_start is not None:
        ref_desc = (
            f"LEFT panel  = BEFORE  (t=[{before_start},{before_start+WIN-1}]) — "
            f"score={inter_normal_scores[0]:.4f}\n"
            f"RIGHT panel = AFTER   (t=[{after_start},{after_start+WIN-1}]) — "
            f"score={inter_normal_scores[-1]:.4f}\n"
            "These are the nearest non-anomalous windows immediately before and after the candidate.\n"
            "They represent WHAT THIS MACHINE NORMALLY LOOKS LIKE at this time period."
        )
    elif before_start is not None:
        ref_desc = (
            f"LEFT panel  = BEFORE  (t=[{before_start},{before_start+WIN-1}]) — "
            f"score={inter_normal_scores[0]:.4f}\n"
            "This is the nearest non-anomalous window immediately before the candidate."
        )
    elif after_start is not None:
        ref_desc = (
            f"RIGHT panel = AFTER   (t=[{after_start},{after_start+WIN-1}]) — "
            f"score={inter_normal_scores[0]:.4f}\n"
            "This is the nearest non-anomalous window immediately after the candidate."
        )
    else:
        ref_desc = (
            "REFERENCE panels = local non-anomalous windows near the candidate.\n"
            "These represent normal operation in this time region."
        )

    ratio = inter_cand_score / np.mean(inter_normal_scores) if inter_normal_scores else 1.0

    return f"""Entity: {entity} | Candidate interval [{cs}, {ce}] (length {ce-cs+1} steps)

━━━ WHY THESE CHANNELS ━━━
Stage1 (DINOv2 inter-overlay) flagged this interval. Then we identified which individual
channels had the highest anomaly scores WITHIN this specific window:
{ch_score_lines}
These {len(channels)} channels (out of {n_ch_total} total) showed the strongest anomalous
signal during this candidate interval.

━━━ VISUALIZATION ━━━
All panels use IDENTICAL global normalization:
  y=0 → the channel's minimum value over the ENTIRE test series
  y=1 → the channel's maximum value over the ENTIRE test series
This means: if a channel's absolute value changes, its y-position changes.
A channel at y=0.8 in panel A and y=0.8 in panel B IS at the same absolute value.

━━━ REFERENCE PANELS ━━━
{ref_desc}
MIDDLE panel = CANDIDATE interval (score={inter_cand_score:.4f} = {ratio:.1f}x normal score)

━━━ YOUR TASK ━━━
Compare the CANDIDATE panel to the normal reference panel(s):

1. Do the channels maintain their relative positions?
   (e.g., if Ch5 is normally above Ch13, is it still above in the candidate?)

2. Do the channels show the same tracking behavior?
   (e.g., if Ch5 and Ch0 normally move together, do they still co-vary?)

3. Does any channel shift to a position it NEVER occupies in the normal panels?
   (absolute shift visible because of global normalization)

4. Is the candidate pattern consistent with natural temporal variation,
   or is there a structural change that BOTH before and after lack?

KEY PRINCIPLE:
  If candidate looks like it COULD COME FROM the normal reference → NORMAL
  If candidate shows structural differences ABSENT in both before and after → ANOMALY

Reply ONLY with JSON (no markdown fences):
{{
  "verdict": "ANOMALY" or "NORMAL",
  "confidence": 1, 2, or 3,
  "observed_difference": "the specific visual change you see (or absence of change)",
  "key_channels": "which channels show the difference (e.g., Ch5, Ch0)"
}}
Confidence: 1=ambiguous, 2=moderate evidence, 3=strong clear evidence"""


# ══════════════════════════════════════════════
# GPT-4o Query
# ══════════════════════════════════════════════

def query_vlm(img_b64: str, prompt: str, max_attempts: int = 5):
    """
    GPT-4o API 호출. JSON 파싱 실패 시 regex fallback.
    quota 소진 시 None 반환.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(max_attempts):
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
                                "detail": "high",
                            }},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

            # Try direct JSON parse
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            # Try extracting first JSON block
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

            # Last resort: keyword extraction
            verdict = "ANOMALY" if "ANOMALY" in raw.upper() else "NORMAL"
            return {"verdict": verdict, "confidence": 1,
                    "observed_difference": raw[:200],
                    "key_channels": "parse error"}

        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = (attempt + 1) * 30
                print(f"      [rate limit] waiting {wait}s ...", flush=True)
                time.sleep(wait)
            elif "insufficient_quota" in err or "quota" in err:
                print("      [quota exhausted]", flush=True)
                return None
            else:
                print(f"      [api error attempt {attempt+1}] {exc}", flush=True)
                time.sleep(5)

    return None


# ══════════════════════════════════════════════
# Per-entity runner
# ══════════════════════════════════════════════

def run_entity(entity: str) -> dict | None:
    print(f"\n{'='*65}")
    print(f"  {entity}")
    print(f"{'='*65}", flush=True)

    train, test, labels = load_smd(entity)
    T     = len(labels)
    n_ch  = test.shape[1]

    ch_scores, ov_scores = load_scores(entity)
    available_ch_indices = sorted(ch_scores.keys())
    print(f"  T={T}  channels_total={n_ch}  "
          f"ch_scores_cached={len(available_ch_indices)} {available_ch_indices}",
          flush=True)

    # ── Stage1 (inter-overlay) ──
    inter_agg, loose_ivs, gt_ivs, oracle_f1, oracle_ivs = get_stage1_results(
        ov_scores, T, labels)

    loose_f1, loose_p, loose_r = interval_f1(gt_ivs, loose_ivs)
    print(f"  GT={len(gt_ivs)}  oracle={oracle_f1:.4f} ({len(oracle_ivs)} pred)  "
          f"loose={loose_f1:.4f} P={loose_p:.2f} R={loose_r:.2f} ({len(loose_ivs)} cand)",
          flush=True)

    # ── 저장 디렉토리 ──
    img_dir = RESULTS_DIR / "plots" / entity
    img_dir.mkdir(parents=True, exist_ok=True)

    confirmed_ivs = []
    candidate_log = []

    print(f"  Processing {len(loose_ivs)} candidates ...", flush=True)

    for idx, (cs, ce) in enumerate(loose_ivs):
        is_tp  = any(_overlap((cs, ce), g) for g in gt_ivs)
        flag   = "TP" if is_tp else "FP"
        inter_cand_score = float(inter_agg[cs:ce + 1].mean())

        # ── 1) 합의 채널 선택 ──
        consensus_chs, ch_score_in_window = get_consensus_channels(
            ch_scores, (cs, ce), T, n=TOP_K_DISPLAY, test=test)

        # ── 2) Global normalization params ──
        ch_min, ch_max = compute_global_norm(test, consensus_chs)

        # ── 3) Before / After window 탐색 ──
        before_start = find_before_window((cs, ce), loose_ivs, T, inter_agg)
        after_start  = find_after_window((cs, ce),  loose_ivs, T, inter_agg)

        # fallback: before/after 둘 다 없을 때
        fallback_starts = []
        if before_start is None and after_start is None:
            fallback_starts = find_local_fallback((cs, ce), loose_ivs, inter_agg, T, n=2)

        # ── 4) Normal scores for prompt ──
        inter_normal_scores = []
        if before_start is not None:
            inter_normal_scores.append(float(inter_agg[before_start:before_start+WIN].mean()))
        if after_start is not None:
            inter_normal_scores.append(float(inter_agg[after_start:after_start+WIN].mean()))
        for fs in fallback_starts:
            inter_normal_scores.append(float(inter_agg[fs:fs+WIN].mean()))
        if not inter_normal_scores:
            inter_normal_scores = [float(inter_agg.mean())]

        # reference 상황 표시
        ref_status = (
            "BA" if (before_start is not None and after_start is not None) else
            "B_" if before_start is not None else
            "_A" if after_start is not None else
            f"FB({len(fallback_starts)})"
        )

        # ── 5) 비교 이미지 생성 ──
        img_b64 = generate_comparison_image(
            test, (cs, ce),
            before_start, after_start, fallback_starts,
            consensus_chs, ch_min, ch_max, inter_agg)

        # 처음 10개 이미지 저장
        if idx < 10:
            with open(img_dir / f"cand_{idx:02d}_{cs}_{ce}_{flag}_{ref_status}.png",
                      "wb") as f:
                f.write(base64.b64decode(img_b64))

        # ── 6) 프롬프트 구성 ──
        prompt = build_prompt(
            entity, (cs, ce), consensus_chs, ch_score_in_window,
            before_start, after_start, fallback_starts,
            inter_cand_score, inter_normal_scores, n_ch)

        # ── 7) GPT-4o 호출 ──
        result = query_vlm(img_b64, prompt)

        if result is None:
            # API quota 소진: 현재까지 결과만 사용
            print("      [API quota exhausted — stopping early]", flush=True)
            confirmed_ivs.append((cs, ce))   # keep (conservative: assume anomaly)
            candidate_log.append({
                "entity": entity, "start": cs, "end": ce,
                "length": ce - cs + 1, "cand_score": inter_cand_score,
                "verdict": "ANOMALY", "confidence": -1,
                "observed_difference": "quota_exhausted",
                "key_channels": str(consensus_chs),
                "is_tp": is_tp, "ref_status": ref_status,
                "consensus_chs": str(consensus_chs),
            })
            break

        verdict = result.get("verdict", "ANOMALY").upper()
        conf    = int(result.get("confidence", 1))
        obs     = str(result.get("observed_difference", result.get("reason", "")))[:150]
        key_ch  = str(result.get("key_channels", ""))[:60]

        if verdict == "ANOMALY" and conf >= CONF_THRESHOLD:
            confirmed_ivs.append((cs, ce))

        print(f"    [{cs:6d},{ce:6d}] len={ce-cs+1:4d} "
              f"sc={inter_cand_score:.4f} ch={consensus_chs} "
              f"ref={ref_status} -> {verdict}(c={conf}) [{flag}]",
              flush=True)
        print(f"      {obs}", flush=True)

        candidate_log.append({
            "entity":               entity,
            "start":                cs,
            "end":                  ce,
            "length":               ce - cs + 1,
            "cand_score":           inter_cand_score,
            "verdict":              verdict,
            "confidence":           conf,
            "observed_difference":  obs,
            "key_channels":         key_ch,
            "is_tp":                is_tp,
            "ref_status":           ref_status,
            "consensus_chs":        str(consensus_chs),
        })

    # ── 평가 ──
    s2_f1, s2_p, s2_r = interval_f1(gt_ivs, confirmed_ivs)
    n_removed = len([iv for iv in loose_ivs if iv not in confirmed_ivs])
    n_added   = len([iv for iv in confirmed_ivs
                     if not any(_overlap(iv, lv) for lv in loose_ivs)])

    print(f"\n  ── Results ({entity}) ──")
    print(f"  Stage1 oracle : F1={oracle_f1:.4f}  ({len(oracle_ivs)} pred)")
    print(f"  Stage1 loose  : F1={loose_f1:.4f}  ({len(loose_ivs)} cand)")
    print(f"  Stage2 perchan: F1={s2_f1:.4f}  P={s2_p:.2f} R={s2_r:.2f}  "
          f"({len(confirmed_ivs)} confirmed)")
    print(f"  vs oracle: {s2_f1 - oracle_f1:+.4f}  "
          f"vs loose: {s2_f1 - loose_f1:+.4f}  "
          f"removed={n_removed}  added_new={n_added}", flush=True)

    return {
        "entity":              entity,
        "n_gt":                len(gt_ivs),
        "oracle_f1":           oracle_f1,
        "oracle_n":            len(oracle_ivs),
        "loose_f1":            loose_f1,
        "loose_p":             loose_p,
        "loose_r":             loose_r,
        "loose_n":             len(loose_ivs),
        "stage2_f1":           s2_f1,
        "stage2_p":            s2_p,
        "stage2_r":            s2_r,
        "stage2_n":            len(confirmed_ivs),
        "n_removed":           n_removed,
        "n_added":             n_added,
        "change_vs_oracle":    s2_f1 - oracle_f1,
        "change_vs_loose":     s2_f1 - loose_f1,
        "candidate_log":       candidate_log,
    }


# ══════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results, all_logs = [], []

    for entity in SMD_ENTITIES:
        try:
            r = run_entity(entity)
        except Exception as exc:
            print(f"\n  [ERROR] {entity}: {exc}", flush=True)
            import traceback; traceback.print_exc()
            r = None

        if r is not None:
            all_logs.extend(r.pop("candidate_log"))
            all_results.append(r)

    # ── Summary ──
    if all_results:
        print(f"\n{'='*75}")
        print("FINAL: Per-Channel Consensus Before/After Stage2")
        print(f"{'='*75}")
        print(f"{'Entity':<15} {'Oracle':>8} {'Loose':>8} {'Stage2':>8} "
              f"{'dOracle':>8} {'dLoose':>7}  n_conf/n_cand")
        print("-" * 75)
        for r in all_results:
            print(f"{r['entity']:<15} {r['oracle_f1']:>8.4f} {r['loose_f1']:>8.4f} "
                  f"{r['stage2_f1']:>8.4f} {r['change_vs_oracle']:>+8.4f} "
                  f"{r['change_vs_loose']:>+7.4f}  "
                  f"{r['stage2_n']}/{r['loose_n']}")
        print("-" * 75)
        o_avg  = np.mean([r["oracle_f1"]  for r in all_results])
        l_avg  = np.mean([r["loose_f1"]   for r in all_results])
        s2_avg = np.mean([r["stage2_f1"]  for r in all_results])
        print(f"{'AVG':<15} {o_avg:>8.4f} {l_avg:>8.4f} {s2_avg:>8.4f} "
              f"{s2_avg - o_avg:>+8.4f} {s2_avg - l_avg:>+7.4f}")

        # ── CSV 저장 ──
        pd.DataFrame(all_results).to_csv(
            RESULTS_DIR / "summary.csv", index=False)
        pd.DataFrame(all_logs).to_csv(
            RESULTS_DIR / "candidate_verdicts.csv", index=False)

        print(f"\nSaved: {RESULTS_DIR / 'summary.csv'}")
        print(f"Plots (first 10 per entity): {RESULTS_DIR / 'plots/'}")
