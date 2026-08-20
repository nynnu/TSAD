"""
Stage2 K5 (탐지 트랙): K3/K4의 heatmap+overlay+index-aware 프롬프팅 구조를
"어느 채널이냐"(진단)가 아니라 "이 구간이 진짜 이상이냐"(탐지)에 적용.

배경
----
K2/K3/K4는 전부 진단(diagnosis) 트랙이었다 -- 이미 GT가 알려진 윈도우 안에서
채널 집합을 맞히는 과제라, VLM4TS 논문이 실제로 하는 일(탐지: 이상 시간구간을
찾는 것, Max-F1 지표)과 다른 우리 자체 확장이었다(report20 참고).

Stage2에도 탐지 트랙이 필요하다는 논의 끝에, 새로 v16(ViT-B, 별도 캐시, 후보별
개별 판정) 스타일을 따로 만드는 대신, **이미 검증된 K3/K4의 heatmap+overlay+
index-aware 포맷을 그대로 재사용**하고 출력만 채널목록 대신 "이상 여부 + 경계
재조정"으로 바꿨다.

구조
----
1. Stage1(K4 detect, experiment_stage1_k4_adaptive.py)이 만든 후보 구간(전체
   시계열 슬라이딩 -> threshold sweep으로 얻은 candidate intervals)을 그대로
   받는다.
2. 각 후보 구간마다: 224틱 중심윈도우로 자르고, K4의 채널선택(fixed 또는
   hysteresis, experiment_stage2_k4_adaptive에서 그대로 import)으로 "관련
   채널"을 뽑는다.
3. 이미지(heatmap 38채널 + overlay 선택채널) + index-aware 텍스트를 K3/K4와
   동일하게 만들되, 프롬프트만 바꿔서 GPT-4o에게 (a) 이 구간이 진짜 이상인지
   (b) 상대적 시작/끝(경계 재조정)을 물어본다.
4. ANOMALY로 판정된 구간들(재조정된 경계 반영)을 최종 탐지 결과로 모아서,
   interval-overlap F1 + point-wise Max-F1(둘 다, report20의 dual-metric 규칙)
   로 GT와 비교한다.

사용법
------
  python experiment_stage2_k5_detect.py --stage1 --entity machine-1-1          # 후보/콜 수만 확인, VLM 없음
  python experiment_stage2_k5_detect.py --run --entity machine-1-1 --method hysteresis
"""
import argparse
import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, N_CHANNELS
from smd_3way_baseline_comparison import call_vlm
# v16(구 Stage2 파이프라인)의 검증된 정규화/정상참조 로직 재사용 -- K5의 heatmap+overlay/
# subplot_grid 전부 윈도우 자기자신의 min/max로 정규화했는데, 이러면 작은 정상 변동도
# [0,1] 전체를 채우게 늘어나서 진짜 큰 이상처럼 보인다(sanity-check 100% 오탐의 유력한
# 원인). v16은 train 전체의 min/max를 고정 기준으로 쓰고, 정상 참조 윈도우를 나란히
# 보여준다 -- 새로 구현하지 않고 그대로 import.
import experiment_stage2_v16 as v16

# K4 진단에서 채널선택/렌더링 로직 재사용 (새로 안 만듦) -- render_heatmap_overlay는
# K4와 완전히 동일해서 그대로 import (아래서 재정의하지 않음)
from experiment_stage2_k4_adaptive import (
    constant_channels, compute_zscores, select_channels_fixed, select_channels_hysteresis,
    get_or_build_channel_calib, render_heatmap_overlay, N_POINTS_PER_CHANNEL,
)

OUT_DIR = BASE / "experiments" / "results_stage2_k5_detect"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SMD_DIR = BASE / "mv_data" / "SMD"

STRIDE, WIN = 56, 224
cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════
# 1. Stage1 후보 구간 생성 (K4 detect 스코어러 재사용)
# ══════════════════════════════════════════════════════════════════

def get_intervals(binary):
    ivs, seg, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not seg:
            s, seg = i, True
        elif not v and seg:
            ivs.append((s, i - 1))
            seg = False
    if seg:
        ivs.append((s, len(binary) - 1))
    return ivs


def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def eval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0.0
    TP = sum(1 for d in pred_ivs if any(_ov(d, a) for a in gt_ivs))
    FP = sum(1 for d in pred_ivs if not any(_ov(d, a) for a in gt_ivs))
    FN = sum(1 for a in gt_ivs if not any(_ov(a, d) for d in pred_ivs))
    p = TP / (TP + FP) if (TP + FP) else 0
    r = TP / (TP + FN) if (TP + FN) else 0
    return 2 * p * r / (p + r) if (p + r) else 0


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return 2 * p * r / (p + r) if (p + r) else 0


def win_to_ts(win_scores, n_ts):
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i, s in enumerate(win_scores):
        st = i * STRIDE
        en = min(st + WIN, n_ts)
        scores[st:en] += s
        counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


def stage1_candidates(entity, train, test, degenerate_ch, loose_pct=90.0):
    """K4 detect 점수(윈도우당 alpha=0.1 넘는 채널 개수)를 슬라이딩해서 느슨한
    후보 구간을 만든다 (Stage1 K3/v16과 같은 관용: 널널하게 뽑고 Stage2가 거름)."""
    T_test = len(test)
    starts = list(range(0, T_test - WIN + 1, STRIDE))
    entity_channel_calib = {}
    z_thr = norm.ppf(1 - 0.1)

    # 윈도우별 점수를 .npy로 체크포인트 -- 중간에 끊겨도(타임아웃/인터럽트) 재실행 시
    # 이미 계산된 윈도우는 건너뛴다 (안 채워진 자리는 NaN으로 표시).
    cache_path = OUT_DIR / f"{entity}_candidates_winscore.npy"
    if cache_path.exists():
        win_score = np.load(cache_path)
        n_done = int(np.sum(~np.isnan(win_score)))
        print(f"    [candidates] 체크포인트에서 재개: {n_done}/{len(starts)} 완료", flush=True)
    else:
        win_score = np.full(len(starts), np.nan)

    t0 = time.time()
    for wi, s in enumerate(starts):
        if not np.isnan(win_score[wi]):
            continue
        window = test[s:s + WIN]
        n_sel = 0
        for c in range(N_CHANNELS):
            if c in degenerate_ch:
                continue
            if c not in entity_channel_calib:
                entity_channel_calib[c] = get_or_build_channel_calib(entity, c, train)
            tr_cls, tr_patches, stats = entity_channel_calib[c]
            test_img = cm.ts_to_image_fast(window[:, c])
            te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
            sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
            z = (float(sc["sum"][0]) - stats["mu"]) / stats["sigma"]
            if z > z_thr:
                n_sel += 1
        win_score[wi] = n_sel
        if wi % 20 == 0:
            np.save(cache_path, win_score)
            print(f"    [candidates] window {wi}/{len(starts)} ({time.time()-t0:.0f}s elapsed)", flush=True)
    np.save(cache_path, win_score)

    inter = win_to_ts(win_score, T_test)
    thr = np.percentile(inter, loose_pct)
    loose_ivs = get_intervals((inter > thr).astype(int))
    return loose_ivs, inter, entity_channel_calib


# ══════════════════════════════════════════════════════════════════
# 2. 렌더링 + 프롬프트
#    - render_heatmap_overlay: K4에서 import(기존 방식)
#    - render_subplot_grid: VLM4TS 논문의 실제 다변량 시각화(38채널을 각각
#      독립된 작은 칸에 따로 그림, attachments-19/presentation_A_vlm4ts_
#      subplot_grid.png 참고) -- overlay(겹쳐그리기)가 sanity-check에서
#      100% 오탐(정상 컨트롤 5/5 ANOMALY)을 낸 것과 비교하기 위한 대조군.
# ══════════════════════════════════════════════════════════════════

def render_subplot_grid(window, n_cols=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = -(-N_CHANNELS // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.1), dpi=100)
    axes = axes.flatten()
    for c in range(N_CHANNELS):
        ax = axes[c]
        ax.plot(window[:, c], color="black", linewidth=0.6)
        ax.set_title(f"ch{c}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for c in range(N_CHANNELS, len(axes)):
        axes[c].axis("off")
    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_heatmap_subplot_grid(window, ranked, selected, n_cols=6):
    """상단 = 기존과 동일한 38채널 heatmap(전체 개요), 하단 = overlay(겹쳐그리기) 대신
    선택된 채널만 각각 독립된 작은 칸에 따로 그림(논문의 subplot grid 방식을 detail
    패널에 적용) -- heatmap의 전체 개요 기능은 유지하면서, overlay가 오탐(FPR=100%)을
    낸 "겹쳐그리기" 부분만 subplot grid로 교체해서 원인을 좁히기 위한 버전."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_sel = len(selected)
    n_rows = -(-n_sel // n_cols)
    fig = plt.figure(figsize=(6, 3.5 + n_rows * 1.1), dpi=100)
    gs = GridSpec(2, 1, height_ratios=[1.4, n_rows * 0.9], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    heat = np.zeros((len(ranked), window.shape[0]))
    for i, c in enumerate(ranked):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
    ax1.imshow(heat, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(ranked)))
    ax1.set_yticklabels([str(c) for c in ranked], fontsize=5)
    ax1.set_xticks([])
    ax1.set_title("Heatmap: 38 channels, sorted by adaptive z-score", fontsize=7)

    gs_bottom = gs[1].subgridspec(n_rows, n_cols, hspace=0.6, wspace=0.3)
    for i, c in enumerate(selected):
        ax = fig.add_subplot(gs_bottom[i // n_cols, i % n_cols])
        ax.plot(window[:, c], color="black", linewidth=0.6)
        ax.set_title(f"ch{c}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Bottom: {n_sel} candidate channels, each in its own independent panel (no overlay)",
                 fontsize=7, y=0.5 - n_rows * 0.02)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_train_referenced(window, train, selected, cmin, cmax, train_cal_starts,
                             ranked=None, cmin_all=None, cmax_all=None):
    """v16 스타일: train min/max 고정 정규화 + train에서 뽑은 확인된 정상 윈도우를
    후보와 나란히 보여줌 (v16.gn_train/_n/get_train_cal_windows를 그대로 사용해서
    계산한 cmin/cmax/train_cal_starts를 받는다 -- 새로 계산 로직 만들지 않음).

    ranked/cmin_all/cmax_all이 주어지면 맨 위에 38채널 heatmap도 추가(K2~K5가 계속
    써온 개요 패널) -- 단, window 자기자신 min/max가 아니라 여기서도 train min/max로
    정규화해서 같은 자기참조 정규화 버그가 heatmap에도 생기지 않게 함."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    n_cols = max(len(train_cal_starts), 1)
    ylim_max = 1.05
    for c in selected:
        for src, s, length in [(train, s0, v16.WIN) for s0 in train_cal_starts] + [(window, 0, len(window))]:
            vals = v16._n(src[s:s + length, c], cmin[c], cmax[c])
            ylim_max = max(ylim_max, float(vals.max()) + 0.1)
    ylim_max = min(ylim_max, 3.0)

    def _panel(ax, data, s, length, title, face, edge):
        for i, c in enumerate(selected):
            vals = v16._n(data[s:s + length, c], cmin[c], cmax[c])
            ax.plot(np.arange(length), vals, color=v16.LC[i % len(v16.LC)], lw=0.9, label=f"Ch{c}")
        ax.axhline(y=1.0, color="#ff9800", lw=0.9, ls="--", alpha=0.75)
        ax.set_ylim(-0.05, ylim_max)
        ax.set_xlim(0, length - 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=7, color=edge, fontweight="bold")
        ax.set_facecolor(face)
        for sp in ax.spines.values():
            sp.set_edgecolor(edge); sp.set_linewidth(1.2)

    has_heatmap = ranked is not None
    height_ratios = ([1.3] if has_heatmap else []) + [1, 1]
    fig = plt.figure(figsize=(3.8 * n_cols, 4.5 + (1.6 if has_heatmap else 0)), dpi=100)
    gs = gridspec.GridSpec(3 if has_heatmap else 2, n_cols, figure=fig,
                            height_ratios=height_ratios, hspace=0.65, wspace=0.28)
    row = 0

    if has_heatmap:
        ax0 = fig.add_subplot(gs[0, :])
        heat = np.zeros((len(ranked), len(window)))
        for i, c in enumerate(ranked):
            heat[i] = np.clip(v16._n(window[:, c], cmin_all[c], cmax_all[c]), 0, 1.5)
        ax0.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1.5)
        ax0.set_yticks(range(len(ranked)))
        ax0.set_yticklabels([str(c) for c in ranked], fontsize=4.5)
        ax0.set_xticks([])
        ax0.set_title("Heatmap: 38 channels, TRAIN min/max normalized, sorted by adaptive z-score "
                       "(bright/capped = exceeds training range)", fontsize=6.5)
        row = 1

    for i, s in enumerate(train_cal_starts):
        ax = fig.add_subplot(gs[row, i])
        _panel(ax, train, s, v16.WIN, f"TRAIN NORMAL {i+1}", "#f5f5f5", "#555")
    for i in range(len(train_cal_starts), n_cols):
        fig.add_subplot(gs[row, i]).axis("off")

    offset = (n_cols - 1) // 2
    ax = fig.add_subplot(gs[row + 1, offset])
    _panel(ax, window, 0, len(window), "CANDIDATE", "#fff8e1", "#b71c1c")
    ax.legend(fontsize=5, loc="upper right", ncol=2)
    for j in list(range(offset)) + list(range(offset + 1, n_cols)):
        fig.add_subplot(gs[row + 1, j]).axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_train_referenced(selected, width, n_cal, window=None, cmin=None, cmax=None):
    """v16 스타일 이중가설 프롬프트를 K5 스키마(verdict/start/end/confidence)에 맞게 축약.
    train min/max 고정 정규화 + 정상참조 패널을 설명하고, 정상/이상 두 가설을 각각
    논증한 뒤 종합 판단하게 시킨다 (여전히 1콜, JSON 하나로 응답).

    window/cmin/cmax가 주어지면 K2/K3/K4에서 이미 검증된(2x2 factorial, +0.042 유의)
    index-aware 텍스트를 top-25 포인트로 추가 -- 눈대중으로 y=1.0 선 높이를 읽다가
    틀리는 할루시네이션(예: 실제 0.65인데 "1.0 넘었다"고 주장)을 정확한 숫자로 방지."""
    text_block = ""
    if window is not None:
        lines = []
        for c in selected:
            nv = v16._n(window[:, c], cmin[c], cmax[c])
            top_idx = np.sort(np.argsort(-nv)[:N_POINTS_PER_CHANNEL])
            pts = ", ".join(f"({idx}, {nv[idx]:.3f})" for idx in top_idx)
            lines.append(f"  Channel {c}, top-{N_POINTS_PER_CHANNEL} highest-normalized points (index, value; "
                         f"value>1.0 = above confirmed training max): {pts}")
        text_block = "\n\n--- EXACT NUMBERS (cross-check against the image; do not eyeball the orange line) ---\n" + "\n".join(lines)

    return f"""=== SYSTEM ANOMALY VERIFICATION -- DUAL HYPOTHESIS ANALYSIS ===
Candidate window width: {width} ticks (relative indices 0 to {width-1}). Channels shown: {selected}.

--- NORMALIZATION (CRITICAL for interpretation) ---
y=0.0 = confirmed TRAINING minimum for each channel (machine in known-normal operation)
y=1.0 = confirmed TRAINING maximum for each channel (machine in known-normal operation)
The dashed ORANGE LINE marks y=1.0 -- the normal operating ceiling.
Values ABOVE the orange line indicate the channel has EXCEEDED its confirmed normal range.
Values between 0 and 1 are within the machine's confirmed normal operating envelope.
{text_block}

--- IMAGE LAYOUT ---
TOP (heatmap, if present): all 38 channels, one row each, TRAIN min/max normalized like everything
  else here -- use it only as a rough overview of which channels look most elevated; the line
  panels below and the EXACT NUMBERS table are the reliable source for precise judgments.
MIDDLE ROW (gray, "TRAIN NORMAL 1/2/3"): {n_cal} CONFIRMED NORMAL windows from TRAINING data.
  Training data is guaranteed anomaly-free -- these show the machine's true normal operation.
  These are your ground-truth baseline: compare the candidate's values against these.
BOTTOM (yellow/red border, "CANDIDATE"): the window you are judging.

Most candidates shown to you may in fact be normal (an automated screening step over-selects
loosely) -- do not assume anomaly just because a window was shown to you.

=== REQUIRED: DUAL HYPOTHESIS ANALYSIS ===
Before reaching a verdict, work through BOTH hypotheses:

STEP 1 -- HYPOTHESIS: CANDIDATE IS NORMAL
  (a) Are the candidate channel values within y=[0,1] (below the orange training-max line)?
  (b) Could any exceedance above y=1.0 be explained by the variation shown in the TRAIN NORMAL panels?
  (c) How strong is the normal-hypothesis evidence? (weak / moderate / strong)

STEP 2 -- HYPOTHESIS: CANDIDATE IS ANOMALOUS
  (a) Which EXACT CHANNELS show values clearly above the orange y=1.0 line, and by how much?
      Use the EXACT NUMBERS table above -- only claim an exceedance if a listed value is actually >1.0.
  (b) Is this exceedance ABSENT in ALL of the TRAIN NORMAL panels?
  (c) How strong is the anomaly-hypothesis evidence? (weak / moderate / strong)

STEP 3 -- VERDICT: pick NORMAL unless the anomaly-hypothesis is clearly stronger than the
normal-hypothesis. If evidence is tied or ambiguous, default to NORMAL.
If genuine, what is the TIGHTEST relative sub-range [start, end] (0 to {width-1}) that
captures the core anomalous behavior?

Respond ONLY with valid JSON (no markdown, no extra text):
{{"normal_hypothesis": "...", "anomaly_hypothesis": "...", "normal_strength": "weak/moderate/strong",
"anomaly_strength": "weak/moderate/strong", "verdict": "ANOMALY" or "NORMAL",
"start": <int>, "end": <int>, "confidence": "low" or "medium" or "high"}}"""


def build_prompt(selected, width, window, train, primed=True, viz="heatmap_overlay"):
    blocks = []
    for i, c in enumerate(selected):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        lo, hi = float(v.min()), float(v.max())
        nv = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {nv[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c} (rank {i+1}), top-{N_POINTS_PER_CHANNEL} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    if viz == "subplot_grid":
        # VLM4TS 논문 방식: 38채널을 각각 독립된 작은 칸에 그림(겹쳐그리기 없음)
        flag_note = ("" if primed else " (this is a screening heuristic, not a verdict -- most windows may be normal)")
        setup = (f"for a {width}-tick window (relative indices 0 to {width-1}) from a multivariate industrial "
                 f"time series with {N_CHANNELS} channels.\n\n"
                 f"The image is a grid of {N_CHANNELS} small subplots, one per channel (labeled ch0-ch{N_CHANNELS-1}), "
                 "each showing that channel's raw values over this window independently -- no channels are "
                 f"overlaid together. Channels {selected} were flagged by an automated per-channel screening "
                 f"step for closer attention{flag_note}.")
        context_note = ""
    elif viz == "heatmap_subplot_grid":
        # heatmap(개요, 38채널 전부)은 유지하고, overlay(겹쳐그리기)만 selected 채널 각각
        # 독립된 작은 칸으로 교체 -- overlay의 "겹쳐그리기" 요소만 분리해서 원인을 좁히기 위함.
        flag_note = ("" if primed else " -- this selection does NOT imply they are anomalous")
        setup = (f"for a {width}-tick window (relative indices 0 to {width-1}) from a multivariate industrial "
                 f"time series with {N_CHANNELS} channels.\n\n"
                 "Top panel: heatmap of all 38 channels, sorted by an automated per-channel deviation score "
                 "(this ranking is a screening heuristic, not a verdict).\n"
                 f"Bottom panel: a grid of {len(selected)} small subplots, one per candidate channel "
                 f"({selected}), each showing that channel's raw values independently -- no channels are "
                 f"overlaid together{flag_note}.")
        context_note = ""
    elif primed:
        setup = (f"for a Stage-1 candidate window of width {width} (relative indices 0 to {width-1}) "
                 "flagged by an adaptive per-channel anomaly scorer.\n\n"
                 "Top panel: heatmap of all 38 channels, sorted by adaptive z-score.\n"
                 f"Bottom panel: overlay of the {len(selected)} channels ({selected}) that an adaptive "
                 "per-channel threshold flagged as statistically unusual in this window.")
        context_note = ("Stage-1's window-merging often produces candidates WIDER than the true anomaly "
                         "(padded with quiet, normal periods). ")
    else:
        # 프라이밍 제거 버전 (sanity check용): "이미 이상하다고 판정됨"을 암시하는 문구를 전부 중립화
        setup = (f"for a {width}-tick window (relative indices 0 to {width-1}) from a multivariate industrial "
                 f"time series with {N_CHANNELS} channels.\n\n"
                 "Top panel: heatmap of all 38 channels, sorted by an automated per-channel deviation score "
                 "(this ranking is a screening heuristic, not a verdict).\n"
                 f"Bottom panel: overlay of {len(selected)} channels ({selected}) selected by that same "
                 "heuristic for closer inspection -- this selection does NOT imply they are anomalous.")
        context_note = ""

    return f"""You are shown a composite image for a multivariate industrial system, {setup}

For each of these channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

{context_note}Judge independently, from the data itself:
(a) is this window a genuine anomaly, or does it just contain normal variation / an isolated non-anomalous blip? Most windows shown to you may in fact be normal -- do not assume otherwise.
(b) if genuine, what is the TIGHTEST relative sub-range [start, end] (0 to {width-1}) that captures the core anomalous behavior? (use the full range if the whole window looks anomalous)

Respond ONLY with valid JSON (no markdown, no extra text):
{{"verdict": "ANOMALY" or "NORMAL", "start": <int>, "end": <int>, "confidence": "low" or "medium" or "high"}}"""


def parse_detect_response(raw):
    if raw is None:
        return None
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════
# 3. 실행
# ══════════════════════════════════════════════════════════════════

def run(entity, execute=False, method="fixed", alpha=0.1, alpha_strict=0.01, corr_thr=0.5,
        viz="heatmap_overlay"):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    degenerate_ch = constant_channels(train)
    corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0) if method == "hysteresis" else None

    train_cal_starts = v16.get_train_cal_windows(train) if viz == "train_referenced" else None
    cmin_all, cmax_all = (v16.gn_train(train, range(N_CHANNELS)) if viz == "train_referenced" else (None, None))

    print(f"[{entity}] Stage1 후보 구간 생성 중 (전체 시계열 슬라이딩)...", flush=True)
    loose_ivs, inter, entity_channel_calib = stage1_candidates(entity, train, test, degenerate_ch)
    print(f"  후보 {len(loose_ivs)}개 = 예상 VLM 콜 수", flush=True)

    if not execute:
        print("[STOP] --run 플래그로 실행하세요.")
        return

    checkpoint_path = OUT_DIR / f"checkpoint_{entity}_{method}_{viz}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    confirmed = []

    for cs, ce in loose_ivs:
        key = f"{cs}_{ce}"
        # 224틱보다 좁으면 중심 확장, 넓으면 그대로(폭 가변 허용 -- 넓은 후보를 그대로 보여주는 게 이번 취지).
        # ws = 실제 표시되는 창의 절대 시작점 -- 확장된 경우 cs와 다르므로 이걸 기준으로 좌표를 복원해야 함
        # (예전 버그: 여기서 cs를 썼는데, 확장된 창(예: 폭 56 -> ws=cs-84)에서는 84틱씩 밀려서 저장됐음).
        ws = cs if (ce - cs + 1) >= WIN else max(0, cs - (WIN - (ce - cs + 1)) // 2)
        window = test[ws:ws + WIN] if (ce - cs + 1) < WIN else test[cs:ce + 1]
        width = len(window)

        zs = compute_zscores(entity, train, window, entity_channel_calib, degenerate_ch)
        if method == "fixed":
            ranked, selected = select_channels_fixed(zs, alpha)
        else:
            ranked, selected = select_channels_hysteresis(zs, corr, alpha_strict, alpha, corr_thr)

        if checkpoint.get(key, {}).get("status") == "OK":
            pred = checkpoint[key]["pred"]
        else:
            if viz == "train_referenced":
                img = render_train_referenced(window, train, selected, cmin_all, cmax_all, train_cal_starts,
                                               ranked=ranked, cmin_all=cmin_all, cmax_all=cmax_all)
                prompt = build_prompt_train_referenced(selected, width, len(train_cal_starts),
                                                        window, cmin_all, cmax_all)
            else:
                img = render_heatmap_overlay(window, ranked, selected)
                prompt = build_prompt(selected, width, window, train)
            raw = call_vlm(prompt, img)
            pred = parse_detect_response(raw)
            checkpoint[key] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{key}] pred={pred}", flush=True)

        if pred and pred.get("verdict") == "ANOMALY":
            s = ws + max(0, min(width - 1, int(pred.get("start", 0))))
            e = ws + max(0, min(width - 1, int(pred.get("end", width - 1))))
            if e < s:
                s, e = cs, ce
            confirmed.append((s, e))

    gt_ivs = get_intervals(labels)
    iv_f1 = eval_f1(gt_ivs, confirmed)
    pred_binary = np.zeros(len(labels), dtype=int)
    for s, e in confirmed:
        pred_binary[s:e + 1] = 1
    point_f1 = pt_f1(labels, pred_binary)

    print(f"\n=== [{entity}, method={method}, viz={viz}] 결과 ===")
    print(f"후보 {len(loose_ivs)}개 -> 확정 {len(confirmed)}개")
    print(f"interval-F1 = {iv_f1:.4f}")
    print(f"point-F1    = {point_f1:.4f}")
    (OUT_DIR / f"summary_{entity}_{method}_{viz}.json").write_text(json.dumps({
        "n_candidates": len(loose_ivs), "n_confirmed": len(confirmed),
        "interval_f1": iv_f1, "point_f1": point_f1, "confirmed": confirmed,
    }, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# 4. Sanity check: 14/14 후보가 전부 ANOMALY로 나온 게 프롬프트 프라이밍
#    ("Stage-1이 이미 이상하다고 플래그했다") 때문인지, 아니면 VLM이 이 시각화
#    포맷에서는 원래 거의 항상 ANOMALY라고 하는지 구분하기 위한 진단.
#    GT와 전혀 안 겹치는 확실한 정상 윈도우에, 프라이밍 문구를 뺀 프롬프트로
#    같은 채널선택/시각화를 그대로 돌려서 오탐률(FPR)을 직접 잰다.
# ══════════════════════════════════════════════════════════════════

def pick_normal_windows(labels, n=5, seed=0):
    starts = list(range(0, len(labels) - WIN + 1, STRIDE))
    safe = [s for s in starts if labels[s:s + WIN].sum() == 0]
    rng = np.random.default_rng(seed)
    if len(safe) <= n:
        return safe
    return sorted(int(s) for s in rng.choice(safe, size=n, replace=False))


def sanity_check(entity, n=5, method="fixed", alpha=0.1, alpha_strict=0.01, corr_thr=0.5, seed=0,
                  viz="heatmap_overlay"):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    degenerate_ch = constant_channels(train)
    corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0) if method == "hysteresis" else None

    starts = pick_normal_windows(labels, n=n, seed=seed)
    print(f"[{entity}] 정상 컨트롤 윈도우 {len(starts)}개 (GT와 전혀 안 겹침), viz={viz}: {starts}", flush=True)

    train_cal_starts, cmin, cmax = None, None, None
    if viz == "train_referenced":
        train_cal_starts = v16.get_train_cal_windows(train)

    checkpoint_path = OUT_DIR / f"sanity_{entity}_{method}_{viz}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    entity_channel_calib = {}
    n_anomaly = 0
    for s in starts:
        window = test[s:s + WIN]
        key = f"{s}_{s + WIN - 1}"
        zs = compute_zscores(entity, train, window, entity_channel_calib, degenerate_ch)
        if method == "fixed":
            ranked, selected = select_channels_fixed(zs, alpha)
        else:
            ranked, selected = select_channels_hysteresis(zs, corr, alpha_strict, alpha, corr_thr)

        if checkpoint.get(key, {}).get("status") == "OK":
            pred = checkpoint[key]["pred"]
        else:
            if viz == "subplot_grid":
                img = render_subplot_grid(window)
            elif viz == "heatmap_subplot_grid":
                img = render_heatmap_subplot_grid(window, ranked, selected)
            elif viz == "train_referenced":
                cmin_all, cmax_all = v16.gn_train(train, range(N_CHANNELS))
                cmin, cmax = cmin_all, cmax_all
                img = render_train_referenced(window, train, selected, cmin, cmax, train_cal_starts,
                                               ranked=ranked, cmin_all=cmin_all, cmax_all=cmax_all)
            else:
                img = render_heatmap_overlay(window, ranked, selected)

            if viz == "train_referenced":
                prompt = build_prompt_train_referenced(selected, WIN, len(train_cal_starts), window, cmin, cmax)
            else:
                prompt = build_prompt(selected, WIN, window, train, primed=False, viz=viz)
            raw = call_vlm(prompt, img)
            pred = parse_detect_response(raw)
            checkpoint[key] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")

        verdict = pred.get("verdict") if pred else None
        if verdict == "ANOMALY":
            n_anomaly += 1
        print(f"  [{key}] verdict={verdict} (실제로는 정상, GT 없음)", flush=True)

    fpr = n_anomaly / len(starts) if starts else 0.0
    print(f"\n=== [{entity}] 정상 컨트롤 오탐률(FPR) = {n_anomaly}/{len(starts)} = {fpr:.2%} "
          f"(탈-프라이밍 프롬프트, method={method}, viz={viz}) ===")
    return fpr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="machine-1-1")
    ap.add_argument("--stage1", action="store_true", help="후보 구간/콜 수만 확인, VLM 없음")
    ap.add_argument("--run", action="store_true", help="VLM 실행")
    ap.add_argument("--sanity-check", action="store_true",
                     help="정상 컨트롤 윈도우에 탈-프라이밍 프롬프트로 오탐률(FPR) 측정, VLM 콜 발생")
    ap.add_argument("--n-sanity", type=int, default=5)
    ap.add_argument("--viz", choices=["heatmap_overlay", "subplot_grid", "heatmap_subplot_grid", "train_referenced"],
                     default="heatmap_overlay",
                     help="subplot_grid=38채널 전부 독립칸(논문 방식) / heatmap_subplot_grid=heatmap 유지+"
                          "selected 채널만 독립칸(overlay의 겹쳐그리기만 교체) / train_referenced=v16 방식"
                          "(train min/max 고정 정규화 + 정상참조 패널 + heatmap + index-aware 텍스트 + 이중가설)")
    ap.add_argument("--method", choices=["fixed", "hysteresis"], default="fixed")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha-strict", type=float, default=0.01)
    ap.add_argument("--corr-thr", type=float, default=0.5)
    args = ap.parse_args()
    if args.sanity_check:
        sanity_check(args.entity, n=args.n_sanity, method=args.method, alpha=args.alpha,
                     alpha_strict=args.alpha_strict, corr_thr=args.corr_thr, viz=args.viz)
    else:
        run(args.entity, execute=args.run, method=args.method, alpha=args.alpha,
            alpha_strict=args.alpha_strict, corr_thr=args.corr_thr, viz=args.viz)
