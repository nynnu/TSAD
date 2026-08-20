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
import json
import re
import sys
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
    win_score = np.zeros(len(starts))
    for wi, s in enumerate(starts):
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

    inter = win_to_ts(win_score, T_test)
    thr = np.percentile(inter, loose_pct)
    loose_ivs = get_intervals((inter > thr).astype(int))
    return loose_ivs, inter, entity_channel_calib


# ══════════════════════════════════════════════════════════════════
# 2. 프롬프트 (렌더링은 K4의 render_heatmap_overlay를 그대로 import해서 재사용,
#    질문/출력스키마만 탐지용으로 교체)
# ══════════════════════════════════════════════════════════════════

def build_prompt(selected, width, window, train):
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

    return f"""You are shown a composite image for a multivariate industrial system, for a Stage-1 candidate window of width {width} (relative indices 0 to {width-1}) flagged by an adaptive per-channel anomaly scorer.

Top panel: heatmap of all 38 channels, sorted by adaptive z-score.
Bottom panel: overlay of the {len(selected)} channels ({selected}) that an adaptive per-channel threshold flagged as statistically unusual in this window.

For each of these channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Stage-1's window-merging often produces candidates WIDER than the true anomaly (padded with quiet, normal periods). Judge:
(a) is this window a genuine anomaly, or does it just contain normal variation / an isolated non-anomalous blip?
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

def run(entity, execute=False, method="fixed", alpha=0.1, alpha_strict=0.01, corr_thr=0.5):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    degenerate_ch = constant_channels(train)
    corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0) if method == "hysteresis" else None

    print(f"[{entity}] Stage1 후보 구간 생성 중 (전체 시계열 슬라이딩)...", flush=True)
    loose_ivs, inter, entity_channel_calib = stage1_candidates(entity, train, test, degenerate_ch)
    print(f"  후보 {len(loose_ivs)}개 = 예상 VLM 콜 수", flush=True)

    if not execute:
        print("[STOP] --run 플래그로 실행하세요.")
        return

    checkpoint_path = OUT_DIR / f"checkpoint_{entity}_{method}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    confirmed = []

    for cs, ce in loose_ivs:
        key = f"{cs}_{ce}"
        window = test[cs:ce + 1] if (ce - cs + 1) >= WIN else test[max(0, cs - (WIN - (ce - cs + 1)) // 2):][:WIN]
        # 224틱보다 좁으면 중심 확장, 넓으면 그대로(폭 가변 허용 -- 넓은 후보를 그대로 보여주는 게 이번 취지)
        width = len(window)

        zs = compute_zscores(entity, train, window, entity_channel_calib, degenerate_ch)
        if method == "fixed":
            ranked, selected = select_channels_fixed(zs, alpha)
        else:
            ranked, selected = select_channels_hysteresis(zs, corr, alpha_strict, alpha, corr_thr)

        if checkpoint.get(key, {}).get("status") == "OK":
            pred = checkpoint[key]["pred"]
        else:
            img = render_heatmap_overlay(window, ranked, selected)
            prompt = build_prompt(selected, width, window, train)
            raw = call_vlm(prompt, img)
            pred = parse_detect_response(raw)
            checkpoint[key] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{key}] pred={pred}", flush=True)

        if pred and pred.get("verdict") == "ANOMALY":
            s = cs + max(0, min(width - 1, int(pred.get("start", 0))))
            e = cs + max(0, min(width - 1, int(pred.get("end", width - 1))))
            if e < s:
                s, e = cs, ce
            confirmed.append((s, e))

    gt_ivs = get_intervals(labels)
    iv_f1 = eval_f1(gt_ivs, confirmed)
    pred_binary = np.zeros(len(labels), dtype=int)
    for s, e in confirmed:
        pred_binary[s:e + 1] = 1
    point_f1 = pt_f1(labels, pred_binary)

    print(f"\n=== [{entity}, method={method}] 결과 ===")
    print(f"후보 {len(loose_ivs)}개 -> 확정 {len(confirmed)}개")
    print(f"interval-F1 = {iv_f1:.4f}")
    print(f"point-F1    = {point_f1:.4f}")
    (OUT_DIR / f"summary_{entity}_{method}.json").write_text(json.dumps({
        "n_candidates": len(loose_ivs), "n_confirmed": len(confirmed),
        "interval_f1": iv_f1, "point_f1": point_f1, "confirmed": confirmed,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="machine-1-1")
    ap.add_argument("--stage1", action="store_true", help="후보 구간/콜 수만 확인, VLM 없음")
    ap.add_argument("--run", action="store_true", help="VLM 실행")
    ap.add_argument("--method", choices=["fixed", "hysteresis"], default="fixed")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha-strict", type=float, default=0.01)
    ap.add_argument("--corr-thr", type=float, default=0.5)
    args = ap.parse_args()
    run(args.entity, execute=args.run, method=args.method, alpha=args.alpha,
        alpha_strict=args.alpha_strict, corr_thr=args.corr_thr)
