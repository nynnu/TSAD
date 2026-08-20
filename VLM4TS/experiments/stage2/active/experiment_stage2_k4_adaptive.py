"""
Stage2 K4: K3(heatmap+overlay+index-aware 텍스트, 채널진단) + 적응형 채널선택
(고정 top-8 대신 가변 개수) -- 나연(공동연구자)의 20주차 실험과 병합.

배경
----
K3(experiment_stage2_k3.py)는 채널을 항상 고정 top-8만 뽑았다. 그런데 교수님
피드백("GT가 8개 넘으면 어떻게 할 거냐")과, 나연이 원본 SMD interpretation_label
48개 세그먼트(필터링 없이 그대로, GT 채널 수 1~34개, 평균 8.2개)로 검증한 결과가
이 문제를 정면으로 지적했다:

  top-8 고정            recall(GT 9~15개 구간) = 0.326, recall(GT 16~38개 구간) = 0.293
  적응형 채널선택(alpha=0.1) recall(GT 9~15개 구간) = 0.703, recall(GT 16~38개 구간) = 0.804

즉 GT가 많은 세그먼트에서 고정 8개는 애초에 못 잡는 게 태반이었다.

이 파일이 하는 일
------------------
K3의 이미지/텍스트/VLM 구조는 그대로 두고, "몇 개 채널을 후보로 보여줄지"만
나연의 방식으로 교체한다:

  1. 채널 선택 (나연 방식, patchknn_channel_select.py 포팅):
     production 스코어러(colab_multivariate_v2, ViT-B)로 채널마다 train bank(60)
     + 별도 calibration셋(30, train에서만 뽑음-test 안 봄)의 정상 sum-score 분포
     (mu, sigma)를 구하고, 실제 세그먼트의 채널 sum-score를 z-표준화한다.
     alpha(기본 0.1, recall이 가장 좋았던 값)에 대응하는 z-threshold를 넘는
     채널만 선택 -- 세그먼트마다 0~38개로 가변.

  2. 렌더링/텍스트 (K3 방식 그대로): 38채널 heatmap(z-score 순 정렬) + 선택된
     채널만 overlay + 각 채널의 index-aware (index,value) 텍스트.

  3. VLM 진단: GPT-4o에게 이미지+텍스트를 주고 {"anomalous_channels":[...]}
     받아서 GT와 F1 비교.

평가셋: 나연의 results_gt_channel_count/segments.json (원본 SMD, 48개, GT 필터링
없음) -- 우리가 기존에 쓰던 74개 세그먼트는 GT<=8로 암묵적으로 필터링돼 있었던
것으로 확인됨(report20 참고), 그래서 이 48개를 새 기준으로 쓴다.

사용법
------
  python experiment_stage2_k4_adaptive.py --stage1          # 선택 결과만 확인, VLM 없음
  python experiment_stage2_k4_adaptive.py --run --alpha 0.1  # VLM 실행(세그먼트당 1콜)
"""
import argparse
import base64
import json
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
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from smd_3way_baseline_comparison import call_vlm, parse_response

OUT_DIR = BASE / "experiments" / "results_stage2_k4_adaptive"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 나연의 48개 원본(필터링 없는) GT 세그먼트를 그대로 재사용
GT_SEGMENTS_PATH = BASE / "experiments" / "20주차실험" / "20주차 주요실험" / "results_gt_channel_count" / "segments.json"

N_CHANNELS = 38
N_BANK = 60
N_CALIB = 30
N_POINTS_PER_CHANNEL = 25

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════
# 1. 적응형 채널 선택 (나연 방식)
# ══════════════════════════════════════════════════════════════════

def _bank_and_calib_idx(n_windows):
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK + N_CALIB + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]
    return sorted(bank_idx), calib_idx


def get_or_build_channel_calib(entity, c, train):
    """채널의 정상 sum-score 분포(mu, sigma)를 train 뱅크/calib에서 구함 (test 안 봄).
    뱅크 임베딩(tr_cls/tr_patches)도 .npy로 캐시 -- 이게 제일 비싼 부분(ViT-B forward,
    채널당 60장)이라 재시작할 때마다 다시 계산하면 매번 15분 넘게 날아간다."""
    ent_cache = CACHE_DIR / entity
    ent_cache.mkdir(parents=True, exist_ok=True)
    stats_path = ent_cache / f"ch{c}_stats.json"
    bank_cls_path = ent_cache / f"ch{c}_bank_cls.npy"
    bank_patches_path = ent_cache / f"ch{c}_bank_patches.npy"

    windows = cm.get_windows(train[:, c])
    bank_idx, calib_idx = _bank_and_calib_idx(len(windows))

    if bank_cls_path.exists() and bank_patches_path.exists():
        tr_cls = np.load(bank_cls_path)
        tr_patches = np.load(bank_patches_path)
    else:
        bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
        tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)
        np.save(bank_cls_path, tr_cls)
        np.save(bank_patches_path, tr_patches)

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
        ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
        sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
        calib_sums = sc["sum"]
        stats = {"mu": float(calib_sums.mean()), "sigma": float(calib_sums.std() + 1e-8)}
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return tr_cls, tr_patches, stats


def score_segment_channel(tr_cls, tr_patches, test_img):
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    return float(sc["sum"][0])


def constant_channels(train, eps=1e-3):
    """train에서 거의 상수(std<eps)인 채널 -- 이런 채널은 DINOv2 임베딩 거리가
    노이즈에도 극단적으로 튀어서(0으로 나누기에 가까운 수치 불안정) 항상 선택되는
    아티팩트가 생긴다(어젯밤 raw_channel_fn_recovery 실험에서 검증됨). 후보에서 제외."""
    stds = train.std(axis=0)
    return {c for c in range(N_CHANNELS) if stds[c] < eps}


def select_channels(entity, train, window, entity_channel_calib, alpha, degenerate_ch):
    """38채널 전부 z-score 계산 -> alpha 임계값 넘는 채널만 선택(가변 개수).
    반환: (ranked_all_38, selected_subset) -- ranked는 heatmap 정렬용, selected는 overlay/텍스트/VLM 후보.
    상수(degenerate) 채널은 애초에 후보에서 제외."""
    zs = {}
    for c in range(N_CHANNELS):
        if c in degenerate_ch:
            zs[c] = -np.inf
            continue
        key = (entity, c)
        if key not in entity_channel_calib:
            entity_channel_calib[key] = get_or_build_channel_calib(entity, c, train)
        tr_cls, tr_patches, stats = entity_channel_calib[key]
        test_img = cm.ts_to_image_fast(window[:, c])
        seg_sum = score_segment_channel(tr_cls, tr_patches, test_img)
        zs[c] = (seg_sum - stats["mu"]) / stats["sigma"]

    ranked = sorted(range(N_CHANNELS), key=lambda c: -zs[c])
    z_thr = norm.ppf(1 - alpha)
    selected = [c for c in ranked if zs[c] > z_thr]
    if not selected:  # 아무것도 안 넘으면 최소 1개(최고점)는 후보로
        selected = ranked[:1]
    return ranked, selected


# ══════════════════════════════════════════════════════════════════
# 2. 렌더링 + 프롬프트 (K3와 동일 구조, 개수만 가변)
# ══════════════════════════════════════════════════════════════════

def render_heatmap_overlay(window, ranked, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=100, gridspec_kw={"height_ratios": [1.4, 1]})
    heat = np.zeros((len(ranked), window.shape[0]))
    for i, c in enumerate(ranked):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
    ax1.imshow(heat, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(ranked)))
    ax1.set_yticklabels([str(c) for c in ranked], fontsize=5)
    ax1.set_xticks([])
    ax1.set_title("Heatmap: 38 channels, sorted by adaptive z-score (top row = most suspicious)", fontsize=7)

    cmap = plt.get_cmap("tab10" if len(selected) <= 10 else "tab20")
    for i, c in enumerate(selected):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm_v = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax2.plot(np.arange(len(norm_v)), norm_v, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax2.legend(fontsize=5, ncol=min(8, len(selected)), loc="upper right")
    ax2.set_xticks([])
    ax2.set_title(f"Overlay: {len(selected)} candidate channels selected by adaptive z-threshold (detail)", fontsize=7)

    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(all_channels, selected, window, train):
    blocks = []
    for i, c in enumerate(selected):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        lo, hi = float(v.min()), float(v.max())
        normv = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {normv[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c} (rank {i+1}), top-{N_POINTS_PER_CHANNEL} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by an adaptive anomaly z-score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of the {len(selected)} candidate channels ({selected}) that an adaptive per-channel threshold flagged as statistically unusual (this number varies per window -- it is NOT a fixed top-k), showing their detailed waveforms with channel numbers labeled.

For each of the {len(selected)} candidate channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Use the panels and this point data together to identify which of these {len(selected)} candidate channels show genuinely anomalous behavior in this window (you may judge that ALL or only SOME of them are truly anomalous). No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {selected} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


# ══════════════════════════════════════════════════════════════════
# 3. 평가 + 실행
# ══════════════════════════════════════════════════════════════════

def f1_of(pred, gt):
    pred = set(pred)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run(execute=False, alpha=0.1):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    print(f"세그먼트 수 = {len(segments)} (나연의 원본 SMD 48개, GT 필터링 없음), alpha={alpha}")

    entity_data, entity_channel_calib, entity_degenerate = {}, {}, {}
    all_channels = list(range(N_CHANNELS))
    checkpoint_path = OUT_DIR / f"checkpoint_a{alpha}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    rows = []

    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        gt = set(d - 1 for d in seg["dims"])
        seg_id = f"{entity}_{cs}_{ce}"

        if entity not in entity_data:
            train, test = load_smd(entity)
            entity_data[entity] = (train, test)
        train, test = entity_data[entity]
        if entity not in entity_degenerate:
            entity_degenerate[entity] = constant_channels(train)
            print(f"  [{entity}] 상수채널(제외) = {sorted(entity_degenerate[entity])}", flush=True)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        t0 = time.time()
        ranked, selected = select_channels(entity, train, window, entity_channel_calib, alpha, entity_degenerate[entity])
        print(f"  {seg_id}: k(GT)={len(gt)} selected={len(selected)}개 {selected} ({time.time()-t0:.1f}s)", flush=True)

        if not execute:
            continue

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_heatmap_overlay(window, ranked, selected)
            prompt = build_prompt(all_channels, selected, window, train)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    [{checkpoint[seg_id]['status']}] pred={pred}", flush=True)

        if pred is None:
            continue
        rows.append({"seg_id": seg_id, "f1": f1_of(pred, gt)})

    if execute and rows:
        mean_f1 = float(np.mean([r["f1"] for r in rows]))
        print(f"\n평균 F1 (n={len(rows)}, alpha={alpha}) = {mean_f1:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="채널 선택 결과만 확인, VLM 없음")
    ap.add_argument("--run", action="store_true", help="VLM 실행")
    ap.add_argument("--alpha", type=float, default=0.1, help="z-threshold 유의수준(작을수록 더 적게 선택)")
    args = ap.parse_args()
    run(execute=args.run, alpha=args.alpha)
