"""
Stage2 K2: heatmap + overlay + index-aware z-score 텍스트 (report18 최종 모델)

채널-진단(diagnosis) 트랙 -- Stage1이 "이 구간이 이상하다"고 찾아낸 윈도우 안에서,
38개 채널 중 어느 채널이 원인인지 GPT-4o에게 진단시킨다. (윈도우 자체가 진짜
이상인지 판정하는 탐지(detection) 트랙 v16과는 다른 별개 과제.)

구조
----
채널 랭킹: Step0 -- 채널마다 train의 대표 구간 딱 하나와 DINOv2 cosine distance로
           비교하는 가장 단순한 방식 (results_adaptive_vs_fixed/step0_scores_cache.json)

이미지 1장:
  위쪽 -- 38채널 heatmap, Step0 점수 순 정렬(가장 의심스러운 채널이 맨 위)
  아래쪽 -- 상위 8채널만 실제 값 overlay(자세한 파형)

텍스트 (index-aware):
  상위 8채널마다, train 대비 |z-score|가 가장 큰 25개 지점을
  "(시간 인덱스, 정규화값)" 쌍으로 나열해서 프롬프트에 추가.

GPT-4o에게 이미지+텍스트를 한 번에 주고 {"anomalous_channels": [...]} 응답을 받는다.

결과: 채널 집합 F1 = 0.327 (n=18, report18)

사용법
------
  python experiment_stage2_k2.py --stage1   # 세그먼트/콜 수만 출력, VLM 호출 없음
  python experiment_stage2_k2.py --run      # VLM 실행(세그먼트당 1콜)
"""
import argparse
import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
from smd_3way_baseline_comparison import load_smd_test, call_vlm, parse_response
from step1v3_dino_graph_smd import load_smd

OUT_DIR = BASE / "experiments" / "results_stage2_k2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
STEP0_SCORE_CACHE = BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]

N_CHANNELS = 38
N_TOP = 8
N_POINTS_PER_CHANNEL = 25
WIN = 224


def parse_seg_id(seg_id):
    m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
    return m.group(1), int(m.group(2)), int(m.group(3))


def find_gt(seg_id, ckpts):
    for ck in ckpts:
        v = ck.get(f"{seg_id}_gt")
        if v:
            return set(v["gt_channels"])
    return None


def get_channel_ranking(seg_id):
    """Step0: 채널마다 train 대표구간 1개와 cosine distance 비교."""
    cache = json.loads(STEP0_SCORE_CACHE.read_text(encoding="utf-8"))
    scores = {int(c): v for c, v in cache[seg_id].items()}
    return sorted(scores, key=lambda c: -scores[c])


def render_heatmap_overlay(window, ranked, n=N_TOP):
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
    ax1.set_title("Heatmap: 38 channels, sorted by Step0 score (top row = most suspicious)", fontsize=7)

    top_n = ranked[:n]
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(top_n):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax2.plot(np.arange(len(norm)), norm, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax2.legend(fontsize=5, ncol=min(6, len(top_n)), loc="upper right")
    ax2.set_xticks([])
    ax2.set_title(f"Overlay: top-{n} candidate channels (detail)", fontsize=7)

    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt(all_channels, top_n, window, train):
    blocks = []
    for i, c in enumerate(top_n):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        pts = ", ".join(f"({idx}, {norm[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c} (rank {i+1}), top-{N_POINTS_PER_CHANNEL} most-deviating points: {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by a preliminary anomaly score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: an overlay line plot of the top-{len(top_n)} candidate channels ({top_n}) from the heatmap, showing their detailed waveforms with channel numbers labeled.

For each of the top-{len(top_n)} candidate channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Use the panels and this point data together to identify which channels show anomalous or deviating behavior in this window. No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {all_channels} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def f1_of(pred, gt):
    pred = set(pred)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run(execute=False):
    selected = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    print(f"세그먼트 수 = 예상 VLM 콜 수 = {len(selected)}")
    if not execute:
        print("[STOP] --run 플래그로 실행하세요.")
        return

    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]
    checkpoint_path = OUT_DIR / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    all_channels = list(range(N_CHANNELS))
    entity_train, entity_test = {}, {}
    rows = []

    for s in selected:
        seg_id = s["segment_id"]
        entity, cs, ce = parse_seg_id(seg_id)
        if entity not in entity_test:
            entity_test[entity] = load_smd_test(entity)
        if entity not in entity_train:
            entity_train[entity], _ = load_smd(entity)
        train, test = entity_train[entity], entity_test[entity]
        center = (cs + ce) // 2
        s_ = max(0, min(len(test) - WIN, center - WIN // 2))
        window = test[s_:s_ + WIN]

        ranked = get_channel_ranking(seg_id)
        top8 = ranked[:N_TOP]

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_heatmap_overlay(window, ranked)
            prompt = build_prompt(all_channels, top8, window, train)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{checkpoint[seg_id]['status']}] {seg_id}: pred={pred}", flush=True)

        gt = find_gt(seg_id, ckpts)
        if gt is None or pred is None:
            continue
        rows.append({"seg_id": seg_id, "f1": f1_of(pred, gt)})

    mean_f1 = float(np.mean([r["f1"] for r in rows])) if rows else 0.0
    print(f"\n평균 F1 (n={len(rows)}) = {mean_f1:.4f}  (참고: report18 = 0.327)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="콜 수만 확인, VLM 호출 없음")
    ap.add_argument("--run", action="store_true", help="VLM 실행")
    args = ap.parse_args()
    run(execute=args.run)
