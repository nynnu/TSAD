"""
Stage2 K3: heatmap + overlay + index-aware z-score 텍스트, patch-KNN 채널 랭킹
(report20 최종 모델 -- K2에서 채널 랭킹 소스만 교체)

K2(experiment_stage2_k2.py)와 이미지/텍스트 구조는 완전히 동일하고,
"채널 랭킹을 어떻게 매기는가"만 바뀐다.

  K2: Step0 -- 채널마다 train 대표구간 1개와 cosine distance 비교
  K3: patch-KNN -- 채널마다 train에서 15개 참조 윈도우(뱅크)를 뽑고,
      DINOv2 patch 임베딩끼리 최근접 거리(K=1, topk10 집계)로 순위

결과 (n=39 = 파일럿18 + held-out21, 2x2 요인설계로 검증):
  K2(Step0+텍스트)      = 0.344
  K3(patchKNN+텍스트)   = 0.433   <- 이 파일

  랭킹 교체 자체의 효과 = +0.083 (텍스트 효과 +0.042의 2배, 완전 가산적)

주의 (report20 원인분석 요약): 이 결과(0.433)는 아직 VLM 없이 patch-KNN
top-8을 그대로 답으로 낸 것(F1=0.442)을 넘지 못한다. VLM이 채널을 지울 때
(a) patch-KNN 순위가 낮은 채널일수록 지우는 경향(Mann-Whitney p=0.0034),
(b) 여러 채널이 동시에 튀면 안 지우고 혼자 튀면 지우는 "다수결 동조 편향"
(오즈비 4.38, p=0.0295)에 편향돼 진짜 이상까지 같이 지우기 때문. 순위/동조
편향을 프롬프트로 역보정하는 시도는 둘 다 95% 신뢰구간이 0을 포함해 실패.
다음 단계 후보는 production(5-참조) 채널 선택 스코어러 비교 -- report20 참고.

사용법
------
  python experiment_stage2_k3.py --stage1   # patch-KNN 랭킹만 계산, VLM 없음(비용 0)
  python experiment_stage2_k3.py --run      # VLM 실행(세그먼트당 1콜)
"""
import argparse
import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from transformers import Dinov2Model

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "analysis"))
from smd_3way_baseline_comparison import load_smd_test, call_vlm, parse_response
from step1v3_dino_graph_smd import load_smd

OUT_DIR = BASE / "experiments" / "results_stage2_k3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PILOT_SELECTION = BASE / "experiments" / "results_pilot_layout" / "pilot_selection.json"
CKPT_PATHS = [
    BASE / "experiments" / "results_smd_3way_baseline" / "checkpoint.json",
    BASE / "experiments" / "results_3way_broad" / "checkpoint.json",
    BASE / "experiments" / "results_full_smd_3way" / "checkpoint.json",
]

N_CHANNELS = 38
N_TOP = 8
N_POINTS_PER_CHANNEL = 25
WIN = 224
N_BANK = 15  # patch-KNN 참조 뱅크 크기

_dino_model = None
_dino_tfm = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _get_dino_model():
    global _dino_model
    if _dino_model is None:
        _dino_model = Dinov2Model.from_pretrained("facebook/dinov2-small").eval()
    return _dino_model


def parse_seg_id(seg_id):
    m = re.match(r"^(machine-\d+-\d+)_(\d+)_(\d+)$", seg_id)
    return m.group(1), int(m.group(2)), int(m.group(3))


def find_gt(seg_id, ckpts):
    for ck in ckpts:
        v = ck.get(f"{seg_id}_gt")
        if v:
            return set(v["gt_channels"])
    return None


def _ts_to_image(window, size=WIN):
    lo, hi = float(window.min()), float(window.max())
    normed = (window - lo) / (hi - lo + 1e-8)
    n = len(normed)
    xs = (np.arange(n) * (size - 1) / (n - 1)).astype(int)
    ys = size - 1 - (normed * (size - 5) + 2).astype(int)
    ys = np.clip(ys, 0, size - 1)
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(0, 0, 0), width=2)
    return img


def _extract_patches(imgs):
    model = _get_dino_model()
    x = torch.stack([_dino_tfm(im) for im in imgs])
    with torch.no_grad():
        out = model(pixel_values=x)
    return out.last_hidden_state[:, 1:].numpy()


def get_channel_ranking(entity, train, window, bank_cache):
    """patch-KNN: 채널마다 train에서 15개 참조 윈도우와 최근접 patch 거리(topk10)로 순위."""
    scores = {}
    for c in range(N_CHANNELS):
        key = (entity, c)
        if key not in bank_cache:
            starts = np.linspace(0, len(train) - WIN, N_BANK).astype(int)
            bank_imgs = [_ts_to_image(train[s:s + WIN, c]) for s in starts]
            bank_patches = _extract_patches(bank_imgs).reshape(-1, 384)
            bank_cache[key] = bank_patches / (np.linalg.norm(bank_patches, axis=1, keepdims=True) + 1e-8)
        bank_n = bank_cache[key]

        cand_patches = _extract_patches([_ts_to_image(window[:, c])])[0]
        cand_n = cand_patches / (np.linalg.norm(cand_patches, axis=1, keepdims=True) + 1e-8)
        dist = 1.0 - cand_n @ bank_n.T
        nn_dist = dist.min(axis=1)
        k10 = max(1, int(len(nn_dist) * 0.10))
        scores[c] = float(np.sort(nn_dist)[-k10:].mean())
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
    ax1.set_title("Heatmap: 38 channels, sorted by patch-KNN score (top row = most suspicious)", fontsize=7)

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

    entity_train, entity_test, bank_cache = {}, {}, {}
    rankings = {}
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
        rankings[seg_id] = get_channel_ranking(entity, train, window, bank_cache)
        print(f"  {seg_id}: patch-KNN top8={rankings[seg_id][:8]}", flush=True)

    if not execute:
        print("[STOP] 랭킹 계산 완료 (VLM 호출 없음). --run 플래그로 실행하세요.")
        return

    ckpts = [json.loads(p.read_text(encoding="utf-8")) for p in CKPT_PATHS]
    checkpoint_path = OUT_DIR / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    all_channels = list(range(N_CHANNELS))
    rows = []

    for s in selected:
        seg_id = s["segment_id"]
        entity, cs, ce = parse_seg_id(seg_id)
        train, test = entity_train[entity], entity_test[entity]
        center = (cs + ce) // 2
        s_ = max(0, min(len(test) - WIN, center - WIN // 2))
        window = test[s_:s_ + WIN]
        ranked = rankings[seg_id]
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
    print(f"\n평균 F1 (n={len(rows)}) = {mean_f1:.4f}  (참고: report20, n=39 기준 = 0.433)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="patch-KNN 랭킹만 계산, VLM 없음")
    ap.add_argument("--run", action="store_true", help="VLM 실행")
    args = ap.parse_args()
    run(execute=args.run)
