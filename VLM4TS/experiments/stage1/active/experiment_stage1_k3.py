"""
Stage1 K3: patch-KNN 전체시계열 이상탐지 스코어러 (ViT-S, final layer, K=1)

Stage2의 K2/K3(experiments/stage2/active/experiment_stage2_k3.py)가 채널 랭킹에
쓰는 것과 정확히 같은 스코어러를, "이 시간대가 이상인가" 탐지(detection) 과제에
그대로 적용한 버전이다. 채널마다 train에서 15개 참조 윈도우(뱅크)를 뽑고, test를
슬라이딩하며 DINOv2 patch 임베딩 최근접 거리(K=1, topk10 집계)로 점수를 매긴 뒤,
38채널 평균으로 하나의 점수곡선을 만들고 threshold sweep으로 최적 F1을 찾는다.

지표는 항상 둘 다 계산한다(interval-overlap F1 + point-wise Max-F1, 논문
VLM4TS의 다변량 확장 섹션이 실제로 쓰는 지표는 후자) -- report20 참고.

결과 (machine-1-1, 1개 entity):
  interval-MaxF1 = 0.6154
  point-MaxF1    = 0.3978

참고: 같은 machine-1-1에서 colab_multivariate_v2.py 기반 production 캐시
(ViT-B, 멀티레이어, residual, K=5)의 point-MaxF1 = 0.4217로 이게 더 높다.
ViT-S/K=1/residual없음 조합(K3)을 ViT-S 그대로 두고 멀티레이어+residual+K=5만
추가한 버전은 오히려 point-MaxF1=0.2558로 더 나빠짐 -- report20 5절 참고.
아직 5개 entity 중 1개(machine-1-1)만 검증된 상태.

사용법
------
  python experiment_stage1_k3.py --entity machine-1-1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from transformers import Dinov2Model
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[3]
OUT_DIR = BASE / "experiments" / "results_stage1_k3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SMD_DIR = BASE / "mv_data" / "SMD"

STRIDE, WIN, N_BANK = 56, 224, 15
N_CHANNELS = 38

_dino_model = None
_tfm = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _get_model():
    global _dino_model
    if _dino_model is None:
        _dino_model = Dinov2Model.from_pretrained("facebook/dinov2-small").eval()
    return _dino_model


def ts_to_image(window, size=WIN):
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


def extract_patches_batched(imgs, batch_size=64):
    model = _get_model()
    outs = []
    for i in range(0, len(imgs), batch_size):
        x = torch.stack([_tfm(im) for im in imgs[i:i + batch_size]])
        with torch.no_grad():
            out = model(pixel_values=x)
        outs.append(out.last_hidden_state[:, 1:].numpy())
    return np.concatenate(outs, axis=0)


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
    """interval-overlap F1 (겹치기만 하면 TP)."""
    if not gt_ivs:
        return 0.0
    TP = sum(1 for d in pred_ivs if any(_ov(d, a) for a in gt_ivs))
    FP = sum(1 for d in pred_ivs if not any(_ov(d, a) for a in gt_ivs))
    FN = sum(1 for a in gt_ivs if not any(_ov(a, d) for d in pred_ivs))
    p = TP / (TP + FP) if (TP + FP) else 0
    r = TP / (TP + FN) if (TP + FN) else 0
    return 2 * p * r / (p + r) if (p + r) else 0


def pt_f1(labels, pred):
    """point-wise F1 (VLM4TS 논문의 Max-F1과 같은 정의, threshold sweep으로 최댓값)."""
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


def score_entity(entity):
    t0 = time.time()
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    T_test = len(test)

    # 채널별 뱅크(train, N_BANK) 구축
    bank_by_ch = {}
    for c in range(N_CHANNELS):
        starts = np.linspace(0, len(train) - WIN, N_BANK).astype(int)
        imgs = [ts_to_image(train[s:s + WIN, c]) for s in starts]
        p = extract_patches_batched(imgs).reshape(-1, 384)
        bank_by_ch[c] = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)

    # test 슬라이딩 윈도우, 채널별 topk10 patch-KNN 점수
    starts = list(range(0, T_test - WIN + 1, STRIDE))
    n_win = len(starts)
    ch_scores = np.zeros((n_win, N_CHANNELS))
    for c in range(N_CHANNELS):
        imgs = [ts_to_image(test[s:s + WIN, c]) for s in starts]
        p = extract_patches_batched(imgs)
        pn = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
        bank_n = bank_by_ch[c]
        for i in range(n_win):
            dist = 1.0 - pn[i] @ bank_n.T
            nn_dist = dist.min(axis=1)
            k10 = max(1, int(len(nn_dist) * 0.10))
            ch_scores[i, c] = float(np.sort(nn_dist)[-k10:].mean())

    win_score = ch_scores.mean(axis=1)  # 38채널 평균으로 단일 점수곡선
    inter = win_to_ts(win_score, T_test)
    gt_ivs = get_intervals(labels)

    all_ws = np.array([inter[s:s + WIN].mean() for s in starts])
    best_iv_f1 = 0.0
    for alpha in [0.1, 0.01, 0.001]:
        mu, sig = all_ws.mean(), all_ws.std()
        if sig < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sig
        pred_ivs = get_intervals((inter > thr).astype(int))
        best_iv_f1 = max(best_iv_f1, eval_f1(gt_ivs, pred_ivs))

    best_pt_f1 = 0.0
    for q in np.linspace(0.80, 0.999, 50):
        thr = np.quantile(inter, q)
        pred = (inter > thr).astype(int)
        best_pt_f1 = max(best_pt_f1, pt_f1(labels, pred))

    elapsed = time.time() - t0
    np.save(OUT_DIR / f"{entity}_inter.npy", inter)
    result = {
        "n_gt": len(gt_ivs), "interval_maxf1": best_iv_f1, "point_maxf1": best_pt_f1,
        "elapsed_sec": elapsed,
    }
    print(f"[{entity}] interval-MaxF1={best_iv_f1:.4f}  point-MaxF1={best_pt_f1:.4f}  ({elapsed:.0f}s)", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="machine-1-1")
    args = ap.parse_args()

    result = score_entity(args.entity)
    out_path = OUT_DIR / "results.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    results[args.entity] = result
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n참고: production 캐시(ViT-B, 멀티레이어) machine-1-1 point-MaxF1=0.4217")
