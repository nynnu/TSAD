"""
동일 환경(같은 세그먼트, 같은 지표)에서 돌려보는 순수 통계 baseline.

방법: 채널을 이미지로 렌더링하거나 DINOv2를 쓰지 않고, entity의 train 데이터로
38채널 공분산 행렬(Sigma)을 추정한 뒤, 각 시점의 Mahalanobis distance
D^2(t) = (x(t)-mu)^T Sigma^-1 (x(t)-mu) 를 채널별로 분해(contribution decomposition,
SPC/MSPC 문헌의 표준 기법 -- contribution_i(t) = (x_i-mu_i) * [Sigma^-1(x-mu)]_i,
합하면 D^2(t)와 같아짐)해서 채널별 원인 기여도 점수를 얻는다.

세그먼트(224틱 창) 안에서 각 채널의 contribution 최댓값을 그 채널의 점수로 사용해
랭킹하고, HitRate@100%(top-k, k=|GT|)와 top-8 강제선택(P@8/R@8)을 계산한다.
"""
import json
from pathlib import Path

import numpy as np

from step1v3_dino_graph_smd import load_smd, _centered_window, WIN

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results_baseline_mahalanobis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fit_mahalanobis(train):
    mu = train.mean(axis=0)
    cov = np.cov(train, rowvar=False)
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6 * np.trace(cov) / cov.shape[0]
    cov_inv = np.linalg.inv(cov_reg)
    return mu, cov_inv


def channel_scores(window, mu, cov_inv):
    resid = window - mu  # (T, C)
    contrib = resid * (resid @ cov_inv.T)  # (T, C), row t sums to D^2(t)
    return contrib.max(axis=0)  # (C,) -- 채널별 창 내 최대 기여도


def f1_of(pred, gt):
    pred, gt = set(pred), set(gt)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run(segments_path, out_name):
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    entity_cache = {}
    rows = []
    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        gt_channels = [d - 1 for d in seg["dims"]]
        k = len(gt_channels)

        if entity not in entity_cache:
            train, test = load_smd(entity)
            mu, cov_inv = fit_mahalanobis(train)
            entity_cache[entity] = (train, test, mu, cov_inv)
        train, test, mu, cov_inv = entity_cache[entity]

        center = (cs + ce) // 2
        ws, we = _centered_window(len(test), center, WIN)
        window = test[ws:we]

        scores = channel_scores(window, mu, cov_inv)
        ranked = np.argsort(-scores)
        gt_set = set(gt_channels)

        top_k = set(ranked[:k].tolist()) if k > 0 else set()
        hitrate_100 = len(top_k & gt_set) / k if k else None

        top8 = set(ranked[:8].tolist())
        hit8 = len(top8 & gt_set)
        p8 = hit8 / 8
        r8 = hit8 / k if k else None

        rows.append({
            "entity": entity, "start": cs, "end": ce, "k": k,
            "gt_channels": gt_channels, "ranked": ranked.tolist(),
            "hitrate_100": hitrate_100, "p8": p8, "r8": r8,
        })

    hitrates = [r["hitrate_100"] for r in rows if r["hitrate_100"] is not None]
    p8s = [r["p8"] for r in rows]
    r8s = [r["r8"] for r in rows if r["r8"] is not None]
    summary = {
        "n": len(rows),
        "hitrate_100": float(np.mean(hitrates)),
        "mean_p8": float(np.mean(p8s)),
        "mean_r8": float(np.mean(r8s)),
    }
    print(f"[{out_name}] n={summary['n']}  HitRate@100%={summary['hitrate_100']:.4f}  "
          f"P@8={summary['mean_p8']:.4f}  R@8={summary['mean_r8']:.4f}")

    (RESULTS_DIR / f"{out_name}_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS_DIR / f"{out_name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run(BASE / "results_gt_channel_count" / "segments.json", "tuning48")
    run(BASE / "results_gt_channel_count" / "segments_val.json", "heldout40")
