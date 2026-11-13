"""
Mahalanobis distance 기반 Stage1 -- residual-sum(DINOv2) 대신 Mahalanobis
contribution score로 z-score 3-tier 분류를 만들어서, VLM 파이프라인과
공정 비교(Mahalanobis+VLM vs residual-sum+VLM)가 가능하게 함.

방법: entity마다 train 전체로 mu, Sigma^-1 계산(baseline_mahalanobis.py 재사용).
train 슬라이딩 윈도우 30개(calib)에서 채널별 contribution 최댓값 분포(mu_c, sigma_c)
추정 -> 세그먼트의 채널별 contribution을 z = (score-mu_c)/sigma_c로 표준화.
이후 기존과 동일한 3-tier(High/Border/Low, alpha=0.01, z<=0.0) + 8개 규칙 적용.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from colab_multivariate_v2 import get_windows
from baseline_mahalanobis import fit_mahalanobis, channel_scores

BASE = Path(__file__).resolve().parent
SEGMENTS_PATH = BASE / "results_gt_channel_count" / "segments_val.json"
RESULTS_DIR = BASE / "results_mahalanobis_stage1_val40"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

N_CALIB = 30
Z_HIGH = norm.ppf(1 - 0.01)
Z_LOW = 0.0
N_HIGH_MERGE_THRESHOLD = 8


def calib_stats(train, mu, cov_inv):
    windows = get_windows(train)
    idx = np.linspace(0, len(windows) - 1, min(N_CALIB, len(windows))).astype(int)
    calib_scores = np.array([channel_scores(windows[i], mu, cov_inv) for i in idx])  # (N_CALIB, 38)
    return calib_scores.mean(axis=0), calib_scores.std(axis=0) + 1e-8


def run():
    segments = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8")) if CHECKPOINT.exists() else {}
    entity_cache = {}

    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            continue
        gt_channels = [d - 1 for d in seg["dims"]]

        if entity not in entity_cache:
            train, test = load_smd(entity)
            mu, cov_inv = fit_mahalanobis(train)
            cal_mu, cal_sigma = calib_stats(train, mu, cov_inv)
            entity_cache[entity] = (train, test, mu, cov_inv, cal_mu, cal_sigma)
        train, test, mu, cov_inv, cal_mu, cal_sigma = entity_cache[entity]

        center = (cs + ce) // 2
        ws, we = _centered_window(len(test), center, WIN)
        window = test[ws:we]
        scores = channel_scores(window, mu, cov_inv)
        zs = {c: float((scores[c] - cal_mu[c]) / cal_sigma[c]) for c in range(38)}

        high = sorted(c for c, z in zs.items() if z > Z_HIGH)
        low = sorted(c for c, z in zs.items() if z <= Z_LOW)
        border = sorted(set(zs) - set(high) - set(low))
        overlay = sorted(set(border) | set(high)) if len(high) <= N_HIGH_MERGE_THRESHOLD else border

        checkpoint[seg_id] = {
            "entity": entity, "start": cs, "end": ce, "k": len(gt_channels),
            "gt_channels": gt_channels, "z_scores": zs,
            "high": high, "border": border, "low": low, "overlay_for_vlm": overlay,
            "f1_ours_no_vlm": None,
        }
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {seg_id}: k={len(gt_channels)} high={len(high)} border={len(border)} low={len(low)}", flush=True)

    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
