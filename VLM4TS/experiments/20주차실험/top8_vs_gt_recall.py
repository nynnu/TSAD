"""
20주차실험: 우리가 고른 top-8 채널이 SMD 공식 GT(interpretation_label) 채널과
얼마나 겹치는지 측정.

방법: step1v3_dino_graph_smd.py의 Step0 스코어러(정적 train 기준 윈도우 vs 동적
세그먼트 윈도우, DINOv2 vits14 cosine distance, 채널당 line plot 1장)를 그대로
재사용 -- 새 스코어링 방식을 만들지 않음. 이전엔 9개(손선별, k=4-8만) 세그먼트만
했지만, 이번엔 실제 배포 interpretation_label 기반 48개 세그먼트(5개 entity)
전체로 확장.

평가: production 파이프라인(unified_detect_diagnose_pilot.py)이 실제로 쓰는
고정 top-8을 그대로 재현 -- k(=GT 채널 수)에 맞춘 adaptive recall@k가 아니라
precision@8 = |top8 ∩ GT| / 8, recall@8 = |top8 ∩ GT| / k.
k 구간별(1-3/4-8/9-15/16+, results_gt_channel_count/summary.json의 버킷과 동일)
로도 분해.

VLM 호출 없음, 로컬 DINOv2만 사용. 세그먼트 단위 체크포인트로 중단 시 재개 가능.
"""

import json
import time
from pathlib import Path

import numpy as np

from step1v3_dino_graph_smd import (
    N_CHANNELS, WIN, load_smd, _centered_window, embed_channel_window, cosine_dist,
)

BASE = Path(__file__).resolve().parent
GT_SEGMENTS_PATH = BASE / "results_gt_channel_count" / "segments.json"
RESULTS_DIR = BASE / "results_top8_vs_gt"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
TOP_K = 8

BUCKETS = [(1, 3), (4, 8), (9, 15), (16, 38)]


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def bucket_of(k):
    for lo, hi in BUCKETS:
        if lo <= k <= hi:
            return f"{lo}-{hi}"
    return "?"


def run():
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    print(f"GT 세그먼트 {len(segments)}개 로드 ({GT_SEGMENTS_PATH})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()

    entity_static = {}
    entity_data = {}

    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        gt_dims = seg["dims"]
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}")
            continue

        if entity not in entity_data:
            train, test = load_smd(entity)
            entity_data[entity] = (train, test)
        train, test = entity_data[entity]

        if entity not in entity_static:
            print(f"  Static reference embeddings for {entity} (38채널) ...", flush=True)
            t0 = time.time()
            s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
            entity_static[entity] = {c: embed_channel_window(train[:, c], s_static, e_static) for c in range(N_CHANNELS)}
            print(f"  done in {time.time()-t0:.1f}s", flush=True)
        static = entity_static[entity]

        gt_channels = [d - 1 for d in gt_dims]  # 1-indexed -> 0-indexed
        k = len(gt_channels)
        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        t0 = time.time()
        dynamic = {c: embed_channel_window(test[:, c], s_dyn, e_dyn) for c in range(N_CHANNELS)}
        step0 = {c: cosine_dist(static[c], dynamic[c]) for c in range(N_CHANNELS)}
        ranked = sorted(step0, key=lambda c: -step0[c])

        top8 = set(ranked[:TOP_K])
        gt_set = set(gt_channels)
        hit = len(top8 & gt_set)

        checkpoint[seg_id] = {
            "entity": entity, "start": cs, "end": ce, "k": k, "bucket": bucket_of(k),
            "gt_channels": gt_channels, "top8": sorted(top8),
            "hit": hit, "precision_at_8": hit / TOP_K, "recall_at_8": hit / k if k else None,
            "ranked_top15": ranked[:15],
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} bucket={bucket_of(k)} hit={hit}/8 "
              f"P@8={hit/TOP_K:.2f} R@8={hit/k:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # ---- summary ----
    rows = list(checkpoint.values())
    print(f"\n{'='*60}\nTOP-8 vs GT SUMMARY (n={len(rows)})\n{'='*60}")
    mean_p = np.mean([r["precision_at_8"] for r in rows])
    mean_r = np.mean([r["recall_at_8"] for r in rows if r["recall_at_8"] is not None])
    print(f"전체 Precision@8 = {mean_p:.4f}")
    print(f"전체 Recall@8    = {mean_r:.4f}")

    print("\nk 구간별:")
    by_bucket = {}
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}"
        sub = [r for r in rows if r["bucket"] == b]
        if not sub:
            continue
        p = np.mean([r["precision_at_8"] for r in sub])
        r_ = np.mean([r["recall_at_8"] for r in sub if r["recall_at_8"] is not None])
        by_bucket[b] = {"n": len(sub), "precision_at_8": float(p), "recall_at_8": float(r_)}
        print(f"  k={b:6s} n={len(sub):>2}  P@8={p:.4f}  R@8={r_:.4f}")

    summary = {
        "n": len(rows), "top_k": TOP_K,
        "mean_precision_at_8": float(mean_p), "mean_recall_at_8": float(mean_r),
        "by_bucket": by_bucket,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
