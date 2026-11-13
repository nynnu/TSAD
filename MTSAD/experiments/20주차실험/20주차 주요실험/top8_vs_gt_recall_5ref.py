"""
production이 실제로 쓰는 5-reference DINOv2 스코어러(f1_track_pilot_intra_multiref.py,
unified_detect_diagnose_pilot.py의 Stage 1.5와 동일 로직)를 48개 GT 세그먼트
전체로 재현. top8_vs_gt_recall.py(1-ref 간소화 버전, Step0)와 동일한 평가 방식
(fixed top-8, k-구간별 P@8/R@8)으로 비교 가능하게 만듦.

production 코드는 세그먼트마다 5개 참조 임베딩을 매번 새로 계산하지만, 5개
참조 윈도우가 entity당(=train_len당) 항상 동일하므로 여기서는 entity당 한 번만
계산해서 캐싱 -- 결과는 production과 동일, 계산량만 절약.

VLM 호출 없음, 로컬 DINOv2만 사용.
"""

import json
import time
from pathlib import Path

import numpy as np

from step1v3_dino_graph_smd import N_CHANNELS, WIN, load_smd, _centered_window, _get_model, _device
from f1_track_pilot_intra_multiref import (
    static_ref_windows, render_single, normed_window, embed_batch, cosine_dist_batch, N_STATIC_REFS,
)

BASE = Path(__file__).resolve().parent
GT_SEGMENTS_PATH = BASE / "results_gt_channel_count" / "segments.json"
RESULTS_DIR = BASE / "results_top8_vs_gt_5ref"
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
    print(f"GT 세그먼트 {len(segments)}개 로드")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    model = _get_model()

    entity_data = {}
    entity_refs = {}  # entity -> {c: (5, D) ndarray}

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

        if entity not in entity_refs:
            print(f"  5-ref embeddings for {entity} (38채널 x 5장) ...", flush=True)
            t0 = time.time()
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            refs = {}
            for c in range(N_CHANNELS):
                imgs = [render_single(normed_window(train[:, c], s, e)) for s, e in ref_windows]
                refs[c] = embed_batch(imgs, model, _device)  # (5, D)
            entity_refs[entity] = refs
            print(f"  done in {time.time()-t0:.1f}s", flush=True)
        refs = entity_refs[entity]

        gt_channels = [d - 1 for d in gt_dims]
        k = len(gt_channels)
        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        t0 = time.time()
        scores = {}
        for c in range(N_CHANNELS):
            dyn_img = render_single(normed_window(test[:, c], s_dyn, e_dyn))
            dyn_emb = embed_batch([dyn_img], model, _device)  # (1, D)
            dists = [cosine_dist_batch(dyn_emb, ref)[0] for ref in refs[c]]
            scores[c] = float(np.mean(dists))
        ranked = sorted(scores, key=lambda c: -scores[c])

        top8 = set(ranked[:TOP_K])
        gt_set = set(gt_channels)
        hit = len(top8 & gt_set)

        checkpoint[seg_id] = {
            "entity": entity, "start": cs, "end": ce, "k": k, "bucket": bucket_of(k),
            "gt_channels": gt_channels, "top8": sorted(top8),
            "hit": hit, "precision_at_8": hit / TOP_K, "recall_at_8": hit / k if k else None,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} bucket={bucket_of(k)} hit={hit}/8 "
              f"P@8={hit/TOP_K:.2f} R@8={hit/k:.2f} ({time.time()-t0:.1f}s)", flush=True)

    rows = list(checkpoint.values())
    print(f"\n{'='*60}\nTOP-8 vs GT SUMMARY -- 5-ref production 스코어러 (n={len(rows)})\n{'='*60}")
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

    summary = {"n": len(rows), "top_k": TOP_K,
               "mean_precision_at_8": float(mean_p), "mean_recall_at_8": float(mean_r),
               "by_bucket": by_bucket}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
