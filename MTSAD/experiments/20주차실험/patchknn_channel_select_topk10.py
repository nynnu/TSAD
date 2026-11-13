"""
patchknn_channel_select.py의 "sum" 집계를 "topk10"(256패치 중 상위 10%만 평균)
으로 바꾼 변형. 오늘 원인분석3에서 찾은 두 실패 패턴(①짧은 스파이크 ②224틱보다
긴 이상구간)이 둘 다 "sum이 소수의 튀는 패치를 희석시킨다"는 같은 메커니즘이었음
-- topk10은 그 소수의 튀는 패치만 남기고 평균 내므로, 이 두 패턴을 직접 겨냥한 수정.

나머지 방법(뱅크 60, 캘리브 30, train-only 캘리브레이션, z-score 표준화, alpha
0.1/0.05/0.01)은 patchknn_channel_select.py와 완전히 동일 -- 집계 방식 하나만
다르므로 두 결과를 그대로 비교 가능.

VLM 호출 없음, 로컬 DINOv2만 사용.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN

BASE = Path(__file__).resolve().parent
GT_SEGMENTS_PATH = BASE / "results_gt_channel_count" / "segments.json"
RESULTS_DIR = BASE / "results_patchknn_channel_select_topk10"
CACHE_DIR = RESULTS_DIR / "cache"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

N_BANK = 60
N_CALIB = 30
ALPHAS = [0.1, 0.05, 0.01]
BUCKETS = [(1, 3), (4, 8), (9, 15), (16, 38)]

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def _bank_and_calib_idx(n_windows):
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK + N_CALIB + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]
    return sorted(bank_idx), calib_idx


def bucket_of(k):
    for lo, hi in BUCKETS:
        if lo <= k <= hi:
            return f"{lo}-{hi}"
    return "?"


def get_or_build_channel_calib(entity, c, train):
    ent_cache = CACHE_DIR / entity
    ent_cache.mkdir(parents=True, exist_ok=True)
    stats_path = ent_cache / f"ch{c}_stats.json"

    windows = cm.get_windows(train[:, c])
    bank_idx, calib_idx = _bank_and_calib_idx(len(windows))
    bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
        ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
        sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
        calib_scores = sc["topk10"]  # (n_calib,)  <- sum 대신 topk10
        stats = {"mu": float(calib_scores.mean()), "sigma": float(calib_scores.std() + 1e-8)}
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return tr_cls, tr_patches, stats


def score_segment_channel(tr_cls, tr_patches, test_img):
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    return float(sc["topk10"][0])


def run():
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    print(f"GT 세그먼트 {len(segments)}개, N_BANK={N_BANK} N_CALIB={N_CALIB}, ALPHAS={ALPHAS}, 집계=topk10")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8")) if CHECKPOINT.exists() else {}

    entity_data = {}
    entity_channel_calib = {}

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

        gt_channels = [d - 1 for d in gt_dims]
        k = len(gt_channels)
        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        t0 = time.time()
        zs = {}
        for c in range(38):
            key = (entity, c)
            if key not in entity_channel_calib:
                entity_channel_calib[key] = get_or_build_channel_calib(entity, c, train)
            tr_cls, tr_patches, stats = entity_channel_calib[key]

            test_img = cm.ts_to_image_fast(test[s_dyn:e_dyn, c])
            seg_score = score_segment_channel(tr_cls, tr_patches, test_img)
            zs[c] = (seg_score - stats["mu"]) / stats["sigma"]

        gt_set = set(gt_channels)
        entry = {"entity": entity, "start": cs, "end": ce, "k": k, "bucket": bucket_of(k),
                 "gt_channels": gt_channels, "z_scores": zs}
        for alpha in ALPHAS:
            z_thr = norm.ppf(1 - alpha)
            sel = {c for c, z in zs.items() if z > z_thr}
            hit = len(sel & gt_set)
            entry[f"a{alpha}"] = {
                "selected": sorted(sel), "n_selected": len(sel), "hit": hit,
                "precision": hit / len(sel) if sel else None,
                "recall": hit / k if k else None,
            }
        checkpoint[seg_id] = entry
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        a = entry[f"a{ALPHAS[0]}"]
        print(f"[OK] {seg_id}: k={k} bucket={bucket_of(k)} | a{ALPHAS[0]}: n_sel={a['n_selected']} "
              f"hit={a['hit']} prec={a['precision']} recall={a['recall']} ({time.time()-t0:.1f}s)", flush=True)

    rows = list(checkpoint.values())
    print(f"\n{'='*70}\nPATCH-KNN topk10 적응형 채널선택 SUMMARY (n={len(rows)})\n{'='*70}")
    summary = {"n": len(rows), "n_bank": N_BANK, "n_calib": N_CALIB, "agg": "topk10", "by_alpha": {}}
    for alpha in ALPHAS:
        key = f"a{alpha}"
        precs = [r[key]["precision"] for r in rows if r[key]["precision"] is not None]
        recs = [r[key]["recall"] for r in rows if r[key]["recall"] is not None]
        n_sels = [r[key]["n_selected"] for r in rows]
        print(f"\n--- alpha={alpha} ---")
        print(f"전체 Precision = {np.mean(precs):.4f}  Recall = {np.mean(recs):.4f}  "
              f"평균 선택 채널 수 = {np.mean(n_sels):.1f} (범위 {min(n_sels)}~{max(n_sels)})")
        by_bucket = {}
        for lo, hi in BUCKETS:
            b = f"{lo}-{hi}"
            sub = [r for r in rows if r["bucket"] == b]
            if not sub:
                continue
            sp = [r[key]["precision"] for r in sub if r[key]["precision"] is not None]
            sr = [r[key]["recall"] for r in sub if r[key]["recall"] is not None]
            sn = [r[key]["n_selected"] for r in sub]
            by_bucket[b] = {"n": len(sub), "precision": float(np.mean(sp)) if sp else None,
                             "recall": float(np.mean(sr)) if sr else None, "mean_n_selected": float(np.mean(sn))}
            print(f"  k={b:6s} n={len(sub):>2}  P={np.mean(sp):.4f}  R={np.mean(sr):.4f}  평균선택개수={np.mean(sn):.1f}")
        summary["by_alpha"][key] = {
            "mean_precision": float(np.mean(precs)), "mean_recall": float(np.mean(recs)),
            "mean_n_selected": float(np.mean(n_sels)), "by_bucket": by_bucket,
        }

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
