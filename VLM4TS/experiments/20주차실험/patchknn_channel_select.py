"""
DINOv2 patch-KNN residual "sum" 점수로 top-8 고정 대신 채널을 적응적으로 선택.

방법 (채널 c, entity마다):
  1. train에서 정상 윈도우 뱅크(60개) 구성 -> DINOv2(vitb14) patch residual bank
  2. 별도 held-out train 윈도우(30개, calib) 각각의 "sum score"(256개 패치
     KNN거리의 합, colab_multivariate_v2.knn_patch_score의 "sum")를 구해서
     이 채널의 평소 sum 분포(mu, sigma)를 잼 (test 안 봄)
  3. 실제 세그먼트의 채널 c sum score를 z = (score-mu)/sigma로 표준화,
     alpha(0.1/0.05/0.01)에 대응하는 z-threshold를 넘으면 그 채널 선택
     (top-8처럼 개수 고정 안 함 -- 세그먼트마다 0~38개 가변)

주의(버전 1의 버그 수정): 패치 단위 "1개라도 넘으면 선택"은 256번 다중비교라
정상 채널도 거의 항상 선택되는 결함이 있었음 -- 이번엔 채널당 스칼라 점수
(sum) 하나로 z-score 표준화해서 이 문제를 피함.

평가: 선택된 채널 집합을 GT와 비교 (precision/recall, k-구간별), 실제로
몇 개씩 뽑혔는지 분포도 같이 봄.

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
RESULTS_DIR = BASE / "results_patchknn_channel_select"
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
    """Return (tr_cls, tr_patches, {"mu":..,"sigma":..}) for this channel's normal sum-score distribution."""
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
        calib_sums = sc["sum"]  # (n_calib,)
        stats = {"mu": float(calib_sums.mean()), "sigma": float(calib_sums.std() + 1e-8)}
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return tr_cls, tr_patches, stats


def score_segment_channel(tr_cls, tr_patches, test_img):
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    return float(sc["sum"][0])


def run():
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    print(f"GT 세그먼트 {len(segments)}개, N_BANK={N_BANK} N_CALIB={N_CALIB}, ALPHAS={ALPHAS}")

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
            seg_sum = score_segment_channel(tr_cls, tr_patches, test_img)
            zs[c] = (seg_sum - stats["mu"]) / stats["sigma"]

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
    print(f"\n{'='*70}\nPATCH-KNN sum 적응형 채널선택 SUMMARY (n={len(rows)})\n{'='*70}")
    summary = {"n": len(rows), "n_bank": N_BANK, "n_calib": N_CALIB, "by_alpha": {}}
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
