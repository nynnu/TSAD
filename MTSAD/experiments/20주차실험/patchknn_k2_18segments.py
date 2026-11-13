"""
K2가 실제로 썼던 18개 세그먼트("환자") 그대로, 우리 patch-KNN residual sum
채널선택 방식을 돌려서 -- VLM 없이(Stage1만) K2의 실제 F1=0.327과 직접 비교.

patchknn_channel_select.py와 완전히 동일한 방법(뱅크60/캘리브30, train-only,
z-score, ViT-B final-layer, sum 집계) -- 대상 세그먼트만 우리 48개 대신
K2의 18개로 교체. entity가 12개 더 필요(machine-1-6/7/8, 2-2/5/7/9,
3-1/2/3/5/8) -- 이번에 mv_data/SMD/test에 28개 전부 받아놔서 가능해짐.

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
GT_PATH = BASE / "k2_18segments_gt.json"
K2_CKPT_PATH = BASE.parents[1] / "experiments" / "results_pilot_reduced_index" / "checkpoint_K2.json"
RESULTS_DIR = BASE / "results_patchknn_k2_18segments"
CACHE_DIR = BASE / "results_patchknn_channel_select" / "cache"  # 기존 48개 캐시 재사용(machine-1-1/2/5는 이미 있음)
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

N_BANK = 60
N_CALIB = 30
Z_HIGH = norm.ppf(1 - 0.01)
Z_LOW = 0.0
N_HIGH_MERGE_THRESHOLD = 8

cm.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def _bank_and_calib_idx(n_windows):
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK + N_CALIB + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB]
    return sorted(bank_idx), calib_idx


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
        calib_sums = sc["sum"]
        stats = {"mu": float(calib_sums.mean()), "sigma": float(calib_sums.std() + 1e-8)}
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return tr_cls, tr_patches, stats


def score_segment_channel(tr_cls, tr_patches, test_img):
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    return float(sc["sum"][0])


def f1_of(pred, gt):
    pred, gt = set(pred), set(gt)
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run():
    gt_map = json.loads(GT_PATH.read_text(encoding="utf-8"))
    k2_ckpt = json.loads(K2_CKPT_PATH.read_text(encoding="utf-8"))
    seg_ids = list(gt_map.keys())
    print(f"K2 세그먼트 {len(seg_ids)}개")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8")) if CHECKPOINT.exists() else {}

    entity_data = {}
    entity_channel_calib = {}

    for seg_id in seg_ids:
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}")
            continue

        parts = seg_id.split("_")
        entity = "_".join(parts[:-2])
        cs, ce = int(parts[-2]), int(parts[-1])
        gt_channels = gt_map[seg_id]
        k = len(gt_channels)

        if entity not in entity_data:
            train, test = load_smd(entity)
            entity_data[entity] = (train, test)
        train, test = entity_data[entity]

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

        high = {c for c, z in zs.items() if z > Z_HIGH}
        low = {c for c, z in zs.items() if z <= Z_LOW}
        border = set(zs) - high - low
        if len(high) <= N_HIGH_MERGE_THRESHOLD:
            overlay = border | high
            final_pred_no_vlm = high  # VLM 없이는 그래도 high만 최종답
        else:
            overlay = border
            final_pred_no_vlm = high

        gt_set = set(gt_channels)
        entry = {
            "entity": entity, "start": cs, "end": ce, "k": k, "gt_channels": gt_channels,
            "z_scores": zs, "high": sorted(high), "border": sorted(border), "low": sorted(low),
            "overlay_for_vlm": sorted(overlay),
            "k2_pred": k2_ckpt[seg_id]["pred"],
        }
        entry["f1_ours_no_vlm"] = f1_of(final_pred_no_vlm, gt_set)
        entry["f1_k2"] = f1_of(k2_ckpt[seg_id]["pred"], gt_set)
        checkpoint[seg_id] = entry
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {seg_id}: k={k}  high={len(high)} border={len(border)} low={len(low)}  "
              f"F1_ours(no-VLM)={entry['f1_ours_no_vlm']:.3f}  F1_K2={entry['f1_k2']:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    rows = list(checkpoint.values())
    f1_ours = np.mean([r["f1_ours_no_vlm"] for r in rows])
    f1_k2 = np.mean([r["f1_k2"] for r in rows])
    print(f"\n{'='*60}\n같은 18개 세그먼트 비교 (n={len(rows)})\n{'='*60}")
    print(f"우리(patch-KNN, VLM 없이 이상확정만) F1 = {f1_ours:.4f}")
    print(f"K2 (실제 VLM 사용)                    F1 = {f1_k2:.4f}")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "f1_ours_no_vlm": float(f1_ours), "f1_k2": float(f1_k2),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
