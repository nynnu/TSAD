"""
Stage1 개선 시도 2/3: 멀티스케일 패치 풀링(논문 2x2/3x3 average pooling)이 "sum 기준
z-score조차 못 넘어 Stage1에서 완전히 놓친 GT 채널"을 구제하는지 파일럿 테스트.

101개 놓친 채널 중 15개 샘플에 대해서만, 원본 패치(k=1)와 2x2/3x3 풀링 패치
각각의 z-score를 계산해서 비교(VLM 콜 없음, DINOv2 forward만).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, CACHE_DIR, _bank_and_calib_idx,
)

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pool_patches(patches, k):
    """patches: (N, 256, D) -> reshape (N,16,16,D), average pool kxk stride1,
    flatten back to (N, (16-k+1)^2, D). k=1 = 원본 그대로."""
    if k == 1:
        return patches
    N, P, D = patches.shape
    side = int(np.sqrt(P))
    grid = patches.reshape(N, side, side, D)
    out_side = side - k + 1
    pooled = np.zeros((N, out_side, out_side, D), dtype=grid.dtype)
    for i in range(out_side):
        for j in range(out_side):
            pooled[:, i, j, :] = grid[:, i:i + k, j:j + k, :].mean(axis=(1, 2))
    return pooled.reshape(N, out_side * out_side, D)


def score_sum(tr_cls, tr_patches, te_cls, te_patches, k):
    tr_p = pool_patches(tr_patches, k)
    te_p = pool_patches(te_patches, k)
    sc = cm.knn_patch_score(tr_p, te_p, tr_cls, te_cls)
    return float(sc["sum"][0])


def get_bank(entity, c):
    ent_cache = CACHE_DIR / entity
    bank_cls_path = ent_cache / f"ch{c}_bank_cls.npy"
    bank_patches_path = ent_cache / f"ch{c}_bank_patches.npy"
    if not (bank_cls_path.exists() and bank_patches_path.exists()):
        return None
    return np.load(bank_cls_path), np.load(bank_patches_path)


def main():
    zs_cache = json.loads((BASE / "experiments" / "results_stage2_k4_adaptive" / "zs_cache.json").read_text(encoding="utf-8"))
    z_loose = norm.ppf(1 - 0.1)
    missed = []
    for sid, d in zs_cache.items():
        entity = d["entity"]
        zs = {int(c): (v if v is not None else -1e9) for c, v in d["zs"].items()}
        gt = set(d["gt"])
        for c in gt:
            if -1e8 < zs[c] <= z_loose:
                missed.append((entity, c, sid, zs[c]))
    targets = missed[:15]
    print(f"파일럿 대상 {len(targets)}개 (놓친 채널)")

    entity_data = {}
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    KS = [1, 2, 3]
    rescued = {k: 0 for k in KS}
    for entity, c, sid, z_sum in targets:
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        bank = get_bank(entity, c)
        if bank is None:
            continue
        tr_cls, tr_patches = bank

        windows = cm.get_windows(train[:, c])
        bank_idx, calib_idx = _bank_and_calib_idx(len(windows))
        calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
        ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)

        test_img = cm.ts_to_image_fast(window[:, c])
        te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)

        row = {"entity": entity, "ch": c, "seg": sid, "z_sum_k1(기존)": round(z_sum, 3)}
        for k in KS:
            calib_scores = np.array([
                score_sum(tr_cls, tr_patches,
                          ca_cls[i:i + 1], ca_patches[i:i + 1], k)
                for i in range(len(calib_idx))
            ])
            mu, sigma = calib_scores.mean(), calib_scores.std() + 1e-8
            test_score = score_sum(tr_cls, tr_patches, te_cls, te_patches, k)
            z = (test_score - mu) / sigma
            row[f"z_k{k}"] = round(float(z), 3)
            if k != 1 and z > z_loose:
                rescued[k] += 1
        print(row, flush=True)

    print(f"\n=== 15개 놓친 채널 중 각 스케일이 구제(z>loose={z_loose:.2f})한 개수 ===")
    for k in KS:
        print(f"k={k}: {rescued[k]}/{len(targets)}")


if __name__ == "__main__":
    main()
