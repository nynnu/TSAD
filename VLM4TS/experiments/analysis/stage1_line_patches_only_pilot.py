"""
Stage1 개선 시도: "sum" 집계가 256개 패치 전부(대부분 흰 배경, 정보 없음)를 더해서
신호를 희석시키는 게 아닌지 확인. VLM4TS 논문 부록(Fig.7b 캡션)이 "whitespace
patch가 균일하게 낮은 dissimilarity를 준다"고 명시.

선이 실제로 지나가는 패치만 골라서(흰 배경 아닌 패치) sum 하면 놓친 GT 채널이
구제되는지 확인 (VLM 콜 없음, 같은 15개 대상 재사용).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, _bank_and_calib_idx, CACHE_DIR


def get_bank(entity, c):
    ent_cache = CACHE_DIR / entity
    bank_cls_path = ent_cache / f"ch{c}_bank_cls.npy"
    bank_patches_path = ent_cache / f"ch{c}_bank_patches.npy"
    if not (bank_cls_path.exists() and bank_patches_path.exists()):
        return None
    return np.load(bank_cls_path), np.load(bank_patches_path)


def line_patch_mask(window_1d, patch_size=14, img_size=224, n_side=16):
    """ts_to_image_fast와 동일한 렌더링 로직으로 라인이 지나가는 픽셀 위치를 구하고,
    각 14x14 패치 안에 라인 픽셀이 하나라도 있으면 True."""
    img = cm.ts_to_image_fast(window_1d)
    arr = np.array(img.convert("L"))  # (224,224), 흰=255, 선(검정)=0 근처
    is_line = arr < 200  # 흰 배경(255) 아닌 픽셀
    mask = np.zeros((n_side, n_side), dtype=bool)
    for i in range(n_side):
        for j in range(n_side):
            block = is_line[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            mask[i, j] = block.any()
    return mask.reshape(-1)  # (256,)


def score_line_patches_only(tr_cls, tr_patches, train_ch, window_ch):
    windows = cm.get_windows(train_ch)
    bank_idx, calib_idx = _bank_and_calib_idx(len(windows))

    def knn_line_sum(test_img, test_ch_window):
        te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
        sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, return_win=True)
        knn_win = sc["knn_win"][0]  # (256,)
        mask = line_patch_mask(test_ch_window)
        if not mask.any():
            return float(knn_win.sum())
        return float(knn_win[mask].sum())

    calib_imgs_windows = [windows[i] for i in calib_idx]
    calib_scores = np.array([knn_line_sum(cm.ts_to_image_fast(w), w) for w in calib_imgs_windows])
    mu, sigma = calib_scores.mean(), calib_scores.std() + 1e-8

    test_img = cm.ts_to_image_fast(window_ch)
    test_score = knn_line_sum(test_img, window_ch)
    return (test_score - mu) / sigma


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
    print(f"파일럿 대상 {len(targets)}개 (앞선 시도들과 동일한 15개)")

    entity_data = {}
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    rescued = 0
    n_line_patches_list = []
    for ti, (entity, c, sid, z_orig) in enumerate(targets):
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

        mask = line_patch_mask(window[:, c])
        n_line_patches_list.append(mask.sum())

        z_line = score_line_patches_only(tr_cls, tr_patches, train[:, c], window[:, c])
        rescued_flag = z_line > z_loose
        if rescued_flag:
            rescued += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_원본(256패치)={z_orig:.3f} "
              f"z_라인패치만({mask.sum()}개)={z_line:.3f} {'구제!' if rescued_flag else ''}", flush=True)

    print(f"\n평균 라인패치 수 = {np.mean(n_line_patches_list):.1f} / 256")
    print(f"=== 15개 놓친 채널 중 라인패치만 sum이 구제한 개수: {rescued}/{len(targets)} "
          f"(z>loose={z_loose:.2f}) ===")


if __name__ == "__main__":
    main()
