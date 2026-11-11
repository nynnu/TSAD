"""
Stage1 개선 시도 4: 1차 차분(diff) 신호를 병렬로 스코어링해서, "sum 기준 z-score조차
못 넘어 Stage1에서 완전히 놓친 GT 채널"을 구제하는지 확인.

레벨시프트(계단형 변화)는 원본 값보다 변화율(diff)에서 더 도드라진다는 가설.
원본 채널 파이프라인(bank/calib/test 전부 diff 신호로)을 그대로 재사용, VLM 콜 없음.

앞선 세 시도(분위수 집계, 멀티스케일 풀링, 작은윈도우)와 정확히 같은 15개 샘플로
비교해서 공정하게 비교.
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
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, _bank_and_calib_idx

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_diff_channel(train_ch, window_ch):
    """train_ch, window_ch: 1차원 원본 채널 시계열(diff 하기 전). 내부에서 diff 계산
    후 기존 bank/calib/test 파이프라인 그대로 적용."""
    train_diff = np.diff(train_ch)
    window_diff = np.diff(window_ch)

    windows = cm.get_windows(train_diff)  # 기본 WINDOW_SIZE/STEP 그대로
    bank_idx, calib_idx = _bank_and_calib_idx(len(windows))
    bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
    ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
    calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
    mu, sigma = float(calib_sc["sum"].mean()), float(calib_sc["sum"].std() + 1e-8)

    test_img = cm.ts_to_image_fast(window_diff)
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    test_sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    return (float(test_sc["sum"][0]) - mu) / sigma


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
    print(f"파일럿 대상 {len(targets)}개 (앞선 3개 시도와 동일한 15개, 공정비교)")

    entity_data = {}
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    rescued = 0
    combined_rescued = 0
    for ti, (entity, c, sid, z_orig) in enumerate(targets):
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        z_diff = score_diff_channel(train[:, c], window[:, c])
        rescued_flag = z_diff > z_loose
        combined_flag = max(z_orig, z_diff) > z_loose
        if rescued_flag:
            rescued += 1
        if combined_flag:
            combined_rescued += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_원본={z_orig:.3f} "
              f"z_diff={z_diff:.3f} {'구제!' if rescued_flag else ''}", flush=True)

    print(f"\n=== 15개 놓친 채널 중 diff 채널이 구제한 개수: {rescued}/{len(targets)} "
          f"(z>loose={z_loose:.2f}) ===")
    print(f"=== 원본 OR diff 중 하나라도 넘으면(combined): {combined_rescued}/{len(targets)} ===")


if __name__ == "__main__":
    main()
