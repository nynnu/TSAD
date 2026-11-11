"""
Stage1 개선 시도 3/3: 작은 윈도우(짧은 시간구간을 224x224 이미지에 확대해서 그림)로
scoring하면 "sum 기준 z-score조차 못 넘어 Stage1에서 완전히 놓친 GT 채널"을
구제하는지 파일럿 테스트.

논문 부록: A3/A4에서 기본 윈도우(224)의 Max-F1(0.614/0.565)이 윈도우를 28로
줄이면 0.906/0.844로 폭등. "여러 점 이상이 큰 윈도우 안에서 뭉개진다"는 원인.

이건 앞선 두 시도(분위수 집계, 멀티스케일 풀링)와 달리 bank/calib를 완전히 새
크기(SMALL_WIN=56)로 다시 뽑아야 함(캐시 재사용 불가) -- VLM 콜은 없음.

101개 놓친 채널 중 같은 15개 샘플로 비교.
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
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMALL_WIN = 56
SMALL_STEP = 14
N_BANK_SMALL = 60
N_CALIB_SMALL = 30


def bank_and_calib_idx_small(n_windows):
    bank_idx = set(np.linspace(0, n_windows - 1, min(N_BANK_SMALL, n_windows)).astype(int).tolist())
    fine = np.linspace(0, n_windows - 1, min(N_BANK_SMALL + N_CALIB_SMALL + 20, n_windows)).astype(int)
    calib_idx = [i for i in fine.tolist() if i not in bank_idx][:N_CALIB_SMALL]
    return sorted(bank_idx), calib_idx


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

    rescued = 0
    for ti, (entity, c, sid, z_orig) in enumerate(targets):
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, SMALL_WIN)
        small_window = test[s_:e_, c]

        windows_small = cm.get_windows(train[:, c], window_size=SMALL_WIN, step=SMALL_STEP)
        bank_idx, calib_idx = bank_and_calib_idx_small(len(windows_small))
        bank_imgs = [cm.ts_to_image_fast(windows_small[i]) for i in bank_idx]
        tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

        calib_imgs = [cm.ts_to_image_fast(windows_small[i]) for i in calib_idx]
        ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
        calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
        mu, sigma = float(calib_sc["sum"].mean()), float(calib_sc["sum"].std() + 1e-8)

        test_img = cm.ts_to_image_fast(small_window)
        te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
        test_sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
        z_small = (float(test_sc["sum"][0]) - mu) / sigma

        rescued_flag = z_small > z_loose
        if rescued_flag:
            rescued += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_원래(224)={z_orig:.3f} "
              f"z_작은윈도우(56)={z_small:.3f} {'구제!' if rescued_flag else ''}", flush=True)

    print(f"\n=== 15개 놓친 채널 중 작은윈도우(56)가 구제한 개수: {rescued}/{len(targets)} "
          f"(z>loose={z_loose:.2f}) ===")


if __name__ == "__main__":
    main()
