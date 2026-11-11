"""
작은윈도우(56)와 diff 채널을 같은 15개 놓친 채널에 같이 적용해서, 합쳤을 때
(둘 중 하나라도 구제) 총 몇 개가 구제되는지 확인 (VLM 콜 없음).
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
from step1v3_dino_graph_smd import load_smd, _centered_window
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH
from stage1_small_window_pilot import SMALL_WIN
from stage1_diff_channel_pilot import score_diff_channel

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_small_window(entity, c, train, test, center):
    from stage1_small_window_pilot import bank_and_calib_idx_small
    windows_small = cm.get_windows(train[:, c], window_size=SMALL_WIN, step=14)
    bank_idx, calib_idx = bank_and_calib_idx_small(len(windows_small))
    bank_imgs = [cm.ts_to_image_fast(windows_small[i]) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    calib_imgs = [cm.ts_to_image_fast(windows_small[i]) for i in calib_idx]
    ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
    calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
    mu, sigma = float(calib_sc["sum"].mean()), float(calib_sc["sum"].std() + 1e-8)

    s_, e_ = _centered_window(len(test), center, SMALL_WIN)
    small_window = test[s_:e_, c]
    test_img = cm.ts_to_image_fast(small_window)
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
    print(f"대상 {len(targets)}개")

    entity_data = {}
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    small_only, diff_only, both, either = 0, 0, 0, 0
    for ti, (entity, c, sid, z_orig) in enumerate(targets):
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        center = (cs + ce) // 2
        from step1v3_dino_graph_smd import WIN
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        z_small = score_small_window(entity, c, train, test, center)
        z_diff = score_diff_channel(train[:, c], window[:, c])

        r_small = z_small > z_loose
        r_diff = z_diff > z_loose
        if r_small and r_diff:
            both += 1
        if r_small and not r_diff:
            small_only += 1
        if r_diff and not r_small:
            diff_only += 1
        if r_small or r_diff:
            either += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_small={z_small:.2f}({'O' if r_small else 'X'}) "
              f"z_diff={z_diff:.2f}({'O' if r_diff else 'X'})", flush=True)

    print(f"\n=== 15개 중: small만={small_only} diff만={diff_only} 둘다={both} "
          f"합쳐서구제(either)={either}/{len(targets)} ===")


if __name__ == "__main__":
    main()
