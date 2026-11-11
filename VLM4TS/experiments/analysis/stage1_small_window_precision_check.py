"""
Stage1 작은 윈도우(56틱) precision 확인 -- 놓친 GT 채널만 보면 recall이 극적으로
좋아졌지만(9/15), 정상 채널까지 다 같이 민감해져서 precision이 무너지는 건
아닌지 전체 38채널로 확인 (VLM 콜 없음).

세그먼트 2개(GT 작은 것 1개 + 큰 것 1개)를 골라서, 38채널 전부 작은윈도우(56)로
스코어링 -> z>z_loose(alpha=0.1) 단순 임계값으로 선택했을 때 precision/recall을
기존 224윈도우(zs_cache에 이미 있음)와 비교.
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
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, N_CHANNELS, constant_channels
from stage1_small_window_pilot import SMALL_WIN, SMALL_STEP, bank_and_calib_idx_small

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_channel_small(entity, c, train, window):
    windows_small = cm.get_windows(train[:, c], window_size=SMALL_WIN, step=SMALL_STEP)
    bank_idx, calib_idx = bank_and_calib_idx_small(len(windows_small))
    bank_imgs = [cm.ts_to_image_fast(windows_small[i]) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    calib_imgs = [cm.ts_to_image_fast(windows_small[i]) for i in calib_idx]
    ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
    calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
    mu, sigma = float(calib_sc["sum"].mean()), float(calib_sc["sum"].std() + 1e-8)

    test_img = cm.ts_to_image_fast(window[:, c])
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    test_sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
    return (float(test_sc["sum"][0]) - mu) / sigma


def main():
    zs_cache = json.loads((BASE / "experiments" / "results_stage2_k4_adaptive" / "zs_cache.json").read_text(encoding="utf-8"))
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    # k(GT)가 작은 것 1개, 큰 것 1개 선택
    sids_sorted = sorted(zs_cache.keys(), key=lambda s: len(zs_cache[s]["gt"]))
    target_sids = [sids_sorted[len(sids_sorted) // 4], sids_sorted[-len(sids_sorted) // 4]]
    print("대상 세그먼트:", target_sids)

    z_loose = norm.ppf(1 - 0.1)
    entity_data = {}

    for sid in target_sids:
        d = zs_cache[sid]
        entity = d["entity"]
        gt = set(d["gt"])
        zs_224 = {int(c): (v if v is not None else -1e9) for c, v in d["zs"].items()}
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        degenerate = constant_channels(train)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, SMALL_WIN)
        window_small_full = test[s_:e_]

        zs_56 = {}
        for c in range(N_CHANNELS):
            if c in degenerate:
                zs_56[c] = -1e9
                continue
            zs_56[c] = score_channel_small(entity, c, train, window_small_full)
            print(f"  {sid} ch{c}: z224={zs_224[c]:.2f} z56={zs_56[c]:.2f} {'[GT]' if c in gt else ''}", flush=True)

        def prf(zs):
            pred = {c for c in range(N_CHANNELS) if zs[c] > z_loose}
            tp = len(pred & gt)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(gt) if gt else 0.0
            return p, r, len(pred)

        p224, r224, n224 = prf(zs_224)
        p56, r56, n56 = prf(zs_56)
        print(f"\n=== {sid} (k(GT)={len(gt)}) ===")
        print(f"224윈도우: precision={p224:.3f} recall={r224:.3f} n_selected={n224}")
        print(f"56윈도우:  precision={p56:.3f} recall={r56:.3f} n_selected={n56}")
        print()


if __name__ == "__main__":
    main()
