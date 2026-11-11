"""
Stage1 개선 시도: DINOv2는 학습 막바지에 518x518 고해상도로도 훈련됐다(논문 6.6절,
"Adapting the resolution"). 지금까지 우리는 항상 224x224로 렌더링했는데, 같은
224틱 시간축을 유지한 채(작은윈도우처럼 시간을 줄이는 게 아니라) 픽셀 해상도만
518로 올리면 패치 그리드가 16x16(256)->37x37(1369)로 훨씬 촘촘해져서 미세한
지역 신호를 더 잘 잡을 수 있는지 확인.

dinov2_transform이 Resize((224,224))로 고정돼있어 별도 transform 필요.
사전 확인: DINOv2 hub 모델이 518 입력을 위치임베딩 보간으로 정상 처리함(1369 패치
확인됨).

같은 15개 놓친 채널로 비교(VLM 콜 없음).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from torchvision import transforms

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, _bank_and_calib_idx

cm.DEVICE = torch.device("cpu")

HIRES = 518
hires_transform = transforms.Compose([
    transforms.Resize((HIRES, HIRES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_dinov2_hires(images):
    model = cm.get_model()
    with torch.no_grad():
        batch = torch.stack([hires_transform(img) for img in images])
        out = model.forward_features(batch)
        return out["x_norm_clstoken"].cpu().numpy(), out["x_norm_patchtokens"].cpu().numpy()


def score_hires(train_ch, window_ch):
    windows = cm.get_windows(train_ch)  # 시간축은 그대로(224틱), 해상도만 올림
    bank_idx, calib_idx = _bank_and_calib_idx(len(windows))
    bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
    tr_cls, tr_patches = extract_dinov2_hires(bank_imgs)

    calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
    ca_cls, ca_patches = extract_dinov2_hires(calib_imgs)
    calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
    mu, sigma = float(calib_sc["sum"].mean()), float(calib_sc["sum"].std() + 1e-8)

    test_img = cm.ts_to_image_fast(window_ch)
    te_cls, te_patches = extract_dinov2_hires([test_img])
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
    print(f"파일럿 대상 {len(targets)}개 (동일한 15개), 해상도={HIRES} (패치 1369개)")

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
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        z_hi = score_hires(train[:, c], window[:, c])
        rescued_flag = z_hi > z_loose
        if rescued_flag:
            rescued += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_224px={z_orig:.3f} "
              f"z_518px={z_hi:.3f} {'구제!' if rescued_flag else ''}", flush=True)

    print(f"\n=== 15개 놓친 채널 중 고해상도(518)가 구제한 개수: {rescued}/{len(targets)} "
          f"(z>loose={z_loose:.2f}) ===")


if __name__ == "__main__":
    main()
