"""
지역-문맥 Stage1 버그 수정판.

수정 1: sigma 폭주 방지 -- calib 점수의 분포가 그 채널의 스코어 스케일 대비 너무
좁으면(sigma가 평균의 1% 미만) 이 채널을 "판별 불가"로 처리(z=0, 후보에서 자동
제외되게). 기존 "+1e-8"만으로는 나누기가 터짐(z=17억 같은 값 확인됨).

수정 2: 지역 은행 robust filtering -- 지역 문맥 윈도우들 중 상위 20%(자체적으로
튀어 보이는 것들)를 먼저 거르고, 나머지 80%로만 은행/calib를 지음(v16의 stage1
robust threshold-fitting과 동일 원리). 근처에 라벨 안 된 잔잔한 이상이 섞여
있어도 오염을 줄임 -- TimeRCD 논문 Limitations가 지적한 "근처 이상치가 지역
정상기준에 흡수되는" 문제에 대한 대응.

machine-1-2_15540_15605(직전 검증에서 지역문맥 recall이 0.5로 떨어졌던 세그먼트)로
재검증.
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
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, N_CHANNELS, constant_channels
from stage1_local_context_pilot import build_local_windows, RADIUS

cm.DEVICE = torch.device("cpu")
N_BANK, N_CALIB = 60, 30


def score_with_local_bank_v2(test, c, valid_ranges, window_for_test, robust_pct=20):
    windows = []
    for (s, e) in valid_ranges:
        seg = test[s:e, c]
        windows.extend(cm.get_windows(seg))
    if len(windows) < N_BANK + N_CALIB + 20:
        return None, "표본부족"

    # 1차 스크리닝(가벼움, DINOv2 안 씀): 각 윈도우의 원시값 표준편차로 "자체적으로
    # 튀는" 상위 robust_pct%를 거칠게 제외 -- DINOv2 self-KNN 스크리닝은 CPU에서
    # 감당 안 될 만큼 느려서(28160x28160 규모 매트릭스) 값 기반 프록시로 대체.
    raw_std = np.array([w.std() for w in windows])
    cutoff = np.percentile(raw_std, 100 - robust_pct)
    clean_local_idx = [i for i in range(len(windows)) if raw_std[i] <= cutoff]
    if len(clean_local_idx) < N_BANK + N_CALIB:
        clean_local_idx = list(range(len(windows)))

    rng = np.random.default_rng(0)
    idx = np.array(clean_local_idx)
    rng.shuffle(idx)
    bank_idx, calib_idx = idx[:N_BANK], idx[N_BANK:N_BANK + N_CALIB]

    bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
    ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
    calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)["sum"]
    mu, sigma_raw = float(calib_sc.mean()), float(calib_sc.std())

    # 수정1: sigma 폭주 방지 -- mu 대비 sigma가 너무 작으면(1% 미만) 판별불가
    min_sigma = max(abs(mu) * 0.01, 1e-6)
    if sigma_raw < min_sigma:
        return 0.0, f"판별불가(sigma폭주 방지, sigma={sigma_raw:.2e})"
    sigma = sigma_raw

    test_img = cm.ts_to_image_fast(window_for_test)
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    test_sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)["sum"]
    z = (float(test_sc[0]) - mu) / sigma
    return z, "ok"


def main():
    zs_cache = json.loads((BASE / "experiments" / "results_stage2_k4_adaptive" / "zs_cache.json").read_text(encoding="utf-8"))
    z_loose = norm.ppf(1 - 0.1)
    all_segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    segments_by_id = {f"{s['entity']}_{s['start']}_{s['end']}": s for s in all_segments}
    segments_by_entity = {}
    for s in all_segments:
        segments_by_entity.setdefault(s["entity"], []).append((s["start"], s["end"]))

    sid = "machine-1-2_15540_15605"
    d = zs_cache[sid]
    entity = d["entity"]
    gt = set(d["gt"])
    z_train_raw = {int(c): (v if v is not None else -1e9) for c, v in d["zs"].items()}
    # 참고: train z도 같은 버그(폭주)가 있을 수 있으니 비정상치는 표시만
    seg = segments_by_id[sid]
    cs, ce = seg["start"], seg["end"]

    train, test = load_smd(entity)
    degenerate = constant_channels(train)
    center = (cs + ce) // 2
    s_, e_ = _centered_window(len(test), center, WIN)
    window = test[s_:e_]
    other_segs = [(a, b) for (a, b) in segments_by_entity[entity] if not (a == cs and b == ce)]
    valid_ranges = build_local_windows(len(test), center, cs, ce, other_segs)

    print(f"=== {sid} (k(GT)={len(gt)}) v2(버그수정판) ===", flush=True)
    z_local_v2 = {}
    for c in range(N_CHANNELS):
        if c in degenerate:
            z_local_v2[c] = -1e9
            continue
        z, status = score_with_local_bank_v2(test, c, valid_ranges, window[:, c])
        z_local_v2[c] = z if z is not None else -1e9
        z_train_note = z_train_raw.get(c, -1e9)
        train_note = f"{z_train_note:.2f}" if abs(z_train_note) < 1e6 else f"{z_train_note:.2e}(폭주의심)"
        print(f"  ch{c}: train={train_note} local_v2={z_local_v2[c]:.2f} [{status}] {'[GT]' if c in gt else ''}", flush=True)

    def prf(pred):
        tp = len(pred & gt)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt) if gt else 0.0
        return p, r, len(pred)

    pred_train = {c for c in range(N_CHANNELS) if abs(z_train_raw.get(c, -1e9)) < 1e6 and z_train_raw.get(c, -1e9) > z_loose}
    pred_local = {c for c in range(N_CHANNELS) if z_local_v2.get(c, -1e9) > z_loose}
    pred_or = pred_train | pred_local
    pred_and = pred_train & pred_local

    for label, pred in [("train기준(폭주필터링)", pred_train), ("지역문맥v2(수정판)", pred_local),
                         ("OR", pred_or), ("AND", pred_and)]:
        p, r, n = prf(pred)
        print(f"{label}: precision={p:.3f} recall={r:.3f} n={n}")


if __name__ == "__main__":
    main()
