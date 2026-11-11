"""
Stage1 근본 재설계: train 고정 정상성 -> RCD식 "문맥 상대적" 정상성.

배경
----
TimeRCD 논문의 핵심 동기(representation conflict, Fig.2): "정상성은 패턴 자체의
속성이 아니라 문맥에 달렸다"는 것을 우리 구조가 정면으로 위반하고 있었다 --
Stage1 z-score도(train bank/calib), Stage2 calibration(gn_train)도 전부 "train
시절의 고정 정상 분포"와 비교한다. SMD는 시간이 지나며 정상 자체가 바뀌는
concept-drift 데이터라서, sanity-check에서 확인했듯 진짜 정상 구간도 train
범위를 크게 벗어나는 경우가 흔했다(한 윈도우에서 17/20 채널이 train max를 최대
2.6배 초과) -- 이게 TimeRCD가 경고한 representation conflict의 실제 사례.

이 실험은 patch-KNN bank/calib를 TRAIN이 아니라 후보 구간 근처의 TEST 데이터로
지어서("지역 은행", local bank), "이 채널이 자기 자신의 최근 문맥과 비교해서
얼마나 다른가"로 재정의한다. 리키지 방지: 후보 구간 자체 + 패딩은 은행에서 제외,
같은 엔티티의 다른 GT 세그먼트가 반경 안에 있으면 그것도 제외.

같은 15개 놓친 채널로 recall + 같은 precision-check 세그먼트로 precision, 둘 다
확인(한 번에 -- recall만 보고 낚이지 않기 위해).
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

cm.DEVICE = torch.device("cpu")

RADIUS = 8000  # 후보 중심 기준 좌우 반경(틱)
PAD = WIN  # 후보 구간 자체 주변 제외 버퍼
N_BANK, N_CALIB = 60, 30


def build_local_windows(entity_test_len, center, cs, ce, other_segments, radius=RADIUS):
    """후보 중심(center) 기준 [center-radius, center+radius] 범위에서, 후보구간
    자체(+PAD)와 같은 엔티티의 다른 GT세그먼트를 제외한 유효 구간(리스트 of (s,e))을 반환."""
    lo, hi = max(0, center - radius), min(entity_test_len, center + radius)
    exclude = [(max(0, cs - PAD), min(entity_test_len, ce + PAD))]
    for (ocs, oce) in other_segments:
        exclude.append((max(0, ocs - PAD), min(entity_test_len, oce + PAD)))
    exclude.sort()

    valid = []
    cur = lo
    for (es, ee) in exclude:
        if es > cur:
            valid.append((cur, min(es, hi)))
        cur = max(cur, ee)
        if cur >= hi:
            break
    if cur < hi:
        valid.append((cur, hi))
    return [(s, e) for s, e in valid if e - s >= WIN]


def score_with_local_bank(test, c, valid_ranges, window_for_test):
    windows = []
    for (s, e) in valid_ranges:
        seg = test[s:e, c]
        windows.extend(cm.get_windows(seg))
    if len(windows) < N_BANK + N_CALIB:
        return None  # 표본 부족

    rng = np.random.default_rng(0)
    idx = rng.choice(len(windows), size=N_BANK + N_CALIB, replace=False)
    bank_idx, calib_idx = idx[:N_BANK], idx[N_BANK:]

    bank_imgs = [cm.ts_to_image_fast(windows[i]) for i in bank_idx]
    tr_cls, tr_patches = cm.extract_dinov2(bank_imgs, multilayer=False)

    calib_imgs = [cm.ts_to_image_fast(windows[i]) for i in calib_idx]
    ca_cls, ca_patches = cm.extract_dinov2(calib_imgs, multilayer=False)
    calib_sc = cm.knn_patch_score(tr_patches, ca_patches, tr_cls, ca_cls)
    mu, sigma = float(calib_sc["sum"].mean()), float(calib_sc["sum"].std() + 1e-8)

    test_img = cm.ts_to_image_fast(window_for_test)
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
    print(f"파일럿 대상 {len(targets)}개 (동일한 15개)", flush=True)

    all_segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    segments_by_id = {f"{s['entity']}_{s['start']}_{s['end']}": s for s in all_segments}
    segments_by_entity = {}
    for s in all_segments:
        segments_by_entity.setdefault(s["entity"], []).append((s["start"], s["end"]))

    entity_data = {}
    print("\n=== Part 1: 놓친 채널 15개에서 지역-문맥 z-score ===")
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

        other_segs = [(a, b) for (a, b) in segments_by_entity[entity] if not (a == cs and b == ce)]
        valid_ranges = build_local_windows(len(test), center, cs, ce, other_segs)
        n_avail = sum(e - s for s, e in valid_ranges)
        z_local = score_with_local_bank(test, c, valid_ranges, window[:, c])

        if z_local is None:
            print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: 표본부족(가용틱={n_avail}), 스킵", flush=True)
            continue
        rescued_flag = z_local > z_loose
        if rescued_flag:
            rescued += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_train기준={z_orig:.3f} "
              f"z_지역문맥={z_local:.3f} {'구제!' if rescued_flag else ''}", flush=True)

    print(f"\nthreshold=z_loose 기준 구제: {rescued}/{len(targets)}")

    print("\n=== Part 2: 전체 38채널 precision 확인 (machine-1-4_20193_20233) ===")
    sid2 = "machine-1-4_20193_20233"
    d2 = zs_cache[sid2]
    entity2 = d2["entity"]
    gt2 = set(d2["gt"])
    seg2 = segments_by_id[sid2]
    cs2, ce2 = seg2["start"], seg2["end"]
    if entity2 not in entity_data:
        entity_data[entity2] = load_smd(entity2)
    train2, test2 = entity_data[entity2]
    degenerate2 = constant_channels(train2)
    center2 = (cs2 + ce2) // 2
    s2_, e2_ = _centered_window(len(test2), center2, WIN)
    window2 = test2[s2_:e2_]
    other_segs2 = [(a, b) for (a, b) in segments_by_entity[entity2] if not (a == cs2 and b == ce2)]
    valid_ranges2 = build_local_windows(len(test2), center2, cs2, ce2, other_segs2)

    zs_local2 = {}
    for c in range(N_CHANNELS):
        if c in degenerate2:
            zs_local2[c] = -1e9
            continue
        z = score_with_local_bank(test2, c, valid_ranges2, window2[:, c])
        zs_local2[c] = z if z is not None else -1e9
        print(f"  ch{c}: z_지역문맥={zs_local2[c]:.2f} {'[GT]' if c in gt2 else ''}", flush=True)

    z224 = {int(cc): (v if v is not None else -1e9) for cc, v in d2["zs"].items()}
    for label, zs in [("train기준(기존)", z224), ("지역문맥(신규)", zs_local2)]:
        pred = {c for c in range(N_CHANNELS) if zs.get(c, -1e9) > z_loose}
        tp = len(pred & gt2)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt2) if gt2 else 0.0
        print(f"{label}: precision={p:.3f} recall={r:.3f} n_selected={len(pred)}")


if __name__ == "__main__":
    main()
