"""
Stage1 새 구조 제안: "관계 깨짐(correlation breakdown)" 신호.

배경
----
지금까지 7가지 magnitude 축 방법(분위수집계/멀티스케일풀링/라인패치/diff채널/
주파수/작은윈도우/TimeRCD)이 전부 "그 채널 자체 값이 얼마나 벗어났는지"만 쟀다.
근데 놓친 GT 채널(101개, 89%)은 자기 값 자체가 안 튀는 경우가 대부분 -- GT는
관계형이라는 게 이미 검증됐으니(GT 채널쌍 상관계수 0.60 vs 무작위 0.19,
p<0.0001), "이 채널이 평소 파트너와 같이 움직였는데 지금 그 동조가 깨졌는지"를
직접 재는 게 다른 축의 신호일 수 있다.

방법: 후보 윈도우 안에서 채널쌍별 상관계수를 다시 계산해서, train 상관계수와
비교 -- "평소 이 채널과 강하게 연관됐던 채널들과의 관계가 지금 얼마나 깨졌는지"를
점수화. seed 채널(이미 z-score로 확정된 강한 신호)과의 관계 변화를 우선 본다.

같은 15개 놓친 채널로 recall 확인 + 정상채널 섞인 세그먼트로 precision도 같이 확인
(이번엔 recall만 보고 낚이지 않도록 둘 다 한 번에).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, N_CHANNELS, constant_channels


def windowed_corr(window, c1, c2):
    v1, v2 = window[:, c1], window[:, c2]
    if v1.std() < 1e-9 or v2.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(v1, v2)[0, 1])


def relational_breakdown_score(window, c, train_corr, seed_channels, corr_thr=0.3):
    """seed 채널들 중 train 기준 이 채널과 상관관계가 있는 것들만 골라서, 지금
    윈도우에서의 상관계수와 train 상관계수 차이(절댓값)의 최댓값을 반환.
    관련 seed가 하나도 없으면 0."""
    relevant = [s for s in seed_channels if s != c and abs(train_corr[c, s]) > corr_thr]
    if not relevant:
        return 0.0
    diffs = []
    for s in relevant:
        now_corr = windowed_corr(window, c, s)
        diffs.append(abs(now_corr - train_corr[c, s]))
    return max(diffs)


def main():
    zs_cache = json.loads((BASE / "experiments" / "results_stage2_k4_adaptive" / "zs_cache.json").read_text(encoding="utf-8"))
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    z_loose = norm.ppf(1 - 0.1)
    z_strict = norm.ppf(1 - 0.01)
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

    entity_data, entity_corr = {}, {}

    def get_corr(entity):
        if entity not in entity_corr:
            train, _ = entity_data[entity]
            entity_corr[entity] = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
        return entity_corr[entity]

    # === Part 1: 15개 놓친 채널 recall ===
    print("\n=== Part 1: 놓친 채널 15개에서 관계깨짐 점수 ===")
    rescued_count = 0
    for ti, (entity, c, sid, z_orig) in enumerate(targets):
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        corr = get_corr(entity)

        d = zs_cache[sid]
        zs = {int(cc): (v if v is not None else -1e9) for cc, v in d["zs"].items()}
        seed_channels = [cc for cc in range(N_CHANNELS) if zs.get(cc, -1e9) > z_strict]

        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        score = relational_breakdown_score(window, c, corr, seed_channels)
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_원본={z_orig:.3f} "
              f"관계깨짐점수={score:.3f} (seed 개수={len(seed_channels)})", flush=True)
        if score > 0.3:  # 초기 threshold, 뒤에서 분포 보고 조정
            rescued_count += 1

    print(f"\nthreshold=0.3 기준 구제: {rescued_count}/{len(targets)}")

    # === Part 2: 정상 채널 포함 precision 확인 (같은 세그먼트, 이전 파일럿들과 동일) ===
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
    corr2 = get_corr(entity2)
    degenerate2 = constant_channels(train2)

    zs2 = {int(cc): (v if v is not None else -1e9) for cc, v in d2["zs"].items()}
    seed_channels2 = [cc for cc in range(N_CHANNELS) if zs2.get(cc, -1e9) > z_strict]
    center2 = (cs2 + ce2) // 2
    s2_, e2_ = _centered_window(len(test2), center2, WIN)
    window2 = test2[s2_:e2_]

    scores2 = {}
    for c in range(N_CHANNELS):
        if c in degenerate2:
            scores2[c] = -1.0
            continue
        scores2[c] = relational_breakdown_score(window2, c, corr2, seed_channels2)
        print(f"  ch{c}: 관계깨짐점수={scores2[c]:.3f} {'[GT]' if c in gt2 else ''}", flush=True)

    for thr in [0.2, 0.3, 0.4, 0.5]:
        pred = {c for c in range(N_CHANNELS) if scores2[c] > thr}
        tp = len(pred & gt2)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt2) if gt2 else 0.0
        print(f"threshold={thr}: precision={p:.3f} recall={r:.3f} n_selected={len(pred)}")


if __name__ == "__main__":
    main()
