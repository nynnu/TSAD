"""
관계깨짐 신호 v2 -- Fisher z-검정 기반 통계적 유의성 + 다중 seed 평균(다중비교 완화).

v1(단순 |상관계수 차이| 최댓값) 문제: 224틱 표본 노이즈로 상관계수가 흔들려서
GT 채널이 비-GT 채널보다 오히려 낮은 순위를 받음(precision 붕괴, threshold=0.4에서
recall 0으로 날아감).

v2 변경점:
1. Fisher z-변환 + 검정통계량으로 "표본크기 대비 통계적으로 유의한 차이인지" 측정
   (단순 절댓값 차이 대신, SE=1/sqrt(n-3)로 정규화된 z-stat 사용 -- 노이즈가 큰
   케이스는 자동으로 낮은 점수를 받음)
2. seed 여러 개 중 최댓값이 아니라 "관련 seed들의 |z-stat| 평균" 사용
   (하나만 튀어도 걸리는 다중비교 취약성 완화)

같은 15개 놓친 채널 + 같은 precision 세그먼트로 재검증.
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


def fisher_z(r):
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))


def relational_breakdown_score_v2(window, c, train_corr, seed_channels, corr_thr=0.3):
    """관련 seed들과의 상관계수 변화를 Fisher z-검정통계량으로 측정, 평균 반환."""
    relevant = [s for s in seed_channels if s != c and abs(train_corr[c, s]) > corr_thr]
    if not relevant:
        return 0.0, 0
    n = window.shape[0]
    se = 1.0 / np.sqrt(max(n - 3, 1))
    zstats = []
    for s in relevant:
        now_corr = windowed_corr(window, c, s)
        z_now = fisher_z(now_corr)
        z_train = fisher_z(train_corr[c, s])
        zstat = abs(z_now - z_train) / se
        zstats.append(zstat)
    return float(np.mean(zstats)), len(relevant)


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

    print("\n=== Part 1: 놓친 채널 15개에서 관계깨짐 z-stat(v2) ===")
    all_scores = []
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

        score, n_rel = relational_breakdown_score_v2(window, c, corr, seed_channels)
        all_scores.append(score)
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_원본={z_orig:.3f} "
              f"관계z-stat(v2)={score:.3f} (관련seed={n_rel})", flush=True)

    print(f"\nscore 분포: min={min(all_scores):.2f} median={np.median(all_scores):.2f} max={max(all_scores):.2f}")
    for thr in [1.96, 2.5, 3.0]:
        n_res = sum(1 for s in all_scores if s > thr)
        print(f"threshold={thr}(z-stat, {'p<0.05' if thr==1.96 else ''}): 구제 {n_res}/{len(targets)}")

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
        score, n_rel = relational_breakdown_score_v2(window2, c, corr2, seed_channels2)
        scores2[c] = score
        print(f"  ch{c}: 관계z-stat(v2)={score:.3f} (관련seed={n_rel}) {'[GT]' if c in gt2 else ''}", flush=True)

    for thr in [1.96, 2.5, 3.0, 4.0]:
        pred = {c for c in range(N_CHANNELS) if scores2[c] > thr}
        tp = len(pred & gt2)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt2) if gt2 else 0.0
        print(f"threshold={thr}: precision={p:.3f} recall={r:.3f} n_selected={len(pred)}")


if __name__ == "__main__":
    main()
