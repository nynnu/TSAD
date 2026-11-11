"""
Stage1 대체 후보: TimeRCD(제로샷 이상탐지 전용 파운데이션 모델)로 DINOv2 patch-KNN을
대체했을 때, "sum 기준 z-score조차 못 넘어 Stage1에서 완전히 놓친 GT 채널"을
구제하는지 확인 (VLM 콜 없음, TimeRCD도 파인튜닝 없이 제로샷 그대로 사용).

TimeRCD는 다변량 입력(T,C)을 줘도 출력이 (T,) 하나뿐(전 채널 통합)이라, 채널별
점수를 얻으려면 단변량(uni) variant로 채널마다 개별 실행해야 한다 -- 이게 우리
DINOv2 스코어러가 채널마다 독립적으로 도는 것과 구조적으로 동일해서 공정 비교 가능.

TimeRCD predict()는 train 참조 없이 test 시퀀스 자체만 보고(RCD: 문맥 상대적 판단)
전체 길이의 이상확률(0~1)을 준다. 같은 채널의 test 전체 점수 분포에서 상위 20%
(이상 가능성 높은 구간)를 제외한 나머지로 robust mu/sigma를 잡고, GT 세그먼트
구간의 평균 점수를 z-표준화 -- DINOv2 파일럿과 동일한 방법론으로 공정 비교.

같은 15개 놓친 채널(분위수집계/멀티스케일풀링/라인패치/diff/주파수/작은윈도우
파일럿과 전부 동일한 대상)로 비교.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
from step1v3_dino_graph_smd import load_smd
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH

from time_rcd import TimeRCDDetector


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
    print(f"파일럿 대상 {len(targets)}개 (기존 시도들과 동일한 15개)", flush=True)

    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    print("TimeRCD(uni) 로딩...", flush=True)
    detector = TimeRCDDetector.from_pretrained(variant="uni", device="cpu")
    print("로딩 완료", flush=True)

    entity_data = {}
    score_cache = {}  # (entity, ch) -> full-length score array (재사용, 같은 채널 중복 있음)
    rescued = 0
    for ti, (entity, c, sid, z_orig) in enumerate(targets):
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        key = (entity, c)
        if key not in score_cache:
            score_cache[key] = detector.predict(test[:, c:c + 1].astype(np.float32))
        scores = score_cache[key]

        cutoff = np.percentile(scores, 80)
        clean = scores[scores <= cutoff]
        if len(clean) < 10:
            clean = scores
        mu, sigma = float(clean.mean()), float(clean.std() + 1e-8)

        seg_score = float(scores[cs:ce + 1].mean())
        z_rcd = (seg_score - mu) / sigma

        rescued_flag = z_rcd > z_loose
        if rescued_flag:
            rescued += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_원본(DINOv2)={z_orig:.3f} "
              f"z_TimeRCD={z_rcd:.3f} {'구제!' if rescued_flag else ''}", flush=True)

    print(f"\n=== 15개 놓친 채널 중 TimeRCD가 구제한 개수: {rescued}/{len(targets)} "
          f"(z>loose={z_loose:.2f}) ===")


if __name__ == "__main__":
    main()
