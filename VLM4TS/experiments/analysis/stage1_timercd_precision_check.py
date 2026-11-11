"""
TimeRCD Stage1 precision 확인 -- 15개 놓친 채널만 보면 15/15 구제였는데, z값이
수천 단위로 비정상적으로 커서(정상 DINOv2 z는 -2~5 범위) 계산 방식 문제(점수
분포가 극단적으로 치우쳐 robust sigma가 0에 가까워져 z가 인위적으로 폭발) 의심됨.
정상 채널도 전부 이런 극단값이 나오는지 확인 (VLM 콜 없음).

세그먼트 1개(machine-1-4_20193_20233, k(GT)=2 -- 작은윈도우/diff precision 체크
때와 동일, 비교 가능하게) 38채널 전부 TimeRCD로 스코어링.
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
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, N_CHANNELS, constant_channels

from time_rcd import TimeRCDDetector


def main():
    zs_cache = json.loads((BASE / "experiments" / "results_stage2_k4_adaptive" / "zs_cache.json").read_text(encoding="utf-8"))
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    sid = "machine-1-4_20193_20233"
    d = zs_cache[sid]
    entity = d["entity"]
    gt = set(d["gt"])
    zs_224 = {int(c): (v if v is not None else -1e9) for c, v in d["zs"].items()}
    seg = segments_by_id[sid]
    cs, ce = seg["start"], seg["end"]

    train, test = load_smd(entity)
    degenerate = constant_channels(train)

    print("TimeRCD(uni) 로딩...", flush=True)
    detector = TimeRCDDetector.from_pretrained(variant="uni", device="cpu")
    print("로딩 완료", flush=True)

    z_loose = norm.ppf(1 - 0.1)
    zs_rcd = {}
    raw_stats = {}
    for c in range(N_CHANNELS):
        if c in degenerate:
            zs_rcd[c] = -1e9
            continue
        scores = detector.predict(test[:, c:c + 1].astype(np.float32))
        cutoff = np.percentile(scores, 80)
        clean = scores[scores <= cutoff]
        if len(clean) < 10:
            clean = scores
        mu, sigma = float(clean.mean()), float(clean.std() + 1e-8)
        seg_score = float(scores[cs:ce + 1].mean())
        z = (seg_score - mu) / sigma
        zs_rcd[c] = z
        raw_stats[c] = (mu, sigma, seg_score, float(scores.mean()), float(scores.std()))
        print(f"  ch{c}: z224(DINOv2)={zs_224[c]:.2f} z_TimeRCD={z:.2f} "
              f"(mu={mu:.4f} sigma={sigma:.6f} seg_score={seg_score:.4f}) {'[GT]' if c in gt else ''}", flush=True)

    def prf(zs):
        pred = {c for c in range(N_CHANNELS) if zs[c] > z_loose}
        tp = len(pred & gt)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt) if gt else 0.0
        return p, r, len(pred)

    p224, r224, n224 = prf(zs_224)
    prcd, rrcd, nrcd = prf(zs_rcd)

    print(f"\n=== {sid} (k(GT)={len(gt)}) ===")
    print(f"224윈도우(DINOv2): precision={p224:.3f} recall={r224:.3f} n_selected={n224}")
    print(f"TimeRCD:           precision={prcd:.3f} recall={rrcd:.3f} n_selected={nrcd}")


if __name__ == "__main__":
    main()
