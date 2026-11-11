"""
지역-문맥(RCD식) Stage1의 OR/AND 2단계 구조가 다른 세그먼트에서도 재현되는지
검증. GT 크기가 다른 3개 세그먼트(k=2,8,14), 다른 엔티티들로 확대.

각 세그먼트에서 38채널 전부 지역-문맥 z-score 계산 -> train기준/지역문맥/OR/AND
precision-recall 비교. (VLM 콜 없음)
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
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, N_CHANNELS, constant_channels
from stage1_local_context_pilot import build_local_windows, score_with_local_bank

import colab_multivariate_v2 as cm
cm.DEVICE = torch.device("cpu")

TARGET_SEGS = [
    "machine-1-2_15540_15605",  # k=2
    "machine-1-1_18071_18528",  # k=8
    "machine-1-5_12765_12786",  # k=14
]


def main():
    zs_cache = json.loads((BASE / "experiments" / "results_stage2_k4_adaptive" / "zs_cache.json").read_text(encoding="utf-8"))
    z_loose = norm.ppf(1 - 0.1)
    all_segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    segments_by_id = {f"{s['entity']}_{s['start']}_{s['end']}": s for s in all_segments}
    segments_by_entity = {}
    for s in all_segments:
        segments_by_entity.setdefault(s["entity"], []).append((s["start"], s["end"]))

    entity_data = {}
    summary = []

    for sid in TARGET_SEGS:
        d = zs_cache[sid]
        entity = d["entity"]
        gt = set(d["gt"])
        z_train = {int(c): (v if v is not None else -1e9) for c, v in d["zs"].items()}
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        degenerate = constant_channels(train)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]
        other_segs = [(a, b) for (a, b) in segments_by_entity[entity] if not (a == cs and b == ce)]
        valid_ranges = build_local_windows(len(test), center, cs, ce, other_segs)

        print(f"\n=== {sid} (entity={entity}, k(GT)={len(gt)}) ===", flush=True)
        z_local = {}
        for c in range(N_CHANNELS):
            if c in degenerate:
                z_local[c] = -1e9
                continue
            z = score_with_local_bank(test, c, valid_ranges, window[:, c])
            z_local[c] = z if z is not None else -1e9
            print(f"  ch{c}: train={z_train.get(c,-1e9):.2f} local={z_local[c]:.2f} {'[GT]' if c in gt else ''}", flush=True)

        def prf(pred):
            tp = len(pred & gt)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(gt) if gt else 0.0
            return p, r, len(pred)

        pred_train = {c for c in range(N_CHANNELS) if z_train.get(c, -1e9) > z_loose}
        pred_local = {c for c in range(N_CHANNELS) if z_local.get(c, -1e9) > z_loose}
        pred_or = pred_train | pred_local
        pred_and = pred_train & pred_local

        p_t, r_t, n_t = prf(pred_train)
        p_l, r_l, n_l = prf(pred_local)
        p_or, r_or, n_or = prf(pred_or)
        p_and, r_and, n_and = prf(pred_and)

        print(f"train기준:  precision={p_t:.3f} recall={r_t:.3f} n={n_t}")
        print(f"지역문맥:   precision={p_l:.3f} recall={r_l:.3f} n={n_l}")
        print(f"OR:        precision={p_or:.3f} recall={r_or:.3f} n={n_or}")
        print(f"AND(강한확신): precision={p_and:.3f} recall={r_and:.3f} n={n_and}")

        summary.append((sid, len(gt), p_t, r_t, p_or, r_or, p_and, r_and))

    print("\n\n=== 종합 (3세그먼트) ===")
    print(f"{'seg':30s} {'k':>3s} | {'train_P':>8s} {'train_R':>8s} | {'OR_P':>6s} {'OR_R':>6s} | {'AND_P':>6s} {'AND_R':>6s}")
    for sid, k, p_t, r_t, p_or, r_or, p_and, r_and in summary:
        print(f"{sid:30s} {k:3d} | {p_t:8.3f} {r_t:8.3f} | {p_or:6.3f} {r_or:6.3f} | {p_and:6.3f} {r_and:6.3f}")


if __name__ == "__main__":
    main()
