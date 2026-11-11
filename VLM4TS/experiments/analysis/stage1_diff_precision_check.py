"""
diff 채널 precision 확인 -- 15개 놓친 채널만 보면 4/15 구제였는데, 정상 채널까지
포함해서 무차별적으로 민감해지는 건 아닌지 확인 (VLM 콜 없음).

세그먼트 1개(machine-1-4_20193_20233, k(GT)=2, 작은윈도우 precision 체크 때
이미 썼던 것과 동일 -- 비교 가능하게) 38채널 전부 diff로 스코어링.
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
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, N_CHANNELS, constant_channels, WIN
from stage1_diff_channel_pilot import score_diff_channel

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    center = (cs + ce) // 2
    s_, e_ = _centered_window(len(test), center, WIN)
    window = test[s_:e_]

    z_loose = norm.ppf(1 - 0.1)
    zs_diff = {}
    for c in range(N_CHANNELS):
        if c in degenerate:
            zs_diff[c] = -1e9
            continue
        zs_diff[c] = score_diff_channel(train[:, c], window[:, c])
        print(f"  ch{c}: z224={zs_224[c]:.2f} zdiff={zs_diff[c]:.2f} {'[GT]' if c in gt else ''}", flush=True)

    def prf(zs):
        pred = {c for c in range(N_CHANNELS) if zs[c] > z_loose}
        tp = len(pred & gt)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gt) if gt else 0.0
        return p, r, len(pred)

    p224, r224, n224 = prf(zs_224)
    pdiff, rdiff, ndiff = prf(zs_diff)
    zs_combined = {c: max(zs_224[c], zs_diff[c]) for c in range(N_CHANNELS)}
    pc, rc, nc = prf(zs_combined)

    print(f"\n=== {sid} (k(GT)={len(gt)}) ===")
    print(f"224 원본:        precision={p224:.3f} recall={r224:.3f} n_selected={n224}")
    print(f"diff만:          precision={pdiff:.3f} recall={rdiff:.3f} n_selected={ndiff}")
    print(f"원본+diff(max):  precision={pc:.3f} recall={rc:.3f} n_selected={nc}")


if __name__ == "__main__":
    main()
