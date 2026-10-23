"""
2단계 준비: SMD 28대 train+test 모든 채널의 spike 목록 → spikes_smd.csv
  spike: spike_detect.detect (규칙) + test의 GT spike(≤15틱) 합침
  붙이는 특징: 높이(채널 범위 대비), 폭, 방향, 갑작스러움, 잔잔한 흔들림(noise), 자주 튀는 정도(density: 같은 채널 ±100틱 안 다른 spike 수)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from exp_patch_groups import interp_channels  # noqa: E402
from spike_detect import detect, gt_spikes, merge_gt, local_stats  # noqa: E402

SMD = HERE.parents[2] / "mv_data" / "SMD"
CTX = 100


def add_context(lst, x, rng):
    if not lst:
        return
    _, dev, noise = local_stats(x, 0.01 * rng)
    ts = np.array([s["t"] for s in lst])
    for s in lst:
        t = s["t"]
        s["noise"] = float(noise[t] / rng)
        s.setdefault("height", float(np.abs(dev[t]) / rng)); s.setdefault("up", bool(dev[t] > 0))
        s.setdefault("sudden", float(np.abs(dev[t]) / noise[t]))
        s["density"] = int(((ts >= t - CTX) & (ts <= t + CTX)).sum() - 1)


def main():
    rows = []
    for e in sorted(p.stem for p in (SMD / "test").glob("*.txt")):
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        rng = tr.max(0) - tr.min(0)
        for split, X in [("train", tr), ("test", te)]:
            S = {c: detect(X[:, c], rng[c]) for c in range(X.shape[1])}
            if split == "test":
                S = merge_gt(S, gt_spikes(te, tr, y, lambda a, b: interp_channels(e, a, b)))
            for c, lst in S.items():
                add_context(lst, X[:, c], rng[c])
                for s in lst:
                    in_gt = bool(split == "test" and y[s["start"]:s["end"]].any())
                    rows.append(dict(entity=e, split=split, ch=c, t=s["t"], start=s["start"], end=s["end"], width=s["width"],
                                     height=s["height"], up=s["up"], sudden=s["sudden"], noise=s["noise"], density=s["density"],
                                     rule=s.get("rule", True), is_gt_spike=s.get("is_gt", False), in_gt=in_gt))
        print(e, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "spikes_smd.csv", index=False)
    print(df.groupby(["split", "in_gt"]).size())


if __name__ == "__main__":
    main()
