"""
spike가 많은 채널 중 GT 채널로도 나오는 채널 — 같은 채널 안에서 정상 spike vs GT spike 높이 비교
  spike = 갑작스러움 ≥ 10인 틱 묶음, 높이 = 그 묶음 최댓값 − 주변 ±100틱 중앙값 (train 범위 비율)
  train spike 수 ≥ 50 (자주 튀는 채널) 이고 GT 채널로 나온 짧은 GT(≤15틱) ≥ 3개인 채널
  GT spike 높이가 train 정상 spike 높이 분포에서 상위 몇 %인지
"""
from pathlib import Path

import numpy as np
import pandas as pd

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent


def spikes(x, G, rg):
    med = pd.Series(x).rolling(201, center=True, min_periods=50).median().values
    return [(u, v, (x[u:v].max() - med[u + np.argmax(x[u:v])]) / rg) for u, v in segs((G >= 10).astype(int))]


def main():
    L = [f"{'machine':13s} ch | train spike 수 | train 정상 spike 높이 중앙 (90%) | GT spike: 높이 → train 정상 중 상위 몇 % | test 정상 spike 높이 중앙"]
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0); Gte = np.nan_to_num(suddenness(te, tr), nan=0.0)
        gt = [(a, b) for a, b in segs(y) if b - a <= 15]
        for c in range(tr.shape[1]):
            rg = np.ptp(tr[:, c])
            if rg < 1e-6:
                continue
            gts = [(a, b) for a, b in gt if c in interp_channels(e, a, b)]
            if len(gts) < 3:
                continue
            ts = spikes(tr[:, c], Gtr[c], rg)
            if len(ts) < 50:
                continue
            th = np.array([h for *_, h in ts]); te_s = spikes(te[:, c], Gte[c], rg)
            normal = [h for u, v, h in te_s if not y[max(0, u - 5):v + 5].any()]
            rows = []
            for a, b in gts:
                med = np.median(te[max(0, a - 100):a, c]); h = (te[a:b, c].max() - med) / rg
                rows.append(f"{h:.2f}→{100 * np.mean(th >= h):.0f}%")
            L.append(f"{e:13s} {c:2d} | {len(ts):4d} | {np.median(th):.2f} ({np.percentile(th, 90):.2f}) | " + ", ".join(rows) + f" | {np.median(normal) if normal else float('nan'):.2f}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "spiky_gt_channels.txt").write_text(txt)


if __name__ == "__main__":
    main()
