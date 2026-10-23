"""
가정 12 이어서: train 값을 있는 그대로 (정규화 없이) — train 범위가 드문 큰 값 때문에 넓어졌는지 보기
  사용: python3 stage12_train_raw.py entity ch1 ch2 ...
  회색 점선 = train 값 0.5%~99.5% 지점 (값의 99%가 들어가는 폭)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import SMD

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent / "stage12_check"


def main():
    E, chs = sys.argv[1], [int(c) for c in sys.argv[2:]]
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=",")
    fig, ax = plt.subplots(len(chs), 1, figsize=(22, 4 * len(chs)), squeeze=False)
    for A, c in zip(ax[:, 0], chs):
        x = tr[:, c]; lo, hi = np.percentile(x, [0.5, 99.5])
        A.plot(x, color="k", lw=.5)
        A.axhline(lo, color="gray", ls="--", lw=.8); A.axhline(hi, color="gray", ls="--", lw=.8)
        A.set_title(f"{E} train ch{c} | 최소 {x.min():.3f} · 최대 {x.max():.3f} · 값의 99%가 {lo:.3f}~{hi:.3f} 안 (범위의 {(hi - lo) / (np.ptp(x) + 1e-12) * 100:.0f}%)", fontsize=11)
        A.set_xlim(0, len(x))
    fig.tight_layout(); fig.savefig(OUT / f"{E}_train_raw_{'_'.join(map(str, chs))}.png", dpi=65)


if __name__ == "__main__":
    main()
