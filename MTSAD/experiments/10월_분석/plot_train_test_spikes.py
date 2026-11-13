"""
한 machine의 채널들: train과 test에서 spike가 얼마나 반복되나
  spike = 갑작스러움 ≥ 10인 묶음, 높이 = (최댓값 − 주변 ±100틱 중앙값) / train 범위
  그림: 채널마다 train 전체 / test 전체 (train 범위 눈금), test의 GT 구간 빨강 배경
  사용: python3 plot_train_test_spikes.py machine-3-5 16 17
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SMD
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels
from diag_spiky_gt import spikes

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent


def main():
    e, chs = sys.argv[1], [int(c) for c in sys.argv[2:]]
    y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
    Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0); Gte = np.nan_to_num(suddenness(te, tr), nan=0.0)
    gt = segs(y); fig, ax = plt.subplots(2 * len(chs), 1, figsize=(24, 3.2 * len(chs)), squeeze=False)
    for k, c in enumerate(chs):
        rg = np.ptp(tr[:, c]); g0 = tr[:, c].min()
        ts = spikes(tr[:, c], Gtr[c], rg); xs = spikes(te[:, c], Gte[c], rg)
        lab = [(a, b) for a, b in gt if c in interp_channels(e, a, b)]
        isg = lambda u, v: any(u < b + 5 and v > a - 5 for a, b in lab)
        hg = [h for u, v, h in xs if isg(u, v)]; hn = [h for u, v, h in xs if not isg(u, v)]
        th = np.array([h for *_, h in ts])
        print(f"{e} ch{c}: train spike {len(ts)}개 (높이 중앙 {np.median(th):.2f}, 상위 10% {np.percentile(th, 90):.2f}, 최대 {th.max():.2f}) | "
              f"test spike {len(xs)}개 — GT {len(hg)}개 (높이 {', '.join(f'{h:.2f}' for h in sorted(hg))}), 정상 {len(hn)}개 (높이 중앙 {np.median(hn):.2f}, 최대 {max(hn):.2f})")
        A = ax[2 * k, 0]; A.plot((tr[:, c] - g0) / rg, color="0.3", lw=.4); A.set_xlim(0, len(tr))
        A.set_title(f"train ch{c} — spike {len(ts)}개 (갑작 ≥ 10)", fontsize=10)
        B = ax[2 * k + 1, 0]; B.plot((te[:, c] - g0) / rg, color="k", lw=.4); B.set_xlim(0, len(te))
        for a, b in lab:
            B.axvspan(a, max(b, a + 30), color="red", alpha=.25, lw=0)
        B.set_title(f"test ch{c} — spike {len(xs)}개 (이 채널이 GT 채널인 GT 구간 = 빨강, {len(lab)}개)", fontsize=10)
    fig.suptitle(f"{e}: train과 test에서 spike 반복 (train 범위 눈금)", fontsize=12)
    fig.tight_layout(); fig.savefig(HERE / "stage12_check" / "above_gt" / e / f"train_test_ch{'_'.join(map(str, chs))}.png", dpi=60); plt.close(fig)


if __name__ == "__main__":
    main()
