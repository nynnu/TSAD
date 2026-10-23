"""
채널마다 보정 spike 점수(열마다 1개)를 큰 순서로 세운 곡선 — 채널 안에서는 정상과 GT 사이에 틈이 있나
  윗줄: GT spike가 있는 채널 / 아랫줄: 잡음 채널 (GT 0개). 위에서 100번째까지. 빨강 점 = GT 구간 열이고 이 채널이 그 GT의 GT 채널
  기준: 1번 + 눈금 넓힘, test 보정. 회색 점선 = 이 machine의 고정 0.5% dt
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
P, TOP = 14, 100
ROWS = [[("machine-1-6", 9), ("machine-2-7", 33), ("machine-3-2", 13), ("machine-1-8", 14)],
        [("machine-1-5", 29), ("machine-3-1", 10), ("machine-2-2", 15), ("machine-3-9", 15)]]


def main():
    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    for r, row in enumerate(ROWS):
        for A, (e, c) in zip(ax[r], row):
            y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
            tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
            raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
            S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S); dt = np.percentile(S[fin], 99.5)
            lab = np.zeros(T, bool)
            for a, b in segs(y):
                if c in interp_channels(e, a, b):
                    lab[a:b] = True
            ncol = T // P; v = S[c, :ncol * P:P]; g = lab[:ncol * P].reshape(ncol, P).any(1)
            ok = np.isfinite(v); v, g = v[ok], g[ok]; o = np.argsort(-v)[:TOP]
            A.plot(np.arange(1, len(o) + 1), v[o], color="k", lw=1)
            A.scatter(np.where(g[o])[0] + 1, v[o][g[o]], color="red", s=30, zorder=3, label="GT 열")
            A.axhline(dt, color="gray", ls="--", lw=1)
            A.set_title(f"{e} ch{c} | GT 열 {int(g.sum())}개 (상위 {TOP} 안 {int(g[o].sum())}개)", fontsize=10)
            A.set_xlabel("순위"); A.set_ylabel("보정 spike 점수")
    ax[0, 0].text(0.02, 0.95, "윗줄: GT spike가 있는 채널", transform=ax[0, 0].transAxes, fontsize=11, va="top")
    ax[1, 0].text(0.02, 0.95, "아랫줄: 잡음 채널 (GT 0개)", transform=ax[1, 0].transAxes, fontsize=11, va="top")
    fig.suptitle("채널마다 보정 spike 점수를 큰 순서로 (위에서 100번째까지) | 빨강 = GT 열, 회색 점선 = machine 고정 0.5% dt", fontsize=13)
    fig.tight_layout(); fig.savefig(HERE / "coverage_groups" / "channel_sorted_curves.png", dpi=70)


if __name__ == "__main__":
    main()
