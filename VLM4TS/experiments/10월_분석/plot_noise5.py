"""
GT에 한 번도 안 나온 잡음 채널 5개 — 실제 값 그대로 (세로축 = 원래 값)
  1줄: train 전체 / 2줄: test 전체 / 3줄: 막대 — 빨강 = GT, 주황 = 이 채널이 켜진 구간(이상으로 잡음), 파랑 = 그중 GT와 안 겹친 오탐
  기준: 1번 + 눈금 넓힘, spike 쪽 dt 0.5% · st 1%. 막대는 보이도록 최소 폭 30틱
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SMD
from eval_spike_detector import suddenness
from eval_flat_fix import calib, drop_flat

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check" / "noise_channels"
CASES = [("machine-1-5", 29), ("machine-2-2", 29), ("machine-2-2", 15), ("machine-3-1", 10), ("machine-3-9", 15)]
MINW = 30


def main():
    for e, c in CASES:
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99))
        on = segs(m[c].astype(int)); fp = [(u, v) for u, v in on if not y[u:v].any()]
        fig, ax = plt.subplots(3, 1, figsize=(26, 9), gridspec_kw=dict(height_ratios=[1, 1, 0.6]))
        ax[0].plot(tr[:, c], color="0.3", lw=.4); ax[0].set_xlim(0, len(tr)); ax[0].set_title(f"train ch{c} 실제 값 (범위 {tr[:, c].min():.3f} ~ {tr[:, c].max():.3f})", fontsize=10)
        ax[1].plot(te[:, c], color="k", lw=.4); ax[1].set_xlim(0, T); ax[1].set_title(f"test ch{c} 실제 값 (범위 {te[:, c].min():.3f} ~ {te[:, c].max():.3f})", fontsize=10)
        B = ax[2]
        for k, (lst, col, nm) in enumerate([(segs(y), "red", "GT"), (on, "orange", f"이 채널이 잡은 구간 {len(on)}개"), (fp, "royalblue", f"그중 오탐 {len(fp)}개")]):
            lo = 2 - k
            for u, v in lst:
                B.add_patch(plt.Rectangle((u, lo + 0.1), max(v - u, MINW), 0.8, color=col, lw=0))
            B.text(T * 1.003, lo + 0.5, nm, va="center", fontsize=9)
        B.set_xlim(0, T); B.set_ylim(0, 3); B.set_yticks([])
        fig.suptitle(f"{e} ch{c} — GT에 한 번도 안 나온 채널 | spike 쪽 dt 0.5% · st 1% (1번 + 눈금 넓힘)", fontsize=13)
        fig.tight_layout(); fig.savefig(OUT / f"raw5_{e}_ch{c}.png", dpi=60); plt.close(fig)
        print(e, c, len(on), len(fp))


if __name__ == "__main__":
    main()
