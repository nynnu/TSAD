"""
가정 12 넘김 관점: 구간 오탐(GT 앞뒤 30틱 밖 후보)을 하나씩
  사용: python3 stage12_fp.py entity
  채널 = 그 후보 안에서 켜진 채널 (최대 6개, 켜진 틱이 많은 순), DINOv2 눈금(train 최소~최대, 잘림), 회색 점선 = 224틱 창 경계
  띠 3줄: 빨강 = GT / 보라 = 탈주기 / 주황 = spike
  채널 이름 옆: 켜진 이유 (탈 = 탈주기, 스 = spike)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, PH, SMD
from stage12_check import best_thr, search_thr
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check"
E = sys.argv[1] if len(sys.argv) > 1 else "machine-2-5"
NEAR, CTX, MAXCH = 30, 150, 6


def main():
    y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
    D = np.load(PH / f"{E}_phase_deseason.npz")["scores"]
    mD = D > best_thr(D.max(0), y)
    mS = search_thr(mD.any(0), calibrate(np.load(HERE / "spike_scores" / f"{E}.npz")["scores"].astype(float)),
                    np.nan_to_num(suddenness(te, tr), nan=0.0), y)
    pD, pS = mD.any(0), mS.any(0)
    near = np.convolve(y, np.ones(2 * NEAR + 1), "same") > 0
    fps = [(a, b) for a, b in segs((pD | pS).astype(int)) if not near[a:b].any()]
    nc = 4; nr = -(-len(fps) // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(24, 5.5 * nr), squeeze=False)
    for i, (A, (a, b)) in enumerate(zip(ax.flat, fps)):
        s0, s1 = max(0, a - CTX), min(len(y), b + CTX); xs = np.arange(s0, s1)
        on = (mD | mS)[:, a:b].sum(1); chs = [int(c) for c in np.argsort(-on) if on[c] > 0][:MAXCH]
        for k, c in enumerate(chs):
            g0, g1 = tr[:, c].min(), tr[:, c].max()
            x = np.clip((te[s0:s1, c] - g0) / (g1 - g0 + 1e-8), 0, 1) * .8 + k
            A.axhline(k, color="0.85", lw=.5); A.axhline(k + .8, color="0.85", lw=.5)
            A.plot(xs, x, color="red", lw=.8)
            why = ("탈" if mD[c, a:b].any() else "") + ("스" if mS[c, a:b].any() else "")
            A.text(s0, k + .4, f"ch{c}({why}) ", ha="right", va="center", fontsize=8, color="red")
        A.axvspan(a, b, color="orange", alpha=.12)
        for w in range((s0 // 224 + 1) * 224, s1, 224):
            A.axvline(w, color="gray", ls="--", lw=.8)
        for j, (m, col, name) in enumerate([(y[s0:s1] == 1, "red", "GT"), (pD[s0:s1], "purple", "탈주기"), (pS[s0:s1], "orange", "spike")]):
            lo = -0.4 - 0.35 * j
            A.fill_between(xs, lo - .25, lo, where=m, color=col, step="mid")
            A.text(s1, lo - .12, f" {name}", va="center", fontsize=8)
        A.set_ylim(-1.5, max(len(chs), 1)); A.set_yticks([]); A.set_xlim(s0, s1)
        A.set_title(f"오탐 {i + 1} [{a},{b}) {b - a}틱 | 켜진 채널 {int((on > 0).sum())}개", fontsize=10)
    for A in ax.flat[len(fps):]:
        A.axis("off")
    fig.suptitle(f"{E} 구간 오탐 {len(fps)}개 (GT 앞뒤 {NEAR}틱 밖 후보, 주황 배경 = 후보 구간, 채널 = 그 안에서 켜진 채널, 탈 = 탈주기 / 스 = spike)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"{E}_fp.png", dpi=55)
    print(E, len(fps), [(int(a), int(b - a)) for a, b in fps])


if __name__ == "__main__":
    main()
