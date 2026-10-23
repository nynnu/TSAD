"""
오탐을 많이 만든 채널만 전체 test 시계열로 (색깔_설명.png와 같은 스타일)
  선 색: 파랑 = 그 machine에서 GT 채널이고 GT 안에서 켜진 적 있음 / 초록 = GT 채널인데 못 켬 / 빨강 = GT 채널 아님
  점: 주황 = 이 채널이 GT 안에서 켜진 틱, 검정 = 이 채널이 GT 밖에서 켜진 틱(오탐)
  눈금 = train min~max (DINOv2가 보는 것), 연한 빨강 배경 = GT
  아래 막대: 빨강 = GT / 보라 = 탈주기(상위 3%) / 주황 = spike 쪽(보정 spike 상위 0.5% AND 갑작 상위 1%, 아무 채널)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, PH, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check" / "thr_dt05_st1" / "fp_channels"
CASES = {"machine-3-4": [32], "machine-3-11": [9, 29], "machine-1-5": [29, 12], "machine-2-9": [9, 24], "machine-1-2": [31]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for e, chs in CASES.items():
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); D = np.load(PH / f"{e}_phase_deseason.npz")["combined"]
        T = min(len(y), S.shape[1], len(D)); y, S, te, D = y[:T], S[:, :T], te[:T], D[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); fin = np.isfinite(S)
        m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99))
        pD = D > np.percentile(D, 97); gt = segs(y); gch = [set(interp_channels(e, a, b)) for a, b in gt]
        allg = set().union(*gch) if gch else set()
        fig, A = plt.subplots(figsize=(26, 1.3 * len(chs) + 2.2)); xs = np.arange(T)
        for k, c in enumerate(chs):
            g0, g1 = tr[:, c].min(), tr[:, c].max()
            base = len(chs) - 1 - k
            x = np.clip((te[:, c] - g0) / (g1 - g0 + 1e-8), 0, 1) * .8 + base
            hit = any(c in g and m[c, a:b].any() for (a, b), g in zip(gt, gch))
            col = "royalblue" if hit else ("green" if c in allg else "red")
            A.axhline(base, color="0.88", lw=.5); A.axhline(base + .8, color="0.88", lw=.5)
            A.plot(xs, x, color=col, lw=.5)
            on = np.where(m[c])[0]
            A.scatter(on, x[on], s=14, color=np.where(y[on] == 1, "orange", "black"), zorder=3)
            nfp = sum(1 for u, v in segs(m[c].astype(int)) if not y[u:v].any())
            A.text(-T * 0.004, base + .4, f"{'GT ' if c in allg else ''}ch{c}", ha="right", va="center", fontsize=10, color=col)
            A.text(T * 1.003, base + .4, f"오탐 {nfp}", va="center", fontsize=10, color=col)
        for a, b in gt:
            A.axvspan(a, max(b, a + 10), color="red", alpha=.1, lw=0)
        for j, (msk, col, nm) in enumerate([(y == 1, "red", "GT"), (pD, "purple", "탈주기 3%"), (m.any(0), "orange", "spike 쪽")]):
            lo = -0.35 - 0.4 * j
            A.fill_between(xs, lo - .28, lo, where=msk, color=col, step="mid")
            A.text(T * 1.003, lo - .14, nm, va="center", fontsize=9)
        A.set_xlim(0, T); A.set_ylim(-1.6, len(chs)); A.set_yticks([])
        A.set_title(f"{e} — 오탐을 많이 만든 채널 (눈금 = train 범위) | 점: 주황 = GT 안에서 켜짐, 검정 = 오탐 | 막대: 빨강 = GT, 보라 = 탈주기 3%, 주황 = spike 쪽 (보정 spike 0.5% AND 갑작 1%)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / f"{e}.png", dpi=70); plt.close(fig)


if __name__ == "__main__":
    main()
