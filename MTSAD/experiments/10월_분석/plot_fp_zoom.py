"""
오탐을 많이 만든 채널의 실제 생김새 — 켜진 곳 주변 ±150틱 확대 (색깔_설명.png 스타일)
  윗줄: 오탐 예시 (GT 밖에서 켜짐) 최대 6개, 아랫줄: GT 안에서 켜진 예시 최대 6개 (없으면 비움)
  눈금 = train min~max (DINOv2가 보는 것), 연한 빨강 배경 = GT, 회색 점선 = 224틱 그림 경계, 주황 세로띠 = 이 채널이 켜진 틱
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels
from plot_fp_channels import CASES

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check" / "thr_dt05_st1" / "fp_channels"
W, N = 150, 6


def pick(lst, n):
    return [lst[i] for i in np.linspace(0, len(lst) - 1, min(n, len(lst))).astype(int)] if lst else []


def main():
    for e, chs in CASES.items():
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); T = min(len(y), S.shape[1]); y, S, te = y[:T], S[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); fin = np.isfinite(S)
        m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99))
        gt = segs(y); allg = set().union(*[set(interp_channels(e, a, b)) for a, b in gt]) if gt else set()
        for c in chs:
            bl = segs(m[c].astype(int))
            fp = pick([(u, v) for u, v in bl if not y[u:v].any()], N); tp = pick([(u, v) for u, v in bl if y[u:v].any()], N)
            g0, g1 = tr[:, c].min(), tr[:, c].max()
            fig, ax = plt.subplots(2, N, figsize=(3.6 * N, 5.6), squeeze=False)
            for row, (lst, nm, col) in enumerate([(fp, "오탐", "red"), (tp, "GT 안에서 켜짐", "royalblue")]):
                for j in range(N):
                    A = ax[row, j]
                    if j >= len(lst):
                        A.axis("off"); continue
                    u, v = lst[j]; s0, s1 = max(0, u - W), min(T, v + W); xs = np.arange(s0, s1)
                    x = np.clip((te[s0:s1, c] - g0) / (g1 - g0 + 1e-8), 0, 1)
                    for a, b in gt:
                        if a < s1 and b > s0:
                            A.axvspan(max(a, s0), min(b, s1), color="red", alpha=.12, lw=0)
                    for w in range((s0 // 224 + 1) * 224, s1, 224):
                        A.axvline(w, color="gray", ls="--", lw=.7)
                    A.axvspan(u, max(v, u + 1), color="orange", alpha=.35, lw=0)
                    A.plot(xs, x, color=col, lw=1)
                    A.set_ylim(-0.03, 1.03); A.set_xlim(s0, s1); A.tick_params(labelsize=7)
                    A.set_title(f"{nm} t={u}  높이 {x.max() - np.median(x):.2f}", fontsize=9)
            fig.suptitle(f"{e} {'GT ' if c in allg else ''}ch{c} — 실제 값 확대 (±{W}틱, 눈금 = train 범위 0~1) | 윗줄 오탐 {sum(1 for u, v in bl if not y[u:v].any())}개 중 {len(fp)}개, "
                         f"아랫줄 GT 안 {sum(1 for u, v in bl if y[u:v].any())}개 중 {len(tp)}개 | 주황 띠 = 켜진 곳, 연한 빨강 = GT", fontsize=11)
            fig.tight_layout(); fig.savefig(OUT / f"zoom_{e}_ch{c}.png", dpi=70); plt.close(fig)


if __name__ == "__main__":
    main()
