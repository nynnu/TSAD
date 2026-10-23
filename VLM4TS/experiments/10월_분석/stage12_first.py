"""
가정 12 이어서: entity 하나의 GT 이상 중 앞에서부터 N개를 자세히
  사용: python3 stage12_first.py [entity] [N] [시작 번호(0부터)]  (기본 machine-2-5 2 0)
  그림 = DINOv2가 보는 눈금: 채널마다 train 최솟값 = 칸 바닥(0), 최댓값 = 칸 꼭대기(1), 벗어나면 잘림 (exp_C_globalnorm.ts_to_image_global 과 같음)
  회색 점선 세로줄 = DINOv2 224틱 창 경계, 칸마다 옅은 가로줄 = 0과 1
  채널 이름 앞 "GT" = GT 채널, 오른쪽 숫자 = GT 구간 높이 / train 범위
  채널 색: 파랑 = 맞춘 채널(GT ∩ 의심), 빨강 = 오탐(의심인데 GT 아님), 초록 = 못 맞춘 채널(GT인데 의심 아님)
  띠 3줄: 빨강 = GT / 보라 = 탈주기가 잡은 틱 / 주황 = spike 탐지기가 잡은 틱
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, PH, SMD
from exp_patch_groups import interp_channels
from stage12_check import best_thr, search_thr
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check"
E = sys.argv[1] if len(sys.argv) > 1 else "machine-2-5"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
K0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def main():
    y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
    D = np.load(PH / f"{E}_phase_deseason.npz")["scores"]
    mD = D > best_thr(D.max(0), y)
    mS = search_thr(mD.any(0), calibrate(np.load(HERE / "spike_scores" / f"{E}.npz")["scores"].astype(float)),
                    np.nan_to_num(suddenness(te, tr), nan=0.0), y)
    pD, pS = mD.any(0), mS.any(0); pred = pD | pS; bl = segs(pred.astype(int))
    gts = segs(y)[K0:K0 + N]
    fig, ax = plt.subplots(1, len(gts), figsize=(12 * len(gts), 13), squeeze=False)
    for i, (A, (a, b)) in enumerate(zip(ax.flat, gts)):
        L = b - a; s0, s1 = max(0, a - max(150, L)), min(len(y), b + max(150, L)); xs = np.arange(s0, s1)
        ov = [(u, v) for u, v in bl if u < b and v > a]
        gch = set(interp_channels(E, a, b))
        sus = set(np.where((mD | mS)[:, ov[0][0]:ov[-1][1]].any(1))[0]) if ov else set()
        chs = sorted(gch | sus)
        for k, c in enumerate(chs):
            g0, g1 = tr[:, c].min(), tr[:, c].max()
            x = np.clip((te[s0:s1, c] - g0) / (g1 - g0 + 1e-8), 0, 1) * .8 + k
            A.axhline(k, color="0.85", lw=.5); A.axhline(k + .8, color="0.85", lw=.5)
            col = "royalblue" if c in gch and c in sus else ("red" if c in sus else "green")
            A.plot(xs, x, color=col, lw=.9)
            A.text(s0, k + .4, f"{'GT ' if c in gch else ''}ch{c} ", ha="right", va="center", fontsize=9, color=col)
            h = np.abs(te[a:b, c] - np.median(te[max(0, a - 100):a, c])).max() / max(np.ptp(tr[:, c]), 1e-9)
            A.text(s1, k + .4, f" 높이 {h:.2f}", va="center", fontsize=9, color=col)
        A.axvspan(a, b, color="r", alpha=.07)
        for w in range((s0 // 224 + 1) * 224, s1, 224):
            A.axvline(w, color="gray", ls="--", lw=.8)
        for j, (m, col, name) in enumerate([(y[s0:s1] == 1, "red", "GT"), (pD[s0:s1], "purple", "탈주기"), (pS[s0:s1], "orange", "spike")]):
            lo = -0.5 - 0.45 * j
            A.fill_between(xs, lo - .3, lo, where=m, color=col, step="mid")
            A.text(s1, lo - .15, f" {name}", va="center", fontsize=9)
        A.set_ylim(-1.9, len(chs)); A.set_yticks([]); A.set_xlim(s0, s1)
        hit = len(gch & sus)
        A.set_title(f"#{K0 + i + 1} [{a},{b}) {L}틱 | 겹친 후보 {len(ov)}개\n맞춘 {hit} (파랑) · 오탐 {len(sus - gch)} (빨강) · 못 맞춤 {len(gch - sus)} (초록) / GT 채널 {len(gch)}", fontsize=11)
    fig.suptitle(f"{E} GT 이상 #{K0 + 1}~#{K0 + len(gts)}", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / f"{E}_first{K0 + 1}-{K0 + len(gts)}.png", dpi=70)


if __name__ == "__main__":
    main()
