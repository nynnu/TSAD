"""
오탐 덩어리 자세히 (stage12_first.py 그림 스타일) — 설정: 탈주기 상위 3% / spike 쪽 = 보정 spike 상위 0.5% AND 갑작 상위 1%
  CASES의 채널이 켜진 오탐 덩어리(GT 밖) 4개를 고르게 골라 한 칸씩. 칸 = 덩어리 ±150틱
  채널 = 그 덩어리에서 켜진 채널 전부 (+ 대상 채널). 눈금 = train min~max (0~1로 자름)
  채널 색: 빨강 = 켜졌지만 GT 아님(오탐), 파랑/초록 = GT 구간이 겹칠 때만 (맞춤/못 맞춤)
  오른쪽 숫자 = 높이 (켜진 덩어리 안 최댓값 − 직전 100틱 중앙값, train 범위 비율)
  띠 3줄: 빨강 = GT / 보라 = 탈주기 3% / 주황 = spike 쪽. 회색 점선 = 224틱 그림 경계
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
import sys
from plot_fp_channels import CASES

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check" / "thr_dt05_st1" / "fp_channels"
N, W = 4, 150


def main():
    cases, out = CASES, OUT
    if len(sys.argv) > 2:                      # 사용: python3 plot_fp_events.py 9 machine-1-1,machine-2-9,...
        cases = {e: [int(sys.argv[1])] for e in sys.argv[2].split(",")}; out = OUT / f"ch{sys.argv[1]}"; out.mkdir(parents=True, exist_ok=True)
    for e, targets in cases.items():
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); D = np.load(PH / f"{e}_phase_deseason.npz")["combined"]
        T = min(len(y), S.shape[1], len(D)); y, S, te, D = y[:T], S[:, :T], te[:T], D[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); fin = np.isfinite(S)
        m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99))
        pD = D > np.percentile(D, 97); pS = m.any(0)
        for tc in targets:
            fps = [(u, v) for u, v in segs(pS.astype(int)) if not y[u:v].any() and m[tc, u:v].any()]
            if not fps:
                continue
            pick = [fps[i] for i in np.linspace(0, len(fps) - 1, min(N, len(fps))).astype(int)]
            fig, ax = plt.subplots(1, len(pick), figsize=(9 * len(pick), 10), squeeze=False)
            for A, (u, v) in zip(ax.flat, pick):
                s0, s1 = max(0, u - W), min(T, v + W); xs = np.arange(s0, s1)
                on = sorted(set(np.where(m[:, u:v].any(1))[0]) | {tc})
                gch = set()
                for a, b in segs(y):
                    if a < s1 and b > s0:
                        gch |= set(interp_channels(e, a, b))
                chs = sorted(set(on) | gch, reverse=True)
                for k, c in enumerate(chs):
                    g0, g1 = tr[:, c].min(), tr[:, c].max()
                    x = np.clip((te[s0:s1, c] - g0) / (g1 - g0 + 1e-8), 0, 1) * .8 + k
                    A.axhline(k, color="0.9", lw=.5); A.axhline(k + .8, color="0.9", lw=.5)
                    col = "red" if c not in gch else ("royalblue" if c in on else "green")
                    A.plot(xs, x, color=col, lw=.9)
                    A.text(s0, k + .4, f"{'GT ' if c in gch else ''}ch{c}{' ◀' if c == tc else ''} ", ha="right", va="center", fontsize=9, color=col)
                    h = np.abs(te[u:v, c] - np.median(te[max(0, u - 100):u, c])).max() / max(np.ptp(tr[:, c]), 1e-9)
                    A.text(s1, k + .4, f" 높이 {h:.2f}", va="center", fontsize=9, color=col)
                for a, b in segs(y):
                    if a < s1 and b > s0:
                        A.axvspan(max(a, s0), min(b, s1), color="r", alpha=.07)
                for w in range((s0 // 224 + 1) * 224, s1, 224):
                    A.axvline(w, color="gray", ls="--", lw=.8)
                for j, (msk, col, nm) in enumerate([(y[s0:s1] == 1, "red", "GT"), (pD[s0:s1], "purple", "탈주기"), (pS[s0:s1], "orange", "spike")]):
                    lo = -0.5 - 0.45 * j
                    A.fill_between(xs, lo - .3, lo, where=msk, color=col, step="mid")
                    A.text(s1, lo - .15, f" {nm}", va="center", fontsize=9)
                A.set_ylim(-1.9, len(chs)); A.set_yticks([]); A.set_xlim(s0, s1)
                A.set_title(f"오탐 [{u},{v}) {v - u}틱 | 켜진 채널 {len(on)}개 (빨강) / GT 채널 {len(gch)}", fontsize=11)
            fig.suptitle(f"{e} ch{tc} 이 켜진 오탐 {len(fps)}개 중 {len(pick)}개 (◀ = 대상 채널) | spike 상위 0.5% AND 갑작 상위 1%, 탈주기 3%", fontsize=13)
            fig.tight_layout(); fig.savefig(out / f"event_{e}_ch{tc}.png", dpi=70); plt.close(fig)


if __name__ == "__main__":
    main()
