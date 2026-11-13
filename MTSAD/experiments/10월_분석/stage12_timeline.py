"""
가정 12 넘김 구상: entity test 전체를 시간 순으로 38채널 모두 (DINOv2 눈금, 224틱 창 경계 점선)
  사용: python3 stage12_timeline.py entity [이미지당 틱 수, 기본 2240 = 224틱 창 10개]
  채널 색: 검정 = 평소 / 파랑 = GT 구간의 GT 채널 중 맞춘 것 / 초록 = GT 구간의 GT 채널 중 못 잡은 것 /
          빨강 굵게 = 켜졌지만 그 시각 GT 채널이 아닌 곳(오탐: GT 밖이거나 GT 채널 목록에 없음)
  "맞춤"은 stage12_check 와 같음: GT와 겹친 후보 안에서 그 채널이 켜졌으면 맞춤
  띠 3줄: 빨강 = GT / 보라 = 탈주기 / 주황 = spike, 위에 GT 번호
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
OUT = HERE / "stage12_check" / "timeline"
OUT.mkdir(parents=True, exist_ok=True)
E = sys.argv[1] if len(sys.argv) > 1 else "machine-3-10"
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 2240


def main():
    y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
    D = np.load(PH / f"{E}_phase_deseason.npz")["scores"]
    mD = D > best_thr(D.max(0), y)
    mS = search_thr(mD.any(0), calibrate(np.load(HERE / "spike_scores" / f"{E}.npz")["scores"].astype(float)),
                    np.nan_to_num(suddenness(te, tr), nan=0.0), y)
    M = mD | mS; pD, pS = mD.any(0), mS.any(0); bl = segs(M.any(0).astype(int))
    T, C = te.shape
    g0, g1 = tr.min(0), tr.max(0)
    X = np.clip((te - g0) / np.where(g1 - g0 > 1e-8, g1 - g0, 1), 0, 1).T
    ok = np.zeros((C, T), bool)                         # GT 구간 × GT 채널 (오탐에서 뺄 곳)
    hit = np.zeros((C, T), bool); miss = np.zeros((C, T), bool)
    gts = segs(y)
    for a, b in gts:
        gch = interp_channels(E, a, b)
        ov = [(u, v) for u, v in bl if u < b and v > a]
        sus = set(np.where(M[:, ov[0][0]:ov[-1][1]].any(1))[0]) if ov else set()
        for c in gch:
            ok[c, a:b] = True
            (hit if c in sus else miss)[c, a:b] = True
    fp = M & ~ok
    for p, s0 in enumerate(range(0, T, CHUNK)):
        s1 = min(T, s0 + CHUNK); xs = np.arange(s0, s1)
        fig, A = plt.subplots(figsize=(30, 26))
        for c in range(C):
            k = C - 1 - c; x = X[c, s0:s1] * .8 + k
            A.plot(xs, x, color="k", lw=.4)
            for m, col, lw in [(hit, "royalblue", 1.0), (miss, "green", 1.0), (fp, "red", 1.6)]:
                A.plot(xs, np.where(m[c, s0:s1], x, np.nan), color=col, lw=lw)
            A.text(s0, k + .4, f"ch{c} ", ha="right", va="center", fontsize=7)
        for w in range((s0 // 224 + 1) * 224, s1, 224):
            A.axvline(w, color="gray", ls="--", lw=.5)
        for i, (a, b) in enumerate(gts):
            if b > s0 and a < s1:
                A.axvspan(a, b, color="r", alpha=.06)
                A.text((max(a, s0) + min(b, s1)) / 2, C + .3, f"GT#{i + 1}", ha="center", fontsize=9, color="red")
        for j, (m, col, name) in enumerate([(y[s0:s1] == 1, "red", "GT"), (pD[s0:s1], "purple", "탈주기"), (pS[s0:s1], "orange", "spike")]):
            lo = -0.5 - 0.6 * j
            A.fill_between(xs, lo - .45, lo, where=m, color=col, step="mid")
            A.text(s1, lo - .22, f" {name}", va="center", fontsize=9)
        A.set_ylim(-2.4, C + 1); A.set_yticks([]); A.set_xlim(s0, s1)
        A.set_title(f"{E} test {s0}~{s1} ({p + 1}/{-(-T // CHUNK)}) | 파랑 = GT 채널 맞춤, 초록 = GT 채널 못 잡음, 빨강 굵게 = 오탐(켜졌는데 그 시각 GT 채널 아님), 검정 = 평소 | DINOv2 눈금", fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / f"{E}_{p + 1:02d}_{s0}-{s1}.png", dpi=55); plt.close(fig)
    print("saved", -(-T // CHUNK))


if __name__ == "__main__":
    main()
