"""
가정 12 이어서: entity 하나의 GT 이상을 하나씩 — 위치가 어떻게 틀리나
  사용: python3 stage12_m29.py [entity] [최대 길이]  (기본 machine-2-9, 길이 제한 없음)
  채널: GT 채널(interpretation_label) + 의심 채널(겹친 후보 안에서 임계값 넘은 채널), 채널마다 높이 맞춰 쌓음
  띠 2개: GT / 지금 예측 (보라 = 탈주기로 켜짐, 주황 = spike로 켜짐)
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
E = sys.argv[1] if len(sys.argv) > 1 else "machine-2-9"
MAXLEN = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9


def main():
    y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
    D = np.load(PH / f"{E}_phase_deseason.npz")["scores"]
    mD = D > best_thr(D.max(0), y)
    mS = search_thr(mD.any(0), calibrate(np.load(HERE / "spike_scores" / f"{E}.npz")["scores"].astype(float)),
                    np.nan_to_num(suddenness(te, tr), nan=0.0), y)
    pD, pS = mD.any(0), mS.any(0); pred = pD | pS; bl = segs(pred.astype(int))
    gts = [(a, b) for a, b in segs(y) if b - a <= MAXLEN]
    nc = 5; nr = -(-len(gts) // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(30, 7 * nr), squeeze=False)
    for i, (A, (a, b)) in enumerate(zip(ax.flat, gts)):
        L = b - a; s0, s1 = max(0, a - max(150, L)), min(len(y), b + max(150, L)); xs = np.arange(s0, s1)
        ov = [(u, v) for u, v in bl if u < b and v > a] or [(a, b)]; u0, v0 = ov[0][0], ov[-1][1]
        gch = set(interp_channels(E, a, b)); sus = set(np.where((mD | mS)[:, u0:v0].any(1))[0])
        chs = sorted(gch | sus)
        for k, c in enumerate(chs):
            x = te[s0:s1, c]; x = (x - x.min()) / (np.ptp(x) + 1e-9) * .8 + k
            col = "darkred" if c in gch and c in sus else ("gray" if c in gch else "royalblue")
            A.plot(xs, x, color=col, lw=.8)
            A.text(s0, k + .4, f"ch{c} ", ha="right", va="center", fontsize=8, color=col)
        A.axvspan(a, b, color="r", alpha=.08)
        A.fill_between(xs, -.6, -.3, where=y[s0:s1] == 1, color="red", step="mid")
        A.fill_between(xs, -1.2, -.9, where=pD[s0:s1], color="purple", step="mid")
        A.fill_between(xs, -1.2, -.9, where=pS[s0:s1] & ~pD[s0:s1], color="orange", step="mid")
        A.text(s1, -.45, " GT", va="center", fontsize=8); A.text(s1, -1.05, " 예측", va="center", fontsize=8)
        A.set_ylim(-1.4, len(chs)); A.set_yticks([]); A.set_xlim(s0, s1)
        inter = pred[a:b].sum(); iou = inter / (pred[u0:v0].sum() + L - inter)
        A.set_title(f"#{i + 1} [{a},{b}) {L}틱 | IoU {iou:.2f} · 조각 {len(ov)}\nGT 채널 {len(gch)} · 의심 {len(sus)} · 겹침 {len(gch & sus)}", fontsize=10)
    for A in ax.flat[len(gts):]:
        A.axis("off")
    fig.suptitle(f"{E} GT 이상 {len(gts)}개 (진빨강 = GT 채널이면서 의심 채널, 회색 = GT 채널인데 못 고름, 파랑 = 의심인데 GT 채널 아님 | 예측 띠: 보라 = 탈주기, 주황 = spike)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / ("m29_events.png" if E == "machine-2-9" and MAXLEN > 10 ** 8 else f"{E}_events{'_le' + str(MAXLEN) if MAXLEN < 10 ** 8 else ''}.png"), dpi=60)


if __name__ == "__main__":
    main()
