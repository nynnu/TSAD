"""
눈으로 보이는데 놓친 GT 15개 (spike 쪽: 보정 spike 상위 0.5% AND 갑작 상위 1%) — DINOv2가 보는 눈금(train min~max, 0~1로 자름)
  채널 = 그 GT의 GT 채널 (초록 = 못 맞춤). 오른쪽 = 높이 | spike 점수/dt | 갑작/st (O 통과, X 탈락)
  회색 점선 = DINOv2 224틱 그림 경계 (spike 점수는 이 그림 단위로 계산), 연한 빨강 = GT
  띠: 빨강 = GT / 주황 = spike 쪽 켜짐 (다른 채널 포함)
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

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
D = HERE / "stage12_check" / "thr_dt05_st1" / "miss_wide_range"
FLAT = {"machine-3-10 [2875,2951)", "machine-3-2 [15577,15582)", "machine-3-3 [19081,19088)"}
PER = 4


def main():
    cases = [" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(D / "list.txt").read().splitlines()[1:]]
    cases = [c for c in cases if c not in FLAT]
    cache = {}
    for pg in range(0, len(cases), PER):
        chunk = cases[pg:pg + PER]
        fig, ax = plt.subplots(1, len(chunk), figsize=(8.5 * len(chunk), 11), squeeze=False)
        for A, s in zip(ax.flat, chunk):
            e, rg = s.split(" "); a, b = map(int, rg.strip("[)").split(","))
            if e not in cache:
                y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
                tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
                S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); T = min(len(y), S.shape[1]); y, S, te = y[:T], S[:, :T], te[:T]
                G = np.nan_to_num(suddenness(te, tr), nan=0.0); fin = np.isfinite(S); dt, st = np.percentile(S[fin], 99.5), np.percentile(G, 99)
                cache[e] = (y, tr, te, S, G, dt, st, ((np.where(fin, S, -np.inf) > dt) & (G > st)).any(0))
            y, tr, te, S, G, dt, st, p = cache[e]
            L = b - a; s0, s1 = max(0, a - max(150, L)), min(len(y), b + max(150, L)); xs = np.arange(s0, s1)
            chs = [c for c in sorted(interp_channels(e, a, b), reverse=True) if np.ptp(tr[:, c]) > 1e-6]
            for k, c in enumerate(chs):
                g0, g1 = tr[:, c].min(), tr[:, c].max()
                x = np.clip((te[s0:s1, c] - g0) / (g1 - g0 + 1e-8), 0, 1) * .8 + k
                A.axhline(k, color="0.9", lw=.5); A.axhline(k + .8, color="0.9", lw=.5)
                A.plot(xs, x, color="green", lw=.9)
                h = np.abs(te[a:b, c] - np.median(te[max(0, a - 100):a, c])).max() / np.ptp(tr[:, c])
                sv = np.nanmax(S[c, a:b]) if np.isfinite(S[c, a:b]).any() else np.nan; gv = G[c, a:b].max()
                A.text(s0, k + .4, f"GT ch{c} ", ha="right", va="center", fontsize=9, color="green")
                A.text(s1, k + .4, f" 높이 {h:.2f} | spike {sv:.1f}/{dt:.1f} {'O' if sv > dt else 'X'} | 갑작 {gv:.0f}/{st:.0f} {'O' if gv > st else 'X'}",
                       va="center", fontsize=8, color="black" if h >= 0.3 else "gray")
            A.axvspan(a, b, color="r", alpha=.08)
            for w in range((s0 // 224 + 1) * 224, s1, 224):
                A.axvline(w, color="gray", ls="--", lw=1)
            for j, (msk, col, nm) in enumerate([(y[s0:s1] == 1, "red", "GT"), (p[s0:s1], "orange", "spike")]):
                lo = -0.5 - 0.45 * j
                A.fill_between(xs, lo - .3, lo, where=msk, color=col, step="mid"); A.text(s1, lo - .15, f" {nm}", va="center", fontsize=9)
            A.set_ylim(-1.5, len(chs)); A.set_yticks([]); A.set_xlim(s0, s1)
            A.set_title(f"{e} [{a},{b}) {L}틱 놓침", fontsize=11)
        fig.suptitle("눈으로 보이는데 놓친 GT — DINOv2가 보는 눈금 | 오른쪽: 높이 | spike 점수/dt | 갑작스러움/st | 회색 점선 = 224틱 그림 경계", fontsize=12)
        fig.tight_layout(); fig.savefig(D / f"visible_{pg // PER + 1:02d}.png", dpi=65); plt.close(fig)


if __name__ == "__main__":
    main()
