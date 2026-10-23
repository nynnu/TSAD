"""
미탐 중 "train 범위가 드물게 튄 값 때문에 넓어진" 경우 (spike 쪽: 보정 spike 상위 0.5% AND 갑작 상위 1%, 탈주기 없음)
  넓어짐 = GT 채널의 train 값 99%가 들어가는 폭(0.5~99.5%) / 전체 범위 < 0.3
  칸마다 GT 채널 두 줄씩: 위 = DINOv2가 보는 눈금(train min~max), 아래 = train 99% 폭 눈금 (그림 순서 기준) (같은 데이터)
  오른쪽: 높이 = (GT 안 최댓값 − 직전 100틱 중앙값) / train 범위,  (99%폭 기준 높이)
  띠: 빨강 = GT / 주황 = spike 쪽 켜짐. 회색 점선 = 224틱 그림 경계
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check" / "thr_dt05_st1" / "miss_wide_range"
R_MAX, PER = 0.3, 4


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, lines, nmiss = [], [], 0
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); T = min(len(y), S.shape[1]); y, S, te = y[:T], S[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); fin = np.isfinite(S)
        m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99)); p = m.any(0)
        lo, hi = np.percentile(tr, 0.5, 0), np.percentile(tr, 99.5, 0); rg = np.ptp(tr, 0)
        for a, b in segs(y):
            if p[a:b].any():
                continue
            nmiss += 1
            gch = interp_channels(e, a, b)
            r = {c: (hi[c] - lo[c]) / rg[c] for c in gch if rg[c] > 1e-6}
            if r and np.median(list(r.values())) < R_MAX:
                cases.append((e, a, b, gch, r, tr, te, y, p))
                lines.append(f"{e} [{a},{b}) {b - a}틱 | GT 채널 99%폭/범위 " + ", ".join(f"ch{c} {v:.2f}" for c, v in sorted(r.items())))
    lines.insert(0, f"놓친 GT {nmiss}개 중 GT 채널 train 범위가 넓어진 경우(99%폭/범위 중앙 < {R_MAX}) {len(cases)}개")
    (OUT / "list.txt").write_text("\n".join(lines)); print("\n".join(lines))
    for pg in range(0, len(cases), PER):
        chunk = cases[pg:pg + PER]
        fig, ax = plt.subplots(1, len(chunk), figsize=(9 * len(chunk), 3 + 1.6 * max(len(c[3]) for c in chunk)), squeeze=False)
        for A, (e, a, b, gch, r, tr, te, y, p) in zip(ax.flat, chunk):
            L = b - a; s0, s1 = max(0, a - max(150, L)), min(len(y), b + max(150, L)); xs = np.arange(s0, s1)
            k = 0
            for c in sorted(gch, reverse=True):
                lo99, hi99 = np.percentile(tr[:, c], [0.5, 99.5]); g0, g1 = tr[:, c].min(), tr[:, c].max()
                base = np.median(te[max(0, a - 100):a, c]); mx = np.abs(te[a:b, c] - base).max()
                for sc0, sc1, nm, col in [(lo99, hi99, "99%폭", "darkgreen"), (g0, g1, "DINOv2 눈금", "green")]:
                    x = np.clip((te[s0:s1, c] - sc0) / (sc1 - sc0 + 1e-8), 0, 1) * .8 + k
                    A.axhline(k, color="0.9", lw=.5); A.axhline(k + .8, color="0.9", lw=.5)
                    A.plot(xs, x, color=col, lw=.9)
                    A.text(s0, k + .4, f"GT ch{c} ({nm}) ", ha="right", va="center", fontsize=8, color=col)
                    A.text(s1, k + .4, f" 높이 {mx / (sc1 - sc0 + 1e-12):.2f}", va="center", fontsize=8, color=col)
                    k += 1
                k += 0.3
            A.axvspan(a, b, color="r", alpha=.08)
            for w in range((s0 // 224 + 1) * 224, s1, 224):
                A.axvline(w, color="gray", ls="--", lw=.8)
            for j, (msk, col, nm) in enumerate([(y[s0:s1] == 1, "red", "GT"), (p[s0:s1], "orange", "spike")]):
                lo_ = -0.5 - 0.45 * j
                A.fill_between(xs, lo_ - .3, lo_, where=msk, color=col, step="mid"); A.text(s1, lo_ - .15, f" {nm}", va="center", fontsize=8)
            A.set_ylim(-1.5, k); A.set_yticks([]); A.set_xlim(s0, s1)
            A.set_title(f"{e} [{a},{b}) {L}틱 놓침 | 99%폭/범위: " + ", ".join(f"ch{c} {v:.2f}" for c, v in sorted(r.items())), fontsize=10)
        fig.suptitle("미탐 — train 범위가 드물게 튄 값 때문에 넓어진 경우 | 채널마다 위 = DINOv2가 보는 train 범위 눈금, 아래 = 99%폭 눈금", fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / f"page_{pg // PER + 1:02d}.png", dpi=65); plt.close(fig)


if __name__ == "__main__":
    main()
