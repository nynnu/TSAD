"""
knee 위치가 보는 범위(W)에 따라 바뀌는 그림 (나연 주간 리포트, 2026-10-22)
  eval_knee_dt.py와 같은 점수(보정 spike, 1번 + 눈금 넓힘)·같은 knee 함수. machine 4대, 정렬 곡선 상위 10%까지 + W = 2·5·10%의 knee
  출력: coverage_groups/knee_by_window.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import SMD
from eval_flat_fix import calib, drop_flat
from eval_knee_dt import knee, SHOW, P, WS

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
COL = {2: "tab:blue", 5: "tab:orange", 10: "tab:red"}


def main():
    fig, ax = plt.subplots(1, len(SHOW), figsize=(5 * len(SHOW), 4))
    for a, e in zip(ax, sorted(SHOW)):
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(te), raw.shape[1])
        S, _ = calib(drop_flat(raw[:, :T], te[:T], tr)); v = S[:, ::P]; v = v[np.isfinite(v)]
        s = np.sort(v)[::-1]; n = int(len(s) * 0.10); a.plot(100 * np.arange(1, n + 1) / len(s), s[:n], "k", lw=1.2)
        for w in WS:
            _, r, _ = knee(v, w)
            a.axvspan(0, w, color=COL[w], alpha=0.06); a.axvline(r, color=COL[w], ls="--", lw=1.5, label=f"상위 {w}%까지 보면 → knee {r:.2f}%")
        a.set_title(e); a.set_xlabel("상위 몇 % (정렬 순위)"); a.set_ylabel("보정 spike 점수"); a.set_xlim(0, 10); a.legend(fontsize=8)
    fig.suptitle("같은 점수 곡선인데, 보는 범위(색 영역)를 넓힐수록 knee(점선)가 뒤로 밀려남", fontsize=12)
    fig.tight_layout(); fig.savefig(HERE / "coverage_groups" / "knee_by_window.png", dpi=150)


if __name__ == "__main__":
    main()
