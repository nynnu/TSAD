"""
한 채널의 오탐 전부 확대 — 실제 값 그대로, 오탐 앞뒤 60틱. 파랑 칸 = 오탐 구간, 회색 점선 = DINOv2 224틱 그림 경계
  칸 제목: 시작 틱 | 값 | 보정 spike | 갑작스러움.  기준: 1번 + 눈금 넓힘, spike 쪽 dt 0.5% · st 1%
  사용: python3 plot_fp_zoom_all.py machine-1-5 29
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SMD
from eval_spike_detector import suddenness
from eval_flat_fix import calib, drop_flat

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check" / "noise_channels"
W, COLS = 60, 8


def main():
    e, c = sys.argv[1], int(sys.argv[2])
    y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
    raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
    G = np.nan_to_num(suddenness(te, tr), nan=0.0); S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
    dt, st = np.percentile(S[fin], 99.5), np.percentile(G, 99)
    m = (np.where(fin, S, -np.inf) > dt) & (G > st)
    fp = [(u, v) for u, v in segs(m[c].astype(int)) if not y[u:v].any()]
    rows = int(np.ceil(len(fp) / COLS)); lo, hi = min(tr[:, c].min(), te[:, c].min()), max(tr[:, c].max(), te[:, c].max())
    fig, ax = plt.subplots(rows, COLS, figsize=(3.2 * COLS, 2.4 * rows), squeeze=False)
    for A, (u, v) in zip(ax.flat, fp):
        s0, s1 = max(0, u - W), min(T, v + W)
        A.plot(np.arange(s0, s1), te[s0:s1, c], color="k", lw=1)
        A.add_patch(plt.Rectangle((u - 0.5, lo), max(v - u, 1), hi - lo, color="royalblue", alpha=.3, lw=0))
        for w in range((s0 // 224 + 1) * 224, s1, 224):
            A.axvline(w, color="gray", ls="--", lw=.8)
        A.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)); A.set_xlim(s0, s1); A.tick_params(labelsize=6)
        A.set_title(f"t={u} | 값 {te[u:v, c].max():.2f} | spike {np.nanmax(S[c, u:v]):.1f} | 갑작 {G[c, u:v].max():.0f}", fontsize=7)
    for A in ax.flat[len(fp):]:
        A.axis("off")
    fig.suptitle(f"{e} ch{c} 오탐 {len(fp)}개 전부 (실제 값, 앞뒤 {W}틱) | 파랑 = 오탐, 회색 점선 = 224틱 그림 경계 | dt {dt:.1f}, st {st:.1f}", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"fpzoom_{e}_ch{c}.png", dpi=60); plt.close(fig)


if __name__ == "__main__":
    main()
