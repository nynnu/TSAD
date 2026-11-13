"""
GT에 거의 안 나오는 잡음 채널 — 기준: 1번 + 눈금 넓힘, spike 쪽 dt 0.5% · st 1%
  채널마다 한 장: (1) train 전체 (2) test 전체 (검정 점 = 오탐으로 켜진 곳, 빨강 배경 = GT) (3) 오탐 확대 4개 (±150틱, 회색 점선 = 224틱 그림 경계)
  눈금 = train min~max (DINOv2가 보는 것, 넘으면 위로 넘어가게 그림)
"""
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
CASES = [("machine-1-5", 29), ("machine-3-1", 10), ("machine-1-2", 31), ("machine-3-4", 9), ("machine-3-4", 15), ("machine-2-8", 9), ("machine-2-2", 29), ("machine-2-2", 15), ("machine-3-9", 15)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for e, c in CASES:
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); S, mad = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99))
        fpb = [(u, v) for u, v in segs(m.any(0).astype(int)) if not y[u:v].any() and m[c, u:v].any()]
        g0, g1 = tr[:, c].min(), tr[:, c].max(); sc = lambda x: (x - g0) / (g1 - g0 + 1e-12)
        fig = plt.figure(figsize=(24, 10)); gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.3])
        A = fig.add_subplot(gs[0, :]); A.plot(sc(tr[:, c]), color="gray", lw=.4); A.set_title(f"train 전체 (DINOv2 눈금)", fontsize=10); A.set_xlim(0, len(tr))
        B = fig.add_subplot(gs[1, :]); B.plot(sc(te[:, c]), color="red", lw=.4)
        for a, b in segs(y):
            B.axvspan(a, max(b, a + 10), color="r", alpha=.12, lw=0)
        on = np.array([u for u, v in fpb]); B.scatter(on, sc(te[on, c]) if len(on) else [], s=12, color="k", zorder=3)
        B.set_title(f"test 전체 — 검정 점 = 이 채널이 켜진 오탐 {len(fpb)}개, 연한 빨강 = GT", fontsize=10); B.set_xlim(0, T)
        for j, (u, v) in enumerate([fpb[i] for i in np.linspace(0, len(fpb) - 1, min(4, len(fpb))).astype(int)] if fpb else []):
            Cx = fig.add_subplot(gs[2, j]); s0, s1 = max(0, u - 150), min(T, v + 150); xs = np.arange(s0, s1)
            Cx.plot(xs, sc(te[s0:s1, c]), color="red", lw=.9); Cx.axvspan(u, max(v, u + 1), color="orange", alpha=.35, lw=0)
            for w in range((s0 // 224 + 1) * 224, s1, 224):
                Cx.axvline(w, color="gray", ls="--", lw=.8)
            Cx.set_title(f"오탐 t={u} | 높이 {sc(te[u:v, c]).max() - np.median(sc(te[max(0, u - 100):u, c])):.2f} | 보정 spike {np.nanmax(S[c, u:v]):.1f} | 갑작 {G[c, u:v].max():.0f}", fontsize=9)
        fig.suptitle(f"{e} ch{c} — 잡음 채널 (MAD {mad[c]:.2f})", fontsize=13)
        fig.tight_layout(); fig.savefig(OUT / f"{e}_ch{c}.png", dpi=60); plt.close(fig)
        print(e, c, len(fpb))


if __name__ == "__main__":
    main()
