"""
채널 관계가 train과 같은지 — 같은 시각으로 맞춘 train 창 vs 채널마다 따로 맞춘 train 창 (나연, 2026-11-05)

  창     = Stage2와 같은 창 (Stage1 구간 ± max(60, 길이))
  따로   = 채널마다 train에서 값 차이(제곱합)가 가장 작은 시점 → 거리 합  D_ind
  같이   = 모든 채널을 같은 시각으로 자른 train 창 중 거리 합이 가장 작은 시점 → D_joint
  비율   = D_joint / D_ind  (1에 가까울수록 그 채널 조합이 train 한 시점에 함께 있었음)
  기준   = GT에서 200틱 이상 떨어진 test 창 200개의 비율 분포 → GT 창의 백분위
  채널   = (a) GT 채널만, (b) train에서 변하는 채널 전부
  --event = 창을 Stage1 구간 ± 10틱으로 줄이고, 채널마다 창 중앙값을 빼고 train 평소 흔들림(MAD)으로 나눔
            (긴 창에서는 배경과 크게 흔들리는 채널이 거리를 좌우해서 이상 부분이 묻힘)

  실행:  python joint_match.py machine-1-8 714:742 2100:2170 2716:2730 3066:3080 3584:3598
  출력:  raw_view/{machine}/analysis/joint_{start}_{end}.png + 숫자 요약(화면)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, MARGIN_MIN, load, gt_channel_ranges

plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def dist_all(trc, v, scale=None):
    """train 한 채널의 모든 시작 시점에 대해 v와의 거리(제곱합). scale이 있으면 창마다 중앙값을 빼고 scale로 나눔"""
    W = np.lib.stride_tricks.sliding_window_view(trc, len(v))
    if scale is not None:
        W = (W - np.median(W, 1, keepdims=True)) / scale; v = (v - np.median(v)) / scale
    return (W ** 2).sum(1) - 2 * W @ v + v @ v


def ratio(tr, te, w0, w1, chs, scale=None):
    d = np.array([dist_all(tr[:, c], te[w0:w1, c], None if scale is None else scale[c]) for c in chs])
    ind = d.min(1).sum(); t = int(np.argmin(d.sum(0)))
    return d[:, t].sum() / max(ind, 1e-9), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("machine"); ap.add_argument("intervals", nargs="+", help="Stage1 구간 start:end")
    ap.add_argument("--n-base", type=int, default=200)
    ap.add_argument("--event", action="store_true")
    a = ap.parse_args(); e = a.machine
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    out = Path(__file__).resolve().parent / "raw_view" / e / "analysis"
    moving = [c for c in range(tr.shape[1]) if np.ptp(tr[:, c]) > 1e-6]
    far = np.convolve(y, np.ones(401), "same") == 0
    rng = np.random.default_rng(0)
    scale = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01) if a.event else None
    for iv in a.intervals:
        s, t = map(int, iv.split(":")); m = 10 if a.event else max(MARGIN_MIN, t - s); w0, w1 = max(0, s - m), min(len(te), t + m); L = w1 - w0
        G = sorted(set().union(*[c for p, q, c in gtr if p < w1 and q > w0]))
        G = [c for c in G if c in moving]
        res = {}
        for nm, chs in (("GT 채널", G), ("전체 채널", moving)):
            r, tj = ratio(tr, te, w0, w1, chs, scale)
            starts = [x for x in rng.integers(0, len(te) - L, 4 * a.n_base) if far[x:x + L].all()][:a.n_base]
            base = np.array([ratio(tr, te, x, x + L, chs, scale)[0] for x in starts])
            res[nm] = (r, tj, (base < r).mean(), np.median(base))
            print(f"[{s}:{t}] {nm:5s} {len(chs):2d}개: 비율 {r:.2f} (정상 test 창 중앙 {np.median(base):.2f}) → 정상 창 중 이보다 낮은 비율 {100 * (base < r).mean():.0f}% | 같은 시각 train {tj}")
        # 그림: GT 채널(+ch0) — test 창 | 같은 시각 train 창(GT 채널 기준)
        tj = res["GT 채널"][1]; show = G + ([0] if 0 not in G and 0 in moving else [])
        fig, axs = plt.subplots(len(show), 2, figsize=(16, 1.1 * len(show) + 0.8), dpi=110, sharey="row", squeeze=False)
        for i, c in enumerate(show):
            axs[i, 0].plot(range(w0, w1), te[w0:w1, c], color="#dc2626" if c in G else "#6b7280", lw=0.8)
            axs[i, 0].axvspan(s - 0.5, t - 0.5, color="#f59e0b", alpha=0.2)
            axs[i, 1].plot(range(tj, tj + L), tr[tj:tj + L, c], color="#374151", lw=0.8)
            axs[i, 0].set_ylabel(f"ch {c}" + (" [GT]" if c in G else ""), rotation=0, labelpad=22, fontsize=8)
            for ax in axs[i]:
                ax.tick_params(labelsize=6)
        r, _, p, med = res["GT 채널"]
        axs[0, 0].set_title(f"test 창 [{w0}, {w1}) · 주황 = Stage1 구간", fontsize=9)
        axs[0, 1].set_title(f"train 같은 시각 [{tj}, {tj + L}) — GT 채널 전체가 가장 비슷한 한 시점", fontsize=9)
        fig.suptitle(f"{e} · Stage1 구간 [{s}, {t}) · GT 채널 같은 시각 비율 {r:.2f} (정상 test 창 중앙 {med:.2f}, 백분위 {100 * p:.0f}%)", fontsize=10)
        fig.savefig(out / f"joint{'_event' if a.event else ''}_{s}_{t}.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
