"""
원인 후보 vs 이 채널 산점도 — train 전체 = 회색 구름, test 창 = 시간 순서 색 궤적 (나연, 2026-11-10) — API 호출 없음
  원인 후보 = Stage1 채널 점수 상위 2개 (위 줄 x = 1위, 아래 줄 x = 2위)
  test 창   = Stage1 구간 ± 30틱, 색 = 시간 (연함 → 진함), 테두리 굵은 점 = Stage1 구간 안
  눈금      = 값 그대로 (train 최소 0 ~ 최대 1, 넘으면 넓힘)
  실행:  python scatter_view.py machine-1-2 40 29 9 10 5 6 11   → raw_view_scatter/{machine}/{k:03d}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, STAGE1_OUT, GT_C, load, gt_channel_ranges, stage1_scale

CTX = 30
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def main():
    e, k = sys.argv[1], int(sys.argv[2]); extra = [int(x) for x in sys.argv[3:]]
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    iv = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"][k - 1]; a, b = iv["start"], iv["end"]
    s1 = {c["channel"]: (c["spike_score"] if c["spike_score"] is not None else -1e9) for c in iv["channels"]}
    roots = sorted(s1, key=lambda c: -s1[c])[:2]; G = set().union(*[c for s, t, c in gtr if s < b and t > a])
    cols = roots + [c for c in (extra or sorted(s1)) if c not in roots]; w0, w1 = max(a - CTX, 0), min(b + CTX, len(te))
    t = np.arange(w0, w1); inside = (t >= a) & (t < b)
    fig, axs = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.6), dpi=105, squeeze=False, gridspec_kw=dict(wspace=0.3, hspace=0.35))
    for r, xr in enumerate(roots):
        for j, c in enumerate(cols):
            ax = axs[r, j]
            if c == xr:
                ax.axis("off"); continue
            ax.scatter(tr[:, xr], tr[:, c], s=2, color="#9ca3af", alpha=0.25, lw=0, rasterized=True)
            ax.scatter(te[w0:w1, xr], te[w0:w1, c], s=np.where(inside, 16, 7), c=np.linspace(0, 1, w1 - w0), cmap="Reds",
                       vmin=-0.3, vmax=1, edgecolors=np.where(inside, "black", "none"), linewidths=0.4)
            ax.plot(te[w0:w1, xr], te[w0:w1, c], color=GT_C if c in G else "black", lw=0.4, alpha=0.5)
            xl, yl = stage1_scale(np.r_[tr[:, xr], te[w0:w1, xr]]), stage1_scale(np.r_[tr[:, c], te[w0:w1, c]])
            ax.set_xlim(xl[0] - 0.05, xl[1] + 0.05); ax.set_ylim(yl[0] - 0.05, yl[1] + 0.05); ax.tick_params(labelsize=6)
            ax.set_xlabel(f"ch {xr} (원인 후보)", fontsize=7)
            ax.set_title(f"ch {c} · {'GT' if c in G else 'GT 아님'}", fontsize=8.5, color=GT_C if c in G else "black")
    fig.suptitle(f"{e} · Stage1 구간 #{k} [{a}, {b}) · 원인 후보 ch {roots} · 회색 = train 전체, 빨강 = test 창 (연함→진함 = 시간, 테두리 = 구간 안) · GT {sorted(G)}",
                 fontsize=9.5, y=1.0)
    out = Path(__file__).resolve().parent / "raw_view_scatter" / e / f"{k:03d}.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print(out)


if __name__ == "__main__":
    main()
