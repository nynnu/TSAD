"""
보고서 4장(데이터 자체의 문제)용 그림 — 보기 쉽게 채널 몇 개만, 설명을 그림 위에 (나연, 2026-11-05)

  1) 예측 가능: machine-2-9 #8 — test vs train 같은 사건(모양 같음, 높이 약 3배)
  2) 예측 불가: machine-1-8 GT 710–1081 — test GT vs train의 똑같은 사건(하루 중 같은 시각)
  3) 라벨 불일치 ① 크게 튀었는데 GT 아님: machine-2-9 #14 ch 9 vs GT 채널 ch 11
  4) 라벨 불일치 ② GT인데 안 움직임: machine-1-8 GT 19579–19583
  5) 라벨 불일치 ③ 같은 사건인데 라벨이 다름: machine-1-8 test ch 0 계단 사건 3번
  값은 Stage1과 같은 정규화(0 = train 최솟값, 1 = train 최댓값, 빨간 점선)

  실행:  python report_figures.py   → results/report_figures/fig{1..5}.png + report_figures.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, load, segs

plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent / "results" / "report_figures"
RED, GRAY, ORANGE = "#dc2626", "#374151", "#f59e0b"
DATA = BASE / "mv_data" / "SMD"


def frame(ax, lim=None):
    for v in (0, 1):
        ax.axhline(v, color="#b91c1c", lw=0.6, ls="--")
    if lim:
        ax.set_ylim(*lim)
    ax.tick_params(labelsize=8)


def shade_gt(ax, y, s, t):
    for p, q in segs(y[s:t]):
        ax.axvspan(s + p, s + q, color=RED, alpha=0.12, lw=0)


def side_by_side(title, sub, rows, left, right, lt, rt, y=None):
    """rows = [(채널, 빨강 여부)], left/right = (데이터, 시작, 끝)"""
    fig, axs = plt.subplots(len(rows), 2, figsize=(14, 1.9 * len(rows) + 1.3), sharey="row", squeeze=False, gridspec_kw=dict(wspace=0.05))
    for i, (c, gt) in enumerate(rows):
        for j, (X, s, t) in enumerate((left, right)):
            ax = axs[i, j]; v = X[s:t, c]; ax.plot(range(s, t), v, color=RED if (gt and j == 0) else GRAY, lw=1.0)
            if j == 0 and y is not None:
                shade_gt(ax, y, s, t)
            ax.text(0.99, 0.9, f"최고 {v.max():.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    color=RED if j == 0 else GRAY, weight="bold")
            frame(ax); ax.set_xlim(s, t)
        axs[i, 0].set_ylabel(f"ch {c}", rotation=0, labelpad=25, fontsize=11)
    axs[0, 0].set_title(lt, fontsize=11); axs[0, 1].set_title(rt, fontsize=11)
    fig.subplots_adjust(top=0.82)
    fig.suptitle(title, fontsize=14, weight="bold", y=0.985); fig.text(0.5, 0.925, sub, ha="center", fontsize=10.5, color=GRAY)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True); figs = []
    tr, te, y = load(DATA, "machine-2-9")
    figs.append(side_by_side("① 예측할 수 있었던 경우 — 모양은 평소에도 있는데 더 큼",
                             "여러 채널이 함께 올랐다가 떨어지는 모양은 train에도 있음. test에서는 높이가 약 3배(train 최댓값 1을 크게 넘음)",
                             [(0, True), (2, True), (18, True)], (te, 15186, 15320), (tr, 17945, 18079),
                             "test (빨간 배경 = GT 구간)", "train — 같은 채널들이 같은 모양으로 움직인 때", y))
    tr8, te8, y8 = load(DATA, "machine-1-8")
    figs.append(side_by_side("② 예측할 수 없었던 경우 — train에 똑같은 사건이 있음",
                             "약 340틱 동안 계단처럼 오르내리는 사건. 길이·모양·높이·함께 움직이는 채널, 하루 중 시각(0.96)까지 같음. test만 GT",
                             [(0, True), (8, True), (9, True)], (te8, 560, 1230), (tr8, 11315, 11962),
                             "test — GT", "train — 라벨 없음(정상으로 취급)", y8))
    # ③ 크게 튀었는데 GT 아님
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.6), gridspec_kw=dict(wspace=0.15))
    for ax, c, gt in ((axs[0], 9, False), (axs[1], 11, True)):
        v = te[18000:18148, c]; ax.plot(range(18000, 18148), v, color=RED if gt else "black", lw=1.0); shade_gt(ax, y, 18000, 18148); frame(ax)
        ax.set_title(f"ch {c} — {'GT 채널' if gt else 'GT 아님'}", fontsize=12, color=RED if gt else "black", weight="bold")
        ax.text(0.99, 0.9, f"최고 {v.max():.1f}\n(train 최댓값의 약 {v.max():.0f}배)" if v.max() > 2 else f"최고 {v.max():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11, weight="bold")
    fig.suptitle("③ 라벨이 이상한 경우 — 크게 튀었는데 GT 채널이 아님", fontsize=14, weight="bold", y=1.06)
    fig.text(0.5, 0.97, "같은 GT 구간(빨간 배경)에서 ch 9는 train 최댓값의 약 96배까지 튀었지만 라벨에 없음. 라벨에 있는 ch 11은 최고 1.9",
             ha="center", fontsize=10.5, color=GRAY)
    figs.append(fig)
    # ④ GT인데 안 움직임
    a, b = 19579, 19584; s, t = a - 100, b + 100; chs = [0, 2, 3, 18, 19, 20]
    fig, axs = plt.subplots(len(chs), 1, figsize=(14, 1.15 * len(chs) + 1.3), sharex=True)
    for ax, c in zip(axs, chs):
        v = te8[s:t, c]; ax.plot(range(s, t), v, color=RED, lw=1.0); ax.axvspan(a, b, color=RED, alpha=0.15, lw=0)
        lo, hi = min(0, v.min()), max(1, v.max()); frame(ax, (lo - 0.05, hi + 0.05)); ax.set_ylabel(f"ch {c}", rotation=0, labelpad=25, fontsize=11)
        ax.text(0.99, 0.85, f"GT 구간 안 변동 {np.ptp(te8[a:b, c]):.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=GRAY)
    fig.suptitle("④ 라벨이 이상한 경우 — GT 채널인데 거의 움직이지 않음", fontsize=14, weight="bold", y=1.03)
    fig.text(0.5, 0.985, "5틱짜리 GT(빨간 배경)에 채널 21개가 라벨됐는데, 그중 14개는 그 순간 거의 움직이지 않음 (그림은 그중 6개)", ha="center", fontsize=10.5, color=GRAY)
    figs.append(fig)
    # ⑤ 같은 사건인데 라벨이 다름
    ev = [(719, 1059, "GT 100%"), (2159, 2428, "GT 4%"), (3592, 3892, "GT 1%")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.4), sharey=True, gridspec_kw=dict(wspace=0.05))
    for ax, (p, q, g) in zip(axs, ev):
        s, t = p - 150, q + 150; ax.plot(range(s, t), te8[s:t, 0], color=GRAY, lw=0.8); shade_gt(ax, y8, s, t); frame(ax)
        ax.set_title(f"test [{p}, {q}) · {g}", fontsize=12, weight="bold", color=RED if g == "GT 100%" else "black")
    axs[0].set_ylabel("ch 0", rotation=0, labelpad=25, fontsize=11)
    fig.suptitle("⑤ 라벨이 이상한 경우 — 같은 사건인데 한 번만 GT", fontsize=14, weight="bold", y=1.08)
    fig.text(0.5, 0.98, "test에서 똑같은 계단 사건이 3번 일어남. 빨간 배경 = GT. 첫 번째는 사건 전체가 GT, 나머지는 시작 몇 틱만 GT", ha="center", fontsize=10.5, color=GRAY)
    figs.append(fig)
    with PdfPages(OUT / "report_figures.pdf") as pdf:
        for i, f in enumerate(figs, 1):
            f.savefig(OUT / f"fig{i}.png", bbox_inches="tight", dpi=130); pdf.savefig(f, bbox_inches="tight"); plt.close(f)
    print(OUT)


if __name__ == "__main__":
    main()
