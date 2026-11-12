"""
GT 채널의 특징 찾기용 그림 (나연, 2026-11-05) — API 호출 없음

  줄   = 위: GT 채널(빨강, Stage1이 놓친 GT는 이름 회색) / 아래: GT 아닌 Stage1 후보(검정). 각 묶음 안은 채널 번호 순
  칸   = 1) Stage2 창 trace (Stage1 구간 ± max(60, 길이), 주황 = Stage1 구간, 빨간 빗금 = GT 구간)
         2) train 중 가장 비슷한 창   3) test 다른 곳 중 가장 비슷한 창 (자기 창 앞뒤 창 길이 + 60틱은 뺌)
            2)·3)은 핵심(Stage1 구간 ± 10틱)만으로 찾음(창마다 중앙값 빼고 MAD로 나눔), 핵심이 같은 위치에 오게 그림
         4) 그 채널의 test 전체 (빨간 세로선 = 지금 위치)
         1)–3) 세로 눈금 같음 (Stage1 규칙: train 최솟값 0 – 최댓값 1, 넘으면 넓힘)

  실행:  python feature_view.py machine-1-8 8 16      # Stage1 구간 번호(1부터)
  출력:  raw_view_feature/{machine}/{k:03d}.png (줄이 많으면 _p2 …)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, MARGIN_MIN, STAGE1_OUT, GT_C, load, segs, gt_channel_ranges, stage1_scale

PAD, EXCL, PAGE = 10, 60, 6
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def nearest(ref, v, sc, ban=None):
    W = np.lib.stride_tricks.sliding_window_view(ref, len(v))
    W = (W - np.median(W, 1, keepdims=True)) / sc; vv = (v - np.median(v)) / sc
    d = (W ** 2).sum(1) - 2 * W @ vv
    if ban is not None:
        d[max(0, ban[0]):max(0, ban[1])] = np.inf
    return int(np.argmin(d))


def draw(e, k, tr, te, y, gtr, iv, out):
    sc_tr = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    sc_te = np.maximum(1.4826 * np.median(np.abs(te - np.median(te, 0)), 0), 0.01)
    a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a); w0, w1 = max(0, a - m), min(len(te), b + m)
    p0, p1 = max(0, a - PAD), min(len(te), b + PAD); off, L = p0 - w0, w1 - w0
    gts = [(s, t) for s, t in segs(y) if s < w1 and t > w0]
    G = set().union(*[c for s, t, c in gtr if s < w1 and t > w0]) if gts else set()
    cand = {c["channel"] for c in iv["channels"]}
    rows = [(c, True, c in cand) for c in sorted(G)] + [(c, False, True) for c in sorted(cand - G)]
    pages = [rows[i:i + PAGE] for i in range(0, len(rows), PAGE)]
    for p, page in enumerate(pages):
        fig, axs = plt.subplots(len(page), 4, figsize=(24, 1.25 * len(page) + 0.7), dpi=105, squeeze=False,
                                gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.2, 2.2], wspace=0.04, hspace=0.12))
        for i, (c, is_gt, passed) in enumerate(page):
            core = te[p0:p1, c]
            t_tr = min(max(nearest(tr[:, c], core, sc_tr[c]) - off, 0), len(tr) - L)
            t_te = min(max(nearest(te[:, c], core, sc_te[c], ban=(p0 - (p1 - p0) - EXCL, p1 + EXCL)) - off, 0), len(te) - L)
            v, vtr, vte = te[w0:w1, c], tr[t_tr:t_tr + L, c], te[t_te:t_te + L, c]
            lo, hi = stage1_scale(np.r_[v, vtr, vte]); pad = 0.05 * (hi - lo); lim = (lo - pad, hi + pad)
            col = GT_C if is_gt else "black"
            for j, (x0, vv, nm) in enumerate([(w0, v, "test 창"), (t_tr, vtr, "train"), (t_te, vte, "test 다른 곳")]):
                ax = axs[i, j]; ax.plot(range(x0, x0 + L), vv, color=col if j == 0 else "#374151", lw=0.8)
                ax.axvspan(x0 + (a - w0) - 0.5, x0 + (b - w0) - 0.5, color="#f59e0b", alpha=0.22 if j == 0 else 0.08)
                if j == 0:
                    for s, t in gts:
                        ax.axvspan(max(s, w0) - 0.5, min(t, w1) - 0.5, facecolor="none", edgecolor=GT_C, hatch="////", lw=0, alpha=0.5)
                ax.set_ylim(*lim); ax.tick_params(labelsize=6, pad=1, length=2)
                ax.text(0.995, 0.93, f"{nm} [{x0}, {x0 + L})", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#374151")
                if j:
                    ax.set_yticklabels([])
            ax = axs[i, 3]; ax.plot(te[:, c], color=col if is_gt else "#374151", lw=0.3); ax.axvline((a + b) / 2, color="#2563eb", lw=1.0)
            ax.set_xlim(0, len(te)); ax.set_ylim(min(lim[0], te[:, c].min()), max(lim[1], te[:, c].max())); ax.tick_params(labelsize=6, pad=1, length=2)
            lab = f"ch {c}\n" + ("GT" if is_gt else "후보") + ("" if passed else "\n(S1 놓침)")
            axs[i, 0].set_ylabel(lab, rotation=0, labelpad=26, fontsize=8, va="center", color=(GT_C if passed else "#6b7280") if is_gt else "black")
            if i > 0 and is_gt != page[i - 1][1]:
                for ax in axs[i]:
                    ax.spines["top"].set_linewidth(2.5); ax.spines["top"].set_color("#2563eb")
        for j, t in enumerate(["1) Stage2 창 (주황 = Stage1 구간, 빨간 빗금 = GT 구간)", "2) 같은 채널 · train 중 가장 비슷한 창",
                               "3) 같은 채널 · test 다른 곳 중 가장 비슷한 창", "4) 같은 채널 · test 전체 (파란 선 = 지금 위치)"]):
            axs[0, j].set_title(t, fontsize=9)
        fig.suptitle(f"{e} · Stage1 구간 #{k} [{a}, {b}) · GT 채널 {len(G)}개(빨강, 위) / GT 아닌 후보 {len(cand - G)}개(검정, 아래)"
                     + (f" · {p + 1}/{len(pages)}" if len(pages) > 1 else ""), fontsize=11, y=1.0)
        fig.savefig(out / (f"{k:03d}" + (f"_p{p + 1}" if len(pages) > 1 else "") + ".png"), bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    return len(pages)


def main():
    e, ks = sys.argv[1], [int(x) for x in sys.argv[2:]]
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    out = Path(__file__).resolve().parent / "raw_view_feature" / e; out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        print(f"{e} #{k}: {draw(e, k, tr, te, y, gtr, ivs[k - 1], out)}장", flush=True)


if __name__ == "__main__":
    main()
