"""
블라인드 판정용 그림 — GT·기존 VLM 선택·Stage1 점수 없이, 채널 번호 순서 (나연, 2026-11-05) — API 호출 없음

  채널마다 4칸 (세로 눈금 모두 같음, Stage1 규칙: train 최솟값 0 – 최댓값 1, 넘으면 넓힘)
    1) test 창 = Stage2 창(Stage1 구간 ± max(60, 길이)), 주황 = Stage1 구간
    2) test의 다른 곳 중 가장 비슷한 창 — 자기 창 앞뒤 (창 길이 + 60)틱은 뺌
    3) train 중 가장 비슷한 창
       2)·3)은 핵심 부분(Stage1 구간 ± 10틱)만으로 찾고(창마다 중앙값 빼고 MAD로 나눔), 핵심이 같은 위치에 오게 창을 그림
    4) 그 채널의 test 전체 (빨간 세로선 = 지금 위치)

  실행:  python blind_view.py machine-1-8 8        # 8번째 Stage1 구간
  출력:  raw_view_blind/{machine}/{k:03d}.png (채널이 많으면 _p2 …)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, MARGIN_MIN, STAGE1_OUT, load, stage1_scale

PAD, EXCL, PAGE = 10, 60, 10
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def nearest(ref, v, sc, ban=None):
    W = np.lib.stride_tricks.sliding_window_view(ref, len(v))
    W = (W - np.median(W, 1, keepdims=True)) / sc; vv = (v - np.median(v)) / sc
    d = (W ** 2).sum(1) - 2 * W @ vv
    if ban is not None:
        d[max(0, ban[0]):ban[1]] = np.inf
    return int(np.argmin(d))


def main():
    e, k = sys.argv[1], int(sys.argv[2])
    tr, te, _ = load(BASE / "mv_data" / "SMD", e)
    sc_tr = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    sc_te = np.maximum(1.4826 * np.median(np.abs(te - np.median(te, 0)), 0), 0.01)
    iv = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"][k - 1]
    a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a); w0, w1 = max(0, a - m), min(len(te), b + m)
    p0, p1 = max(0, a - PAD), min(len(te), b + PAD); off, L = p0 - w0, w1 - w0
    chs = sorted(c["channel"] for c in iv["channels"])
    out = Path(__file__).resolve().parent / "raw_view_blind" / e; out.mkdir(parents=True, exist_ok=True)
    pages = [chs[i:i + PAGE] for i in range(0, len(chs), PAGE)]
    for p, page in enumerate(pages):
        fig, axs = plt.subplots(len(page), 4, figsize=(24, 1.25 * len(page) + 0.7), dpi=105, squeeze=False,
                                gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.2, 2.2], wspace=0.04, hspace=0.12))
        for i, c in enumerate(page):
            core = te[p0:p1, c]
            t_te = nearest(te[:, c], core, sc_te[c], ban=(p0 - (p1 - p0) - EXCL, p1 + EXCL)) - off
            t_tr = nearest(tr[:, c], core, sc_tr[c]) - off
            t_te, t_tr = min(max(t_te, 0), len(te) - L), min(max(t_tr, 0), len(tr) - L)
            v, vte, vtr = te[w0:w1, c], te[t_te:t_te + L, c], tr[t_tr:t_tr + L, c]
            lo, hi = stage1_scale(np.r_[v, vte, vtr]); pad = 0.05 * (hi - lo); lim = (lo - pad, hi + pad)
            for j, (x0, vv, nm) in enumerate([(w0, v, "test 창"), (t_te, vte, "test 다른 곳"), (t_tr, vtr, "train")]):
                ax = axs[i, j]; ax.plot(range(x0, x0 + L), vv, color="black", lw=0.8)
                ax.axvspan(x0 + (a - w0) - 0.5, x0 + (b - w0) - 0.5, color="#f59e0b", alpha=0.22 if j == 0 else 0.08)
                ax.set_ylim(*lim); ax.tick_params(labelsize=6, pad=1, length=2)
                ax.text(0.995, 0.93, f"{nm} [{x0}, {x0 + L})", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#374151")
                if j:
                    ax.set_yticklabels([])
            ax = axs[i, 3]; ax.plot(te[:, c], color="#374151", lw=0.3); ax.axvline((a + b) / 2, color="#dc2626", lw=1.0)
            ax.set_xlim(0, len(te)); ax.set_ylim(min(lim[0], te[:, c].min()), max(lim[1], te[:, c].max())); ax.tick_params(labelsize=6, pad=1, length=2)
            axs[i, 0].set_ylabel(f"ch {c}", rotation=0, labelpad=20, fontsize=9)
        for j, t in enumerate(["1) test 창 (주황 = Stage1 구간)", "2) 같은 채널 · test 다른 곳 중 가장 비슷한 창", "3) 같은 채널 · train 중 가장 비슷한 창",
                               "4) 같은 채널 · test 전체 (빨간 선 = 지금 위치)"]):
            axs[0, j].set_title(t, fontsize=9)
        fig.suptitle(f"{e} · Stage1 구간 #{k} [{a}, {b}) · 후보 {len(chs)}채널 (번호 순)" + (f" · {p + 1}/{len(pages)}" if len(pages) > 1 else ""), fontsize=11, y=1.0)
        fig.savefig(out / (f"{k:03d}" + (f"_p{p + 1}" if len(pages) > 1 else "") + ".png"), bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    print(out, len(pages), "장")


if __name__ == "__main__":
    main()
