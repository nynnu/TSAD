"""
씨앗 채널 + 평소 같이 변하는 그룹 + train 평소 모습 (나연, 2026-11-06) — API 호출 없음, 사람이 눈으로 보는 용

  씨앗  = 구간 Stage1 점수 1위 채널
  그룹  = train 한 틱 변화 상관(|Spearman|)이 씨앗과 높은 채널 순 7개 (상관 ≥ 0.2만)
  칸    = 1) test 창 (Stage1 구간 ± max(60, 길이)), 주황 = Stage1 구간, 빨간 빗금 = GT 구간
          2) train 평소 모습 = 창 앞 60틱(그룹 채널 전부)이 가장 비슷한 train 시점 8곳의 이어지는 창
             파랑 = 시점마다 median, 띠 = 10–90% (이상 구간 자체로는 찾지 않음)
  줄 이름 = 채널 · 씨앗과 상관 · Stage1 점수(후보 아니면 −) · GT면 빨강
  실행:  python group_view.py machine-2-9 8 11 12   → raw_view_group/{machine}/{k:03d}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, MARGIN_MIN, STAGE1_OUT, GT_C, load, segs, gt_channel_ranges, stage1_scale

PRE, NREF, NGRP = 60, 8, 7
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def refs(tr, te, w0, L, chs):
    q = te[w0 - PRE:w0, chs]
    win = sliding_window_view(tr[:len(tr) - L, chs], PRE, axis=0)            # (n, C, PRE)
    d = ((win - q.T[None]) ** 2).sum((1, 2)); out = []
    for t in np.argsort(d):
        if all(abs(t - u) >= L for u in out):
            out.append(t)
        if len(out) == NREF:
            break
    return [t + PRE for t in out]


def main():
    e, ks = sys.argv[1], [int(x) for x in sys.argv[2:]]
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    ok = np.ptp(tr, 0) > 1e-6
    R = np.abs(np.nan_to_num(spearmanr(np.diff(tr, axis=0)).statistic)); R[:, ~ok] = 0
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    out = Path(__file__).resolve().parent / "raw_view_group" / e; out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        iv = ivs[k - 1]; a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a)
        w0, w1 = max(PRE, a - m), min(len(te), b + m); L = w1 - w0
        s1 = {c["channel"]: c["spike_score"] for c in iv["channels"]}; seed = iv["channels"][0]["channel"]
        G = set().union(*[c for s, t, c in gtr if s < b and t > a])
        peers = [c for c in np.argsort(-R[seed]) if c != seed and R[seed, c] >= 0.2][:NGRP]
        rows = [seed] + peers; T = refs(tr, te, w0, L, rows)
        gts = [(s, t) for s, t in segs(y) if s < w1 and t > w0]
        fig, axs = plt.subplots(len(rows), 2, figsize=(16, 1.25 * len(rows) + 0.9), dpi=105, squeeze=False,
                                gridspec_kw=dict(wspace=0.04, hspace=0.15))
        for i, c in enumerate(rows):
            v = te[w0:w1, c]; B = np.stack([tr[t:t + L, c] for t in T]); med, lo10, hi90 = np.median(B, 0), *np.percentile(B, [10, 90], 0)
            lo, hi = stage1_scale(np.r_[v, B.ravel()]); pad = 0.05 * (hi - lo); x = np.arange(L)
            ax = axs[i, 0]; ax.plot(x, v, color=GT_C if c in G else "black", lw=0.9)
            ax.axvspan(a - w0 - 0.5, b - w0 - 0.5, color="#f59e0b", alpha=0.22)
            for s, t in gts:
                ax.axvspan(max(s, w0) - w0 - 0.5, min(t, w1) - w0 - 0.5, facecolor="none", edgecolor=GT_C, hatch="////", lw=0, alpha=0.5)
            ax = axs[i, 1]; ax.fill_between(x, lo10, hi90, color="#93c5fd", alpha=0.5, lw=0); ax.plot(x, med, color="#2563eb", lw=0.9)
            ax.plot(x, v, color="black", lw=0.5, alpha=0.35); ax.axvspan(a - w0 - 0.5, b - w0 - 0.5, color="#f59e0b", alpha=0.08)
            for ax in axs[i]:
                for yv in (0, 1):
                    ax.axhline(yv, color="#b91c1c", lw=0.5, ls="--")
                ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(0, L - 1); ax.tick_params(labelsize=6, pad=1, length=2)
            axs[i, 1].set_yticklabels([])
            sc = "−" if c not in s1 else ("탈주기" if s1[c] is None else f"{s1[c]:.1f}")
            axs[i, 0].set_ylabel(f"ch {c}{' (씨앗)' if c == seed else f' r={R[seed, c]:.2f}'}\nS1 {sc}{'  GT' if c in G else ''}",
                                 rotation=0, labelpad=34, fontsize=8, va="center", color=GT_C if c in G else "black")
        axs[0, 0].set_title(f"1) test 창 [{w0}, {w1}) · 주황 = Stage1 구간, 빨간 빗금 = GT", fontsize=9)
        axs[0, 1].set_title(f"2) train 평소 모습 — 앞 {PRE}틱이 비슷한 {NREF}곳 · 파랑 median, 띠 10–90% · 회색 = test", fontsize=9)
        fig.suptitle(f"{e} · 구간 #{k} [{a}, {b}) · 씨앗 ch {seed} + 평소 같이 변하는 {len(peers)}개 · GT 채널 {sorted(G)}", fontsize=10, y=1.0)
        fig.savefig(out / f"{k:03d}.png", bbox_inches="tight", pad_inches=0.08); plt.close(fig)
        print(out / f"{k:03d}.png", "GT∩그룹", sorted(G & set(rows)), "후보", sorted(s1), flush=True)


if __name__ == "__main__":
    main()
