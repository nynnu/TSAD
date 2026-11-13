"""
어떤 GT event가 어떤 세트인지 보는 그림 (나연, 2026-11-10) — API 호출 없음, 사람이 라벨 구조를 보는 용

  묶음  = GT 채널 집합끼리 Jaccard ≥ 0.3이면 같은 묶음 (이어지면 하나로)
  채널  = 묶음 안 GT 채널 합집합 전부 + 묶음 event 중 하나에서라도 앞 60틱 대비 10배 넘게 튄 GT 아닌 채널 (튄 정도 순 최대 6개), 번호 순
  ① ev{n}.png      = 왼쪽 test event 창 (GT ± max(30, 길이)) · 오른쪽 그 창이 가장 비슷한 train 시점 (표시 채널 전부, 창마다 중앙값 빼고 train MAD로 나눔)
                     빨강 = 이 event의 GT 채널, 검정 = 아님, 연한 빨강 배경 = GT 구간
  ② group{g}.png   = 같은 묶음의 event들을 열로 나란히 (같은 채널 줄), 빨강 = 그 event의 GT 채널
  실행:  python set_view.py machine-2-3   → raw_view_set/{machine}/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, GT_C, load, gt_channel_ranges, stage1_scale

PRE, EXTRA = 60, 6
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def groups(gtr):
    n = len(gtr); par = list(range(n))
    find = lambda i: i if par[i] == i else find(par[i])
    for i in range(n):
        for j in range(i + 1, n):
            A, B = gtr[i][2], gtr[j][2]
            if len(A & B) / len(A | B) >= 0.3:
                par[find(i)] = find(j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return sorted(out.values(), key=lambda g: g[0])


def main():
    e = sys.argv[1]; data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    sc = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    out = Path(__file__).resolve().parent / "raw_view_set" / e; out.mkdir(parents=True, exist_ok=True)
    win = lambda s, t: (max(PRE, s - max(30, t - s)), min(len(te), t + max(30, t - s)))
    def zmax(s, t):
        a, b = win(s, t); pre = te[a - PRE:a]; med = np.median(pre, 0)
        return (np.abs(te[a:b] - med) / np.maximum(1.4826 * np.median(np.abs(pre - med), 0), 0.01)).max(0)
    for gi, g in enumerate(groups(gtr), 1):
        U = set().union(*[gtr[i][2] for i in g]); Z = np.max([zmax(*gtr[i][:2]) for i in g], 0)
        rows = sorted(U | set(sorted((c for c in range(te.shape[1]) if Z[c] > 10 and c not in U), key=lambda c: -Z[c])[:EXTRA]))
        print(f"묶음 {gi}: event {[i + 1 for i in g]} · 채널 {rows}", flush=True)
        for i in g:                                                                          # ① event vs 가장 비슷한 train
            s, t, G = gtr[i]; a, b = win(s, t); L = b - a
            q = (te[a:b, rows] - np.median(te[a:b, rows], 0)) / sc[rows]
            W = sliding_window_view(tr[:, rows], L, axis=0); W = (W - np.median(W, 2, keepdims=True)) / sc[rows][None, :, None]
            t0 = int(np.argmin(((W - q.T[None]) ** 2).mean((1, 2))))
            fig, axs = plt.subplots(len(rows), 2, figsize=(15, 1.05 * len(rows) + 1), dpi=100, squeeze=False, sharex="col",
                                    gridspec_kw=dict(wspace=0.04, hspace=0.18))
            for r, c in enumerate(rows):
                v, vr = te[a:b, c], tr[t0:t0 + L, c]; lo, hi = stage1_scale(np.r_[v, vr]); pad = 0.05 * (hi - lo)
                for j, (vv, col) in enumerate(((v, GT_C if c in G else "black"), (vr, "#2563eb"))):
                    ax = axs[r, j]; ax.plot(np.arange(L), vv, color=col, lw=0.8); ax.set_ylim(lo - pad, hi + pad)
                    ax.axvspan(s - a, t - a, color=GT_C if j == 0 else "#f59e0b", alpha=0.10, lw=0); ax.tick_params(labelsize=6)
                    if j:
                        ax.set_yticklabels([])
                axs[r, 0].set_ylabel(f"ch {c}", rotation=0, labelpad=18, fontsize=8, color=GT_C if c in G else "black")
            axs[0, 0].set_title(f"test GT event {i + 1} [{s}, {t})", fontsize=10); axs[0, 1].set_title(f"가장 비슷한 train [{t0}, {t0 + L})", fontsize=10)
            fig.suptitle(f"{e} · 묶음 {gi} · event {i + 1} · GT 채널 {sorted(G)} (빨강)", fontsize=11, y=1.0)
            fig.savefig(out / f"ev{i + 1:02d}.png", bbox_inches="tight"); plt.close(fig)
        if len(g) < 2:
            continue
        fig, axs = plt.subplots(len(rows), len(g), figsize=(3.6 * len(g) + 1, 1.0 * len(rows) + 1.2), dpi=100, squeeze=False,
                                gridspec_kw=dict(wspace=0.08, hspace=0.18))                 # ② 같은 묶음 event 나란히
        for j, i in enumerate(g):
            s, t, G = gtr[i]; a, b = win(s, t)
            for r, c in enumerate(rows):
                ax = axs[r, j]; v = te[a:b, c]; lo, hi = stage1_scale(v); pad = 0.05 * (hi - lo)
                ax.plot(np.arange(a, b), v, color=GT_C if c in G else "black", lw=0.8); ax.axvspan(s, t, color=GT_C, alpha=0.10, lw=0)
                ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(a, b - 1); ax.tick_params(labelsize=5.5)
                if j == 0:
                    ax.set_ylabel(f"ch {c}", rotation=0, labelpad=18, fontsize=8)
            axs[0, j].set_title(f"event {i + 1} [{s}, {t})\nGT {sorted(G)}", fontsize=8)
        fig.suptitle(f"{e} · 묶음 {gi} · 같은 세트처럼 보이는 event 나란히 · 빨강 = 그 event의 GT 채널", fontsize=11, y=1.0)
        fig.savefig(out / f"group{gi}.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
