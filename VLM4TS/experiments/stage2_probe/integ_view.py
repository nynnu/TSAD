"""
통합 모델(S-int) VLM 입력 그림 + 프롬프트용 세트 정보 (나연, 2026-11-10)

  기준 채널 = Stage1 채널 점수 상위 2개 (탈주기만 = 맨 뒤)
  창        = Stage1 구간 [a, b) 앞 60틱 ~ 뒤 60틱 (구간에 맞춤)
  시작(0)   = 구간 시작 −10틱부터 기준 채널이 창 앞 60틱 대비 처음 10배를 넘은 틱 (없으면 구간 시작), 보라 점선
  heatmap   = Stage1 후보 채널만, log10(1 + |값 − 창 앞 240틱 중앙값| / MAD), 굵은 이름 = 아래에 그린 채널
  왼쪽 줄   = test만, 눈금 = 0 = train 최솟값, 1 = train 최댓값 (넘으면 넓힘), 빨간 점선 = 0과 1
  오른쪽 2칸 = Stage1 구간 ±10틱, 기준 채널(회색 점선)과 이 채널 — train 칸 / test 칸, 같은 눈금 (값 그대로, 두 채널 범위 합침)
              train 시점 = 기준 채널 2개가 창 전체에서 가장 비슷한 곳 (모든 채널 같은 시점)
  세트      = 후보 채널끼리 train 1차 차분 상관 |r| ≥ 0.5 → 이어지면 한 묶음 (GT 안 씀, 프롬프트에 글로)
  show=True면 사람 확인용으로 GT 빨강 (VLM 입력에는 False)
  실행:  python integ_view.py machine-1-2 40 [--show]  → raw_view_integ/{machine}/{k:03d}_p{n}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, STAGE1_OUT, GT_C, load, gt_channel_ranges, stage1_scale

PRE, CTX, BASEWIN, PER, RTHR = 60, 60, 240, 6, 0.5
HERE = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def sets(tr, cand):
    d = np.diff(tr[:, cand], axis=0); ok = d.std(0) > 1e-9; R = np.zeros((len(cand),) * 2)
    R[np.ix_(ok, ok)] = np.abs(np.corrcoef(d[:, ok].T)); par = list(range(len(cand)))
    find = lambda i: i if par[i] == i else find(par[i])
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if R[i, j] >= RTHR:
                par[find(i)] = find(j)
    g = {}
    for i, c in enumerate(cand):
        g.setdefault(find(i), []).append(c)
    return [sorted(v) for v in g.values() if len(v) > 1]


def prep(e, k):
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    sc = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    iv = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"][k - 1]; a, b = iv["start"], iv["end"]
    s1 = {c["channel"]: (c["spike_score"] if c["spike_score"] is not None else -1e9) for c in iv["channels"]}
    roots = sorted(s1, key=lambda c: -s1[c])[:2]
    w0, w1 = max(a - CTX, BASEWIN + PRE), min(b + CTX, len(te))
    pre = te[w0 - PRE:w0]; med = np.median(pre, 0); mad = np.maximum(1.4826 * np.median(np.abs(pre - med), 0), 0.01)
    s0 = max(a - 10, w0); Z = np.abs(te[s0:b] - med) / mad
    hit = [np.flatnonzero(Z[:, c] > 10) for c in roots]
    on = s0 + int(min(h[0] for h in hit if len(h))) if any(len(h) for h in hit) else a
    L = w1 - w0; q = te[w0:w1, roots] / sc[roots]
    W = sliding_window_view(tr[:, roots], L, axis=0) / sc[roots][None, :, None]
    t0 = int(np.argmin(((W - q.T[None]) ** 2).sum((1, 2))))                              # train 창 시작
    G = set().union(*[c for s, t, c in gtr if s < b and t > a])
    rest = [c for c in sorted(s1) if c not in roots]
    pages = [roots + rest[i:i + PER] for i in range(0, max(len(rest), 1), PER)]
    return dict(e=e, k=k, tr=tr, te=te, a=a, b=b, s1=s1, roots=roots, on=on, w0=w0, w1=w1, t0=t0, G=G, pages=pages,
                sets=sets(tr, sorted(s1)))


def draw(ctx, rows, path, show=False, mark=None, note=""):
    """mark = {채널: 색} (채점 확인용, 주면 show보다 우선) · note = 제목 첫 줄"""
    tr, te, on, w0, w1, t0, G, roots, a, b = (ctx[x] for x in ("tr", "te", "on", "w0", "w1", "t0", "G", "roots", "a", "b"))
    x = np.arange(w0, w1) - on; cand = sorted(ctx["s1"])
    base = te[w0 - BASEWIN:w0]; med = np.median(base, 0); mad = np.maximum(1.4826 * np.median(np.abs(base - med), 0), 0.01)
    H = np.log10(1 + np.abs(te[w0:w1, cand] - med[cand]) / mad[cand]).T
    r0, r1 = max(a - 10, w0), min(b + 10, w1); xs = np.arange(r0, r1) - on                  # 오른쪽 두 칸 = Stage1 구간 ±10틱
    fig = plt.figure(figsize=(17, 0.16 * len(cand) + 1.5 + 1.15 * len(rows)), dpi=105)
    gs = fig.add_gridspec(1 + len(rows), 3, height_ratios=[0.16 * len(cand) + 0.6] + [1] * len(rows), width_ratios=[3.2, 1, 1],
                          hspace=0.4, wspace=0.12, top=0.95)
    ax = fig.add_subplot(gs[0, 0]); im = ax.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=2.5, interpolation="nearest",
                                                   extent=(x[0] - 0.5, x[-1] + 0.5, len(cand) - 0.5, -0.5))
    ax.set_yticks(range(len(cand))); ax.set_yticklabels([f"ch {c}" for c in cand], fontsize=6)
    for c, lab in zip(cand, ax.get_yticklabels()):
        lab.set_fontweight("bold" if c in rows else "normal"); lab.set_color(mark.get(c, "black") if mark else (GT_C if show and c in G else "black"))
    ax.axvline(0, color="#22d3ee", ls=":", lw=1.4); ax.tick_params(axis="x", labelsize=6)
    ax.set_title("후보 채널 heatmap — 색이 밝을수록 평소보다 크게 벗어남 (굵은 이름 = 아래에 그린 채널) · 점선 = 기준 채널이 벗어나기 시작한 순간", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.005)
    lim = lambda c: stage1_scale(np.r_[te[w0:w1, c], tr[t0:t0 + (w1 - w0), c]])
    for i, c in enumerate(rows):
        col = mark.get(c, "black") if mark else (GT_C if show and c in G else "black")
        ax = fig.add_subplot(gs[1 + i, 0]); v = te[w0:w1, c]; lo, hi = stage1_scale(v); pad = 0.05 * (hi - lo)
        ax.plot(x, v, color=col, lw=0.9)
        for yv in (0, 1):
            ax.axhline(yv, color="#b91c1c", lw=0.5, ls="--")
        ax.axvline(0, color="#7c3aed", ls=":", lw=1.2); ax.set_xlim(x[0], x[-1]); ax.set_ylim(lo - pad, hi + pad); ax.tick_params(labelsize=6)
        ax.set_ylabel(f"ch {c}\n{'기준 채널' if c in roots else ''}", rotation=0, labelpad=26, fontsize=8.5, va="center",
                      weight="bold" if c in roots else "normal", color=col)
        q0, q1 = lim(roots[0]), lim(c); lo2, hi2 = min(q0[0], q1[0]), max(q0[1], q1[1]); pad2 = 0.05 * (hi2 - lo2)
        for j, (X, off, cc) in enumerate(((tr, t0 - w0, "#2563eb"), (te, 0, col)), 1):
            ax = fig.add_subplot(gs[1 + i, j]); ax.plot(xs, X[r0 + off:r1 + off, roots[0]], color="#9ca3af", lw=1.6, ls="--")
            if c != roots[0]:
                ax.plot(xs, X[r0 + off:r1 + off, c], color=cc, lw=1.0)
            ax.axvline(0, color="#7c3aed", ls=":", lw=1.0); ax.set_ylim(lo2 - pad2, hi2 + pad2); ax.tick_params(labelsize=6)
            if j == 2:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(("평소(train) 관계\n회색 점선 = ch %d, 파랑 = 이 채널" if j == 1 else "지금(test) 관계\n회색 점선 = ch %d, 검정 = 이 채널") % roots[0], fontsize=8.5)
    head = (f"{ctx['e']} · 채널 {rows} · 왼쪽 = 지금(test), 눈금 0 = 평소 최솟값, 1 = 평소 최댓값 (빨간 점선) · "
            f"오른쪽 = 기준 채널 ch {roots[0]}이 지금과 가장 비슷하게 움직였던 평소 시점 vs 지금 (같은 눈금)")
    if note:
        head = note + "\n" + head
    elif show:
        head = f"[확인용] Stage1 구간 #{ctx['k']} [{a}, {b}) · train {t0} · GT 채널 {sorted(G)} (빨강)\n" + head
    fig.suptitle(head, fontsize=9.5, y=1.02 if note else 0.985)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    e, k = sys.argv[1], int(sys.argv[2]); show = "--show" in sys.argv
    ctx = prep(e, k); print("기준", ctx["roots"], "· 시작", ctx["on"], "· 창", (ctx["w0"], ctx["w1"]), "· train", ctx["t0"], "· 세트", ctx["sets"])
    for p, rows in enumerate(ctx["pages"], 1):
        path = HERE / "raw_view_integ" / e / f"{k:03d}_p{p}{'_show' if show else ''}.png"; draw(ctx, rows, path, show); print(path.name, rows)
