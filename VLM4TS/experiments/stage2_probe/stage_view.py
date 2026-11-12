"""
단계별 VLM 입력 그림 (S1 → S2 → …) — 한 장 = heatmap + 채널 8줄 (기준 채널 2 + 나머지 후보 6) (나연, 2026-11-10)

  기준 채널 = Stage1 채널 점수 상위 2개 (탈주기만 = 맨 뒤) — 추가 호출 없음
  창      = 기준 채널이 앞 60틱 대비 처음 10배를 넘은 순간(on) 기준 −60 ~ +80틱 (relation_view와 같음)
  heatmap = 38채널 전부, 색 = log10(1 + |값 − 창 앞 240틱 중앙값| / MAD) (0 = 평소, 2 = 100배)
  채널 줄 = test만, 채널마다 같은 눈금: train 최솟값 0 ~ 최댓값 1 (넘으면 넓힘), 빨간 점선 = 0과 1
  S1a     = 채널 줄만 (heatmap·세로선·train·세트 없음)
  S1b     = heatmap + 채널 줄 (세로선·train·세트 없음)
  S2      = + 기준 채널이 벗어나기 시작한 순간(0) 보라 점선 (heatmap에도)
  show=True면 사람 확인용으로 GT 채널을 빨강 표시 (VLM 입력에는 False)
  실행:  python stage_view.py S1 machine-2-3 8 [--show]   → raw_view_stage/{S}/{machine}/{k:03d}_p{n}.png
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

PRE, BEFORE, AFTER, BASEWIN, PER = 60, 60, 80, 240, 6
HERE = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def prep(e, k):
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    iv = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"][k - 1]; a, b = iv["start"], iv["end"]
    s1 = {c["channel"]: (c["spike_score"] if c["spike_score"] is not None else -1e9) for c in iv["channels"]}
    roots = sorted(s1, key=lambda c: -s1[c])[:2]
    pre = te[a - 2 * PRE:a - PRE]; med = np.median(pre, 0); mad = np.maximum(1.4826 * np.median(np.abs(pre - med), 0), 0.01)
    Z = np.abs(te[a - PRE:b + PRE] - med) / mad
    hit = [np.flatnonzero(Z[:, c] > 10) for c in roots]
    on = a - PRE + int(min(h[0] for h in hit if len(h))) if any(len(h) for h in hit) else a
    on = int(np.clip(on, BASEWIN + BEFORE, len(te) - AFTER))
    G = set().union(*[c for s, t, c in gtr if s < b + PRE and t > a - PRE])
    rest = [c for c in sorted(s1) if c not in roots]
    pages = [roots + rest[i:i + PER] for i in range(0, max(len(rest), 1), PER)]
    return dict(e=e, k=k, tr=tr, te=te, a=a, b=b, s1=s1, roots=roots, on=on, G=G, pages=pages)


def draw(ctx, rows, path, stage="S1", show=False, heat=True):
    te, on, G, roots = ctx["te"], ctx["on"], ctx["G"], ctx["roots"]
    w0, w1 = on - BEFORE, on + AFTER; x = np.arange(-BEFORE, AFTER)
    base = te[w0 - BASEWIN:w0]; med = np.median(base, 0); mad = np.maximum(1.4826 * np.median(np.abs(base - med), 0), 0.01)
    H = np.log10(1 + np.abs(te[w0:w1] - med) / mad).T
    fig = plt.figure(figsize=(13, (6.0 if heat else 0.8) + 1.15 * len(rows)), dpi=105)
    gs = fig.add_gridspec(heat + len(rows), 1, height_ratios=[5.6] * heat + [1] * len(rows), hspace=0.35, top=0.95 if heat else 0.93)
    if heat:
        ax = fig.add_subplot(gs[0]); im = ax.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=2.5, interpolation="nearest",
                                                    extent=(x[0] - 0.5, x[-1] + 0.5, H.shape[0] - 0.5, -0.5))
        ax.set_yticks(range(H.shape[0])); ax.set_yticklabels([f"ch {c}" for c in range(H.shape[0])], fontsize=5.5)
        for c, lab in enumerate(ax.get_yticklabels()):
            lab.set_fontweight("bold" if c in rows else "normal"); lab.set_color(GT_C if show and c in G else "black")
        ax.tick_params(axis="x", labelsize=6); ax.set_title("전체 38채널 — 색이 밝을수록 평소보다 크게 벗어남 (굵은 이름 = 아래에 그린 채널)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.005)
        if stage >= "S2":
            ax.axvline(0, color="#22d3ee", ls=":", lw=1.4)
    for i, c in enumerate(rows):
        ax = fig.add_subplot(gs[heat + i]); v = te[w0:w1, c]; lo, hi = stage1_scale(v); pad = 0.05 * (hi - lo)
        ax.plot(x, v, color=GT_C if show and c in G else "black", lw=0.9)
        for yv in (0, 1):
            ax.axhline(yv, color="#b91c1c", lw=0.5, ls="--")
        if stage >= "S2":
            ax.axvline(0, color="#7c3aed", ls=":", lw=1.2)
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(lo - pad, hi + pad); ax.tick_params(labelsize=6)
        tag = "기준 채널" if c in roots else ""
        ax.set_ylabel(f"ch {c}\n{tag}", rotation=0, labelpad=26, fontsize=8.5, va="center", weight="bold" if c in roots else "normal",
                      color=GT_C if show and c in G else "black")
    head = f"{ctx['e']} · 채널 {rows} · 각 줄 눈금: 0 = 그 채널의 평소(train) 최솟값, 1 = 평소 최댓값 (빨간 점선)"
    if show:
        head = f"[확인용] {ctx['e']} · Stage1 구간 #{ctx['k']} [{ctx['a']}, {ctx['b']}) · {stage} · GT 채널 {sorted(G)} (빨강)\n" + head
    fig.suptitle(head, fontsize=9.5, y=0.985)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    stage, e, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); show = "--show" in sys.argv
    ctx = prep(e, k)
    for p, rows in enumerate(ctx["pages"], 1):
        path = HERE / "raw_view_stage" / stage / e / f"{k:03d}_p{p}{'_show' if show else ''}.png"
        draw(ctx, rows, path, stage, show, heat=stage != "S1a"); print(path)
