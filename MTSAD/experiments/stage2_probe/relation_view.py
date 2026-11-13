"""
원인 채널이 train과 달라지는 순간부터, 다른 채널도 달라지나 + 원인 채널과의 관계가 train과 test에서 다르나 (나연, 2026-11-07)
사람이 눈으로 가설을 확인하는 예시용 그림 — API 호출 없음

  원인  = Stage1 채널 점수 상위 2개 (탈주기만 = 맨 뒤, stage_view와 같음, GT 안 씀)
  시작  = 원인 채널이 처음 평소의 10배를 넘은 틱 (세로 점선)
  train = 정상 채널(Stage1이 후보에서 뺀 채널 전부)이 창 전체(시작 −60 ~ +80틱)에서 가장 비슷한 train 시점 하나
          (값 그대로, 채널마다 train MAD로 나눔) → 모든 채널 같은 시점
          --root면 대신 맨 위 원인 채널 하나가 창 전체에서 가장 비슷한 train 시점 (→ {k:03d}_root.png)
          --root2면 원인 채널 2개가 함께 창 전체에서 가장 비슷한 train 시점 (→ {k:03d}_root2.png)
  왼쪽  = 채널마다 test(빨강 = GT, 검정 = 아님) vs train(파랑), 시작 순간 정렬
  오른쪽 = 시작 −10 ~ +40틱, 원인 채널(회색 점선)과 이 채널을 겹침 — train 칸 / test 칸
          눈금 = 왼쪽과 같음 (값 그대로, train 최소 0 ~ 최대 1, 넘으면 넓힘) — 이 채널과 원인 채널 범위를 합쳐서, train·test 두 칸 같은 눈금
  실행:  python relation_view.py machine-2-3 8 [채널 …]   → raw_view_relation/{machine}/{k:03d}.png
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

PRE, BEFORE, AFTER, SHOW = 60, 60, 80, (-10, 40)
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def prep(e, k, ref="normal"):
    """구간 하나의 원인 후보 · 시작 · train 시점 (GT는 사람 확인용 G에만)"""
    data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    sc = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    iv = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"][k - 1]; a, b = iv["start"], iv["end"]
    cand = sorted(c["channel"] for c in iv["channels"])
    pre = te[a - 2 * PRE:a - PRE]; med = np.median(pre, 0); mad = np.maximum(1.4826 * np.median(np.abs(pre - med), 0), 0.01)
    Z = np.abs(te[a - PRE:b + PRE] - med) / mad
    s1 = {c["channel"]: (c["spike_score"] if c["spike_score"] is not None else -1e9) for c in iv["channels"]}
    roots = sorted(s1, key=lambda c: -s1[c])[:2]
    hit = [np.flatnonzero(Z[:, c] > 10) for c in roots]
    on = a - PRE + int(min(h[0] for h in hit if len(h))) if any(len(h) for h in hit) else a
    on = int(np.clip(on, BEFORE, len(te) - AFTER))
    G = set().union(*[c for s, t, c in gtr if s < b + PRE and t > a - PRE])
    # train 시점: 정상 채널(Stage1 후보가 아닌 채널 전부)이 창 전체에서 가장 비슷한 곳 (값 그대로, 채널마다 train MAD로 나눔)
    normal = [c for c in range(te.shape[1]) if c not in cand]; use = {"normal": normal, "root": roots[:1], "root2": roots}[ref]
    q = te[on - BEFORE:on + AFTER, use] / sc[use]
    W = sliding_window_view(tr[:, use], BEFORE + AFTER, axis=0) / sc[use][None, :, None]
    t0 = int(np.argmin(((W - q.T[None]) ** 2).sum((1, 2)))) + BEFORE                       # train에서 "시작"에 해당하는 틱
    return dict(e=e, k=k, tr=tr, te=te, a=a, b=b, cand=cand, roots=roots, on=on, G=G, normal=normal, t0=t0, ref=ref)


def draw(ctx, rows, path, blind=False):
    """blind = VLM 입력용: GT 색·글자, Stage1 구간 표시 없음"""
    tr, te, a, b, roots, on, G, t0 = (ctx[x] for x in ("tr", "te", "a", "b", "roots", "on", "G", "t0"))
    x = np.arange(-BEFORE, AFTER)
    fig, axs = plt.subplots(len(rows), 3, figsize=(17, 1.45 * len(rows) + 1), dpi=105, squeeze=False,
                            gridspec_kw=dict(width_ratios=[3.2, 1, 1], wspace=0.12, hspace=0.35))
    xs = np.arange(*SHOW); win = lambda X, t, c: X[t + SHOW[0]:t + SHOW[1], c]
    lim = lambda c: stage1_scale(np.r_[te[on - BEFORE:on + AFTER, c], tr[t0 - BEFORE:t0 + AFTER, c]])  # 왼쪽 칸과 같은 눈금
    for i, c in enumerate(rows):
        v, vr = te[on - BEFORE:on + AFTER, c], tr[t0 - BEFORE:t0 + AFTER, c]
        lo, hi = stage1_scale(np.r_[v, vr]); pad = 0.05 * (hi - lo); col = "black" if blind or c not in G else GT_C
        ax = axs[i, 0]; ax.plot(x, vr, color="#2563eb", lw=0.9, alpha=0.85); ax.plot(x, v, color=col, lw=0.9)
        ax.axvline(0, color="#7c3aed", ls=":", lw=1.2)
        if not blind:
            ax.axvspan(a - on, b - on, color="#f59e0b", alpha=0.12, lw=0)
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(lo - pad, hi + pad); ax.tick_params(labelsize=6)
        tag = "원인 후보" if c in roots else ("" if blind else ("GT" if c in G else "GT 아님"))
        ax.set_ylabel(f"ch {c}\n{tag}", rotation=0, labelpad=30, fontsize=8.5, va="center", color=col, weight="bold" if c in roots else "normal")
        r0, r1 = lim(roots[0]), lim(c); lo2, hi2 = min(r0[0], r1[0]), max(r0[1], r1[1]); pad2 = 0.05 * (hi2 - lo2)
        for j, (X, t, cc) in enumerate(((tr, t0, "#2563eb"), (te, on, col)), 1):
            ax = axs[i, j]; ax.plot(xs, win(X, t, roots[0]), color="#9ca3af", lw=1.6, ls="--")
            if c != roots[0]:
                ax.plot(xs, win(X, t, c), color=cc, lw=1.0)
            ax.axvline(0, color="#7c3aed", ls=":", lw=1.0); ax.set_ylim(lo2 - pad2, hi2 + pad2); ax.tick_params(labelsize=6)
            if j == 2:
                ax.set_yticklabels([])
    if blind:
        axs[0, 0].set_title("시간축: 원인 후보가 평소와 크게 달라지기 시작한 순간 = 0 (보라 점선) · 검정 = test, 파랑 = train", fontsize=9)
        axs[0, 2].set_title(f"test 관계\n회색 점선 = ch {roots[0]}, 검정 = 이 채널", fontsize=8.5)
    else:
        axs[0, 0].set_title(f"시간축: 원인 채널이 평소의 10배를 처음 넘은 순간 = 0 (보라 점선) · 파랑 = train [{t0}] 같은 위치 (" + (f"정상 채널 {len(ctx['normal'])}개가" if ctx.get('ref', 'normal') == 'normal' else f"원인 채널 ch {roots if ctx['ref'] == 'root2' else roots[0]}이") + " 창 전체에서 가장 비슷한 곳)", fontsize=9)
        axs[0, 2].set_title(f"test 관계\n회색 점선 = ch {roots[0]}, 색 = 이 채널", fontsize=8.5)
    axs[0, 1].set_title(f"train 관계\n회색 점선 = ch {roots[0]}, 파랑 = 이 채널", fontsize=8.5)
    fig.text(0.79, 0.005, f"오른쪽 두 칸: 0 기준 {SHOW[0]}~+{SHOW[1]}틱, 눈금은 왼쪽과 같음 (train 최소 0 ~ 최대 1, 넘으면 넓힘) · train 칸과 test 칸 같은 눈금", ha="center", fontsize=8, color="#6b7280")
    head = f"{ctx['e']} · 원인 후보 ch {roots}" if blind else f"{ctx['e']} · Stage1 구간 #{ctx['k']} [{a}, {b}) · 원인 후보 ch {roots} (Stage1 점수 상위 2개) · 시작 {on} · GT 채널 {sorted(G)}"
    fig.suptitle(head, fontsize=10.5, y=0.995)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ref = "root2" if "--root2" in sys.argv else ("root" if "--root" in sys.argv else "normal"); args = [x for x in sys.argv[1:] if not x.startswith("--")]
    e, k = args[0], int(args[1]); extra = [int(x) for x in args[2:]]
    ctx = prep(e, k, ref); rows = ctx["roots"] + [c for c in (extra or ctx["cand"]) if c not in ctx["roots"]]
    path = Path(__file__).resolve().parent / "raw_view_relation" / e / f"{k:03d}{'' if ref == 'normal' else '_' + ref}.png"
    draw(ctx, rows, path); print(path, "train", ctx["t0"])


if __name__ == "__main__":
    main()
