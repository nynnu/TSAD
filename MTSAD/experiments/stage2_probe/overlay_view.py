"""
Stage1 후보 채널 전부(한 장에 8개씩) test 창 + train 가장 비슷한 시점 하나를 파랑으로 겹쳐 그림 (나연, 2026-11-07) — API 호출 없음

  구간  = GT와 겹친 Stage1 구간 (구간 번호를 주면 그것만)
  채널  = Stage1 후보 전부, 채널 번호 순으로 8개씩 나눠 여러 장. ★ = 기준 채널(점수 1위, 탈주기만 = 맨 뒤)
  창    = Stage1 구간 ± max(60, 길이)
  train = 기준 채널의 핵심(Stage1 구간 ± 10틱)이 가장 비슷한 train 시점 하나 (창마다 중앙값 빼고 MAD로 나눔)
          → 모든 채널에 같은 시점의 train을 파랑으로 겹침
  색    = test: GT 채널 빨강 / 아니면 검정, train 파랑. 주황 배경 = Stage1 구간, 빨간 빗금 = GT 구간
  실행:  python overlay_view.py machine-2-2 [구간번호 …]   → raw_view_overlay/{machine}/{k:03d}_{TP|FP}_p{n}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_view import nearest
from plot_raw_view import BASE, MARGIN_MIN, STAGE1_OUT, GT_C, load, segs, gt_channel_ranges, stage1_scale

PAD, NCH = 10, 8
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def main():
    e = sys.argv[1]; data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    sc = np.maximum(1.4826 * np.median(np.abs(tr - np.median(tr, 0)), 0), 0.01)
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    ks = [int(x) for x in sys.argv[2:]] or [k for k, iv in enumerate(ivs, 1) if y[iv["start"]:iv["end"]].any()]
    out = Path(__file__).resolve().parent / "raw_view_overlay" / e; out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        iv = ivs[k - 1]; a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a)
        w0, w1 = max(0, a - m), min(len(te), b + m); L = w1 - w0; p0, p1 = max(0, a - PAD), min(len(te), b + PAD)
        chs = sorted(iv["channels"], key=lambda c: -(c["spike_score"] if c["spike_score"] is not None else -1e9))
        top = chs[0]["channel"]; s1 = {c["channel"]: c["spike_score"] for c in chs}
        tref = min(max(nearest(tr[:, top], te[p0:p1, top], sc[top]) - (p0 - w0), 0), len(tr) - L)
        G = set().union(*[c for s, t, c in gtr if s < w1 and t > w0]); gts = [(s, t) for s, t in segs(y) if s < w1 and t > w0]
        tag = "TP" if y[a:b].any() else "FP"; allr = sorted(s1); pages = [allr[i:i + NCH] for i in range(0, len(allr), NCH)]
        for pn, rows in enumerate(pages, 1):
          fig, axs = plt.subplots(len(rows), 1, figsize=(13, 1.35 * len(rows) + 0.9), dpi=105, sharex=True, squeeze=False)
          x = np.arange(w0, w1)
          for ax, c in zip(axs[:, 0], rows):
              v, vr = te[w0:w1, c], tr[tref:tref + L, c]; lo, hi = stage1_scale(np.r_[v, vr]); pad = 0.05 * (hi - lo)
              ax.plot(x, vr, color="#2563eb", lw=0.9, alpha=0.8); ax.plot(x, v, color=GT_C if c in G else "black", lw=0.9)
              ax.axvspan(a - 0.5, b - 0.5, color="#f59e0b", alpha=0.18, lw=0)
              for s, t in gts:
                  ax.axvspan(max(s, w0) - 0.5, min(t, w1) - 0.5, facecolor="none", edgecolor=GT_C, hatch="////", lw=0, alpha=0.45)
              for yv in (0, 1):
                  ax.axhline(yv, color="#b91c1c", lw=0.4, ls="--")
              ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(w0, w1 - 1); ax.tick_params(labelsize=6)
              sc_txt = "탈주기" if s1[c] is None else f"{s1[c]:.1f}"
              ax.set_ylabel(f"ch {c}{' ★' if c == top else ''}\nS1 {sc_txt}{'  GT' if c in G else ''}", rotation=0, labelpad=30, fontsize=8,
                            va="center", color=GT_C if c in G else "black")
          fig.suptitle(f"{e} · Stage1 구간 #{k} [{a}, {b}) · {tag} · 후보 {len(allr)}개 중 {pn}/{len(pages)}장 · GT 채널 {sorted(G)}\n"
                       f"검정/빨강 = test (빨강 = GT 채널), 파랑 = train [{tref}, {tref + L}) — ★ ch {top} 기준으로 가장 비슷한 시점 하나, 모든 채널 같은 시점",
                       fontsize=9.5, y=1.0)
          fig.tight_layout(); fig.savefig(out / f"{k:03d}_{tag}_p{pn}.png", bbox_inches="tight"); plt.close(fig)
          print(out / f"{k:03d}_{tag}_p{pn}.png", flush=True)


if __name__ == "__main__":
    main()
