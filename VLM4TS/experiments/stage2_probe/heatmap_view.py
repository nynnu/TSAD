"""
원인 채널을 heatmap으로 고를 수 있는지 보는 그림 (나연, 2026-11-07) — API 호출 없음

  줄  = 38채널 전부 (Stage1 후보는 이름 굵게, GT 채널은 빨강 — 사람이 확인하는 용, VLM 입력에는 빼야 함)
  칸  = Stage1 창(구간 ± max(60, 길이))의 시점마다 |값 − 앞 60틱 중앙값| / 앞 60틱 MAD, log10 색 (0 = 평소, 2 = 100배)
  선  = 주황 세로선 = Stage1 구간, 빨간 세로선 = GT 구간
  실행:  python heatmap_view.py machine-2-3 [구간번호 …]   → raw_view_heatmap/{machine}/{k:03d}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, MARGIN_MIN, STAGE1_OUT, GT_C, load, segs, gt_channel_ranges

PRE = 60
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False


def main():
    e = sys.argv[1]; data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e)
    ivs = json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]
    ks = [int(x) for x in sys.argv[2:]] or [k for k, iv in enumerate(ivs, 1) if y[iv["start"]:iv["end"]].any()]
    out = Path(__file__).resolve().parent / "raw_view_heatmap" / e; out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        iv = ivs[k - 1]; a, b = iv["start"], iv["end"]; m = max(MARGIN_MIN, b - a)
        w0, w1 = max(PRE, a - m), min(len(te), b + m)
        pre = te[w0 - PRE:w0]; med = np.median(pre, 0); mad = np.maximum(1.4826 * np.median(np.abs(pre - med), 0), 0.01)
        Z = np.log10(1 + np.abs(te[w0:w1] - med) / mad).T                                  # (38, L)
        cand = {c["channel"] for c in iv["channels"]}
        G = set().union(*[c for s, t, c in gtr if s < w1 and t > w0])
        fig, ax = plt.subplots(figsize=(13, 8), dpi=105)
        im = ax.imshow(Z, aspect="auto", cmap="magma", vmin=0, vmax=2.5, extent=(w0 - 0.5, w1 - 0.5, Z.shape[0] - 0.5, -0.5), interpolation="nearest")
        for x in (a, b):
            ax.axvline(x - 0.5, color="#f59e0b", lw=1.2)
        for s, t in segs(y):
            if s < w1 and t > w0:
                for x in (max(s, w0), min(t, w1)):
                    ax.axvline(x - 0.5, color=GT_C, lw=1.0, ls="--")
        ax.set_yticks(range(Z.shape[0]))
        ax.set_yticklabels([f"ch {c}" for c in range(Z.shape[0])], fontsize=6.5)
        for c, lab in enumerate(ax.get_yticklabels()):
            lab.set_color(GT_C if c in G else "black"); lab.set_fontweight("bold" if c in cand else "normal")
        ax.tick_params(axis="x", labelsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("log10(1 + 평소 대비 몇 배)", fontsize=8)
        tag = "TP" if y[a:b].any() else "FP"
        ax.set_title(f"{e} · Stage1 구간 #{k} [{a}, {b}) · {tag} · 색 = 앞 60틱 대비 벗어난 정도 · 주황선 = Stage1 구간, 빨간 점선 = GT\n"
                     f"채널 이름: 빨강 = GT 채널 {sorted(G)}, 굵게 = Stage1 후보 ({len(cand)}개)", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"{k:03d}_{tag}.png", bbox_inches="tight"); plt.close(fig)
        print(out / f"{k:03d}_{tag}.png", flush=True)


if __name__ == "__main__":
    main()
