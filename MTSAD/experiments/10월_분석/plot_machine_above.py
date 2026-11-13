"""
한 machine만: GT vs GT보다 높은 점수를 받은 정상 덩어리 (지금 모델, 보정 spike 점수)
  gt.png: GT마다 점수가 가장 높은 GT 채널의 실제 값 (train 범위 눈금, 앞뒤 150틱)
  normal_N.png: GT 점수 중앙값보다 높은 정상 덩어리, 점수 높은 순 (24개씩)
  timeline.png: 채널 × 시간에 GT(빨강)와 높은 점수 정상 덩어리(파랑) 위치
  사용: python3 plot_machine_above.py machine-3-5
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
P, W = 14, 150


def panel(A, te, tr, c, u, v, title, col):
    s0, s1 = max(0, u - W), min(len(te), v + W); g0, g1 = (tr[:, c].min(), tr[:, c].max()) if np.ptp(tr[:, c]) > 1e-6 else (0.0, 1.0)
    A.plot(np.arange(s0, s1), (te[s0:s1, c] - g0) / (g1 - g0 + 1e-12), color=col, lw=.9)
    A.axvspan(u, max(v, u + 2), color="orange", alpha=.3, lw=0)
    for w in range((s0 // 224 + 1) * 224, s1, 224):
        A.axvline(w, color="gray", ls="--", lw=.7)
    A.set_title(title, fontsize=8); A.tick_params(labelsize=6)


def main():
    e = sys.argv[1]; out = HERE / "stage12_check" / "above_gt" / e; out.mkdir(parents=True, exist_ok=True)
    y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
    raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
    S, _ = calib(drop_flat(raw, te, tr)); gts = []
    for a, b in segs(y):
        g = [c for c in interp_channels(e, a, b) if np.isfinite(S[c, a:b]).any()]
        if g:
            c = max(g, key=lambda k: np.nanmax(S[k, a:b])); gts.append((np.nanmax(S[c, a:b]), c, a, b))
    gmed = np.median([s for s, *_ in gts])
    near = np.zeros(T, bool)
    for a, b in segs(y):
        near[max(0, a - 30):b + 30] = True
    ncol = T // P; Sc = S[:, :ncol * P:P]; nc = near[:ncol * P].reshape(ncol, P).any(1); evs = []
    for c in range(S.shape[0]):
        for u, v in segs((np.nan_to_num(Sc[c], nan=-np.inf) > gmed).astype(int) & ~nc):
            evs.append((float(np.nanmax(Sc[c, u:v])), c, u * P, v * P))
    evs.sort(reverse=True)
    n = len(gts); fig, ax = plt.subplots(int(np.ceil(n / 6)), 6, figsize=(22, 3.2 * int(np.ceil(n / 6))), squeeze=False)
    for A, (s, c, a, b) in zip(ax.flat, sorted(gts, reverse=True)):
        panel(A, te, tr, c, a, b, f"GT [{a},{b}) {b - a}틱 ch{c} | 점수 {s:.1f}", "royalblue")
    for A in ax.flat[n:]:
        A.axis("off")
    fig.suptitle(f"{e} GT {n}개 (GT 채널 중 점수가 가장 높은 채널) | GT 점수 중앙값 {gmed:.1f}", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "gt.png", dpi=60); plt.close(fig)
    for pg in range(0, min(len(evs), 48), 24):
        ch = evs[pg:pg + 24]; fig, ax = plt.subplots(4, 6, figsize=(22, 12), squeeze=False)
        for A, (s, c, u, v) in zip(ax.flat, ch):
            panel(A, te, tr, c, u, v, f"정상 ch{c} t={u} | 점수 {s:.1f}", "red")
        for A in ax.flat[len(ch):]:
            A.axis("off")
        fig.suptitle(f"{e} GT 중앙값({gmed:.1f})보다 높은 정상 덩어리 {len(evs)}개 중 {pg + 1}~{pg + len(ch)}번째 (점수 높은 순)", fontsize=12)
        fig.tight_layout(); fig.savefig(out / f"normal_{pg // 24 + 1}.png", dpi=60); plt.close(fig)
    fig, A = plt.subplots(figsize=(22, 7))
    for s, c, a, b in gts:
        A.add_patch(plt.Rectangle((a, c - 0.4), max(b - a, 40), 0.8, color="red"))
    for s, c, u, v in evs:
        A.add_patch(plt.Rectangle((u, c - 0.4), max(v - u, 40), 0.8, color="royalblue", alpha=.7))
    A.set_xlim(0, T); A.set_ylim(-1, S.shape[0]); A.set_yticks(range(0, S.shape[0], 2)); A.set_ylabel("채널"); A.set_xlabel("틱")
    A.set_title(f"{e} 위치: 빨강 = GT (점수 가장 높은 GT 채널), 파랑 = GT 중앙값보다 높은 정상 덩어리 {len(evs)}개", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "timeline.png", dpi=60); plt.close(fig)
    cnt = {}
    for _, c, *_ in evs:
        cnt[c] = cnt.get(c, 0) + 1
    print(f"GT {n}개, GT 점수 중앙 {gmed:.2f} | 높은 정상 덩어리 {len(evs)}개 | 채널별: " + ", ".join(f"ch{c} {k}" for c, k in sorted(cnt.items(), key=lambda x: -x[1])))
    print("GT:", ", ".join(f"[{a}] ch{c} {s:.1f}" for s, c, a, b in sorted(gts, reverse=True)))


if __name__ == "__main__":
    main()
