"""
GT보다 높은 점수를 받은 정상 spike — 지금 모델 (1번 + 눈금 넓힘 + 상수 0~1 + 갑작 하한 0.01, λ 없음), 보정 spike 점수
  machine마다 GT 점수(= GT 채널·GT 구간 최댓값)의 중앙값 / 최댓값을 기준으로,
  GT 구간(앞뒤 30틱 포함) 밖의 열 중 점수가 기준보다 높은 것 → 같은 채널에서 이어진 열은 한 덩어리
  그림: machine마다 가장 높은 덩어리 2개씩 (실제 값, train 범위 눈금, 앞뒤 150틱)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SC, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check" / "above_gt"
P = 14


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"{'machine':13s} | GT 점수 중앙 / 최대 | GT 중앙보다 높은 정상 덩어리 | GT 최대보다 높은 정상 덩어리 | 많이 나온 채널"]; picks = []; tot = [0, 0]
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr))
        gs = []
        for a, b in segs(y):
            g = [c for c in interp_channels(e, a, b) if np.isfinite(S[c, a:b]).any()]
            if g:
                gs.append(max(np.nanmax(S[c, a:b]) for c in g))
        if not gs:
            continue
        gmed, gmax = np.median(gs), max(gs)
        near = np.zeros(T, bool)
        for a, b in segs(y):
            near[max(0, a - 30):b + 30] = True
        ncol = T // P; Sc = S[:, :ncol * P:P]; nc = near[:ncol * P].reshape(ncol, P).any(1)
        evs = []
        for c in range(S.shape[0]):
            for u, v in segs((np.nan_to_num(Sc[c], nan=-np.inf) > gmed).astype(int) & ~nc):
                evs.append((float(np.nanmax(Sc[c, u:v])), c, u * P, v * P))
        n1, n2 = len(evs), sum(1 for s, *_ in evs if s > gmax); tot[0] += n1; tot[1] += n2
        chs = {}
        for _, c, *_ in evs:
            chs[c] = chs.get(c, 0) + 1
        top = sorted(chs.items(), key=lambda x: -x[1])[:3]
        L.append(f"{e:13s} | {gmed:5.1f} / {gmax:5.1f} | {n1:4d} | {n2:3d} | " + ", ".join(f"ch{c} {k}" for c, k in top))
        for s, c, u, v in sorted(evs, reverse=True)[:2]:
            picks.append((e, c, u, v, s, gmed, gmax, tr, te))
    L += ["", f"전체: GT 중앙보다 높은 정상 덩어리 {tot[0]}개, GT 최대보다도 높은 정상 덩어리 {tot[1]}개"]
    txt = "\n".join(L); print(txt)
    (OUT / "list.txt").write_text(txt)
    for pg in range(0, len(picks), 8):
        chunk = picks[pg:pg + 8]
        fig, ax = plt.subplots(2, 4, figsize=(22, 8), squeeze=False)
        for A, (e, c, u, v, s, gmed, gmax, tr, te) in zip(ax.flat, chunk):
            s0, s1 = max(0, u - 150), min(len(te), v + 150); g0, g1 = tr[:, c].min(), tr[:, c].max()
            A.plot(np.arange(s0, s1), (te[s0:s1, c] - g0) / (g1 - g0 + 1e-12), color="red", lw=.9)
            A.axvspan(u, v, color="orange", alpha=.3, lw=0)
            for w in range((s0 // 224 + 1) * 224, s1, 224):
                A.axvline(w, color="gray", ls="--", lw=.8)
            A.set_title(f"{e} ch{c} t={u} | 점수 {s:.1f} (GT 중앙 {gmed:.1f}, 최대 {gmax:.1f})", fontsize=9)
        for A in ax.flat[len(chunk):]:
            A.axis("off")
        fig.suptitle("GT보다 높은 점수를 받은 정상 spike (machine마다 가장 높은 2개) — 실제 값, train 범위 눈금, 주황 = 높은 점수 열", fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / f"page_{pg // 8 + 1}.png", dpi=60); plt.close(fig)


if __name__ == "__main__":
    main()
