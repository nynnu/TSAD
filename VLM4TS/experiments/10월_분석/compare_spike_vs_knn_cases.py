"""
spike 탐지기로 대체 vs 지금(patch-kNN, 잔차) — event 기준(1틱이라도 걸침) 사례 그림
  ① spike가 놓친 것: patch-kNN은 걸쳤는데 spike는 0틱
  ② spike가 잘 잡은 것: spike는 걸쳤는데 patch-kNN은 0틱
  ③ precision이 낮은 이유: spike에만 새로 생긴 오탐 덩어리 (GT와 안 겹치고, patch-kNN 예측과도 안 겹침)
그림: 그 방법이 실제로 반응한 채널(열점수>dt AND z>zt인 틱이 가장 많은 채널), 아래 띠 = 파랑 patch-kNN / 주황 spike 예측
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import best_pred, segs, PH, SC, SMD
from compare_dino_vs_num_cases import combine_thr

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
SP = HERE / "spike_scores"
RES = HERE / "compare_spike_vs_knn"
RES.mkdir(exist_ok=True)


def draw(items, cache, title, fname):
    pick = [items[i] for i in np.random.default_rng(0).choice(len(items), min(12, len(items)), replace=False)]
    fig, ax = plt.subplots(4, 3, figsize=(18, 13))
    for A, it in zip(ax.flat, pick):
        e, a, b, ch, is_gt = it
        te, y, pO, pS = cache[e]
        pad = max(200, b - a); s, t = max(0, a - pad), min(len(te), b + pad)
        x = te[s:t, ch]; lo, hi = x.min(), x.max(); rg = hi - lo + 1e-9
        A.plot(range(s, t), x, color="k", lw=.8)
        dd = np.diff(np.r_[0, y[s:t], 0])
        for x0, x1 in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
            A.axvspan(s + x0, s + x1, color="r", alpha=.15)
        A.axvspan(a, b, color="r" if is_gt else "orange", alpha=.35)
        A.fill_between(range(s, t), lo - .10 * rg, lo - .05 * rg, where=pO[s:t], color="royalblue", step="mid")
        A.fill_between(range(s, t), lo - .17 * rg, lo - .12 * rg, where=pS[s:t], color="darkorange", step="mid")
        A.set_ylim(lo - .2 * rg, hi + .05 * rg)
        A.set_title(f"{e} ch{ch} [{a},{b}) {b - a}틱", fontsize=9)
    for A in ax.flat[len(pick):]:
        A.axis("off")
    fig.suptitle(f"{title} — 아래 띠: 파랑 = 지금(patch-kNN), 주황 = spike 탐지기 / 빨강 배경 = GT", fontsize=12)
    fig.tight_layout(); fig.savefig(RES / fname, dpi=72); plt.close(fig)


def main():
    lost, won, newfp, cache = [], [], [], {}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        pO, fO = combine_thr(p1, np.load(SC / f"{e}.npz")["D"].astype(float), Z, y)
        pS, fS = combine_thr(p1, np.load(SP / f"{e}.npz")["scores"].astype(float), Z, y)
        cache[e] = (te, y, pO, pS)
        for a, b in segs(y):
            tO, tS = pO[a:b].any(), pS[a:b].any()
            if tO and not tS:
                ch = int(fO[:, a:b].sum(1).argmax()) if fO[:, a:b].any() else int(Z[:, a:b].max(1).argmax())
                lost.append((e, a, b, ch, True))
            if tS and not tO:
                ch = int(fS[:, a:b].sum(1).argmax()) if fS[:, a:b].any() else int(Z[:, a:b].max(1).argmax())
                won.append((e, a, b, ch, True))
        for a, b in segs(pS.astype(int)):
            if not y[a:b].any() and not pO[a:b].any():
                ch = int(fS[:, a:b].sum(1).argmax()) if fS[:, a:b].any() else int(Z[:, a:b].max(1).argmax())
                newfp.append((e, a, b, ch, False))
    print(f"① spike가 놓친 것 (patch-kNN은 걸침): {len(lost)}개  길이: " + ", ".join(f"{n} {sum(lo <= b - a <= hi for _, a, b, _, _ in lost)}" for lo, hi, n in [(1, 15, 'spike'), (16, 224, '16~224'), (225, 10**9, '225+')]))
    print(f"② spike가 잘 잡은 것 (patch-kNN은 0틱): {len(won)}개  길이: " + ", ".join(f"{n} {sum(lo <= b - a <= hi for _, a, b, _, _ in won)}" for lo, hi, n in [(1, 15, 'spike'), (16, 224, '16~224'), (225, 10**9, '225+')]))
    print(f"③ spike에만 새로 생긴 오탐 덩어리: {len(newfp)}개, 길이 중앙값 {int(np.median([b - a for _, a, b, _, _ in newfp]))}틱")
    draw(lost, cache, f"① spike가 놓친 것 ({len(lost)}개 중 12개)", "1_spike가_놓친것.png")
    draw(won, cache, f"② spike가 잘 잡은 것 ({len(won)}개 중 12개)", "2_spike가_잘잡은것.png")
    draw(newfp, cache, f"③ spike에만 새로 생긴 오탐 ({len(newfp)}개 중 12개, 주황 배경 = 그 오탐)", "3_spike_새오탐.png")
    print("saved to", RES)


if __name__ == "__main__":
    main()
