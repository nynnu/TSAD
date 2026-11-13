"""
가정 6: spike 방향(가짜 spike로 학습)은 긴 이상(16~224틱: level shift, trend 등)에는 안 먹힌다.
탈주기가 놓친 16~224틱 GT 이상 중 채널 라벨 있는 것 → 이상 전체가 들어가게 224틱 그래프 (위치 무작위)
  GT 열의 선 patch vs 정상 열의 선 patch를 spike 방향 점수로 AUROC + 히트맵
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, WIN, GRID  # noqa: E402
from exp_patch_groups import WHO, interp_channels, pick_channel  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "exp_long_anomaly_check"
RES.mkdir(exist_ok=True)
P = 224 // GRID


def main():
    with open(HERE / "exp_synthetic_spike" / "direction.pkl", "rb") as f:
        sc, clf = pickle.load(f)
    rows = json.load(open(WHO))
    segs = [r for r in rows if 16 <= r["L"] <= WIN - 2 * P and not r["phase"]]
    model = load_model(); rng = np.random.default_rng(0)
    cache, sA, sN, shows = {}, [], [], []
    for r in segs:
        e, a, b = r["e"], r["a"], r["b"]
        chs = interp_channels(e, a, b)
        if not chs:
            continue
        if e not in cache:
            cache[e] = (np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","),
                        np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=","),
                        np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int))
        tr, te, y = cache[e]
        ch = pick_channel(tr, te, a, b, chs)
        if ch is None:
            continue
        L = b - a
        off = int(rng.integers(P, WIN - P - L + 1))
        s = int(np.clip(a - off, 0, len(te) - WIN))
        img = ts_to_image_global(te[s:s + WIN, ch], tr[:, ch].min(), tr[:, ch].max())
        px = np.asarray(img)[:, :, 0] < 128
        yw = y[s:s + WIN]
        _, pt, _ = extract(model, [img])
        S = clf.decision_function(sc.transform(pt[0])).reshape(GRID, GRID)
        gtcol = np.array([yw[c * P:(c + 1) * P].mean() >= .5 for c in range(GRID)])
        for rr in range(GRID):
            for cc in range(GRID):
                if px[rr * P:(rr + 1) * P, cc * P:(cc + 1) * P].any():
                    (sA if gtcol[cc] else sN).append(S[rr, cc])
        shows.append((img, S, gtcol, e, ch, a, b))
    auc = roc_auc_score(np.r_[np.ones(len(sA)), np.zeros(len(sN))], np.r_[sA, sN])
    print(f"긴 이상 그래프 {len(shows)}개: GT 열 선 patch {len(sA)}개 vs 정상 열 선 patch {len(sN)}개")
    print(f"spike 방향 점수로 AUROC (GT 열 vs 정상 열): {auc:.3f}   (spike일 때 1등 88%, 256칸 0.989)")
    json.dump(dict(n=len(shows), auroc=float(auc)), open(RES / "results.json", "w"))

    pick = rng.choice(len(shows), min(10, len(shows)), replace=False)
    fig, ax = plt.subplots(2, 5, figsize=(20, 8.5))
    for a_, i in zip(ax.flat, pick):
        img, S, gtcol, e, ch, a, b = shows[i]
        a_.imshow(img); a_.imshow(np.kron(S, np.ones((P, P))), cmap="jet", alpha=.5)
        for cc in np.where(gtcol)[0]:
            a_.add_patch(Rectangle((cc * P - .5, -.5), P, 224, fill=False, ec="white", lw=.8))
        a_.set_title(f"{e} ch{ch} GT {b - a}틱", fontsize=9); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"긴 이상(16~224틱)에 spike 방향 점수 — 흰 세로줄 = GT 구간, AUROC {auc:.3f}", fontsize=13)
    fig.tight_layout(); fig.savefig(RES / "긴이상_히트맵.png", dpi=80)
    print("saved to", RES)


if __name__ == "__main__":
    main()
