"""
가정 5-2: "spike 방향"으로 256 patch 전부에 점수를 매기면 spike 자리만 밝아지나?
  (a) 635번 숫자 하나 (학습 없음)
  (b) 768개 방향 = A vs 일반 선(B+C) 분류기. 공정하게: 그 그래프의 entity를 뺀 나머지로 학습한 방향으로만 점수
정량: 창마다 점수 1등 patch가 빨강(A)인 비율 / 배경 포함 256칸 전체에서 A vs 나머지 AUROC
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "exp_spike_heatmap"
RES.mkdir(exist_ok=True)
DIM, GRID, P = 635, 16, 14


def main():
    pt = np.load(HERE / "exp_patch_groups" / "patches.npz", allow_pickle=True)
    win = np.load(HERE / "exp_patch_groups" / "windows.npz", allow_pickle=True)
    X, G, ENT = pt["X"], pt["G"], pt["ENT"]
    V, gw, imgs, meta = win["V"].astype(np.float32), win["g"], win["img"], win["meta"]
    V = V.reshape(len(V), GRID * GRID, -1)                          # (n, 16, 16, 768) -> (n, 256, 768)
    n = len(V)
    m = np.isin(G, ["A", "B", "C"])
    Xl, yl, el = X[m], (G[m] == "A").astype(int), ENT[m]
    sign = 1 if Xl[yl == 1, DIM].mean() > Xl[yl == 0, DIM].mean() else -1

    s635 = sign * V[:, :, DIM]                                     # (n, 256)
    sdir = np.zeros((n, GRID * GRID))
    win_ent = np.array([mt[0] for mt in meta])
    for tri, tei in GroupKFold(5).split(Xl, yl, el):
        held = set(el[tei])
        sc = StandardScaler().fit(Xl[tri])
        clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(Xl[tri]), yl[tri])
        for i in np.where(np.isin(win_ent, list(held)))[0]:
            sdir[i] = clf.decision_function(sc.transform(V[i]))

    res = {}
    for name, S in [("635번", s635), ("768방향", sdir)]:
        top_is_A = np.mean([gw[i].reshape(-1)[S[i].argmax()] == "A" for i in range(n)])
        isA = (gw.reshape(n, -1) == "A").reshape(-1)
        auc_all = roc_auc_score(isA, S.reshape(-1))
        res[name] = (top_is_A, auc_all)
        print(f"[{name}] 창마다 1등 patch가 빨강인 비율: {top_is_A * 100:.0f}%   256칸 전체(배경 포함) A vs 나머지 AUROC: {auc_all:.3f}")

    order = np.random.default_rng(0).permutation(n)
    for f0 in range(0, min(n, 20), 5):
        idx = order[f0:f0 + 5]
        fig, ax = plt.subplots(len(idx), 3, figsize=(15, 4.3 * len(idx)))
        for r, i in enumerate(idx):
            g = gw[i].reshape(GRID, GRID)
            ax[r, 0].imshow(imgs[i])
            for rr in range(GRID):
                for cc in range(GRID):
                    if g[rr, cc] == "A":
                        ax[r, 0].add_patch(Rectangle((cc * P - .5, rr * P - .5), P, P, fill=False, ec="red", lw=1.2))
            e, a, b, ch = meta[i]
            ax[r, 0].set_ylabel(f"{e} ch{ch}\nGT [{a},{b})", fontsize=9)
            for c, (S, t) in enumerate([(s635, "635번 숫자"), (sdir, "768개 방향 (그 entity 빼고 학습)")], start=1):
                ax[r, c].imshow(imgs[i])
                hm = ax[r, c].imshow(np.kron(S[i].reshape(GRID, GRID), np.ones((P, P))), cmap="jet", alpha=.55)
                plt.colorbar(hm, ax=ax[r, c], fraction=.046)
                tr_, tc_ = divmod(int(S[i].argmax()), GRID)
                ax[r, c].add_patch(Rectangle((tc_ * P - .5, tr_ * P - .5), P, P, fill=False, ec="white", lw=2))
                if r == 0:
                    ax[r, c].set_title(f"{t}\n(흰 테두리 = 점수 1등 patch)", fontsize=10)
            if r == 0:
                ax[r, 0].set_title("그래프 (빨간 테두리 = GT spike patch)", fontsize=10)
            for a_ in ax[r]:
                a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f"spike 방향 점수 — 1등이 빨강인 비율: 635번 {res['635번'][0]*100:.0f}%, 768방향 {res['768방향'][0]*100:.0f}%", fontsize=12)
        fig.tight_layout(); fig.savefig(RES / f"히트맵_{f0 // 5 + 1:02d}.png", dpi=80); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
