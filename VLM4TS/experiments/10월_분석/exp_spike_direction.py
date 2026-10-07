"""
가정 5-1: spike(빨강) patch는 768개 숫자 중 소수 몇 개에 몰려 표현되는가, 골고루 퍼져 있는가?
비교: A(GT spike) vs 일반 선(B 작은 튐 + C 평범한 선). T/F(높은 정상 spike)는 모양상 spike라 제외.
  (1) 숫자 하나씩만 써서 AUROC — 혼자서 spike를 가르는 숫자가 있나
  (2) 분류기 가중치 크기 순으로 top-k개만 써서 AUROC (k 선택은 학습 fold 안에서만) — 몇 개면 충분한가
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "exp_spike_direction"
RES.mkdir(exist_ok=True)
KS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 768]


def main():
    d = np.load(HERE / "exp_patch_groups" / "patches.npz", allow_pickle=True)
    X, G, ENT = d["X"], d["G"], d["ENT"]
    m = np.isin(G, ["A", "B", "C"])
    X, y, grp = X[m], (G[m] == "A").astype(int), ENT[m]
    print(f"A(spike) {y.sum()}개 vs 일반 선 {len(y) - y.sum()}개, entity {len(set(grp))}개")

    # (1) 숫자 하나씩
    single = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
    single = np.maximum(single, 1 - single)                       # 방향 상관없이 얼마나 가르나
    order1 = np.argsort(-single)
    print("숫자 하나만으로 AUROC 상위 5개:", [(int(j), round(float(single[j]), 3)) for j in order1[:5]])
    print(f"  AUROC 0.9 넘는 숫자: {(single > 0.9).sum()}개, 0.8 넘는 숫자: {(single > 0.8).sum()}개")

    # (2) top-k (학습 fold에서 가중치로 고르고, 시험 fold에서 평가)
    auc_k = {k: np.zeros(len(y)) for k in KS}
    w_all = np.zeros(X.shape[1])
    for tri, tei in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tri])
        full = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tri]), y[tri])
        w = np.abs(full.coef_[0]); w_all += w
        rank = np.argsort(-w)
        for k in KS:
            idx = rank[:k]
            clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tri])[:, idx], y[tri])
            auc_k[k][tei] = clf.predict_proba(sc.transform(X[tei])[:, idx])[:, 1]
    auc_k = {k: float(roc_auc_score(y, v)) for k, v in auc_k.items()}
    for k in KS:
        print(f"  top-{k:3d}개 숫자로 AUROC: {auc_k[k]:.3f}")
    w_all /= 5
    ws = np.sort(w_all)[::-1]
    cum = np.cumsum(ws) / ws.sum()
    print(f"가중치 합의 50%를 차지하는 숫자 수: {int(np.searchsorted(cum, .5)) + 1}개, 80%: {int(np.searchsorted(cum, .8)) + 1}개")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].hist(single, bins=40, color="gray"); ax[0].axvline(.5, color="k", ls=":")
    ax[0].set_title("① 숫자 하나만으로 spike를 가르는 정도\n(768개 각각의 AUROC 분포)"); ax[0].set_xlabel("AUROC (0.5 = 못 가름)"); ax[0].set_ylabel("숫자 개수")
    ax[1].plot(range(1, 769), cum, color="k"); ax[1].axhline(.5, color="gray", ls=":"); ax[1].axhline(.8, color="gray", ls=":")
    ax[1].set_xscale("log"); ax[1].set_title("② 분류기 가중치: 큰 순서로 몇 개가 대부분을 차지하나"); ax[1].set_xlabel("숫자 개수 (큰 가중치부터, 로그 축)"); ax[1].set_ylabel("가중치 누적 비율")
    ax[2].plot(KS, [auc_k[k] for k in KS], "o-", color="red"); ax[2].set_xscale("log")
    for k in KS:
        ax[2].annotate(f"{auc_k[k]:.3f}", (k, auc_k[k]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax[2].set_title("③ 중요한 숫자 k개만 써도 spike를 가를 수 있나\n(처음 보는 entity에서 시험)"); ax[2].set_xlabel("쓴 숫자 개수 k (로그 축)"); ax[2].set_ylabel("AUROC")
    fig.suptitle("가정 5-1: GT spike patch vs 일반 선 patch — 768개 숫자 중 몇 개가 spike를 담당하나", fontsize=13)
    fig.tight_layout(); fig.savefig(RES / "spike_숫자_몇개.png", dpi=90)
    np.save(RES / "weight_abs_mean.npy", w_all)
    print("saved to", RES)


if __name__ == "__main__":
    main()
