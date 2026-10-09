"""
가정 10 — 관문 2를 엄격하게: 이상이 가장 크게 나타난 채널(구간 안 z 최대 채널)에서 열점수가 dt를 넘었나
  대상: 둘 다 놓친 173개 중, 그 채널에 점수가 있는 것 (관문 1에서 떨어진 것 제외)
  dt: entity마다 결합 F1 최대인 (dt, zt) (eval과 같은 탐색)
  같이 보는 것: 그 채널 train에 이상 구간만큼 높은 잔차가 있었나 (= train에 비슷한 spike가 있었나)
그림: 원래 신호 + 열점수(오른쪽 축) + dt 선
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import best_pred, PH, SC, SMD
from gate_analysis import search
from run_dsz_allch import residual

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "compare_dino_vs_num"


def main():
    rows = json.load(open(RES / "gate_cases.json"))
    by_e = {}
    for r in rows:
        by_e.setdefault(r["e"], []).append(r)
    out = {v: [] for v in ["DINOv2", "숫자"]}
    for e, rs in sorted(by_e.items()):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        d = np.load(SC / f"{e}.npz")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        for v, key in [("DINOv2", "D"), ("숫자", "B")]:
            S = d[key].astype(float)
            bt = search(p1, S, Z, y)
            for r in rs:
                a, b = r["a"], r["b"]
                c = int(Z[:, a:b].max(1).argmax())
                s = S[c, a:b]
                if not np.isfinite(s).any():
                    continue                                          # 관문 1
                if bt is None:
                    continue                                          # 관문 0 (2차 꺼짐)
                rtr, rte = residual(tr[:, c], te[:, c])
                seen = bool(np.abs(rtr - np.median(rtr)).max() >= np.abs(rte[a:b] - np.median(rtr)).max())
                out[v].append(dict(e=e, a=a, b=b, L=b - a, ch=c, smax=float(np.nanmax(s)), dt=float(bt[0]),
                                   fail=bool(np.nanmax(s) <= bt[0]), seen_in_train=seen))
    json.dump(out, open(RES / "gate2_cases.json", "w"), ensure_ascii=False, indent=1)
    for v, lst in out.items():
        f = [x for x in lst if x["fail"]]
        print(f"[{v}] 점수 있는 채널로 판단한 {len(lst)}개 중 관문 2(열점수 ≤ dt)에서 떨어짐: {len(f)}개 "
              f"(spike {sum(x['L'] <= 15 for x in f)}, 16~224 {sum(15 < x['L'] <= 224 for x in f)}, 225+ {sum(x['L'] > 224 for x in f)})")
        if f:
            print(f"     그중 train에 이 구간만큼 크게 튄 잔차가 있던 것: {sum(x['seen_in_train'] for x in f)}개 "
                  f"/ 통과한 것 중에서는 {sum(x['seen_in_train'] for x in lst if not x['fail'])}/{sum(not x['fail'] for x in lst)}개")
            print(f"     열점수/dt 중앙값: {np.median([x['smax'] / x['dt'] for x in f]):.2f}")

    # 그림: DINOv2 버전에서 관문 2로 떨어진 것
    f = [x for x in out["DINOv2"] if x["fail"]]
    pick = [f[i] for i in np.random.default_rng(0).choice(len(f), min(12, len(f)), replace=False)]
    fig, ax = plt.subplots(4, 3, figsize=(18, 13))
    cache = {}
    for A, x in zip(ax.flat, pick):
        if x["e"] not in cache:
            cache[x["e"]] = (np.loadtxt(SMD / "train" / f"{x['e']}.txt", delimiter=","),
                             np.loadtxt(SMD / "test" / f"{x['e']}.txt", delimiter=","), np.load(SC / f"{x['e']}.npz")["D"])
        tr, te, D = cache[x["e"]]
        s, t = max(0, x["a"] - 200), min(len(te), x["b"] + 200)
        A.plot(range(s, t), te[s:t, x["ch"]], color="k", lw=.8)
        A.axvspan(x["a"], x["b"], color="r", alpha=.3)
        A.axhline(tr[:, x["ch"]].max(), color="b", ls="--", lw=.7)
        A2 = A.twinx(); A2.plot(range(s, t), D[x["ch"], s:t], color="royalblue", lw=1.2, alpha=.8)
        A2.axhline(x["dt"], color="royalblue", ls=":", lw=1.2)
        A.set_title(f"{x['e']} ch{x['ch']} {x['L']}틱 | 열점수 최고 {x['smax']:.1f} < dt {x['dt']:.1f} | train에 이만큼 튐: {'있음' if x['seen_in_train'] else '없음'}", fontsize=8.5)
    for A in ax.flat[len(pick):]:
        A.axis("off")
    fig.suptitle("관문 2 (DINOv2 버전): 이상 채널의 열점수가 dt를 못 넘은 이상 — 검정 = 신호, 파랑 = 열점수(오른쪽 축), 파란 점선 = dt, 진한 파란 점선(왼쪽) = train 최대", fontsize=12)
    fig.tight_layout(); fig.savefig(RES / "관문2_열점수부족.png", dpi=72)
    print("saved")


if __name__ == "__main__":
    main()
