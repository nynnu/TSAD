"""
② 가능성 확인: 2차의 열점수를 spike 탐지기 점수로 대체하면 놓친 이상을 잡을 가능성이 있나
  지금: 탈주기 OR (patch-kNN 열점수[잔차 그림] > dt AND z > zt)      ← dsz_scores D
  대체: 탈주기 OR (spike 점수[원본 그림, 가짜 spike로 배운 방향] > dt AND z > zt)  ← spike_scores scores
  구조·임계값 탐색(entity당 (dt, zt) 한 쌍, 오라클) 동일. 문제 1(잔차 희석)과 3(두루뭉실)을 한꺼번에 바꾼 비교.
확인: ① 놓친 173개에서 이상 위치가 그 창(같은 채널)의 1등 열인가  ② 0틱/일부/잡음 그룹 변화  ③ 전체 P/R/F1
"""
import json
from pathlib import Path

import numpy as np

from eval_dsz_allch import prf, f1, best_pred, segs, combine, PH, SC, SMD

HERE = Path(__file__).resolve().parent
SP = HERE / "spike_scores"
WIN, P = 224, 14
GROUPS = ["0틱", "일부(1틱~50%미만)", "잡음(50%이상)"]


def grp(cov):
    return GROUPS[0] if cov == 0 else (GROUPS[1] if cov < .5 else GROUPS[2])


def main():
    old = {(r["e"], r["a"], r["b"]): r for r in json.load(open(HERE / "coverage_groups" / "cases.json"))}
    missed = {k for k, r in old.items() if r["group"] != GROUPS[2]}
    tot = {k: [0, 0, 0] for k in ["지금 (patch-kNN, 잔차)", "대체 (spike 방향, 원본)"]}
    loc = {k: [0, 0, 0] for k in tot}                                   # [1등이 이상 위치, ±1열, 판단 가능 수]
    trans = {}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        S = {"지금 (patch-kNN, 잔차)": np.load(SC / f"{e}.npz")["D"].astype(float),
             "대체 (spike 방향, 원본)": np.load(SP / f"{e}.npz")["scores"].astype(float)}
        pred = {}
        for k, s in S.items():
            pred[k] = combine(p1, s, Z, y)[0]
            c = prf(y, pred[k])
            for i in range(3):
                tot[k][i] += c[i]
        for a, b in segs(y):
            key = (e, int(a), int(b))
            if key not in missed:
                continue
            r = old[key]; ch = r["zmax_ch"]; w0 = (a // WIN) * WIN
            ac = {(t - w0) // P for t in range(a, min(b, w0 + WIN))}
            for k, s in S.items():
                if w0 + WIN > s.shape[1]:
                    continue
                col = s[ch, w0:w0 + WIN:P]
                if not np.isfinite(col).any():
                    continue
                top = int(np.nanargmax(col)); loc[k][2] += 1
                loc[k][0] += int(top in ac); loc[k][1] += int(min(abs(top - x) for x in ac) <= 1)
            new = grp(float(pred["대체 (spike 방향, 원본)"][a:b].mean()))
            trans[(r["group"], new)] = trans.get((r["group"], new), 0) + 1

    print("[③ 전체]")
    for k, (tp, fp, fn) in tot.items():
        print(f"  {k:24s} P {tp / (tp + fp):.3f}  R {tp / (tp + fn):.3f}  F1 {f1(tp, fp, fn):.3f}")
    print("\n[① 놓친 173개 — 같은 채널·같은 창에서 점수 1등 열이 이상 위치인가]")
    for k, (at, near, n) in loc.items():
        print(f"  {k:24s} 정확히 {at}/{n} ({at / n * 100:.0f}%)   ±1열 {near}/{n} ({near / n * 100:.0f}%)")
    print("\n[② 지금 그룹 → 대체했을 때 그룹]")
    for g0 in GROUPS[:2]:
        s = sum(v for (a_, _), v in trans.items() if a_ == g0)
        print(f"  지금 {g0} {s}개 → " + ", ".join(f"{g1} {trans.get((g0, g1), 0)}개" for g1 in GROUPS))


if __name__ == "__main__":
    main()
