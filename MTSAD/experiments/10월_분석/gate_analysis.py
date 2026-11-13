"""
가정 10: DINOv2·숫자 버전이 둘 다 놓친 이상(탈주기가 놓친 212개 중 173개)이 어느 관문에서 떨어졌나
관문 (이상 구간 안, 38채널 중 하나라도 통과하면 통과):
  0. 그 entity에서 2차가 켜졌나 (오라클로 골라보니 아무것도 안 더하는 게 최선이면 2차가 꺼짐)
  1. 열점수가 있는 채널이 있나 (train에서 거의 안 변하는 채널 / 캘리브 점수가 고른 채널은 점수 없음)
  2. 열점수 > dt 인 채널이 있나
  3. z > zt 인 채널이 있나
  4. 같은 채널·같은 시점에 2와 3이 동시에 성립하나
  5. 동시에 성립한 틱이 이상 구간의 50% 이상인가
버전마다 따로 세고, "처음 떨어진 관문"으로 분류.
"""
import json
from pathlib import Path

import numpy as np

from eval_dsz_allch import prf, f1, best_pred, segs, PH, SC, SMD, DQ, ZQ

HERE = Path(__file__).resolve().parent
GATES = ["0. 2차가 꺼진 entity", "1. 점수 있는 채널 없음", "2. 열점수 < dt", "3. z < zt",
         "4. 둘이 같은 채널·시점에 안 겹침", "5. 겹치지만 구간 50% 미만", "(통과 — 잡음)"]


def search(p1, S, Z, y):
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
    dts = np.unique(np.percentile(S[fin], DQ)); zts = np.unique(np.percentile(Z, ZQ))
    best, bt = f1(*prf(y, p1)), None
    for dt in dts:
        Db = Sv > dt
        for zt in zts:
            v = f1(*prf(y, p1 | (Db & (Z > zt)).any(0)))
            if v > best:
                best, bt = v, (dt, zt)
    return bt


def gate(S, Z, bt, a, b):
    if bt is None:
        return 0
    dt, zt = bt
    s, z = S[:, a:b], Z[:, a:b]
    fin = np.isfinite(s)
    if not fin.any():
        return 1
    dpass = np.where(fin, s, -np.inf) > dt
    if not dpass.any():
        return 2
    zpass = z > zt
    if not zpass.any():
        return 3
    both = dpass & zpass
    if not both.any():
        return 4
    return 5 if both.any(0).mean() < .5 else 6


def main():
    cases = json.load(open(HERE / "compare_dino_vs_num" / "cases.json"))
    target = {(c["e"], c["a"], c["b"]) for c in cases if c["cat"] == "둘 다 놓침"}
    cnt = {v: np.zeros(len(GATES), int) for v in ["DINOv2", "숫자"]}
    by_len = {v: {g: [0, 0, 0] for g in range(len(GATES))} for v in ["DINOv2", "숫자"]}
    rows = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        d = np.load(SC / f"{e}.npz")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        S = {"DINOv2": d["D"].astype(float), "숫자": d["B"].astype(float)}
        bts = {v: search(p1, S[v], Z, y) for v in S}
        for a, b in segs(y):
            if (e, int(a), int(b)) not in target:
                continue
            L = b - a; li = 0 if L <= 15 else (1 if L <= 224 else 2)
            r = dict(e=e, a=int(a), b=int(b), L=int(L))
            for v in S:
                g = gate(S[v], Z, bts[v], a, b)
                cnt[v][g] += 1; by_len[v][g][li] += 1; r[v] = g
            rows.append(r)
    json.dump(rows, open(HERE / "compare_dino_vs_num" / "gate_cases.json", "w"), ensure_ascii=False, indent=1)
    n = len(rows)
    print(f"둘 다 놓친 이상 {n}개 — 처음 떨어진 관문 (spike / 16~224 / 225+)\n")
    print(f"{'':32s}{'DINOv2 버전':>22s}{'숫자 버전':>22s}")
    for g, name in enumerate(GATES):
        cells = [f"{cnt[v][g]:4d} ({cnt[v][g] / n * 100:3.0f}%) [{'/'.join(map(str, by_len[v][g]))}]" for v in ["DINOv2", "숫자"]]
        print(f"{name:32s}{cells[0]:>22s}{cells[1]:>22s}")


if __name__ == "__main__":
    main()
