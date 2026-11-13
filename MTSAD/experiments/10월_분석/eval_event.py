"""
event 단위 채점 (Stage2로는 후보 구간 단위로 넘어가므로)
  event recall(걸침): GT 이상 중 1틱이라도 이상으로 표시된 비율 → Stage2에 넘어간 비율
  event recall(50%): GT 이상 중 50% 이상 덮은 비율
  event precision: 예측 덩어리(연속된 이상 틱) 중 GT와 한 틱이라도 겹치는 비율
  event F1: recall(걸침)과 precision으로
  예측 덩어리 수: Stage2가 볼 후보 개수
예측은 지금까지와 같음 (entity별 틱 단위 F1 최대 임계값). 틱 단위 P/R/F1도 같이 표시.
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import prf, f1, best_pred, segs, combine, PH, SC, SMD, ZQ

HERE = Path(__file__).resolve().parent
SP = HERE / "spike_scores"
BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def z_only(p1, Z, y):
    best, bp = f1(*prf(y, p1)), p1
    for zt in np.unique(np.percentile(Z, ZQ)):
        p = p1 | (Z > zt).any(0); v = f1(*prf(y, p))
        if v > best:
            best, bp = v, p
    return bp


def main():
    names = ["탈주기 단독", "탈주기 + z-score", "+ DINOv2 (patch-kNN, 잔차)", "+ 숫자", "+ spike 탐지기 (원본)"]
    pt = {k: [0, 0, 0] for k in names}
    ev = {k: dict(touch=0, half=0, n=0, pred=0, pred_hit=0, touch_len={b[2]: 0 for b in BINS}) for k in names}
    n_len = {b[2]: 0 for b in BINS}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        d = np.load(SC / f"{e}.npz")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        preds = {names[0]: p1, names[1]: z_only(p1, Z, y),
                 names[2]: combine(p1, d["D"].astype(float), Z, y)[0],
                 names[3]: combine(p1, d["B"].astype(float), Z, y)[0],
                 names[4]: combine(p1, np.load(SP / f"{e}.npz")["scores"].astype(float), Z, y)[0]}
        gt = segs(y)
        for a, b in gt:
            n_len[[n for lo, hi, n in BINS if lo <= b - a <= hi][0]] += 1
        for k, p in preds.items():
            c = prf(y, p)
            for i in range(3):
                pt[k][i] += c[i]
            for a, b in gt:
                ev[k]["n"] += 1
                if p[a:b].any():
                    ev[k]["touch"] += 1
                    ev[k]["touch_len"][[n for lo, hi, n in BINS if lo <= b - a <= hi][0]] += 1
                ev[k]["half"] += int(p[a:b].mean() >= .5)
            for a, b in segs(p.astype(int)):
                ev[k]["pred"] += 1; ev[k]["pred_hit"] += int(y[a:b].any())
    print(f"{'':28s}{'틱 F1':>7s} | {'ev R(걸침)':>10s}{'ev R(50%)':>10s}{'ev P':>7s}{'ev F1':>7s}{'후보 수':>8s} | 걸침: spike / 16~224 / 225+")
    for k in names:
        E = ev[k]; R = E["touch"] / E["n"]; P = E["pred_hit"] / max(E["pred"], 1)
        tl = " / ".join(f"{E['touch_len'][b[2]]}/{n_len[b[2]]}" for b in BINS)
        print(f"{k:28s}{f1(*pt[k]):7.3f} | {R:10.3f}{E['half'] / E['n']:10.3f}{P:7.3f}{2 * P * R / (P + R):7.3f}{E['pred']:8d} | {tl}")


if __name__ == "__main__":
    main()
