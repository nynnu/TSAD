"""
가정 9-1: 2차 보완이 진짜 의미가 있나 — z-score 위에 DINOv2/숫자 열점수가 무엇을 더했나 (GT 없이 38채널, 오라클 임계값)
  탈주기 단독 / 탈주기 OR (z > zt) / 탈주기 OR (DINOv2 > dt AND z > zt) / 탈주기 OR (숫자 > dt AND z > zt)
지표: P, R, F1 / 탈주기가 놓친 GT 이상 중 새로 잡은 개수(길이별) / 새로 생긴 오탐(탈주기 예측에 없던, GT와 안 겹치는 추가 덩어리)
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import prf, f1, best_pred, segs, combine, PH, SC, SMD, BINS, ZQ

HERE = Path(__file__).resolve().parent


def z_only(p1, Z, y):
    best, bp = f1(*prf(y, p1)), p1
    for zt in np.unique(np.percentile(Z, ZQ)):
        p = p1 | (Z > zt).any(0); v = f1(*prf(y, p))
        if v > best:
            best, bp = v, p
    return bp


def main():
    ents = sorted(p.stem for p in SC.glob("*.npz"))
    names = ["탈주기 단독", "탈주기 + z-score", "+ DINOv2", "+ 숫자"]
    tot = {k: [0, 0, 0] for k in names}
    rescued = {k: {b[2]: 0 for b in BINS} for k in names}
    missed = {b[2]: 0 for b in BINS}
    newfp = {k: [0, 0] for k in names}                    # [오탐 덩어리 수, 오탐 틱 수]
    added = {k: [0, 0] for k in names}                    # [추가된 덩어리 수, 그중 GT와 겹치는 수]
    for e in ents:
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        d = np.load(SC / f"{e}.npz")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        preds = {"탈주기 단독": p1, "탈주기 + z-score": z_only(p1, Z, y),
                 "+ DINOv2": combine(p1, d["D"].astype(float), Z, y)[0],
                 "+ 숫자": combine(p1, d["B"].astype(float), Z, y)[0]}
        gt = segs(y)
        for a, b in gt:
            if p1[a:b].mean() < .5:
                for lo, hi, nm in BINS:
                    if lo <= b - a <= hi:
                        missed[nm] += 1
        for k, p in preds.items():
            c = prf(y, p)
            for i in range(3):
                tot[k][i] += c[i]
            for a, b in gt:
                if p1[a:b].mean() < .5 and p[a:b].mean() >= .5:
                    for lo, hi, nm in BINS:
                        if lo <= b - a <= hi:
                            rescued[k][nm] += 1
            add = p & ~p1
            for a, b in segs(add.astype(int)):
                added[k][0] += 1
                if y[a:b].any():
                    added[k][1] += 1
                else:
                    newfp[k][0] += 1; newfp[k][1] += b - a
    tm = sum(missed.values())
    print(f"entity {len(ents)}개 / 탈주기가 놓친 GT 이상 {tm}개 (" + ", ".join(f"{k} {v}" for k, v in missed.items()) + ")\n")
    print(f"{'':16s}{'P':>7s}{'R':>7s}{'F1':>7s}   {'놓친 것 새로 잡음':>14s} (spike / 16~224 / 225+)   {'새 오탐 덩어리':>10s} {'새 오탐 틱':>9s}   추가 덩어리 중 GT와 겹침")
    for k in names:
        tp, fp, fn = tot[k]
        rs = rescued[k]; r_all = sum(rs.values())
        hit = f"{added[k][1]}/{added[k][0]} ({added[k][1] / added[k][0] * 100:.0f}%)" if added[k][0] else "-"
        print(f"{k:16s}{tp / max(tp + fp, 1):7.3f}{tp / max(tp + fn, 1):7.3f}{f1(tp, fp, fn):7.3f}   {r_all:6d}/{tm} ({r_all / tm * 100:2.0f}%)  ({' / '.join(str(rs[b[2]]) for b in BINS)})   {newfp[k][0]:10d} {newfp[k][1]:9d}   {hit}")


if __name__ == "__main__":
    main()
