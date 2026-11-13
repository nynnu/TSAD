"""
가정 12 이어서: 왜 못 잡았나 / 왜 이 채널을 골랐나 — GT 구간에서 채널별 점수와 임계값 비교
  사용: python3 stage12_why.py entity a b ch1 ch2 ...
  spike 점수(보정 전 = logit 열 최대, 보정 후 = (점수-채널 중앙값)/(1.4826·MAD)), 갑작스러움, 탈주기 점수
"""
import sys
from pathlib import Path

import numpy as np

from eval_dsz_allch import prf, f1, PH, SMD, DQ, ZQ
from eval_dt_fix import calibrate, P
from eval_spike_detector import suddenness
from stage12_check import best_thr

HERE = Path(__file__).resolve().parent


def main():
    E, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); chs = [int(c) for c in sys.argv[4:]]
    y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
    D = np.load(PH / f"{E}_phase_deseason.npz")["scores"]; tD = best_thr(D.max(0), y); p1 = D.max(0) > tD
    R = np.load(HERE / "spike_scores" / f"{E}.npz")["scores"].astype(float); S = calibrate(R)
    G = np.nan_to_num(suddenness(te, tr), nan=0.0)
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
    best, bt = f1(*prf(y, p1)), (None, None)
    for gt in np.unique(np.percentile(G, ZQ)):
        for dt in np.unique(np.percentile(S[fin], DQ)):
            v = f1(*prf(y, p1 | ((Sv > dt) & (G > gt)).any(0)))
            if v > best:
                best, bt = v, (dt, gt)
    dt, st = bt
    print(f"{E} [{a},{b}) 임계값: 탈주기 > {tD:.2f}, 보정 spike 점수 > {dt:.2f}, 갑작스러움 > {st:.1f}\n")
    rng = tr.max(0) - tr.min(0)
    print(f"{'ch':>4s}{'높이/train범위':>14s}{'보정전 점수':>11s}{'채널 중앙값':>11s}{'채널 MAD':>9s}{'보정후 점수':>11s}{'갑작스러움':>10s}{'탈주기':>8s}  켜짐")
    for c in chs:
        w = slice(a, b); pre = np.median(te[max(0, a - 100):a, c])
        h = np.abs(te[w, c] - pre).max() / max(rng[c], 1e-9)
        v = R[c, ::P]; v = v[np.isfinite(v)]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
        on = ((Sv[c, w] > dt) & (G[c, w] > st))
        print(f"{c:4d}{h:14.2f}{np.nanmax(R[c, w]):11.2f}{med:11.2f}{mad:9.2f}{np.nanmax(S[c, w]):11.2f}{G[c, w].max():10.1f}{D[c, w].max():8.2f}  "
              f"spike {'O' if on.any() else 'X'} (점수{'O' if (Sv[c, w] > dt).any() else 'X'}·갑작{'O' if (G[c, w] > st).any() else 'X'}) / 탈주기 {'O' if (D[c, w] > tD).any() else 'X'}")
    print(f"\n열(14틱) 위치: GT {a}~{b} → 창 {a // 224}, 열 {(a % 224) // P}~{((b - 1) % 224) // P}")


if __name__ == "__main__":
    main()
