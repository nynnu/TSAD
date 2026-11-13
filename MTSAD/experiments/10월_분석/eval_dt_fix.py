"""
가정 11: spike 탐지기 점수에 채널별 보정을 하면, dt에 버려지던 spike를 살릴 수 있다
  보정 B (라벨 안 씀): 채널마다 test 열점수 전체의 중앙값·MAD로 (점수 − 중앙값) / (1.4826·MAD)
     이상은 드물어서 중앙값·MAD는 "평소 점수 수준"을 나타냄
  구조는 그대로: 탈주기 OR (보정 spike 점수 > dt AND z > zt), (dt, zt) entity당 한 쌍 오라클(틱 F1)
확인: ① dt에 버려졌던 spike 10개가 살아나나  ② event 지표 / 후보 수  ③ 틱 P/R/F1
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import prf, f1, best_pred, segs, combine, PH, SC, SMD

HERE = Path(__file__).resolve().parent
SP = HERE / "spike_scores"
P = 14
BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]
DROPPED = [("machine-1-2", 5486), ("machine-1-2", 15540), ("machine-1-2", 18645), ("machine-1-6", 13069),
           ("machine-1-6", 13277), ("machine-1-7", 6031), ("machine-1-7", 13799), ("machine-1-7", 15960),
           ("machine-2-1", 9340), ("machine-3-6", 26138)]


def calibrate(S):
    out = np.full_like(S, np.nan)
    for c in range(S.shape[0]):
        v = S[c, ::P]; v = v[np.isfinite(v)]
        if len(v) < 10:
            continue
        med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
        out[c] = (S[c] - med) / max(mad, 1e-6)
    return out


def main():
    names = ["지금 (patch-kNN)", "spike 탐지기 (보정 없음)", "spike 탐지기 (채널별 보정)"]
    pt = {k: [0, 0, 0] for k in names}
    ev = {k: dict(touch=0, half=0, n=0, pred=0, hit=0, tl={b[2]: 0 for b in BINS}) for k in names}
    nl = {b[2]: 0 for b in BINS}
    rescued = {k: 0 for k in names}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        Sraw = np.load(SP / f"{e}.npz")["scores"].astype(float)
        preds = {names[0]: combine(p1, np.load(SC / f"{e}.npz")["D"].astype(float), Z, y)[0],
                 names[1]: combine(p1, Sraw, Z, y)[0],
                 names[2]: combine(p1, calibrate(Sraw), Z, y)[0]}
        gt = segs(y)
        for a, b in gt:
            nl[[n for lo, hi, n in BINS if lo <= b - a <= hi][0]] += 1
        for k, p in preds.items():
            c = prf(y, p)
            for i in range(3):
                pt[k][i] += c[i]
            for a, b in gt:
                ev[k]["n"] += 1
                if p[a:b].any():
                    ev[k]["touch"] += 1; ev[k]["tl"][[n for lo, hi, n in BINS if lo <= b - a <= hi][0]] += 1
                ev[k]["half"] += int(p[a:b].mean() >= .5)
                if (e, int(a)) in DROPPED and p[a:b].any():
                    rescued[k] += 1
            for a, b in segs(p.astype(int)):
                ev[k]["pred"] += 1; ev[k]["hit"] += int(y[a:b].any())
    print(f"{'':26s}{'틱P':>6s}{'틱R':>6s}{'틱F1':>6s} | {'evR걸침':>7s}{'evR50%':>7s}{'evP':>6s}{'evF1':>6s}{'후보':>6s} | 걸침 spike/16~224/225+ | 버려진 spike 10개 중 살아남")
    for k in names:
        tp, fp, fn = pt[k]; E = ev[k]; R = E["touch"] / E["n"]; Pp = E["hit"] / max(E["pred"], 1)
        print(f"{k:26s}{tp / (tp + fp):6.3f}{tp / (tp + fn):6.3f}{f1(tp, fp, fn):6.3f} | {R:7.3f}{E['half'] / E['n']:7.3f}{Pp:6.3f}{2 * Pp * R / (Pp + R):6.3f}{E['pred']:6d} | "
              + "/".join(str(E["tl"][b[2]]) for b in BINS) + f" | {rescued[k]}/10")


if __name__ == "__main__":
    main()
