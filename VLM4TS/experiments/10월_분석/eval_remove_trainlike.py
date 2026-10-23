"""
train처럼 생긴 튐을 지우면 (GT 안 씀) — 지금 설정: 1번 + 눈금 넓힘, dt 보정 spike 상위 0.5% AND 갑작 상위 1%
  켜진 칸(채널·틱)마다 튐 높이 = |값 − 주변 ±100틱 중앙값| / train 범위
  지움 = 높이 ≤ 그 채널 train 정상 spike 높이(갑작 ≥ 10 묶음)의 기준 지점. 기준: 최댓값 / 상위 1% / 5% / 10%
  train spike 5개 미만인 채널(train 상수 포함)은 기준 0 (안 지움)
  spike 쪽만 / 탈주기(상위 3%) 포함 둘 다
"""
from pathlib import Path

import numpy as np
import pandas as pd

from eval_dsz_allch import segs, SC, SMD, PH
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat
from diag_spiky_gt import spikes

HERE = Path(__file__).resolve().parent
QS = [None, 100, 99, 95, 90]


def main():
    res = {(w, q): 0 for w in [0, 1] for q in QS}; sp = {(w, q): [0, 0] for w in [0, 1] for q in QS}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        D = np.load(PH / f"{e}_phase_deseason.npz")["combined"]
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1], len(D)); y, raw, te, D = y[:T], raw[:, :T], te[:T], D[:T]
        pD = D > np.percentile(D, 97)
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); m0 = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99))
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0); rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
        hs = []
        for c in range(tr.shape[1]):
            h = np.array([x for *_, x in spikes(tr[:, c], Gtr[c], rg[c])]) if np.ptp(tr[:, c]) > 1e-6 else np.array([])
            hs.append(h if len(h) >= 5 else None)
        med = pd.DataFrame(te).rolling(201, center=True, min_periods=50).median().values
        hgt = np.abs(te - med).T / rg[:, None]
        for q in QS:
            if q is None:
                m = m0
            else:
                thr = np.array([np.percentile(h, q) if h is not None else -np.inf for h in hs])
                m = m0 & ~(np.nan_to_num(hgt, nan=0.0) <= thr[:, None])
            for w in [0, 1]:
                p = m.any(0) | (pD if w else False)
                res[(w, q)] = res[(w, q)] + ev2_counts(y, p)
                for a, b in segs(y):
                    if b - a <= 15:
                        sp[(w, q)][1] += 1; sp[(w, q)][0] += int(p[a:b].any())
    L = []
    for w, nm in [(0, "spike 쪽만"), (1, "탈주기 3% 포함")]:
        L += [f"[{nm}] train처럼 생긴 튐 지우기 (SMD 28대)", f"{'지우는 기준':24s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침"]
        for q in QS:
            s = ev2_scores(res[(w, q)]); nmq = "안 지움 (지금)" if q is None else ("train 최댓값 이하" if q == 100 else f"train 상위 {100 - q}% 이하")
            L.append(f"{nmq:24s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[(w, q)][1] - sp[(w, q)][0]:3d}")
        L.append("")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "remove_trainlike.txt").write_text(txt)


if __name__ == "__main__":
    main()
