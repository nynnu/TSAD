"""
spike 기준(dt) 원인 보기 — 갑작스러움은 빼고 보정 spike 점수만 (SMD 28대)
  GT 이상마다: 구간 안 보정 spike 최댓값(모든 채널)이 그 machine 전체 점수(전 채널·전 틱, dt 비율을 정하는 분포) 중 상위 몇 %인가
  → 이 값이 b% 이하면 "dt = 상위 b%"에서 spike 점수로는 걸림
  짧은 spike(≤15틱)와 전체를 따로
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_dt_fix import calibrate

HERE = Path(__file__).resolve().parent


def main():
    L = ["GT 이상 안 보정 spike 최댓값이 machine 전체 점수 중 상위 몇 %인가 (짧은 spike ≤15틱)",
         f"{'machine':13s} | spike 수 | 중앙 상위%  | 0.1% 안 | 0.5% 안 | 1% 안 | 5% 안 | (전체 GT: 0.5% 안)"]
    allp, allq = [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); T = min(len(y), S.shape[1]); y, S = y[:T], S[:, :T]
        v = np.sort(S[np.isfinite(S)]); Sm = np.nanmax(np.where(np.isfinite(S), S, -np.inf), 0)
        top = lambda s: 100 * (len(v) - np.searchsorted(v, s, side="left")) / len(v)
        pos = [(b - a, top(Sm[a:b].max())) for a, b in segs(y)]
        sp = np.array([p for n, p in pos if n <= 15]); al = np.array([p for _, p in pos])
        allp += list(sp); allq += list(al)
        if len(sp):
            L.append(f"{e:13s} | {len(sp):6d}  | {np.median(sp):8.3f}% | {np.mean(sp <= 0.1):6.2f} | {np.mean(sp <= 0.5):6.2f} | {np.mean(sp <= 1):5.2f} | {np.mean(sp <= 5):5.2f} | {np.mean(al <= 0.5):.2f}")
        else:
            L.append(f"{e:13s} |      0  |          - |        |        |       |       | {np.mean(al <= 0.5):.2f}")
    p, q = np.array(allp), np.array(allq)
    L.append(f"{'전체':13s} | {len(p):6d}  | {np.median(p):8.3f}% | {np.mean(p <= 0.1):6.2f} | {np.mean(p <= 0.5):6.2f} | {np.mean(p <= 1):5.2f} | {np.mean(p <= 5):5.2f} | {np.mean(q <= 0.5):.2f}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "diag_spike_dt.txt").write_text(txt)


if __name__ == "__main__":
    main()
