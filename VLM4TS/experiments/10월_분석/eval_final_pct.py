"""
최종 모델 비율 비교 (나연 주간 리포트, 2026-10-22) — 구간 = v3 더하기 상위 r% OR 탈주기(채널 최댓값) 상위 1%, 갑작 없음
  r = 0.1 ~ 0.5%, SMD 28대. 지표: event P / R / F1, R50 / F1(50%), 후보
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD, PH
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat

HERE = Path(__file__).resolve().parent
RS = [0.1, 0.2, 0.3, 0.4, 0.5]


def main():
    cnt = {r: 0 for r in RS}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); dd = np.load(PH / f"{e}_phase_deseason.npz")
        T = min(len(y), d["spike"].shape[1], len(dd["combined"])); y, te = y[:T], te[:T]; Dm = dd["combined"][:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        X = S + Hz; fin = np.isfinite(X); Xv = np.where(fin, X, -np.inf); dp = Dm > np.percentile(Dm, 99)
        for r in RS:
            cnt[r] = cnt[r] + ev2_counts(y, (Xv > np.percentile(X[fin], 100 - r)).any(0) | dp)
    L = ["최종 모델: v3 더하기 상위 r% OR 탈주기 1% (갑작 없음), SMD 28대", f"{'r':5s} | evP   evR   evF1 | R50   F1(50%) | 후보"]
    for r in RS:
        s = ev2_scores(cnt[r]); L.append(f"{str(r) + '%':5s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "final_pct.txt").write_text(txt)


if __name__ == "__main__":
    main()
