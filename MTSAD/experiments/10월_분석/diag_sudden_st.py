"""
갑작스러움 기준(st) 원인 보기 — dt = 상위 0.5% 고정, spike 쪽만 (탈주기 뺌), SMD 28대
  GT 이상마다: "보정 spike > dt"인 채널·틱들 중 갑작스러움 최댓값이 machine 전체 갑작스러움(전 채널·전 틱) 중 상위 몇 %인가
  → 이 값이 c% 이하면 st = 상위 c%에서 걸림. (spike 점수에서 이미 떨어진 GT는 "dt 탈락"으로 따로 셈)
  같이: st별 spike 쪽 결과 (event P / R / F1 / 50% / 후보 수)
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores

HERE = Path(__file__).resolve().parent
CS = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100]


def main():
    pos_sp, pos_all, drop = [], [], [0, 0]; cnt = {c: 0 for c in CS}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); T = min(len(y), S.shape[1])
        y, S = y[:T], S[:, :T]; G = np.nan_to_num(suddenness(te[:T], tr), nan=0.0)
        fin = np.isfinite(S); Sb = np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)
        g = np.sort(G.ravel()); top = lambda s: 100 * (len(g) - np.searchsorted(g, s, side="left")) / len(g)
        for a, b in segs(y):
            m = Sb[:, a:b]
            if not m.any():
                drop[int(b - a <= 15)] += 1; continue
            p = top(G[:, a:b][m].max())
            (pos_sp if b - a <= 15 else pos_all).append(p)
        for c in CS:
            st = np.percentile(G, 100 - c) if c < 100 else -np.inf
            cnt[c] = cnt[c] + ev2_counts(y, (Sb & (G > st)).any(0))
    sp, lo = np.array(pos_sp), np.array(pos_all)
    L = ["dt = 상위 0.5% 고정. spike 점수는 통과한 GT 중, 같은 채널·틱 갑작스러움 최댓값이 상위 몇 %인가",
         f"  짧은 spike: dt 통과 {len(sp)}개 (dt 탈락 {drop[1]}) | 중앙 상위 {np.median(sp):.3f}%",
         f"  16틱 이상: dt 통과 {len(lo)}개 (dt 탈락 {drop[0]}) | 중앙 상위 {np.median(lo):.3f}%",
         "", f"{'st 상위%':>9s} | 짧은 spike 남음 | 16틱+ 남음 | spike쪽만: evP  evR  evF1 | R50  F50 | 후보"]
    for c in CS:
        s = ev2_scores(cnt[c])
        L.append(f"{('없음' if c == 100 else str(c) + '%'):>9s} | {np.sum(sp <= c):4d}/{len(sp)} ({np.mean(sp <= c):.2f}) | {np.sum(lo <= c):3d}/{len(lo)} ({np.mean(lo <= c):.2f}) | "
                 f"{s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f} | {s['n']:5d}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "diag_sudden_st.txt").write_text(txt)


if __name__ == "__main__":
    main()
