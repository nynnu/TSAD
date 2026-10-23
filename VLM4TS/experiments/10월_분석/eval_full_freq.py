"""
전체 모델 = 탈주기(채널 최댓값) > 상위 3%  OR  (새 spike 점수 > 상위 0.5%  AND  갑작스러움 > 상위 st%)
  새 spike 점수 = 1번 + 눈금 넓힘 + test 보정 − λ·log(1 + train spike 빈도)   (eval_freq_penalty.py와 같음)
  st = 1% / 0.05%,  λ = 0 / 0.5 / 1.  SMD 28대, 채점: event F1 / event F1(50%)
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD, PH
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat

HERE = Path(__file__).resolve().parent
STS, LAMS = [1, 0.05], [0, 0.5, 1]
BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def main():
    keys = [("탈주기만", None, None)] + [(f"st {st}% λ {lam}", st, lam) for st in STS for lam in LAMS]
    cnt = {k[0]: 0 for k in keys}; tl = {k[0]: {b[2]: [0, 0] for b in BINS} for k in keys}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        D = np.load(PH / f"{e}_phase_deseason.npz")["combined"]
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1], len(D)); y, raw, te, D = y[:T], raw[:, :T], te[:T], D[:T]
        pD = D > np.percentile(D, 97)
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0)
        rate = np.array([len(segs((Gtr[c] >= 10).astype(int))) * 1000 / len(tr) if np.ptp(tr[:, c]) > 1e-6 else 0.0 for c in range(tr.shape[1])])
        G = np.nan_to_num(suddenness(te, tr), nan=0.0)
        for nm, st, lam in keys:
            if st is None:
                p = pD
            else:
                Sn = S - lam * np.log1p(rate)[:, None]
                p = pD | ((np.where(fin, Sn, -np.inf) > np.percentile(Sn[fin], 99.5)) & (G > np.percentile(G, 100 - st))).any(0)
            cnt[nm] = cnt[nm] + ev2_counts(y, p)
            for a, b in segs(y):
                x = [n2 for lo, hi, n2 in BINS if lo <= b - a <= hi][0]; tl[nm][x][1] += 1; tl[nm][x][0] += int(p[a:b].any())
    L = ["전체 모델 (SMD 28대): 탈주기 상위 3% OR (새 spike 점수 상위 0.5% AND 갑작 상위 st%)",
         f"{'방식':16s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 걸침 spike / 16~224 / 225+"]
    for nm, _, _ in keys:
        s = ev2_scores(cnt[nm])
        L.append(f"{nm:16s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | " + " / ".join(f"{tl[nm][b[2]][0]}/{tl[nm][b[2]][1]}" for b in BINS))
    L += ["참고: 예전 고정 비율 최고 (탈주기 3%, spike 0.5%, 갑작 0.05%, 보정·그림 수정 전) evF1 0.561 / 50% 0.372 / 후보 873 | 오라클(event F1) 0.699 / 0.577"]
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "full_freq_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
