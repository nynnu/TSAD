"""
채널 넘기기(구간 v3 더하기 0.2%, 채널 상위 5%)에서 못 넘긴 GT 채널 — 왜?
  1. 점수 없음 (그 구간에서 점수 칸이 하나도 없음: 작은 움직임 그림 or test에서도 상수)
  2. 후보 범위 밖: GT 구간 전체로 보면 기준을 넘는데, 후보 덩어리 범위 안에서는 못 넘음
  3. 점수가 기준 아래: GT 구간 안 움직임(최대−최소)/train 범위로 나눠서 봄
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent
P = 14


def main():
    cnt = {"1 점수 없음": 0, "2 후보 범위 밖": 0, "3 기준 아래": 0}; mv = []; tot = 0; miss = 0
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); T = min(len(y), d["spike"].shape[1]); y, te = y[:T], te[:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        X = S + Hz; fin = np.isfinite(X); p = (np.where(fin, X, -np.inf) > np.percentile(X[fin], 99.8)).any(0); bl = segs(p.astype(int))
        pool = X[:, ::P]; pool = pool[np.isfinite(pool)]; t5 = np.percentile(pool, 95); rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
        for a, b in segs(y):
            g = interp_channels(e, a, b); ov = [(u, v) for u, v in bl if u < b and v > a]
            if not ov or not g:
                continue
            lo, hi = ov[0][0], ov[-1][1]
            for c in g:
                tot += 1; xr = X[c, lo:hi]
                if np.isfinite(xr).any() and np.nanmax(xr) > t5:
                    continue
                miss += 1; xg = X[c, min(a, lo):max(b, hi)]
                if not np.isfinite(X[c, min(a, lo):max(b, hi)]).any():
                    cnt["1 점수 없음"] += 1
                elif np.nanmax(xg) > t5:
                    cnt["2 후보 범위 밖"] += 1
                else:
                    cnt["3 기준 아래"] += 1; mv.append(np.ptp(te[a:b, c]) / rg[c] if b - a > 1 else abs(te[a, c] - np.median(te[max(0, a - 100):a, c])) / rg[c])
    mv = np.array(mv)
    L = [f"GT 채널 {tot}개 중 못 넘김 {miss}개 ({miss / tot:.0%})"] + [f"  {k}: {v}개 ({v / miss:.0%})" for k, v in cnt.items()]
    L += [f"  3번의 GT 구간 안 움직임 (train 범위 비율): 중앙 {np.median(mv):.3f} | 0.02 미만 {np.mean(mv < 0.02):.0%}, 0.05 미만 {np.mean(mv < 0.05):.0%}, 0.2 이상 {np.mean(mv >= 0.2):.0%}"]
    txt = "\n".join(L); print(txt); (HERE / "coverage_groups" / "channel_miss_why.txt").write_text(txt)


if __name__ == "__main__":
    main()
