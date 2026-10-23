"""
채널 넘기기 (나연 2026-10-21) — 구간 = 마지막 모델: v3 더하기 (보정 spike + 보정 높이 h3), 갑작스러움 없음, 상위 0.2%
  걸친 GT 이상마다: 겹친 후보 덩어리들의 범위 안에서 채널 점수 = 그 채널 v3 더하기 점수 최댓값
  채널 고르는 기준 (구간 기준과 별개, 더 느슨하게):
    ① machine 비율: 채널 점수 > machine 전체 칸 상위 r% (r = 0.5, 1, 2, 5, 10)
    ② 위에서 k개 (k = 3, 5, 8, 10, 15, 20)
    ③ 구간 1등 대비: 채널 점수 ≥ 1등 − f × (1등 − 그 구간 채널 점수 중앙값) 의 형태가 아니라, 간단히 순위 백분위 대신
       "1등 점수와의 차이가 machine 칸 점수 MAD의 m배 이내" (m = 2, 4, 6, 10)
  지표 (걸친 GT 이상 중 채널 라벨 있는 것): 채널 recall / precision (이상마다 평균), GT 채널 전부 맞춤, 넘긴 채널 수 중앙·평균
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent
P = 14
RS, KS, MS = [0.5, 1, 2, 5, 10], [3, 5, 8, 10, 15, 20], [2, 4, 6, 10]


def main():
    keys = [f"machine 상위 {r}%" for r in RS] + [f"위에서 {k}개" for k in KS] + [f"1등과 차이 ≤ MAD×{m}" for m in MS] + ["전부"]
    R = {k: [] for k in keys}; nev = 0; tot = 0
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); T = min(len(y), d["spike"].shape[1]); y, te = y[:T], te[:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        X = S + Hz; fin = np.isfinite(X); Xv = np.where(fin, X, -np.inf)
        p = (Xv > np.percentile(X[fin], 99.8)).any(0); blocks = segs(p.astype(int))
        pool = X[:, ::P]; pool = pool[np.isfinite(pool)]; thr = {r: np.percentile(pool, 100 - r) for r in RS}
        mad = 1.4826 * np.median(np.abs(pool - np.median(pool)))
        chs_all = np.where(fin.any(1))[0]
        for a, b in segs(y):
            tot += 1; g = set(interp_channels(e, a, b)); ov = [(u, v) for u, v in blocks if u < b and v > a]
            if not ov or not g:
                continue
            nev += 1; lo, hi = ov[0][0], ov[-1][1]; cs = np.nanmax(np.where(np.isfinite(X[:, lo:hi]), X[:, lo:hi], -np.inf), 1)
            order = [c for c in np.argsort(-cs) if np.isfinite(cs[c]) and cs[c] > -np.inf]; top = cs[order[0]]
            sel = {f"machine 상위 {r}%": {c for c in order if cs[c] > thr[r]} for r in RS}
            sel.update({f"위에서 {k}개": set(order[:k]) for k in KS})
            sel.update({f"1등과 차이 ≤ MAD×{m}": {c for c in order if top - cs[c] <= m * mad} for m in MS})
            sel["전부"] = set(chs_all)
            for k in keys:
                s = sel[k]; R[k].append((len(g & s), len(g), len(s)))
    L = [f"채널 넘기기 — 구간 = v3 더하기 상위 0.2% (갑작 없음), SMD 28대. 걸친 GT 이상 {nev}개 (채널 라벨 있는 것, 전체 GT {tot}개)",
         f"{'채널 기준':22s} | 채널R  채널P  전부맞춤 | 넘긴 채널 중앙 평균"]
    for k in keys:
        A = np.array(R[k])
        L.append(f"{k:22s} | {np.mean(A[:, 0] / A[:, 1]):.3f} {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f} {np.mean(A[:, 0] == A[:, 1]):8.3f} | {np.median(A[:, 2]):6.0f} {A[:, 2].mean():5.1f}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "channel_handoff.txt").write_text(txt)


if __name__ == "__main__":
    main()
