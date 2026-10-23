"""
채널마다 기준 = 그 채널 train spike 점수(열마다)의 k번째로 높은 값 (나연 2026-10-20) — 기준선: 1번 + 눈금 넓힘, spike 쪽만
  test 열 원점수 > 그 채널 기준  AND  갑작스러움 > machine 상위 1%   (채널별 보정·machine 상위 0.5% dt 없음)
  train 점수가 없거나 50열 미만인 채널(train 상수 등)은 지금 방식: 보정 점수 > machine 상위 0.5%
  k = 1 (train 최댓값), 3, 5, 10
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat, VIS, FLAT
from eval_trainp import train_ref, NOISE

HERE = Path(__file__).resolve().parent
KS = [None, 1, 3, 5, 10]


def main():
    vis = {" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(VIS).read().splitlines()[1:]} - FLAT
    n = len(KS); cnt = [0] * n; sp = [[0, 0] for _ in range(n)]; lg = [[0, 0] for _ in range(n)]; back = [0] * n; nfp = [0] * n; nfb = 0
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        raw = drop_flat(raw, te, tr); S, _ = calib(raw); fin = np.isfinite(S); dt = np.percentile(S[fin], 99.5)
        ref = train_ref(e, tr); fb = np.array([r is None for r in ref]); nfb += int((fb & fin.any(1)).sum())
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        rv = np.where(np.isfinite(raw), raw, -np.inf); Sv = np.where(fin, S, -np.inf)
        for i, k in enumerate(KS):
            if k is None:
                m = (Sv > dt) & Gb
            else:
                thr = np.array([r[-min(k, len(r))] if r is not None else np.inf for r in ref])
                m = np.where(fb[:, None], Sv > dt, rv > thr[:, None]) & Gb
            pr = m.any(0); cnt[i] = cnt[i] + ev2_counts(y, pr)
            for s, b in segs(y):
                t = sp[i] if b - s <= 15 else lg[i]; t[1] += 1; h = pr[s:b].any(); t[0] += int(h); back[i] += int(f"{e} [{s},{b})" in vis and h)
            for e2, c in NOISE:
                if e2 == e:
                    nfp[i] += sum(1 for u, v in segs(m[c].astype(int)) if not y[u:v].any())
    L = [f"spike 쪽만 (1번 + 눈금 넓힘), 채널마다 기준 = train k번째 점수, AND 갑작 상위 1% (SMD 28대) | train 기준 없는 채널 {nfb}개는 지금 방식",
         f"{'방식':12s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | 16틱+ | 잡음 채널 5개 오탐 | 보이는데 놓친 {len(vis)}개 되찾음"]
    for i, k in enumerate(KS):
        s = ev2_scores(cnt[i])
        L.append(f"{'지금 (dt 0.5%)' if k is None else f'k = {k}':12s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[i][1] - sp[i][0]:3d} | {lg[i][0]:3d}/{lg[i][1]} | {nfp[i]:3d} | {back[i]}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "chan_thr_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
