"""
train spike 빈도 벌점 (나연 2026-10-20) — 기준선: 1번 + 눈금 넓힘, spike 쪽만
  새 점수 = test 보정 점수 − λ × log(1 + 그 채널 train spike 빈도)
  train spike 빈도 = train에서 갑작스러움 ≥ 10인 튐 묶음 수 / 1,000틱   (train 상수 채널은 0)
  dt = 새 점수 상위 0.5%, AND 갑작 상위 1%. λ = 0 (지금), 0.5, 1, 2
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat, VIS, FLAT
from eval_trainp import NOISE

HERE = Path(__file__).resolve().parent
LAMS = [0, 0.5, 1, 2]
SHOW = sorted(NOISE) + [("machine-1-6", 10), ("machine-1-6", 9), ("machine-2-7", 33)]


def main():
    vis = {" ".join(l.split(" | ")[0].split(" ")[:2]) for l in open(VIS).read().splitlines()[1:]} - FLAT
    n = len(LAMS); cnt = [0] * n; sp = [[0, 0] for _ in range(n)]; lg = [[0, 0] for _ in range(n)]; back = [0] * n; nfp = [0] * n; rates = {}; allr = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0)
        rate = np.array([len(segs((Gtr[c] >= 10).astype(int))) * 1000 / len(tr) if np.ptp(tr[:, c]) > 1e-6 else 0.0 for c in range(tr.shape[1])])
        allr += list(rate[np.ptp(tr, 0) > 1e-6])
        for k in SHOW:
            if k[0] == e:
                rates[k] = rate[k[1]]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        for i, lam in enumerate(LAMS):
            Sn = S - lam * np.log1p(rate)[:, None]
            m = (np.where(fin, Sn, -np.inf) > np.percentile(Sn[fin], 99.5)) & Gb; pr = m.any(0)
            cnt[i] = cnt[i] + ev2_counts(y, pr)
            for s, b in segs(y):
                t = sp[i] if b - s <= 15 else lg[i]; t[1] += 1; h = pr[s:b].any(); t[0] += int(h); back[i] += int(f"{e} [{s},{b})" in vis and h)
            for e2, c in NOISE:
                if e2 == e:
                    nfp[i] += sum(1 for u, v in segs(m[c].astype(int)) if not y[u:v].any())
    L = [f"train spike 빈도 (1,000틱당): 전체 채널 중앙 {np.median(allr):.2f}, 상위 10% {np.percentile(allr, 90):.2f}",
         "  " + ", ".join(f"{e[8:]} ch{c} {rates[(e, c)]:.2f}" for e, c in SHOW), "",
         "spike 쪽만 (1번 + 눈금 넓힘), 새 점수 = test 보정 − λ·log(1+train spike 빈도), dt 상위 0.5% AND 갑작 상위 1% (SMD 28대)",
         f"{'λ':6s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | 16틱+ | 잡음 채널 5개 오탐 | 보이는데 놓친 {len(vis)}개 되찾음"]
    for i, lam in enumerate(LAMS):
        s = ev2_scores(cnt[i])
        L.append(f"{lam:<6} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[i][1] - sp[i][0]:3d} | {lg[i][0]:3d}/{lg[i][1]} | {nfp[i]:3d} | {back[i]}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "freq_penalty_result.txt").write_text(txt)


if __name__ == "__main__":
    main()
