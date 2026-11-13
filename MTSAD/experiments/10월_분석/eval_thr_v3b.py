"""
v3 + 더하기 점수(보정 spike + 보정 높이 h3)에서 기준 정하는 방법 비교 — spike 쪽만, SMD 28대
  점수 기준: 고정 0.1/0.2/0.3/0.5% / knee(상위 2%·5% 범위에서, Kneedle) / Telemanom(machine 칸 점수에 z 2~10)
  AND 갑작스러움 상위 1% / 0.1%
  지표: event P / R / F1, R50 / F1(50%), 후보, 짧은 spike 놓침, machine마다 고른 비율(최소~최대)
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat
from eval_knee_dt import knee
from eval_telemanom import telemanom

HERE = Path(__file__).resolve().parent
P = 14
THR = ["고정 0.1%", "고정 0.2%", "고정 0.3%", "고정 0.5%", "knee 2%", "knee 5%", "Telemanom"]
STS = [1, 0.1]


def main():
    keys = [(t, s) for s in STS for t in THR]; cnt = {k: 0 for k in keys}; sp = {k: [0, 0] for k in keys}; pct = {k: [] for k in keys}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); T = min(len(y), d["spike"].shape[1]); y, te = y[:T], te[:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        X = S + Hz; fin = np.isfinite(X); Xv = np.where(fin, X, -np.inf); pool = X[:, ::P]; pool = pool[np.isfinite(pool)]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0)
        th = {}
        for t in THR:
            if t.startswith("고정"):
                r = float(t.split()[1][:-1]); th[t] = (np.percentile(X[fin], 100 - r), r)
            elif t.startswith("knee"):
                v, r, _ = knee(pool, int(t.split()[1][:-1])); th[t] = (v, r)
            else:
                v = telemanom(pool); th[t] = (v, 100 * np.mean(pool > v))
        for s in STS:
            Gb = G > np.percentile(G, 100 - s)
            for t in THR:
                v, r = th[t]; p = ((Xv > v) & Gb).any(0); k = (t, s); cnt[k] = cnt[k] + ev2_counts(y, p); pct[k].append(r)
                for a, b in segs(y):
                    if b - a <= 15:
                        sp[k][1] += 1; sp[k][0] += int(p[a:b].any())
    L = ["v3 + 더하기 점수, 기준 정하는 방법 비교 (spike 쪽만, SMD 28대)", f"{'점수 기준':10s} {'갑작':6s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | machine마다 고른 비율"]
    for k in keys:
        s = ev2_scores(cnt[k]); r = pct[k]
        L.append(f"{k[0]:10s} {str(k[1]) + '%':6s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[k][1] - sp[k][0]:3d} | {min(r):.2f}~{max(r):.2f}% (중앙 {np.median(r):.2f}%)")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "thr_v3b.txt").write_text(txt)


if __name__ == "__main__":
    main()
