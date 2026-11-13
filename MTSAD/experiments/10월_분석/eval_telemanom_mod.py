"""
Telemanom 기준 수정 버전 비교 — v3 더하기 점수(보정 spike + 보정 높이 h3), 갑작스러움 없음, spike 쪽만, SMD 28대
  원래: ε = μ + zσ, z ∈ 2~10 (0.5 간격), z* = argmax (Δμ/μ + Δσ/σ) / (|넘은 점| + |덩어리|²), machine 칸 점수 모음
  변형: 벌점 |덩어리|² → |덩어리| / 덩어리 벌점 없음 / z 1~10 / 채널마다 따로 (제곱·약한 벌점)
  점수는 음수가 있어 e − min(e)로 옮겨 계산. 넘는 z가 없으면 기준 = ∞
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat

HERE = Path(__file__).resolve().parent
P = 14


def tm(e, zs, pen):
    e = e[np.isfinite(e)]
    if len(e) < 50 or np.ptp(e) < 1e-12:
        return np.inf
    x = e - e.min(); mu, sd = x.mean(), x.std(); best, bz = 0.0, None
    for z in zs:
        eps = mu + z * sd; a = x > eps; na = a.sum()
        if na == 0:
            continue
        r = x[~a]; ns = len(segs(a.astype(int)))
        den = na + (ns ** 2 if pen == "제곱" else (ns if pen == "약하게" else 0))
        v = ((mu - r.mean()) / mu + (sd - r.std()) / sd) / den
        if v > best:
            best, bz = v, eps
    return np.inf if bz is None else bz + e.min()


VARIANTS = [("원래 (machine, 제곱, z 2~10)", "machine", np.arange(2, 10.01, .5), "제곱"),
            ("machine, 약하게", "machine", np.arange(2, 10.01, .5), "약하게"),
            ("machine, 덩어리 벌점 없음", "machine", np.arange(2, 10.01, .5), "없음"),
            ("machine, 제곱, z 1~10", "machine", np.arange(1, 10.01, .25), "제곱"),
            ("machine, 약하게, z 1~10", "machine", np.arange(1, 10.01, .25), "약하게"),
            ("채널마다, 제곱", "channel", np.arange(2, 10.01, .5), "제곱"),
            ("채널마다, 약하게", "channel", np.arange(2, 10.01, .5), "약하게")]


def main():
    cnt = {v[0]: 0 for v in VARIANTS}; sp = {v[0]: [0, 0] for v in VARIANTS}; pct = {v[0]: [] for v in VARIANTS}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); T = min(len(y), d["spike"].shape[1]); y, te = y[:T], te[:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        X = S + Hz; Xv = np.where(np.isfinite(X), X, -np.inf); cols = X[:, ::P]
        for nm, unit, zs, pen in VARIANTS:
            if unit == "machine":
                thr = np.full((X.shape[0], 1), tm(cols.ravel(), zs, pen))
            else:
                thr = np.array([[tm(cols[c], zs, pen)] for c in range(X.shape[0])])
            on = Xv > thr; p = on.any(0); cnt[nm] = cnt[nm] + ev2_counts(y, p)
            pool = cols[np.isfinite(cols)]; pct[nm].append(100 * np.mean(on[:, ::P][np.isfinite(cols)]))
            for a, b in segs(y):
                if b - a <= 15:
                    sp[nm][1] += 1; sp[nm][0] += int(p[a:b].any())
    L = ["Telemanom 수정 버전 (v3 더하기, 갑작스러움 없음, spike 쪽만, SMD 28대)", f"{'변형':30s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | 켜진 칸 비율 (최소~최대, 중앙)"]
    for nm, *_ in VARIANTS:
        s = ev2_scores(cnt[nm]); r = pct[nm]
        L.append(f"{nm:30s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[nm][1] - sp[nm][0]:3d} | {min(r):.2f}~{max(r):.2f}% (중앙 {np.median(r):.2f}%)")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "telemanom_mod.txt").write_text(txt)


if __name__ == "__main__":
    main()
