"""
채널 넘기기 — 구간 = v3 더하기 상위 0.2% (갑작 없음) OR 탈주기(채널 최댓값) 상위 1%
  채널 점수 (구간 범위 안 최댓값): X = v3 더하기, D = 채널별 탈주기 점수
  기준 A: X > machine X 칸 상위 r% / 기준 B: A 또는 D > machine D (전 채널·전 틱) 상위 r%
  r = 0.5, 1, 2, 5, 10.  지표: 채널 recall / precision (이상마다 평균), GT 채널 전부 맞춤, 넘긴 채널 수
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD, PH
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent
P = 14
RS = [0.5, 1, 2, 5, 10]


def main():
    keys = [(m, r) for m in ["v3 더하기만", "v3 더하기 + 탈주기"] for r in RS]; R = {k: [] for k in keys}; nev = 0
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); dd = np.load(PH / f"{e}_phase_deseason.npz")
        T = min(len(y), d["spike"].shape[1], len(dd["combined"])); y, te = y[:T], te[:T]; Dc = dd["scores"][:, :T]; Dm = dd["combined"][:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        X = S + Hz; fin = np.isfinite(X); Xv = np.where(fin, X, -np.inf)
        p = (Xv > np.percentile(X[fin], 99.8)).any(0) | (Dm > np.percentile(Dm, 99)); bl = segs(p.astype(int))
        pool = X[:, ::P]; pool = pool[np.isfinite(pool)]; tx = {r: np.percentile(pool, 100 - r) for r in RS}; td = {r: np.percentile(Dc, 100 - r) for r in RS}
        for a, b in segs(y):
            g = set(interp_channels(e, a, b)); ov = [(u, v) for u, v in bl if u < b and v > a]
            if not ov or not g:
                continue
            nev += 1; lo, hi = ov[0][0], ov[-1][1]
            xs = Xv[:, lo:hi].max(1); ds = Dc[:, lo:hi].max(1)
            for r in RS:
                A = set(np.where(xs > tx[r])[0]); B = A | set(np.where(ds > td[r])[0])
                for m, s in [("v3 더하기만", A), ("v3 더하기 + 탈주기", B)]:
                    R[(m, r)].append((len(g & s), len(g), len(s)))
    L = [f"채널 넘기기 — 구간 = v3 더하기 0.2% OR 탈주기 1% (갑작 없음), 걸친 GT {nev}개 (채널 라벨 있는 것)", f"{'채널 기준':20s} {'r':5s} | 채널R  채널P  전부맞춤 | 넘긴 채널 중앙 평균"]
    for k in keys:
        A = np.array(R[k])
        L.append(f"{k[0]:20s} {str(k[1]) + '%':5s} | {np.mean(A[:, 0] / A[:, 1]):.3f} {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f} {np.mean(A[:, 0] == A[:, 1]):8.3f} | {np.median(A[:, 2]):6.0f} {A[:, 2].mean():5.1f}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "channel_handoff2.txt").write_text(txt)


if __name__ == "__main__":
    main()
