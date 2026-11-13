"""
spike 점수 + 높이 예측 합치기 비교 (spike_height_scores: spike, h1/h2/h3) — spike 쪽만, AND 갑작 상위 1%, SMD 28대
  공통: 움직임이 train 범위 1% 미만인 그림은 점수 없음 (drop_flat), S = spike 채널별 보정 점수
  지금      : S > S 상위 0.5%
  높이만    : H > H 상위 0.5%
  A 곱하기  : max(S, 0) × H > 그 점수 상위 0.5%
  B 더하기  : S + Hz > 그 점수 상위 0.5%   (Hz = 높이를 채널별 보정한 점수)
  C 거르고 순위: S > S 상위 2% 인 칸 중에서 H > H 상위 0.5%
  지표: event P / R / F1, R50 / F1(50%), 후보, 짧은 spike 놓침, GT 순위 (그 점수로 machine 전체 칸 중 상위 몇 %, 중앙·0.5% 안 비율)
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent
P = 14
METHODS = ["지금", "높이만", "A 곱하기", "B 더하기", "C 거르고 순위"]


def scores(S, H, Hz):
    Sp = np.where(np.isfinite(S), S, np.nan)
    return {"지금": Sp, "높이만": H, "A 곱하기": np.maximum(Sp, 0) * H, "B 더하기": Sp + Hz, "C 거르고 순위": None}


def main():
    keys = [(v, m) for v in ["h1", "h2", "h3"] for m in METHODS if not (m == "지금" and v != "h1")]
    cnt = {k: 0 for k in keys}; sp = {k: [0, 0] for k in keys}; rk = {k: [] for k in keys}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); T = min(len(y), d["spike"].shape[1]); y, te = y[:T], te[:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr))
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        gts = [(a, b, [c for c in interp_channels(e, a, b)]) for a, b in segs(y)]
        for v in ["h1", "h2", "h3"]:
            H = drop_flat(d[v][:, :T].astype(float), te, tr); Hz, _ = calib(H)
            sc = scores(S, H, Hz)
            for m in METHODS:
                if (v, m) not in cnt:
                    continue
                if m == "C 거르고 순위":
                    fin = np.isfinite(S) & np.isfinite(H)
                    gate = fin & (np.where(fin, S, -np.inf) > np.percentile(S[np.isfinite(S)], 98))
                    X = np.where(gate, H, np.nan)
                    thr = np.percentile(H[np.isfinite(H)], 99.5); on = np.nan_to_num(X, nan=-np.inf) > thr
                else:
                    X = sc[m]; fin = np.isfinite(X); thr = np.percentile(X[fin], 99.5); on = np.where(fin, X, -np.inf) > thr
                p = (on & Gb).any(0); k = (v, m); cnt[k] = cnt[k] + ev2_counts(y, p)
                pool = X[:, ::P]; pool = np.sort(pool[np.isfinite(pool)])[::-1]
                for a, b, g in gts:
                    if b - a <= 15:
                        sp[k][1] += 1; sp[k][0] += int(p[a:b].any())
                    g = [c for c in g if np.isfinite(X[c, a:b]).any()]
                    if g and len(pool):
                        val = max(np.nanmax(X[c, a:b]) for c in g); rk[k].append(100 * (np.searchsorted(-pool, -val) + 1) / len(pool))
    L = ["spike 점수 + 높이 예측 합치기 (spike 쪽만, AND 갑작 상위 1%, SMD 28대)",
         f"{'높이 버전':6s} {'방식':12s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | GT 순위: 중앙 상위% / 0.5% 안"]
    for k in keys:
        s = ev2_scores(cnt[k]); r = np.array(rk[k])
        L.append(f"{('-' if k[1] == '지금' else k[0]):6s} {k[1]:12s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[k][1] - sp[k][0]:3d} | {np.median(r):.2f}% / {np.mean(r <= 0.5):.0%}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "spike_height_combo.txt").write_text(txt)


if __name__ == "__main__":
    main()
