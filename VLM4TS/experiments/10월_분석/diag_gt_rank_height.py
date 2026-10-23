"""
GT 순위 비교: 지금(보정 spike) vs 높이 합친 점수 — GT마다 그 machine 전체 칸(채널 × 14틱 열) 중 상위 몇 %
  GT 점수 = GT 채널·GT 구간 최댓값. 높이 버전 h1/h2/h3 × (높이만 / A 곱하기 / B 더하기)
  출력: 전체 분포, 지금보다 올라간 GT 비율, 지금 상위 0.5% 밖이던 GT가 안으로 들어온 수, machine별 중앙
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent
P = 14


def rank(X, gts):
    pool = X[:, ::P]; pool = np.sort(pool[np.isfinite(pool)])[::-1]; out = []
    for a, b, g in gts:
        g = [c for c in g if np.isfinite(X[c, a:b]).any()]
        out.append(100 * (np.searchsorted(-pool, -max(np.nanmax(X[c, a:b]) for c in g)) + 1) / len(pool) if g else np.nan)
    return out


def main():
    keys = ["지금"] + [f"{v} {m}" for v in ["h1", "h2", "h3"] for m in ["높이만", "A 곱하기", "B 더하기"]]
    R = {k: [] for k in keys}; M = []
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); T = min(len(y), d["spike"].shape[1]); y, te = y[:T], te[:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); gts = [(a, b, interp_channels(e, a, b)) for a, b in segs(y)]
        R["지금"] += rank(S, gts); n0 = len(gts)
        for v in ["h1", "h2", "h3"]:
            H = drop_flat(d[v][:, :T].astype(float), te, tr); Hz, _ = calib(H)
            for m, X in [("높이만", H), ("A 곱하기", np.maximum(S, 0) * H), ("B 더하기", S + Hz)]:
                R[f"{v} {m}"] += rank(X, gts)
        M.append((e, n0))
    base = np.array(R["지금"]); ok = np.isfinite(base)
    L = [f"GT {ok.sum()}개 — 상위 몇 % (작을수록 높은 순위)", f"{'점수':14s} | 중앙   | 0.1% 안 | 0.5% 안 | 1% 안 | 지금보다 올라감 / 내려감 | 지금 0.5% 밖({np.sum(base[ok] > 0.5)}개) 중 안으로 들어옴"]
    for k in keys:
        r = np.array(R[k]); o = ok & np.isfinite(r)
        up = np.mean(r[o] < base[o]); dn = np.mean(r[o] > base[o]); came = np.sum((base[o] > 0.5) & (r[o] <= 0.5))
        L.append(f"{k:14s} | {np.median(r[o]):.3f}% | {np.mean(r[o] <= 0.1):5.0%} | {np.mean(r[o] <= 0.5):5.0%} | {np.mean(r[o] <= 1):4.0%} | {up:4.0%} / {dn:4.0%} | {'-' if k == '지금' else came}")
    L += ["", "machine별 GT 순위 중앙 (상위 %): 지금 → h1 B / h3 B"]
    i = 0
    for e, n in M:
        s = slice(i, i + n); i += n
        L.append(f"  {e:13s} GT {n:2d} | {np.nanmedian(base[s]):.3f} → {np.nanmedian(np.array(R['h1 B 더하기'])[s]):.3f} / {np.nanmedian(np.array(R['h3 B 더하기'])[s]):.3f}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "gt_rank_height.txt").write_text(txt)


if __name__ == "__main__":
    main()
