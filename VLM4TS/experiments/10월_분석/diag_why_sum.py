"""
왜 v3 더하기(보정 spike S + 보정 높이 Hz)가 각각보다 나은가 — 열(14틱) 단위, SMD 28대
  순위 = machine 안 전체 칸 중 상위 몇 % (점수마다 따로)
  1. 전체 칸에서 S와 Hz 순위상관
  2. 정상 칸 중 S 상위 0.5% (v0이 잘못 높게 준 칸): Hz 순위 분포, 더하기 상위 0.5%에 남는 비율
     정상 칸 중 Hz 상위 0.5% (높이만이 잘못 높게 준 칸): S 순위 분포, 더하기에 남는 비율
  3. GT (GT 채널·GT 구간 중 더하기 점수가 가장 높은 칸): S 순위, Hz 순위, 더하기 순위
  그림: 산점도 (가로 S 순위, 세로 Hz 순위, 로그) — 회색 = 정상 칸 표본, 빨강 = GT 칸
"""
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SC, SMD
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
P = 14
RNG = np.random.default_rng(0)


def pct(v, pool):
    s = np.sort(pool)[::-1]; return 100 * (np.searchsorted(-s, -v) + 1) / len(s)


def main():
    rho, nS, nH, gt, bg = [], [], [], [], []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(HERE / "spike_height_scores" / f"{e}.npz"); T = min(len(y), d["spike"].shape[1]); y, te = y[:T], te[:T]
        S, _ = calib(drop_flat(d["spike"][:, :T].astype(float), te, tr)); Hz, _ = calib(drop_flat(d["h3"][:, :T].astype(float), te, tr))
        ncol = T // P; s = S[:, :ncol * P:P]; h = Hz[:, :ncol * P:P]; ok = np.isfinite(s) & np.isfinite(h)
        near = np.zeros(T, bool)
        for a, b in segs(y):
            near[max(0, a - 30):b + 30] = True
        nc = np.broadcast_to(near[:ncol * P].reshape(ncol, P).any(1), s.shape)
        sv, hv = s[ok], h[ok]; xv = sv + hv
        rS = np.array([pct(v, sv) for v in sv]) if False else None
        order = lambda v: 100 * (np.argsort(np.argsort(-v)) + 1) / len(v)
        rs, rh, rx = order(sv), order(hv), order(xv); rho.append(spearmanr(sv, hv)[0])
        norm = ~nc[ok]
        m1 = norm & (rs <= 0.5); nS.append(np.c_[rh[m1], rx[m1]])
        m2 = norm & (rh <= 0.5); nH.append(np.c_[rs[m2], rx[m2]])
        pick = np.where(norm)[0]; pick = RNG.choice(pick, min(400, len(pick)), replace=False); bg.append(np.c_[rs[pick], rh[pick]])
        idx = -np.ones(s.shape, int); idx[ok] = np.arange(ok.sum())
        for a, b in segs(y):
            cs = [c for c in interp_channels(e, a, b)]
            cand = [(idx[c, k]) for c in cs for k in range(a // P, min(ncol, (b - 1) // P + 1)) if idx[c, k] >= 0]
            if cand:
                j = max(cand, key=lambda j: xv[j]); gt.append((rs[j], rh[j], rx[j], b - a <= 15))
    nS, nH, gt, bg = np.vstack(nS), np.vstack(nH), np.array(gt), np.vstack(bg)
    L = [f"1. 전체 칸에서 S와 Hz 순위상관 (machine 중앙): {np.median(rho):.2f}",
         f"2. 정상 칸 중 S 상위 0.5% ({len(nS)}칸, v0이 잘못 높게 준 칸): Hz 순위 중앙 상위 {np.median(nS[:, 0]):.2f}% | Hz도 상위 0.5% 안 {np.mean(nS[:, 0] <= 0.5):.0%} | 더하기 상위 0.5%에 남음 {np.mean(nS[:, 1] <= 0.5):.0%}",
         f"   정상 칸 중 Hz 상위 0.5% ({len(nH)}칸, 높이만이 잘못 높게 준 칸): S 순위 중앙 상위 {np.median(nH[:, 0]):.2f}% | S도 상위 0.5% 안 {np.mean(nH[:, 0] <= 0.5):.0%} | 더하기 상위 0.5%에 남음 {np.mean(nH[:, 1] <= 0.5):.0%}",
         f"3. GT {len(gt)}개 (더하기 점수가 가장 높은 GT 칸 기준): S 순위 중앙 {np.median(gt[:, 0]):.3f}% / Hz {np.median(gt[:, 1]):.3f}% / 더하기 {np.median(gt[:, 2]):.3f}%",
         f"   상위 0.5% 안: S {np.mean(gt[:, 0] <= 0.5):.0%} / Hz {np.mean(gt[:, 1] <= 0.5):.0%} / 더하기 {np.mean(gt[:, 2] <= 0.5):.0%} | S와 Hz 둘 다 상위 2% 안 {np.mean((gt[:, 0] <= 2) & (gt[:, 1] <= 2)):.0%}",
         f"   정상 칸 표본: S와 Hz 둘 다 상위 2% 안 {np.mean((bg[:, 0] <= 2) & (bg[:, 1] <= 2)):.2%}"]
    txt = "\n".join(L); print(txt); (HERE / "coverage_groups" / "why_sum.txt").write_text(txt)
    fig, A = plt.subplots(figsize=(8, 7))
    A.scatter(bg[:, 0], bg[:, 1], s=3, c="0.75", label="정상 칸 (표본)")
    A.scatter(nS[:, 0] * 0 + 0, nS[:, 0] * 0, s=0)
    A.scatter(gt[:, 0], gt[:, 1], s=14, c="red", label="GT 칸")
    xs = np.logspace(-3, 2, 200); A.plot(xs, xs, color="k", lw=.5, ls=":")
    A.axvline(0.5, color="tab:blue", ls="--", lw=1, label="S 상위 0.5%"); A.axhline(0.5, color="tab:green", ls="--", lw=1, label="Hz 상위 0.5%")
    A.set_xscale("log"); A.set_yscale("log"); A.set_xlim(100, 0.002); A.set_ylim(100, 0.002)
    tk = [100, 10, 1, 0.1, 0.01]; A.set_xticks(tk); A.set_xticklabels([f"{t:g}%" for t in tk]); A.set_yticks(tk); A.set_yticklabels([f"{t:g}%" for t in tk]); A.minorticks_off()
    A.set_xlabel("보정 spike 점수 순위 (상위 %, 오른쪽일수록 높음)"); A.set_ylabel("보정 높이 점수 순위 (상위 %, 위쪽일수록 높음)")
    A.set_title("칸마다 두 점수의 순위 — 오른쪽 위 모서리 = 둘 다 높음"); A.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(HERE / "coverage_groups" / "why_sum_scatter.png", dpi=90)


if __name__ == "__main__":
    main()
