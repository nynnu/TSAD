"""상수 채널 0~1 추가 후 생긴 오탐 사례 그림 (machine-3-11 ch31 3646·3796 / ch10 GT [27825,27951) 안)
  눈금 = DINOv2가 보는 것: 상수 채널은 0~1, 나머지는 train min~max. 주황 점 = 그 채널에서 spike 규칙(보정 spike > dt AND 갑작스러움 > st)이 켜진 틱
  같이 그린 채널 = 상수 채널 ch10·ch31 + 그 구간에서 갑작스러움 상위 3채널
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, best_pred, prf, f1, PH, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
E = "machine-3-11"
CASES = [(3500, 3950, "ch31 오탐 2개 (3646, 3796)"), (27700, 28100, "GT [27825,27951) 안 ch10 (라벨에 없음)")]


def search_thr(p1, S, G, y):
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); best, bd, bg = f1(*prf(y, p1)), np.inf, np.inf
    for g in np.unique(np.percentile(G, ZQ)):
        Gb = G > g
        for dt in np.unique(np.percentile(S[fin], DQ)):
            v = f1(*prf(y, p1 | ((Sv > dt) & Gb).any(0)))
            if v > best:
                best, bd, bg = v, dt, g
    return bd, bg


def main():
    y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
    tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
    raw = np.load(HERE / "spike_scores" / f"{E}.npz")["scores"].astype(float)
    d = np.load(HERE / "spike_scores_const01" / f"{E}.npz"); chs0 = list(d["chs"]); raw[chs0] = d["scores"][chs0]
    S = calibrate(raw); G = np.nan_to_num(suddenness(te, tr), nan=0.0)
    dt, st = search_thr(best_pred(np.load(PH / f"{E}_phase_deseason.npz")["combined"], y), S, G, y)
    fired = (np.where(np.isfinite(S), S, -np.inf) > dt) & (G > st)
    fig, ax = plt.subplots(1, len(CASES), figsize=(13 * len(CASES), 9), squeeze=False)
    for A, (s0, s1, title) in zip(ax.flat, CASES):
        xs = np.arange(s0, s1)
        gch = set()
        for a, b in segs(y):
            if a < s1 and b > s0:
                gch |= set(interp_channels(E, a, b))
        others = [c for c in np.argsort(-G[:, s0:s1].max(1)) if c not in chs0][:3]
        chs = chs0 + sorted(others)
        for k, c in enumerate(chs):
            g0, g1 = (0.0, 1.0) if c in chs0 else (tr[:, c].min(), tr[:, c].max())
            x = np.clip((te[s0:s1, c] - g0) / (g1 - g0 + 1e-8), 0, 1) * .8 + k
            A.axhline(k, color="0.85", lw=.5); A.axhline(k + .8, color="0.85", lw=.5)
            A.plot(xs, x, color="k", lw=.9)
            f = np.where(fired[c, s0:s1])[0]
            A.scatter(xs[f], x[f], color="orange", s=40, zorder=3)
            tag = "상수, 0~1" if c in chs0 else "train 범위"
            A.text(s0, k + .4, f"{'GT ' if c in gch else ''}ch{c} ({tag}) ", ha="right", va="center", fontsize=9)
        for a, b in segs(y):
            if a < s1 and b > s0:
                A.axvspan(max(a, s0), min(b, s1), color="r", alpha=.08)
        for w in range((s0 // 224 + 1) * 224, s1, 224):
            A.axvline(w, color="gray", ls="--", lw=.8)
        A.set_yticks([]); A.set_xlim(s0, s1); A.set_ylim(-.2, len(chs))
        A.set_title(f"{E} {title}\n빨강 배경 = GT, 주황 점 = spike 규칙이 켜진 틱, 회색 점선 = 224틱 그림 경계, GT 채널 라벨 {sorted(gch) if gch else '없음'}", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "stage12_check" / f"{E}_const01_cases.png", dpi=70)


if __name__ == "__main__":
    main()
