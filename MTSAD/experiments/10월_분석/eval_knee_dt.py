"""
dt를 분포의 꺾이는 지점(knee)으로 — machine마다, GT 없이 (나연 2026-10-21)
  보정 spike 점수(1번 + 눈금 넓힘, 열마다 1개, 전 채널)를 큰 순서로 정렬 → 상위 W%만 떼어 곡선의 처음·끝을 잇는 직선에서 가장 먼 점 = knee → 그 점수 = dt
  (가로·세로 모두 0~1로 맞춘 뒤 거리 계산, Kneedle 방식)
  W = 2%, 5%, 10%.  비교: 고정 0.5%.  spike 쪽만, AND 갑작 상위 1%
  출력: 전체 성능 + machine마다 knee가 고른 비율(%) + 그림(몇 machine의 정렬 곡선과 knee)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_event2 import ev2_counts, ev2_scores
from eval_flat_fix import calib, drop_flat

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
P, WS = 14, [2, 5, 10]
SHOW = ["machine-2-8", "machine-3-11", "machine-1-6", "machine-2-5"]


def knee(v, w):
    s = np.sort(v)[::-1]; n = max(10, int(len(s) * w / 100)); c = s[:n]
    x = np.linspace(0, 1, n); yy = (c - c[-1]) / max(c[0] - c[-1], 1e-12)
    d = np.abs(yy - (1 - x))                       # 직선 (0,1)-(1,0)에서의 거리 (상수배 생략)
    k = int(np.argmax(d)); return c[k], 100 * (k + 1) / len(s), c


def main():
    keys = ["고정 0.5%"] + [f"knee 상위 {w}%" for w in WS]
    cnt = {k: 0 for k in keys}; sp = {k: [0, 0] for k in keys}; pct = {k: [] for k in keys[1:]}; curves = {}
    L2 = [f"{'machine':13s} | GT 수 | " + " | ".join(f"{k} 비율 (후보)" for k in keys)]
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
        v = S[:, ::P]; v = v[np.isfinite(v)]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        row = [f"{e:13s} | {len(segs(y)):4d}"]
        for k in keys:
            if k == "고정 0.5%":
                dt = np.percentile(S[fin], 99.5); r = 0.5
            else:
                dt, r, c = knee(v, int(k.split()[2][:-1])); pct[k].append(r)
                if e in SHOW and k == "knee 상위 5%":
                    curves[e] = (c, dt, r, len(v))
            pr = ((Sv > dt) & Gb).any(0); cnt[k] = cnt[k] + ev2_counts(y, pr)
            for a, b in segs(y):
                if b - a <= 15:
                    sp[k][1] += 1; sp[k][0] += int(pr[a:b].any())
            row.append(f"{r:5.2f}% ({len(segs(pr.astype(int))):3d})")
        L2.append(" | ".join(row))
    L = ["dt를 꺾이는 지점으로 (spike 쪽만, 1번 + 눈금 넓힘, AND 갑작 상위 1%, SMD 28대)",
         f"{'방식':14s} | evP   evR   evF1 | R50   F1(50%) | 후보 | 짧은 spike 놓침 | machine마다 고른 비율 (최소~최대, 중앙)"]
    for k in keys:
        s = ev2_scores(cnt[k]); pr = "0.5% 고정" if k == "고정 0.5%" else f"{min(pct[k]):.2f}~{max(pct[k]):.2f}%, 중앙 {np.median(pct[k]):.2f}%"
        L.append(f"{k:14s} | {s['P']:.3f} {s['R']:.3f} {s['F']:.3f} | {s['R50']:.3f} {s['F50']:.3f}   | {s['n']:4d} | {sp[k][1] - sp[k][0]:3d} | {pr}")
    txt = "\n".join(L + [""] + L2); print(txt)
    (HERE / "coverage_groups" / "knee_dt_result.txt").write_text(txt)
    fig, ax = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4), squeeze=False)
    for A, (e, (c, dt, r, n)) in zip(ax.flat, curves.items()):
        A.plot(100 * np.arange(1, len(c) + 1) / n, c, color="k"); A.axhline(dt, color="r", ls="--"); A.axvline(r, color="r", ls=":")
        A.axvline(0.5, color="gray", ls=":"); A.set_xlabel("상위 몇 % (정렬 순위)"); A.set_ylabel("보정 spike 점수")
        A.set_title(f"{e}: knee {r:.2f}% (dt {dt:.1f})\n회색 점선 = 고정 0.5%", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "coverage_groups" / "knee_dt_curves.png", dpi=80)


if __name__ == "__main__":
    main()
