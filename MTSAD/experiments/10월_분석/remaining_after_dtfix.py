"""
가정 11 이후: spike 탐지기(채널별 보정) 버전에서 아직 한 틱도 못 잡은 GT 이상
  관문(이상이 가장 크게 나타난 채널 = 구간 안 z 최대 채널 기준):
    0. 그 entity에서 2차 꺼짐  1. 그 채널 점수 없음(train 상수 / 마지막 꼬리)  2. 보정 점수 ≤ dt  3. z ≤ zt  4. 같은 틱에 안 겹침
그림: 원본 신호 + 보정 spike 점수(오른쪽 축) + dt
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import best_pred, segs, PH, SC, SMD
from gate_analysis import search
from eval_dt_fix import calibrate

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
SP = HERE / "spike_scores"
RES = HERE / "coverage_groups"
GATES = ["0. 2차 꺼짐", "1. 그 채널 점수 없음", "2. 보정 점수 ≤ dt", "3. z ≤ zt", "4. 같은 틱에 안 겹침"]


def main():
    rows, cache = [], {}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        S = calibrate(np.load(SP / f"{e}.npz")["scores"].astype(float))
        bt = search(p1, S, Z, y)
        pred = p1 if bt is None else p1 | ((np.where(np.isfinite(S), S, -np.inf) > bt[0]) & (Z > bt[1])).any(0)
        cache[e] = (tr, te, y, S, bt, pred)
        for a, b in segs(y):
            if pred[a:b].any():
                continue
            c = int(Z[:, a:b].max(1).argmax()); s = S[c, a:b]
            if bt is None: g = 0
            elif not np.isfinite(s).any(): g = 1
            elif np.nanmax(s) <= bt[0]: g = 2
            elif Z[c, a:b].max() <= bt[1]: g = 3
            else: g = 4
            rows.append(dict(e=e, a=int(a), b=int(b), L=int(b - a), ch=c, gate=g,
                             const=bool(tr[:, c].max() - tr[:, c].min() < 1e-6),
                             s=float(np.nanmax(s)) if np.isfinite(s).any() else None,
                             dt=None if bt is None else float(bt[0]), z=float(Z[c, a:b].max()),
                             zt=None if bt is None else float(bt[1])))
    json.dump(rows, open(RES / "remaining_after_dtfix.json", "w"), ensure_ascii=False, indent=1)
    L = np.array([r["L"] for r in rows])
    print(f"채널별 보정 후에도 한 틱도 못 잡은 GT 이상: {len(rows)}개 (spike {(L <= 15).sum()}, 16~224 {((L > 15) & (L <= 224)).sum()}, 225+ {(L > 224).sum()})")
    for g, name in enumerate(GATES):
        s = [r for r in rows if r["gate"] == g]
        extra = f"  (train에서 상수 {sum(r['const'] for r in s)}개)" if g == 1 else ""
        if g == 2 and s:
            extra = f"  (보정 점수/dt 중앙값 {np.median([r['s'] / r['dt'] for r in s]):.2f})"
        print(f"  {name:22s} {len(s):3d}개{extra}")

    for f0 in range(0, len(rows), 12):
        fig, ax = plt.subplots(4, 3, figsize=(18, 13))
        for A, r in zip(ax.flat, rows[f0:f0 + 12]):
            tr, te, y, S, bt, pred = cache[r["e"]]
            pad = max(200, r["L"]); s0, t0 = max(0, r["a"] - pad), min(len(te), r["b"] + pad)
            A.plot(range(s0, t0), te[s0:t0, r["ch"]], color="k", lw=.8)
            dd = np.diff(np.r_[0, y[s0:t0], 0])
            for x0, x1 in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
                A.axvspan(s0 + x0, s0 + x1, color="r", alpha=.15)
            A.axvspan(r["a"], r["b"], color="r", alpha=.35)
            if np.isfinite(S[r["ch"], s0:t0]).any() and bt is not None:
                A2 = A.twinx(); A2.plot(range(s0, t0), S[r["ch"], s0:t0], color="darkorange", lw=1, alpha=.8)
                A2.axhline(bt[0], color="red", ls="--", lw=1.2)
            A.set_title(f"{r['e']} ch{r['ch']} {r['L']}틱 | {GATES[r['gate']]}", fontsize=9)
        for A in ax.flat[len(rows[f0:f0 + 12]):]:
            A.axis("off")
        fig.suptitle(f"채널별 보정 후에도 못 잡은 이상 ({f0 + 1}~{min(f0 + 12, len(rows))} / {len(rows)}) — 검정 = 원본, 주황 = 보정 spike 점수(오른쪽 축), 빨간 점선 = dt", fontsize=12)
        fig.tight_layout(); fig.savefig(RES / f"남은0틱_{f0 // 12 + 1:02d}.png", dpi=70); plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
