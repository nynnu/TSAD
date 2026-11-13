"""
0.5% 고정이 왜 문제인가 — 지금 설정 (1번 + 눈금 넓힘, spike 쪽만, dt = 보정 점수 상위 0.5%, st = 갑작 상위 1%), SMD 28대
  machine마다: GT 이상 수 / GT 틱 비율 / 켜진 칸 수(= 0.5%라 거의 고정) / 후보 / 오탐 / dt
             짧은 GT spike 중 dt 때문에 놓친 것 (GT 채널 보정 점수 < dt, 갑작은 통과) / 그중 dt의 80% 이상까지 올라온 것
"""
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels

HERE = Path(__file__).resolve().parent


def main():
    L = [f"{'machine':13s} | GT 수 | GT 틱% | 켜진 칸(spike>dt) | 후보 | 오탐 | dt   | 짧은 GT dt 탈락 (그중 dt의 80% 이상)"]
    rows = []
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); dt = np.percentile(S[fin], 99.5)
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); Gb = G > np.percentile(G, 99)
        m = (Sv > dt) & Gb; bl = segs(m.any(0).astype(int)); fp = sum(1 for u, v in bl if not y[u:v].any())
        gt = segs(y); miss = near = 0
        for a, b in gt:
            if b - a > 15 or m[:, a:b].any():
                continue
            g = [c for c in interp_channels(e, a, b) if np.isfinite(S[c, a:b]).any() and Gb[c, a:b].any()]
            if not g:
                continue
            v = max(np.nanmax(S[c, a:b]) for c in g)
            if v <= dt:
                miss += 1; near += int(v >= 0.8 * dt)
        rows.append((len(gt), len(bl), fp)); on = int((Sv > dt).sum())
        L.append(f"{e:13s} | {len(gt):4d} | {y.mean() * 100:5.1f} | {on:6d} ({on / fin.sum() * 100:.2f}%) | {len(bl):4d} | {fp:4d} | {dt:.2f} | {miss} ({near})")
    A = np.array(rows); r1, _ = spearmanr(A[:, 0], A[:, 1]); r2, _ = spearmanr(A[:, 0], A[:, 2])
    L += ["", f"GT 이상 수와 후보 수의 순위상관 {r1:.2f}, GT 이상 수와 오탐 수의 순위상관 {r2:.2f}",
          f"GT 이상 5개 이하 machine {np.sum(A[:, 0] <= 5)}개의 후보 평균 {A[A[:, 0] <= 5, 1].mean():.0f} vs GT 15개 이상 machine {np.sum(A[:, 0] >= 15)}개의 후보 평균 {A[A[:, 0] >= 15, 1].mean():.0f}"]
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "fixed05_why.txt").write_text(txt)


if __name__ == "__main__":
    main()
