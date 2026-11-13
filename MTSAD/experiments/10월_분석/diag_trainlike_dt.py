"""
dt(보정 spike 상위 0.5%)를 "train에서도 흔했던 크기의 튐"이 얼마나 차지하나 — 기준: 1번 + 눈금 넓힘
  열(14틱)마다 튐 높이 = 열 안 |값 − 주변 ±100틱 중앙값| 최댓값 / train 범위
  train에서 흔한 튐 = 그 높이 ≤ 그 채널 train 정상 spike 높이(갑작 ≥ 10인 묶음)의 상위 1% 지점
  출력: machine마다 상위 0.5% 칸 중 "train에서 흔한 튐" 비율, 그 칸을 빼고 다시 구한 dt, 짧은 GT spike 중 spike 점수로 걸리는 수 변화
"""
from pathlib import Path

import numpy as np
import pandas as pd

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_flat_fix import calib, drop_flat
from exp_patch_groups import interp_channels
from diag_spiky_gt import spikes

HERE = Path(__file__).resolve().parent
P = 14


def main():
    L = [f"{'machine':13s} | 상위 0.5% 중 train에서 흔한 튐 | dt → 뺀 dt | 짧은 GT spike 중 spike 점수 통과 (지금 → 뺀 dt)"]
    tot = [0, 0, 0]; fr = []
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr)); C = S.shape[0]
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0)
        rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
        med = pd.DataFrame(te).rolling(201, center=True, min_periods=50).median().values
        dev = np.abs(te - med) / rg; ncol = T // P
        cdev = dev[:ncol * P].reshape(ncol, P, C).max(1).T                     # (C, ncol)
        q = np.zeros(C)
        for c in range(C):
            if np.ptp(tr[:, c]) < 1e-6:
                continue
            h = np.array([x for *_, x in spikes(tr[:, c], Gtr[c], rg[c])])
            q[c] = np.percentile(h, 99) if len(h) >= 5 else 0.0
        Sc = S[:, :ncol * P:P]; fin = np.isfinite(Sc)
        common = fin & (np.nan_to_num(cdev) <= q[:, None])
        dt = np.percentile(Sc[fin], 99.5); top = fin & (Sc > dt)
        Sx = np.where(common, -np.inf, np.where(fin, Sc, -np.inf)); dt2 = np.percentile(Sx[fin], 99.5)
        k1 = k2 = n = 0
        for a, b in segs(y):
            if b - a > 15:
                continue
            g = interp_channels(e, a, b)
            if not g:
                continue
            n += 1; v = np.nanmax(np.where(np.isfinite(S[g, a:b]), S[g, a:b], -np.inf))
            k1 += int(v > dt); k2 += int(v > dt2)
        f = (top & common).sum() / max(top.sum(), 1); fr.append(f); tot = [tot[0] + k1, tot[1] + k2, tot[2] + n]
        L.append(f"{e:13s} | {f:4.0%} | {dt:.2f} → {dt2:.2f} | {k1}/{n} → {k2}/{n}")
    L += ["", f"machine 평균: 상위 0.5% 중 train에서 흔한 튐 {np.mean(fr):.0%} | 짧은 GT spike spike 점수 통과 {tot[0]}/{tot[2]} → {tot[1]}/{tot[2]}"]
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "trainlike_dt.txt").write_text(txt)


if __name__ == "__main__":
    main()
