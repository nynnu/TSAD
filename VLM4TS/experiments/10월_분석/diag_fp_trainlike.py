"""
오탐 덩어리 분류 — 지금 설정 (1번 + 눈금 넓힘, spike 쪽만 dt 0.5% AND 갑작 1%), SMD 28대
  켜진 채널마다 튐 높이 = 덩어리 안 |값 − 주변 ±100틱 중앙값| 최댓값 / train 범위
  train처럼 생김 = 켜진 채널 모두 높이 ≤ 그 채널 train 정상 spike 높이 상위 1% 지점 (train spike 5개 미만이면 기준 0 → train보다 큼으로)
  분류: GT 30틱 안 / train처럼 생김 / train보다 큼(하나라도)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from eval_dsz_allch import segs, SC, SMD
from eval_spike_detector import suddenness
from eval_flat_fix import calib, drop_flat
from diag_spiky_gt import spikes

HERE = Path(__file__).resolve().parent


def main():
    tot = {"GT 30틱 안": 0, "train처럼 생김": 0, "train보다 큼": 0}; rows = []
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)) & (G > np.percentile(G, 99))
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0); rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
        q = np.zeros(tr.shape[1])
        for c in range(tr.shape[1]):
            if np.ptp(tr[:, c]) > 1e-6:
                h = np.array([x for *_, x in spikes(tr[:, c], Gtr[c], rg[c])]); q[c] = np.percentile(h, 99) if len(h) >= 5 else 0.0
        med = pd.DataFrame(te).rolling(201, center=True, min_periods=50).median().values
        gt = segs(y); near = np.zeros(T, bool)
        for a, b in gt:
            near[max(0, a - 30):b + 30] = True
        c0 = {k: 0 for k in tot}
        for u, v in segs(m.any(0).astype(int)):
            if y[u:v].any():
                continue
            if near[u:v].any():
                k = "GT 30틱 안"
            else:
                chs = np.where(m[:, u:v].any(1))[0]
                h = [np.abs(te[u:v, c] - med[u:v, c]).max() / rg[c] for c in chs]
                k = "train처럼 생김" if all(hh <= q[c] for hh, c in zip(h, chs)) else "train보다 큼"
            c0[k] += 1
        for k in tot:
            tot[k] += c0[k]
        rows.append(f"{e:13s} | " + " | ".join(f"{c0[k]:3d}" for k in tot))
    n = sum(tot.values())
    L = [f"오탐 덩어리 {n}개: " + ", ".join(f"{k} {v}개 ({v / n:.0%})" for k, v in tot.items()), "", f"{'machine':13s} | " + " | ".join(tot)] + rows
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "fp_trainlike.txt").write_text(txt)


if __name__ == "__main__":
    main()
