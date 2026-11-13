"""
train처럼 생긴 튐(상위 1% 기준)을 지웠을 때 새로 놓친 짧은 GT spike — 왜 놓쳤나 직접 확인
  각 GT: (1) 가장 크게 튄 GT 채널: 높이 / 그 채널 train 기준(q99) / spike 보정 점수 vs dt / 갑작 vs st / 켜졌나
         (2) 원래 켜졌던 채널들: 높이 / 그 채널 q99 (켜진 틱 기준)
  높이 = |값 − 주변 ±100틱 중앙값| / train 범위 (GT 구간 안 최댓값)
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


def main():
    L = []; stat = {"가장 큰 채널이 train보다 큼": 0, "그중 spike 점수 < dt": 0, "그중 갑작 < st": 0, "그중 켜졌는데도 놓침": 0, "가장 큰 채널도 train처럼 생김": 0}; n = 0
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1]); y, raw, te = y[:T], raw[:, :T], te[:T]
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); dt = np.percentile(S[fin], 99.5)
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); st = np.percentile(G, 99); m0 = (Sv > dt) & (G > st)
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0); rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
        q = np.full(tr.shape[1], -np.inf)
        for c in range(tr.shape[1]):
            if np.ptp(tr[:, c]) > 1e-6:
                h = np.array([x for *_, x in spikes(tr[:, c], Gtr[c], rg[c])])
                if len(h) >= 5:
                    q[c] = np.percentile(h, 99)
        med = pd.DataFrame(te).rolling(201, center=True, min_periods=50).median().values
        H = (np.abs(te - med) / rg).T
        m1 = m0 & ~(np.nan_to_num(H) <= q[:, None])
        for a, b in segs(y):
            if b - a > 15 or not m0[:, a:b].any() or m1[:, a:b].any():
                continue
            n += 1; g = [c for c in interp_channels(e, a, b) if np.ptp(tr[:, c]) > 1e-6]
            if not g:
                continue
            c = max(g, key=lambda k: np.nanmax(H[k, a:b])); hmax = np.nanmax(H[c, a:b])
            sv = np.nanmax(Sv[c, a:b]); gv = G[c, a:b].max(); on = m0[c, a:b].any()
            if hmax > q[c]:
                stat["가장 큰 채널이 train보다 큼"] += 1
                if on:
                    stat["그중 켜졌는데도 놓침"] += 1
                elif sv <= dt:
                    stat["그중 spike 점수 < dt"] += 1
                else:
                    stat["그중 갑작 < st"] += 1
            else:
                stat["가장 큰 채널도 train처럼 생김"] += 1
            fired = np.where(m0[:, a:b].any(1))[0]
            fs = ", ".join(f"ch{k}(높이 {np.nanmax(np.where(m0[k, a:b], H[k, a:b], 0)):.2f} / 기준 {q[k]:.2f})" for k in fired)
            L.append(f"{e} [{a},{b}) | 가장 큰 GT 채널 ch{c}: 높이 {hmax:.2f} / 기준 {q[c]:.2f} | spike {sv:.1f}/dt {dt:.1f} | 갑작 {gv:.0f}/st {st:.0f} | 켜짐 {'O' if on else 'X'} || 원래 켜진 채널: {fs}")
    head = [f"지운 뒤 새로 놓친 짧은 GT spike {n}개"] + [f"  {k}: {v}" for k, v in stat.items()] + [""]
    txt = "\n".join(head + L); print("\n".join(head + L[:25]))
    (HERE / "coverage_groups" / "removed_gt_why.txt").write_text(txt)


if __name__ == "__main__":
    main()
