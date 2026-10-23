"""
eval_full_freq.py와 같은 설정의 채널 정확도 — 걸친 GT 이상마다 겹친 후보 범위 안에서 켜진 채널 vs GT 채널 라벨
  탈주기 채널별 켜짐 = 그 채널 탈주기 점수 > (채널 최댓값 점수의 상위 3% 값), spike 채널별 켜짐 = 새 점수 > 상위 0.5% AND 갑작 > 상위 st%
  지표: 채널 recall / precision (이상마다 평균), GT 채널 전부 맞춤 비율, 이상당 켜진 채널 수 중앙값, 채널 F1 (전체 micro, eval_channel_pr_thr.ch_counts)
"""
from pathlib import Path

import numpy as np

from eval_dsz_allch import segs, SC, SMD, PH
from eval_spike_detector import suddenness
from eval_flat_fix import calib, drop_flat
from eval_channel_pr_thr import gt_info, ch_counts

HERE = Path(__file__).resolve().parent
STS, LAMS = [1, 0.05], [0, 0.5, 1]


def main():
    keys = [("탈주기만", None, None)] + [(f"st {st}% λ {lam}", st, lam) for st in STS for lam in LAMS]
    R = {k[0]: [] for k in keys}; CC = {k[0]: np.zeros(3) for k in keys}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        d = np.load(PH / f"{e}_phase_deseason.npz"); Dc, Dm = d["scores"], d["combined"]
        raw = np.load(HERE / "spike_scores_rescale" / f"{e}.npz")["scores"].astype(float); T = min(len(y), raw.shape[1], len(Dm)); y, raw, te, Dc, Dm = y[:T], raw[:, :T], te[:T], Dc[:, :T], Dm[:T]
        mD = Dc > np.percentile(Dm, 97)
        S, _ = calib(drop_flat(raw, te, tr)); fin = np.isfinite(S)
        Gtr = np.nan_to_num(suddenness(tr, tr), nan=0.0)
        rate = np.array([len(segs((Gtr[c] >= 10).astype(int))) * 1000 / len(tr) if np.ptp(tr[:, c]) > 1e-6 else 0.0 for c in range(tr.shape[1])])
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); gt, gch = gt_info(y, SMD / "interpretation_label" / f"{e}.txt")
        for nm, st, lam in keys:
            if st is None:
                m = mD
            else:
                Sn = S - lam * np.log1p(rate)[:, None]
                m = mD | ((np.where(fin, Sn, -np.inf) > np.percentile(Sn[fin], 99.5)) & (G > np.percentile(G, 100 - st)))
            CC[nm] += ch_counts(m, y, gt, gch); bl = segs(m.any(0).astype(int))
            for (a, b), g in zip(gt, gch):
                ov = [(u, v) for u, v in bl if u < b and v > a]
                if ov and g:
                    sus = set(np.where(m[:, ov[0][0]:ov[-1][1]].any(1))[0]); R[nm].append((len(g & sus), len(g), len(sus)))
    L = ["채널 정확도 (SMD 28대, eval_full_freq.py와 같은 설정)", f"{'방식':16s} | 걸친 GT | 채널R  채널P  전부맞춤 | 켜진 채널 중앙 | 채널F1(전체)"]
    for nm, _, _ in keys:
        A = np.array(R[nm]); tp, fp, fn = CC[nm]
        L.append(f"{nm:16s} | {len(A):3d} | {np.mean(A[:, 0] / A[:, 1]):.3f} {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f} {np.mean(A[:, 0] == A[:, 1]):8.3f} | {np.median(A[:, 2]):5.0f} | {2 * tp / (2 * tp + fp + fn):.3f}")
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "full_freq_channel.txt").write_text(txt)


if __name__ == "__main__":
    main()
