"""
구간은 고정, 채널 고르는 기준만 낮춰서 채널 recall을 최대로 올리면?
  구간 = 채널 F1 기준 오라클로 고른 최종 모델의 예측 덩어리 (eval_channel_pr_thr.py와 같음)
  채널 점수 = 덩어리 안 최댓값 of max(탈주기/탈주기 임계값, min(보정 spike/dt, 갑작스러움/st))  → 1 이상이면 모델이 실제로 켠 채널
  기준 τ를 1 → 0으로 낮춰가며: 채널 점수 > τ 인 채널을 넘김. τ=0 미만 = 전부
  지표(걸친 GT 이상 기준): 채널 recall / precision / GT 채널 전부 맞춤 / 넘긴 채널 수 중앙·평균
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import segs, PH, SC, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_channel_pr_thr import gt_info, ch_counts
from eval_channel_k import interp
import asd_eval

TAUS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -np.inf]


def f1c(m, y, gt, gch):
    tp, fp, fn = ch_counts(m, y, gt, gch); return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def choose_thr(D, S, G, y, gt, gch):
    best, bt = -1, None
    for t in np.linspace(np.percentile(D.max(0), 50), np.percentile(D.max(0), 99.9), 100):
        v = f1c(D > t, y, gt, gch)
        if v > best:
            best, bt = v, t
    mD = D > bt; fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); bd, bg = np.inf, np.inf
    for g in np.unique(np.percentile(G, ZQ)):
        Gb = G > g
        for dt in np.unique(np.percentile(S[fin], DQ)):
            v = f1c(mD | ((Sv > dt) & Gb), y, gt, gch)
            if v > best:
                best, bd, bg = v, dt, g
    return bt, bd, bg


def main():
    sets = {"SMD 28대": [], "ASD 12대": []}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        sets["SMD 28대"].append((y, np.load(PH / f"{e}_phase_deseason.npz")["scores"], calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                                 np.nan_to_num(suddenness(te, tr), nan=0.0), SMD / "interpretation_label" / f"{e}.txt"))
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        sets["ASD 12대"].append((y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["scores"], calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                                 np.nan_to_num(suddenness(te, tr), nan=0.0), asd_eval.ASD / "interpretation_label" / f"{e}.txt"))
    out = []
    for name, items in sets.items():
        R = {t: [] for t in TAUS}; nev = 0
        for y, D, S, G, lab in items:
            T = min(len(y), D.shape[1]); y, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
            gt, gch = gt_info(y, lab)
            td, dt, st = choose_thr(D, S, G, y, gt, gch)
            Sv = np.where(np.isfinite(S), S, -np.inf)
            rS = np.minimum(Sv / dt, G / st) if np.isfinite(dt) else np.full_like(D, -np.inf)
            score = np.maximum(D / td, rS)
            pred = (score > 1).any(0); blocks = segs(pred.astype(int))
            for (a, b), g in zip(gt, gch):
                ov = [(u, v) for u, v in blocks if u < b and v > a]
                if not ov or not g:
                    continue
                nev += 1
                cs = score[:, ov[0][0]:ov[-1][1]].max(1)
                for t in TAUS:
                    sus = set(np.where(cs > t)[0]); R[t].append((len(g & sus), len(g), len(sus)))
        C = items[0][1].shape[0]
        out.append(f"\n[{name}] 걸친 GT 이상 {nev}개, 채널 {C}개 (구간 = 채널F1 기준 최종 모델, 고정)")
        out.append(f"  {'기준 τ':>8s} | 채널R  채널P  전부맞춤 | 넘긴 채널 중앙 평균")
        for t in TAUS:
            A = np.array(R[t])
            out.append(f"  {('전부' if not np.isfinite(t) else f'{t:.1f}'):>8s} | {np.mean(A[:, 0] / A[:, 1]):.3f} {np.mean(A[:, 0] / np.maximum(A[:, 2], 1)):.3f} {np.mean(A[:, 0] == A[:, 1]):8.3f} | {np.median(A[:, 2]):9.0f} {A[:, 2].mean():5.1f}")
    txt = "\n".join(out); print(txt)
    (HERE / "coverage_groups" / "channel_recall_sweep.txt").write_text(txt)


if __name__ == "__main__":
    main()
