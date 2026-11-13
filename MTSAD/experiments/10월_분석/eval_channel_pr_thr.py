"""
채널 recall / precision — 임계값 고르는 기준을 바꿔가며 (모두 GT로 고르는 오라클)
  틱 F1 / event F1 / 채널 F1
  채널 F1 (entity 단위): 예측 덩어리마다 켜진 채널 vs 겹친 GT들의 채널 라벨
    TP = 맞춘 채널, FP = 틀린 채널 (GT 없는 덩어리의 채널은 전부 FP), FN = 못 맞춘 GT 채널 (놓친 GT 이상의 채널 포함)
  탐색 순서는 기존과 같음: 탈주기 임계값 먼저 → (dt, st) 121개 조합
  출력 지표(걸친 GT 이상 기준, eval_channel_pr.py와 같음) + 채널 F1(전체) + event R/P, 후보 수
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import segs, prf, f1, PH, SC, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_eventf1_threshold import ev_counts, ev_f1
from eval_channel_k import interp
import asd_eval


def gt_info(y, lab):
    gt = segs(y); return gt, [set(interp(lab, a, b)) for a, b in gt]


def ch_counts(m, y, gt, gch):
    """m: (C,T) 켜짐 → (TP, FP, FN) 채널 단위"""
    blocks = segs(m.any(0).astype(int)); tp = fp = 0; caught = set()
    for u, v in blocks:
        idx = [i for i, (a, b) in enumerate(gt) if a < v and b > u]
        sus = set(np.where(m[:, u:v].any(1))[0])
        g = set().union(*[gch[i] for i in idx]) if idx else set()
        tp += len(sus & g); fp += len(sus - g); caught |= {(i, c) for i in idx for c in gch[i] & sus}
    fn = sum(len(s) for s in gch) - len(caught)
    return tp, fp, fn


def crit(kind, m, y, gt, gch):
    p = m.any(0)
    if kind == "틱F1":
        return f1(*prf(y, p))
    if kind == "eventF1":
        return ev_f1(ev_counts(y, p, gt))
    tp, fp, fn = ch_counts(m, y, gt, gch)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def choose(kind, D, S, G, y, gt, gch):
    best, bmD = -1, None
    for t in np.linspace(np.percentile(D.max(0), 50), np.percentile(D.max(0), 99.9), 300 if kind != "채널F1" else 100):
        mD = D > t; v = crit(kind, mD, y, gt, gch)
        if v > best:
            best, bmD = v, mD
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf); bm = bmD
    for g in np.unique(np.percentile(G, ZQ)):
        Gb = G > g
        for dt in np.unique(np.percentile(S[fin], DQ)):
            m = bmD | ((Sv > dt) & Gb); v = crit(kind, m, y, gt, gch)
            if v > best:
                best, bm = v, m
    return bm


def main():
    data = {"SMD 28대": [], "ASD 12대": []}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        data["SMD 28대"].append((y, np.load(PH / f"{e}_phase_deseason.npz")["scores"], calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                                 np.nan_to_num(suddenness(te, tr), nan=0.0), SMD / "interpretation_label" / f"{e}.txt"))
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        data["ASD 12대"].append((y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["scores"], calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                                 np.nan_to_num(suddenness(te, tr), nan=0.0), asd_eval.ASD / "interpretation_label" / f"{e}.txt"))
    out = []
    for name, items in data.items():
        out.append(f"\n[{name}] {'기준':8s} | 채널R(macro) 채널P(macro) 전부맞춤 의심채널중앙 | 채널F1(전체 micro) | evR  evP  후보")
        for kind in ["틱F1", "eventF1", "채널F1"]:
            R = []; C = np.zeros(3); E = np.zeros(4)
            for y, D, S, G, lab in items:
                T = min(len(y), D.shape[1]); y, D, S, G = y[:T], D[:, :T], S[:, :T], G[:, :T]
                gt, gch = gt_info(y, lab)
                m = choose(kind, D, S, G, y, gt, gch)
                C += ch_counts(m, y, gt, gch); E += ev_counts(y, m.any(0), gt)
                blocks = segs(m.any(0).astype(int))
                for (a, b), g in zip(gt, gch):
                    ov = [(u, v) for u, v in blocks if u < b and v > a]
                    if ov and g:
                        sus = set(np.where(m[:, ov[0][0]:ov[-1][1]].any(1))[0]); R.append((len(g & sus), len(g), len(sus)))
            R = np.array(R); tp, fp, fn = C
            out.append(f"{'':12s}{kind:8s} | {np.mean(R[:, 0] / R[:, 1]):11.3f} {np.mean(R[:, 0] / np.maximum(R[:, 2], 1)):11.3f} {np.mean(R[:, 0] == R[:, 1]):8.3f} {np.median(R[:, 2]):10.0f} | "
                       f"{2 * tp / (2 * tp + fp + fn):17.3f} | {E[0] / E[1]:.3f} {E[3] / max(E[2], 1):.3f} {int(E[2])}")
            print(out[-1], flush=True)
    (HERE / "coverage_groups" / "channel_pr_thr.txt").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
