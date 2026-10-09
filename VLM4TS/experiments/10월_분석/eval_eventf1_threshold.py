"""
임계값을 틱 F1 대신 event F1이 최대가 되게 골라보기 (여전히 GT로 고르는 오라클 — 상한선 확인용)
  event R = GT 이상 중 1틱이라도 걸친 비율 / event P = 예측 덩어리 중 GT와 겹치는 비율 / event F1 = 둘의 조화평균
  지금 방법: 탈주기 임계값을 event F1로 먼저 고르고 → (dt, st) 121개 조합 중 결합 event F1 최대
  TimeRCD, 탈주기 단독: 임계값 후보 300개 중 event F1 최대
  비교용으로 기존(틱 F1로 고른) 결과도 같이 표시. SMD 28대 + ASD 12대
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "asd"))
from eval_dsz_allch import prf, f1, best_pred, segs, PH, SC, SMD, DQ, ZQ
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from eval_fix_variants import search
import asd_eval

BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def ev_counts(y, p, gt):
    """(걸친 GT 수, GT 수, 예측 덩어리 수, GT와 겹친 덩어리 수)"""
    pc = np.r_[0, np.cumsum(p)]; yc = np.r_[0, np.cumsum(y)]
    touch = sum(pc[b] - pc[a] > 0 for a, b in gt)
    d = np.diff(np.r_[0, p.astype(int), 0]); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    hit = int(((yc[e] - yc[s]) > 0).sum())
    return touch, len(gt), len(s), hit


def ev_f1(c):
    t, n, m, h = c
    R = t / n if n else 0; P = h / m if m else 0
    return 2 * P * R / (P + R) if P + R else 0.0


def best_ev(s, y, gt, n=300):
    best, bp = -1, None
    for t in np.linspace(np.percentile(s, 50), np.percentile(s, 99.9), n):
        p = s > t; v = ev_f1(ev_counts(y, p, gt))
        if v > best:
            best, bp = v, p
    return bp


def search_ev(p1, S, G, y, gt):
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
    dts = np.unique(np.percentile(S[fin], DQ)); gts = np.unique(np.percentile(G, ZQ))
    best, bp = ev_f1(ev_counts(y, p1, gt)), p1
    for g in gts:
        Gb = G > g
        for dt in dts:
            p = p1 | ((Sv > dt) & Gb).any(0); v = ev_f1(ev_counts(y, p, gt))
            if v > best:
                best, bp = v, p
    return bp


def report(name, preds_by_model, ys):
    print(f"\n[{name}]{'':20s}{'틱P':>6s}{'틱R':>6s}{'틱F1':>6s} | {'evR':>6s}{'evP':>6s}{'evF1':>6s}{'후보':>6s} | 걸침 spike / 16~224 / 225+")
    for k, preds in preds_by_model.items():
        pt = [0, 0, 0]; ec = [0, 0, 0, 0]; tl = {b[2]: [0, 0] for b in BINS}
        for y, p in zip(ys, preds):
            gt = segs(y); c = prf(y, p); pt = [pt[i] + c[i] for i in range(3)]
            ec = [ec[i] + v for i, v in enumerate(ev_counts(y, p, gt))]
            for a, b in gt:
                nm = [x for lo, hi, x in BINS if lo <= b - a <= hi][0]; tl[nm][1] += 1; tl[nm][0] += int(p[a:b].any())
        tp, fp, fn = pt; R = ec[0] / ec[1]; P = ec[3] / max(ec[2], 1)
        print(f"{k:28s}{tp / max(tp + fp, 1):6.3f}{tp / max(tp + fn, 1):6.3f}{f1(tp, fp, fn):6.3f} | {R:6.3f}{P:6.3f}{2 * P * R / (P + R):6.3f}{ec[2]:6d} | "
              + " / ".join(f"{tl[b[2]][0]}/{tl[b[2]][1]}" for b in BINS))


def evaluate(name, items):
    models = {k: [] for k in ["탈주기 단독 (틱F1 기준)", "탈주기 단독 (event F1 기준)", "지금 방법 (틱F1 기준)",
                              "지금 방법 (event F1 기준)", "TimeRCD (틱F1 기준)", "TimeRCD (event F1 기준)"]}
    ys = []
    for y, ph, S, sud, trcd in items:
        gt = segs(y); ys.append(y)
        p1t = best_pred(ph, y); p1e = best_ev(ph, y, gt)
        models["탈주기 단독 (틱F1 기준)"].append(p1t); models["탈주기 단독 (event F1 기준)"].append(p1e)
        models["지금 방법 (틱F1 기준)"].append(search(p1t, S, sud, y))
        models["지금 방법 (event F1 기준)"].append(search_ev(p1e, S, sud, y, gt))
        models["TimeRCD (틱F1 기준)"].append(best_pred(trcd, y)); models["TimeRCD (event F1 기준)"].append(best_ev(trcd, y, gt))
    report(name, models, ys)


def main():
    items = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        tc = np.load(HERE.parent / "24주차_원인분석" / "timercd_scores" / f"{e}.npz")["scores"].astype(float)
        items.append((y, np.load(PH / f"{e}_phase_deseason.npz")["combined"],
                      calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                      np.nan_to_num(suddenness(te, tr), nan=0.0), tc))
    evaluate("SMD 28대", items)
    items = []
    for e in asd_eval.ENTS:
        tr, te, y = asd_eval.load(e)
        items.append((y, np.load(HERE / "asd" / "deseason_scores" / f"{e}.npz")["combined"],
                      calibrate(np.load(HERE / "asd" / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                      np.nan_to_num(suddenness(te, tr), nan=0.0),
                      np.load(HERE / "asd" / "timercd_scores" / f"{e}.npz")["scores"].astype(float)))
    evaluate("ASD 12대", items)


if __name__ == "__main__":
    main()
