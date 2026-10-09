"""
ASD(omi-1~12) 공통 채점 — SMD와 같은 방식
  틱 P/R/F1 (오라클 임계값은 각 모델 쪽에서 정해서 예측을 넘김)
  event R(걸침) / R(50%) / P / F1 / 후보 수 / 길이별 걸침 / 이상 종류별 걸침(metric, temporal, metric-temporal)
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASD = HERE.parents[2] / "mv_data" / "ASD"
ENTS = [f"omi-{i}" for i in range(1, 13)]
BINS = [(1, 15, "spike"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]
TYPES = ["metric", "temporal", "metric-temporal"]


def load(e):
    tr = np.loadtxt(ASD / "train" / f"{e}.txt", delimiter=",")
    te = np.loadtxt(ASD / "test" / f"{e}.txt", delimiter=",")
    y = np.loadtxt(ASD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
    return tr, te, y


def anomaly_types():
    t = {}
    for line in open(ASD / "interpretation_label" / "anomaly_type.txt").read().splitlines()[1:]:
        ds, seg, typ = line.split(",")
        a, b = map(int, seg.split("-"))
        t[(ds, a)] = typ
    return t


def segs(y):
    d = np.diff(np.r_[0, y, 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def prf(y, p):
    return int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def best_pred(s, y, n=300):
    best, bp = -1, None
    for t in np.linspace(np.percentile(s, 50), np.percentile(s, 99.9), n):
        p = s > t; v = f1(*prf(y, p))
        if v > best:
            best, bp = v, p
    return bp


def score(preds_by_ent):
    """preds_by_ent: {entity: bool array} → 지표 dict"""
    typ = anomaly_types()
    pt = [0, 0, 0]; touch = half = n = pred = hit = 0
    tl = {b[2]: [0, 0] for b in BINS}; tt = {k: [0, 0] for k in TYPES}
    for e, p in preds_by_ent.items():
        _, _, y = load(e)
        c = prf(y, p); pt = [pt[i] + c[i] for i in range(3)]
        for a, b in segs(y):
            n += 1; t = bool(p[a:b].any()); touch += t; half += int(p[a:b].mean() >= .5)
            nm = [x for lo, hi, x in BINS if lo <= b - a <= hi][0]; tl[nm][1] += 1; tl[nm][0] += t
            k = typ.get((e, int(a)))
            if k in tt:
                tt[k][1] += 1; tt[k][0] += t
        for a, b in segs(p.astype(int)):
            pred += 1; hit += int(y[a:b].any())
    tp, fp, fn = pt; R = touch / n; P = hit / max(pred, 1)
    return dict(P=tp / max(tp + fp, 1), R=tp / max(tp + fn, 1), F1=f1(tp, fp, fn), evR=R, evR50=half / n, evP=P,
                evF1=2 * P * R / (P + R) if P + R else 0, cand=pred, by_len=tl, by_type=tt, n=n)


def show(name, m, header=False):
    if header:
        print(f"{'':30s}{'틱P':>6s}{'틱R':>6s}{'틱F1':>6s} | {'evR걸침':>7s}{'evR50%':>7s}{'evP':>6s}{'evF1':>6s}{'후보':>6s} | 걸침 spike/16~224/225+ | 종류별 걸침 metric/temporal/m-t")
    bl = " / ".join(f"{m['by_len'][b[2]][0]}/{m['by_len'][b[2]][1]}" for b in BINS)
    bt = " / ".join(f"{m['by_type'][k][0]}/{m['by_type'][k][1]}" for k in TYPES)
    print(f"{name:30s}{m['P']:6.3f}{m['R']:6.3f}{m['F1']:6.3f} | {m['evR']:7.3f}{m['evR50']:7.3f}{m['evP']:6.3f}{m['evF1']:6.3f}{m['cand']:6d} | {bl} | {bt}")
