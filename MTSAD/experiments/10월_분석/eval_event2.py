"""
공통 채점 (2026-10-17~): event F1 두 가지
  P = 예측 덩어리 중 GT와 겹친 비율 (두 F1 공통)
  R(걸침) = GT 이상 중 1틱이라도 걸친 비율 → event F1
  R(50%) = GT 이상 중 50% 이상 덮은 비율 → event F1(50%)
"""
import numpy as np

from eval_dsz_allch import segs


def ev2_counts(y, p):
    """(걸친 GT, 50% 덮은 GT, GT 수, 예측 덩어리 수, GT와 겹친 덩어리 수)"""
    gt = segs(y); pc = np.r_[0, np.cumsum(p)]; yc = np.r_[0, np.cumsum(y)]
    cov = [(pc[b] - pc[a]) / (b - a) for a, b in gt]
    d = np.diff(np.r_[0, p.astype(int), 0]); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return np.array([sum(c > 0 for c in cov), sum(c >= 0.5 for c in cov), len(gt), len(s), int(((yc[e] - yc[s]) > 0).sum())], float)


def ev2_scores(c):
    t, h, n, m, k = c; P = k / m if m else 0.0; R, R5 = t / n, h / n
    f = lambda r: 2 * P * r / (P + r) if P + r else 0.0
    return dict(P=P, R=R, F=f(R), R50=R5, F50=f(R5), n=int(m))


def ev2_f1(y, p, half=False):
    s = ev2_scores(ev2_counts(y, p)); return s["F50"] if half else s["F"]
