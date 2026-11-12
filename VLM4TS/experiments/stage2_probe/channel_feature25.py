"""
이상 채널 특징 후보 25개 중 숫자로 계산 가능한 것 — 같은 event 안 "튄 채널"끼리 GT vs GT 아님 (나연, 2026-11-07) — API 호출 없음

  대상  = GT event마다 구간(GT ± 5틱) 안에서 앞 60틱 대비 10배 넘게 튄 채널 (GT 정보는 GT/아님 나누는 데만 씀)
  씨앗  = 그 구간에서 가장 크게 튄 채널
  특징 번호는 대화의 25개 표 번호. 10·22번만 다른 GT 라벨을 씀(방법으로는 반칙, 확인용)
  AUC   = 값이 클수록 GT라고 볼 때. 0.5 = 못 가름, 0.5보다 작으면 반대 방향
  실행:  python channel_feature25.py machine-2-1   → 표 출력, results/channel_feature25_{machine}.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_view import nearest
from plot_raw_view import BASE, load, gt_channel_ranges

PRE, DAY, PAD = 60, 1440, 5
HERE = Path(__file__).resolve().parent
NAMES = {1: "같은 순간 정점(차이, 작을수록)", 2: "test 같이 움직임", 3: "크기(log)", 4: "일찍 움직임(시작 틱, 작을수록)",
         5: "train 평소 같이 움직임", 6: "train 범위 밖", 7: "train에 없는 모양(거리)", 8: "test 안 드문 모양(거리)",
         9: "train에선 평평", 10: "같은 장비 다른 event 라벨 빈도*", 11: "같은 시각 train과 차이", 12: "관계 깨짐(train−test 상관)",
         14: "평소 덜 튐(train 튐 비율, 작을수록)", 15: "위로 튐", 16: "지속(틱)", 17: "제일 먼저 움직임(순위, 작을수록)",
         18: "spike 몰림(구간 안 튐 횟수/평소)", 19: "가장 비슷한 train 사건과 차이", 20: "직선 구간(데이터 끊김)",
         22: "다른 장비 라벨 빈도*", 23: "수준 변화(뒤−앞)", 25: "회복 안 됨(뒤−앞, 구간 뒤 60틱)"}
LOWER = {1, 4, 14, 17}                                     # 작을수록 GT라는 가설 → 부호 뒤집어 AUC


def mad(v, axis=0):
    return np.maximum(1.4826 * np.median(np.abs(v - np.median(v, axis)), axis), 0.01)


def zwin(X, s, t):
    pre = X[s - PRE:s]; return (X[s:t] - np.median(pre, 0)) / mad(pre)


def main():
    e = sys.argv[1]; data = BASE / "mv_data" / "SMD"; tr, te, y = load(data, e); gtr = gt_channel_ranges(data, e); C = te.shape[1]
    sc = mad(tr); Rtr = np.abs(np.nan_to_num(spearmanr(np.diff(tr, axis=0)).statistic))
    blk = np.array([np.abs(zwin(tr, p, p + 10)).max(0) > 10 for p in range(PRE, len(tr) - 10, 10)])
    spiky = blk.mean(0); flat = np.ptp(tr, 0) < 0.05
    others = [m for m in sorted(f.stem for f in (data / "test").glob("machine-*.txt")) if m != e]
    xfreq = np.zeros(C)
    for m in others:
        for _, _, cs in gt_channel_ranges(data, m):
            xfreq[list(cs)] += 1
    rows = []
    for n, (s, t, G) in enumerate(gtr, 1):
        a, b = max(PRE, s - PAD), min(len(te) - PRE, t + PAD); L = b - a
        Z = zwin(te, a, b); A = np.abs(Z); zc = A.max(0)
        moved = [c for c in range(C) if zc[c] > 10]
        if len(moved) < 2:
            continue
        top = int(zc.argmax()); tpk = int(A[:, top].argmax())
        on = {c: int(np.argmax(A[:, c] > 5)) for c in moved}; order = {c: r for r, c in enumerate(sorted(moved, key=lambda c: on[c]), 1)}
        ph = (len(tr) + a) % DAY; td = ph + DAY * ((len(tr) - L - ph) // DAY)
        lf = np.zeros(C)
        for j, (_, _, H) in enumerate(gtr):
            if j != n - 1:
                lf[list(H)] += 1
        cs = sorted(G & set(moved)) or [top]
        Wj = np.lib.stride_tricks.sliding_window_view(tr[:, moved], L, axis=0)              # 튄 채널 전부로 가장 비슷한 train 사건
        q = (te[a:b, moved] - np.median(te[a:b, moved], 0)) / sc[moved]
        dj = (((Wj - np.median(Wj, 2, keepdims=True)) / sc[moved][None, :, None] - q.T[None]) ** 2).mean((1, 2)); tj = int(np.argmin(dj))
        post = te[b:b + PRE]
        for c in moved:
            v = te[a:b, c]
            rt = spearmanr(v, te[a:b, top]).statistic if c != top else 1.0
            rt = 0.0 if np.isnan(rt) else abs(rt)
            t2 = nearest(tr[:, c], v, sc[c]); w2 = tr[t2:t2 + L, c]
            t3 = nearest(te[:, c], v, sc[c], ban=(a - L, b)); w3 = te[t3:t3 + L, c]
            d = lambda w: float(np.sqrt(((((w - np.median(w)) - (v - np.median(v))) / sc[c]) ** 2).mean()))
            dd = np.abs(np.diff(v, 2)); lin = max((len(r) for r in np.split(dd, np.flatnonzero(dd > 1e-6)) if len(r)), default=0)
            base = np.median(te[a - PRE:a, c]); m0 = mad(te[a - PRE:a, c])
            f = {1: abs(int(A[:, c].argmax()) - tpk), 2: rt, 3: np.log10(zc[c]), 4: on[c], 5: Rtr[top, c] if c != top else 1.0,
                 6: float(v.max() > 1 or v.min() < 0), 7: d(w2), 8: d(w3), 9: float(flat[c]), 10: lf[c],
                 11: float(np.abs(v - tr[td:td + L, c]).mean() / sc[c]), 12: (Rtr[top, c] - rt) if c != top else 0.0,
                 14: spiky[c], 15: float(Z[int(A[:, c].argmax()), c] > 0), 16: int((A[:, c] > 5).sum()), 17: order[c],
                 18: float((A[:, c] > 10).sum() / max(spiky[c] * L, 0.1)), 19: float(np.abs(v - tr[tj:tj + L, c]).mean() / sc[c]),
                 20: lin, 22: xfreq[c], 23: abs(np.median(post[:, c]) - base) / m0 if len(post) else 0.0,
                 25: abs(np.median(post[-20:, c]) - base) / m0 if len(post) else 0.0}
            rows.append(dict(ev=n, ch=c, gt=c in G, **{str(k): float(x) for k, x in f.items()}))
    gt = np.array([r["gt"] for r in rows])
    print(f"{e} · 튄 채널(10배+) {len(rows)}개 중 GT {gt.sum()} / GT 아님 {(~gt).sum()} · event {len({r['ev'] for r in rows})}개")
    res = {}
    for k, nm in NAMES.items():
        x = np.array([r[str(k)] for r in rows]); x = -x if k in LOWER else x
        auc = roc_auc_score(gt, x) if len(set(x)) > 1 else 0.5; res[k] = auc
        print(f"  {k:2d} {nm:32s} AUC {auc:.2f}  (GT 중앙 {np.median(np.abs(x[gt])):.3g} / 아님 {np.median(np.abs(x[~gt])):.3g})")
    (HERE / "results" / f"channel_feature25_{e}.json").write_text(json.dumps(dict(auc=res, rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
