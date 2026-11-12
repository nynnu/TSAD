"""
채널 묶음 분석 — SMD 28대 전체 (나연, 2026-11-06) — API 호출 없음

  A) GT 채널 묶음이 반복되는지: machine 안 GT event끼리 채널 조합(Jaccard)이 같은 크기 무작위 조합보다 비슷한지
  B) train에 없던 채널 묶음이 test에 나오면 이상인지
     창 = 10틱씩 겹치지 않게. 창 안에서 채널이 "움직임" = 창 안 최대 |한 틱 변화| > 그 채널 train의 99.5% 분위수 (그리고 > 0.01)
     창의 묶음 = 움직인 채널 집합. train 창들의 묶음을 모아 두고, test 창 묶음마다
       새로움 = 1 − (train 묶음 중 가장 비슷한 것과의 Jaccard), 완전히 새 묶음 = train에 똑같은 묶음 없음
     움직인 채널이 1개 이상인 test 창만 보고, GT 창(창 안에 GT 틱 있음) 비율 · AUC를 냄. 비교: 움직인 채널 수
  C) GT 채널 묶음끼리는 train에서도 평소에 같이 움직이는지
     train 상관(|Spearman|) — 값 자체(level) · 한 틱 변화(diff) 두 가지. 상수 채널은 뺌
     GT event마다 묶음 안 짝 평균 |상관| vs 같은 개수 무작위 묶음(200번) → 무작위보다 높은 비율(백분위)

  실행:  python channel_groups.py [SMD|ASD]   → results/channel_groups.json (ASD는 channel_groups_ASD.json), 표 출력
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, load, gt_channel_ranges

W, Q, MIN_D = 10, 99.5, 0.01
DS = sys.argv[1] if len(sys.argv) > 1 else "SMD"
DATA = BASE / "mv_data" / DS
OUT = Path(__file__).resolve().parent / "results" / ("channel_groups.json" if DS == "SMD" else f"channel_groups_{DS}.json")
rng = np.random.default_rng(0)


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def active(X, thr):
    n = len(X) // W
    d = np.abs(np.diff(X, axis=0, prepend=X[:1]))[:n * W].reshape(n, W, -1).max(1)
    return d > thr


def part_a(gtr, C):
    sets = [c for _, _, c in gtr]
    if len(sets) < 2:
        return None
    real = np.mean([jac(a, b) for a, b in combinations(sets, 2)])
    rand = []
    for _ in range(200):
        rs = [set(rng.choice(C, len(s), replace=False)) for s in sets]
        rand.append(np.mean([jac(a, b) for a, b in combinations(rs, 2)]))
    same = sum(any(s == t for j, t in enumerate(sets) if j != i) for i, s in enumerate(sets))
    return dict(events=len(sets), unique=len({frozenset(s) for s in sets}), exact_repeat=same,
                jaccard=real, jaccard_random=float(np.mean(rand)))


def part_b(tr, te, y):
    thr = np.maximum(np.percentile(np.abs(np.diff(tr, axis=0)), Q, axis=0), MIN_D)
    A_tr, A_te = active(tr, thr), active(te, thr)
    U = np.unique(A_tr[A_tr.any(1)], axis=0)
    yw = y[:len(A_te) * W].reshape(-1, W).any(1)
    m = A_te.any(1); V = A_te[m]; g = yw[m]
    inter = V.astype(int) @ U.T.astype(int); union = V.sum(1)[:, None] + U.sum(1)[None] - inter
    nov = 1 - (inter / union).max(1)
    exact = nov > 0
    auc = lambda s: float(roc_auc_score(g, s)) if 0 < g.sum() < len(g) else None
    return dict(train_sets=len(U), test_windows=int(len(yw)), active_windows=int(m.sum()),
                gt_rate_all=float(yw.mean()), gt_rate_quiet=float(yw[~m].mean()) if (~m).any() else None,
                gt_rate_active=float(g.mean()),
                novel_windows=int(exact.sum()), gt_rate_novel=float(g[exact].mean()) if exact.any() else None,
                gt_rate_seen=float(g[~exact].mean()) if (~exact).any() else None,
                gt_recall_novel=float(exact[g].mean()) if g.any() else None,
                auc_novelty=auc(nov), auc_size=auc(V.sum(1)))


def part_c(tr, gtr):
    ok = np.flatnonzero(np.ptp(tr, 0) > 1e-6); out = {}
    for nm, X in (("level", tr[:, ok]), ("diff", np.diff(tr[:, ok], axis=0))):
        R = np.abs(np.nan_to_num(spearmanr(X).statistic)); np.fill_diagonal(R, np.nan)
        mean = lambda idx: np.nanmean(R[np.ix_(idx, idx)])
        pos = {c: i for i, c in enumerate(ok)}; ev = []
        for _, _, cs in gtr:
            idx = [pos[c] for c in cs if c in pos]
            if len(idx) < 2 or len(idx) == len(ok):
                continue
            real = mean(idx); rand = [mean(rng.choice(len(ok), len(idx), replace=False)) for _ in range(200)]
            ev.append(dict(n=len(idx), corr=float(real), corr_random=float(np.mean(rand)), pct=float(np.mean(np.array(rand) < real))))
        out[nm] = ev
    return out


def main():
    res = {}
    for f in sorted((DATA / "test").glob("*.txt")):
        e = f.stem; tr, te, y = load(DATA, e)
        gtr = gt_channel_ranges(DATA, e)
        res[e] = dict(A=part_a(gtr, te.shape[1]), B=part_b(tr, te, y), C=part_c(tr, gtr))
        a, b = res[e]["A"], res[e]["B"]
        sa = f"A 묶음 {a['unique']}/{a['events']}종 반복 {a['exact_repeat']} J {a['jaccard']:.2f} (무작위 {a['jaccard_random']:.2f})" if a else "A event 1개"
        print(f"{e}: {sa} | "
              f"B 새 묶음 창 {b['novel_windows']}/{b['active_windows']} GT율 새 {b['gt_rate_novel'] or 0:.2f} 본 {b['gt_rate_seen'] or 0:.2f} "
              f"조용 {b['gt_rate_quiet'] or 0:.2f} | AUC 새로움 {b['auc_novelty'] or 0:.2f} 채널수 {b['auc_size'] or 0:.2f}", flush=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
