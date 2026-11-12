"""
Stage1 채널 점수 + 평소 같이 움직이는 채널로 넓히기 — SMD 28대 (나연, 2026-11-06) — API 호출 없음

  가설  = GT는 평소 같이 변하는 채널 그룹 단위로 라벨됨(channel_groups.py C) → 튄 채널과 train 상관이 높은 채널도 GT일 것
  점수  = Stage1: 구간 안 후보 채널의 spike 점수 / 구간 최댓값 (탈주기만 = 0.01, 후보 아님 = 0)
          넓힘:  채널 c 점수 = max_j |train 상관(c, j)| × Stage1(j)   (자기 자신 상관 = 1이라 Stage1 점수보다 작아지지 않음)
          상관은 diff(한 틱 변화) · level(값) 두 가지, 무작위 = 순위 기대값
  평가  = GT와 겹친 Stage1 구간마다 GT 채널 집합 G(겹친 GT 범위의 합집합), 38채널 전체에서
          HitRate@100% = 상위 |G|개 안 GT 비율, @150% = 상위 ⌈1.5|G|⌉개 안 GT 수 / |G|, AUC
  실행:  python group_expand.py   → results/group_expand.json, 표 출력
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_raw_view import BASE, STAGE1_OUT, load, gt_channel_ranges

DATA = BASE / "mv_data" / "SMD"
OUT = Path(__file__).resolve().parent / "results" / "group_expand.json"


def corr(X):
    R = np.abs(np.nan_to_num(spearmanr(X).statistic)); np.fill_diagonal(R, 1.0)
    return R


def evaluate(s, G):
    k = len(G); order = np.argsort(-s, kind="stable")
    h100 = len(set(order[:k]) & G) / k; h150 = len(set(order[:int(np.ceil(1.5 * k))]) & G) / k
    g = np.isin(np.arange(len(s)), list(G))
    return h100, h150, roc_auc_score(g, s) if 0 < g.sum() < len(g) else np.nan


def main():
    names = ["Stage1", "넓힘 diff", "넓힘 level"]; res = {}
    for f in sorted((DATA / "test").glob("machine-*.txt")):
        e = f.stem; tr, te, y = load(DATA, e); gtr = gt_channel_ranges(DATA, e); C = te.shape[1]
        R = {"넓힘 diff": corr(np.diff(tr, axis=0)), "넓힘 level": corr(tr)}
        rows = []
        for iv in json.loads((STAGE1_OUT / f"{e}.json").read_text())["intervals"]:
            a, b = iv["start"], iv["end"]
            G = set().union(*[c for s, t, c in gtr if s < b and t > a]) if y[a:b].any() else set()
            if not G:
                continue
            s1 = np.zeros(C); mx = max([c["spike_score"] or 0 for c in iv["channels"]] + [1e-9])
            for c in iv["channels"]:
                s1[c["channel"]] = (c["spike_score"] or 0) / mx or 0.01
            sc = {"Stage1": s1, **{n: (R[n] * s1[None]).max(1) for n in R}}
            rows.append(dict(start=a, end=b, n_gt=len(G), rand=len(G) / C, **{n: evaluate(sc[n], G) for n in names}))
        res[e] = rows
        m = lambda n, i: np.nanmean([r[n][i] for r in rows])
        print(f"{e} 구간 {len(rows)} | " + " | ".join(f"{n} {m(n, 0):.2f}/{m(n, 1):.2f}/{m(n, 2):.2f}" for n in names), flush=True)
    allr = [r for v in res.values() for r in v]
    print(f"\n전체 구간 {len(allr)} (HitRate@100% / @150% / AUC) · 무작위 @100% {np.mean([r['rand'] for r in allr]):.3f}")
    for n in names:
        print(f"  {n:10s} " + " / ".join(f"{np.nanmean([r[n][i] for r in allr]):.3f}" for i in range(3))
              + f"   (장비 평균 @100% {np.mean([np.mean([r[n][0] for r in v]) for v in res.values() if v]):.3f})")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
