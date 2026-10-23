"""
6단계 Q3: spike의 채널 간 관계
  동시 채널 수 = 같은 entity에서 이 spike ±WIN틱 안에 spike가 있는 '다른' 채널 수 (test)
  비교: 이상(GT ≤15틱) vs 정상 / 조용한 곳의 '생김새로 구분 안 되는' 정상 743개 vs 조용한 곳 이상
  그림: 이상 3개 / 정상(이상처럼 생긴) 3개를 38채널 쌓아 그리기
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
SMD = HERE.parents[2] / "mv_data" / "SMD"
WIN = 5


def main():
    df = pd.read_csv(HERE / "spikes_smd.csv")
    t = df[df.split == "test"].copy()
    co = np.zeros(len(t), int)
    t = t.reset_index(drop=True)
    for e, g in t.groupby("entity"):
        bych = {c: np.sort(gc.t.values) for c, gc in g.groupby("ch")}
        for i, r in g.iterrows():
            n = 0
            for c, ts in bych.items():
                if c == r.ch:
                    continue
                j = np.searchsorted(ts, r.t - WIN)
                n += int(j < len(ts) and ts[j] <= r.t + WIN)
            co[i] = n
    t["co"] = co
    t.to_csv(HERE / "step6_test_cochannel.csv", index=False)
    look = pd.read_csv(HERE / "step4_정상인데_이상처럼생긴것.csv")
    qf = pd.read_csv(HERE / "step4_quiet_features.csv")
    key = lambda d: set(zip(d.entity, d.ch, d.t))
    lk, qa = key(look), key(qf[(qf.abn == 1) & (qf.height2 >= 0.3) & (qf.sudden >= 20)])
    t["k"] = list(zip(t.entity, t.ch, t.t))

    def cmp(name, A, B):
        a, b = A.co.values, B.co.values
        auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
        print(f"{name:42s} 이상 {len(a):4d}개 중앙값 {np.median(a):4.0f} (0개 비율 {np.mean(a == 0) * 100:3.0f}%) | 정상 {len(b):6d}개 중앙값 {np.median(b):4.0f} (0개 {np.mean(b == 0) * 100:3.0f}%) | AUROC {auc:.3f}")

    ab = t[t.kind == "test 이상(GT ≤15틱)"]; nm = t[t.kind == "test 정상"]
    print(f"동시 채널 수 = ±{WIN}틱 안에 spike가 있는 다른 채널 수\n")
    cmp("전체: 이상 vs 정상", ab, nm)
    for g in sorted(t.group.unique()):
        cmp(f"  {g[2:]}", ab[ab.group == g], nm[nm.group == g])
    cmp("핵심: 조용한 곳 이상(규칙 통과) vs 이상처럼 생긴 정상 743", t[t.k.isin(qa)], t[t.k.isin(lk)])
    A, B = t[t.k.isin(qa)], t[t.k.isin(lk)]
    for th in [1, 2, 3, 5]:
        print(f"    동시 채널 ≥ {th}: 이상 {int((A.co >= th).sum())}/{len(A)} 남음, 정상 {int((B.co >= th).sum())}/{len(B)} 남음")

    # 그림: 38채널 쌓기
    rs = np.random.default_rng(0)
    picks = [("이상", A.iloc[i]) for i in rs.choice(len(A), 3, replace=False)] + [("정상(이상처럼 생김)", B.iloc[i]) for i in rs.choice(len(B), 3, replace=False)]
    fig, ax = plt.subplots(1, 6, figsize=(26, 12))
    cache = {}
    for A_, (lab, r) in zip(ax, picks):
        if r.entity not in cache:
            cache[r.entity] = (np.loadtxt(SMD / "train" / f"{r.entity}.txt", delimiter=","), np.loadtxt(SMD / "test" / f"{r.entity}.txt", delimiter=","))
        tr, te = cache[r.entity]; rng = np.maximum(tr.max(0) - tr.min(0), 1e-6)
        a, b = max(0, int(r.t) - 60), min(len(te), int(r.t) + 60)
        for c in range(te.shape[1]):
            x = te[a:b, c]; x = (x - x.min()) / (x.max() - x.min() + 1e-9) * .8
            spk = t[(t.entity == r.entity) & (t.ch == c) & (abs(t.t - r.t) <= WIN)]
            A_.plot(range(a, b), x + c, color="red" if c == r.ch else ("darkorange" if len(spk) else "k"), lw=1.2 if c == r.ch or len(spk) else .5)
        A_.axvline(r.t, color="gray", ls=":")
        A_.set_ylim(-.5, te.shape[1]); A_.set_yticks(range(0, te.shape[1], 5))
        A_.set_title(f"[{lab}]\n{r.entity} ch{int(r.ch)} t={int(r.t)}\n동시 채널 {int(r.co)}개", fontsize=10, color="darkred" if lab == "이상" else "darkgreen")
    fig.suptitle(f"6단계: 같은 시점 38채널 (빨강 = 그 spike 채널, 주황 = ±{WIN}틱 안에 같이 튄 채널, 채널마다 높이를 맞춰 그림)", fontsize=13)
    fig.tight_layout(); fig.savefig(HERE / "step6_채널쌓기.png", dpi=65)
    print("saved")


if __name__ == "__main__":
    main()
