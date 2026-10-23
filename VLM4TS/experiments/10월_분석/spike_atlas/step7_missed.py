"""
7단계 Q4: 지금 최선 모델(탈주기 OR 보정 spike 점수 AND 갑작스러움)이 한 틱도 못 걸친 GT spike(≤15틱)
  GT spike 이벤트마다 대표 spike(spike 표에서 is_gt_spike, 높이가 가장 큰 것)의 특징을 붙여 '잡은 것'과 비교
  4~6단계 규칙을 통과하는지도 표시: 조용한 곳 규칙(높이≥0.3 & 갑작≥20) / 흔들리는 곳 규칙(상대높이≥1.5) / 뚜렷한 동시 채널≥1
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval_dsz_allch import best_pred, segs, PH, SC, SMD  # noqa: E402
from eval_dt_fix import calibrate  # noqa: E402
from eval_spike_detector import suddenness  # noqa: E402
from eval_fix_variants import search  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    sp = pd.read_csv(HERE / "step6_test_cochannel.csv")
    st5 = pd.read_csv(HERE / "step5_test_features.csv")[["entity", "ch", "t", "rel_height"]]
    sp = sp.merge(st5, on=["entity", "ch", "t"], how="left")
    rows = []
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        pred = search(p1, calibrate(np.load(HERE.parent / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                      np.nan_to_num(suddenness(te, tr), nan=0.0), y)
        for a, b in segs(y):
            if b - a > 15:
                continue
            g = sp[(sp.entity == e) & (sp.is_gt_spike) & (sp.start < b) & (sp.end > a)]
            r = g.loc[g.height.fillna(0).idxmax()] if len(g) else None
            rows.append(dict(entity=e, a=int(a), b=int(b), caught=bool(pred[a:b].any()),
                             ch=None if r is None else int(r.ch), t=None if r is None else int(r.t),
                             group=None if r is None else r.group, height=None if r is None else r.height,
                             sudden=None if r is None else r.sudden, density=None if r is None else r.density,
                             rel=None if r is None else r.rel_height, co_clean=None if r is None else r.co_clean))
    d = pd.DataFrame(rows)
    d["rule_quiet"] = (d.group == "1.조용한 곳") & (d.height >= 0.3) & (d.sudden >= 20)
    d["rule_rel"] = (d.group != "1.조용한 곳") & (d.rel.replace(np.inf, 1e3) >= 1.5)
    d["rule_co"] = d.co_clean >= 1
    d.to_csv(HERE / "step7_gt_spikes.csv", index=False)
    m, c = d[~d.caught], d[d.caught]
    print(f"GT spike(≤15틱) {len(d)}개: 최선 모델이 걸침 {len(c)}개 / 한 틱도 못 걸침 {len(m)}개\n")
    print("그룹별 (못 걸침 / 전체):", {g: f"{(m.group == g).sum()}/{(d.group == g).sum()}" for g in sorted(d.group.dropna().unique())})
    print(f"\n{'특징 (중앙값)':22s}{'못 걸침':>10s}{'걸침':>10s}")
    for k in ["height", "sudden", "density", "co_clean"]:
        print(f"{k:22s}{m[k].median():10.2f}{c[k].median():10.2f}")
    print(f"\n못 걸친 {len(m)}개 중 4~6단계 규칙 통과: 조용한 곳 규칙 {int(m.rule_quiet.sum())}개, 흔들리는 곳 규칙 {int(m.rule_rel.sum())}개, "
          f"둘 중 하나 {int((m.rule_quiet | m.rule_rel).sum())}개, 뚜렷한 동시 채널≥1 {int(m.rule_co.sum())}개")

    cache = {}
    for p0 in range(0, len(m), 16):
        chunk = m.iloc[p0:p0 + 16]; fig, ax = plt.subplots(4, 4, figsize=(22, 13))
        for A, (_, r) in zip(ax.flat, chunk.iterrows()):
            if r.entity not in cache:
                cache[r.entity] = (np.loadtxt(SMD / "test" / f"{r.entity}.txt", delimiter=","),
                                   np.loadtxt(SMD / "test_label" / f"{r.entity}.txt", delimiter=",").astype(int))
            X, y = cache[r.entity]; ch = int(r.ch); t = int(r.t)
            s0, t0 = max(0, t - 200), min(len(X), t + 200); x = X[s0:t0, ch]
            A.plot(range(s0, t0), x, color="k", lw=.7)
            dd = np.diff(np.r_[0, y[s0:t0], 0])
            for u, v in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
                A.axvspan(s0 + u, s0 + v, color="r", alpha=.25)
            A.scatter([t], [X[t, ch]], color="red", s=50, zorder=3, edgecolors="k")
            rl = "없음" if not np.isfinite(r.rel) else f"{r.rel:.1f}"
            A.set_title(f"{r.entity} ch{ch} [{r.a},{r.b}) | {str(r.group)[2:]}\n높이 {r.height:.2f} · 갑작 {r.sudden:.0f} · 상대높이 {rl} · 동시채널 {int(r.co_clean)}", fontsize=8)
        for A in ax.flat[len(chunk):]:
            A.axis("off")
        fig.suptitle(f"7단계: 최선 모델이 한 틱도 못 걸친 GT spike ({p0 + 1}~{p0 + len(chunk)} / {len(m)}, 빨강 점 = 대표 spike)", fontsize=12)
        fig.tight_layout(); fig.savefig(HERE / f"step7_못잡은spike_{p0 // 16 + 1}.png", dpi=65); plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
