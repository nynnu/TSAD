"""
가정 12 이어서: 위치(경계)가 왜 틀리나 — 걸친 GT 이상마다
  덮음 = GT 안 예측 비율, 넘침 = 겹친 후보들의 GT 밖 틱 수 / GT 길이, 조각 = 겹친 후보 수
  IoU < 0.5 를 원인별로: 덜 덮음(덮음<0.5) / 넘침(넘침>1) / 둘 다 / 그 외(쪼개짐 등)
  그림: 길이 구간별 IoU 낮은 예 (원래 값의 GT 채널 + 탈주기 점수 + 예측)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, PH, SMD
from exp_patch_groups import interp_channels
from stage12_check import best_thr, search_thr
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE / "stage12_check"


def main():
    ev = pd.read_csv(OUT / "events.csv"); ev = ev[ev.touched].copy()
    rows, preds = [], {}
    for e, g in ev.groupby("entity"):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        D = np.load(PH / f"{e}_phase_deseason.npz")["scores"]
        mD = D > best_thr(D.max(0), y)
        mS = search_thr(mD.any(0), calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)),
                        np.nan_to_num(suddenness(te, tr), nan=0.0), y)
        pD, pS = mD.any(0), mS.any(0); pred = pD | pS; preds[e] = (te, y, D.max(0), pD, pS)
        bl = segs(pred.astype(int))
        for _, r in g.iterrows():
            a, b = int(r.a), int(r.b); ov = [(u, v) for u, v in bl if u < b and v > a]
            out = sum(v - u for u, v in ov) - pred[a:b].sum()
            rows.append(dict(cover=pred[a:b].mean(), over=out / (b - a)))
    ev[["cover", "over"]] = pd.DataFrame(rows, index=ev.index)
    low = ev[ev.iou < 0.5].copy()
    low["cause"] = np.select([(low.cover < .5) & (low.over > 1), low.cover < .5, low.over > 1], ["둘 다", "덜 덮음", "넘침"], "그 외(쪼개짐 등)")
    print(f"걸친 이상 {len(ev)}개 중 IoU<0.5: {len(low)}개\n")
    print(pd.crosstab(low.bin, low.cause, margins=True).reindex(["≤15", "16~224", "225+", "All"]))
    print("\n길이 구간별 중앙값 (걸친 이상 전체)")
    print(ev.groupby("bin")[["cover", "over", "n_blobs", "iou"]].median().reindex(["≤15", "16~224", "225+"]).round(2))
    low.to_csv(OUT / "boundary_low_iou.csv", index=False)

    rs = np.random.default_rng(0)
    fig, ax = plt.subplots(3, 4, figsize=(24, 12))
    for row, bn in zip(ax, ["≤15", "16~224", "225+"]):
        sub = low[low.bin == bn]
        for A, (_, r) in zip(row, sub.iloc[rs.permutation(len(sub))[:4]].iterrows()):
            te, y, dm, pD, pS = preds[r.entity]; a, b = int(r.a), int(r.b); L = b - a
            s0, s1 = max(0, a - max(150, L)), min(len(y), b + max(150, L)); xs = np.arange(s0, s1)
            chs = interp_channels(r.entity, a, b) or [int(np.abs(te[a:b] - te[s0:a].mean(0)).max(0).argmax())]
            for c in chs[:4]:
                x = te[s0:s1, c]; A.plot(xs, (x - x.min()) / (np.ptp(x) + 1e-9), lw=.7)
            A.plot(xs, dm[s0:s1] / (dm[s0:s1].max() + 1e-9) - 1.1, color="purple", lw=.8)
            A.axvspan(a, b, color="r", alpha=.15)
            A.fill_between(xs, -1.35, -1.25, where=pD[s0:s1], color="purple", step="mid")
            A.fill_between(xs, -1.5, -1.4, where=pS[s0:s1], color="orange", step="mid")
            A.set_ylim(-1.6, 1.1); A.set_yticks([])
            A.set_title(f"{r.entity} [{a},{b}) {L}틱 | {r.cause}\nIoU {r.iou:.2f} · 덮음 {r.cover:.2f} · 넘침 {r.over:.1f} · 조각 {int(r.n_blobs)}", fontsize=9)
    fig.suptitle("위치가 틀린 예 (빨강 = GT, 위 = GT 채널 값, 보라 선 = 탈주기 점수, 아래 띠: 보라 = 탈주기 예측, 주황 = spike 예측)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "boundary_low_iou.png", dpi=65)


if __name__ == "__main__":
    main()
