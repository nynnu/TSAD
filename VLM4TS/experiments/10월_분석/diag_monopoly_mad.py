"""
독차지의 뿌리가 MAD 작은 채널인가 — machine마다 보정 spike 상위 0.5% 칸을 채널별로 몇 % 차지하는지 vs 그 채널 MAD
  출력: machine별 (평평 채널 수 / 전체, 평평 채널이 차지한 비율), 전체 상관, 상위 2채널 중 평평 비율. 그림: 산점도
"""
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import SC
from eval_dt_fix import calibrate

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
P = 14


def main():
    L = [f"{'machine':13s} | 평평(MAD<1) 채널 / 점수 있는 채널 | 상위 0.5% 중 평평 채널 몫 | 상위 2채널 (MAD, 몫)"]
    xs, ys, top2flat, top2n = [], [], 0, 0
    for e in sorted([p.stem for p in SC.glob("*.npz")], key=lambda s: [int(x) for x in s.split("-")[1:]]):
        raw = np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float); S = calibrate(raw)
        mad = np.full(raw.shape[0], np.nan)
        for c in range(raw.shape[0]):
            v = raw[c, ::P]; v = v[np.isfinite(v)]
            if len(v) >= 10:
                mad[c] = 1.4826 * np.median(np.abs(v - np.median(v)))
        fin = np.isfinite(S); top = np.where(fin, S, -np.inf) > np.percentile(S[fin], 99.5)
        share = top.sum(1) / top.sum(); ok = np.isfinite(mad); flat = ok & (mad < 1.0)
        xs += list(mad[ok]); ys += list(share[ok])
        o = np.argsort(-share)[:2]; top2flat += int(flat[o].sum()); top2n += 2
        L.append(f"{e:13s} | {flat.sum():2d} / {ok.sum():2d} ({flat.sum() / ok.sum():.0%}) | {share[flat].sum():.0%} | "
                 + ", ".join(f"ch{c} (MAD {mad[c]:.2f}, {share[c]:.0%})" for c in o))
    xs, ys = np.array(xs), np.array(ys); r, pv = spearmanr(xs, ys)
    fl = xs < 1.0
    L += ["", f"전체 채널 {len(xs)}개: 평평 채널 {fl.sum()}개 ({fl.mean():.0%})가 상위 0.5% 자리의 평균 몫 {ys[fl].mean():.1%} vs 나머지 {ys[~fl].mean():.1%} (채널 하나당)",
          f"MAD와 차지 몫의 순위상관 {r:.2f} (p={pv:.1e})", f"machine마다 상위 2채널 {top2n}개 중 평평 채널 {top2flat}개 ({top2flat / top2n:.0%})"]
    txt = "\n".join(L); print(txt)
    (HERE / "coverage_groups" / "monopoly_vs_mad.txt").write_text(txt)
    fig, A = plt.subplots(figsize=(8, 5))
    A.scatter(xs, 100 * ys, s=10, c=np.where(fl, "red", "gray"))
    A.set_xlabel("채널 MAD (spike 원점수)"); A.set_ylabel("그 machine 상위 0.5% 칸 중 이 채널 몫 (%)")
    A.set_title(f"채널마다: MAD vs 상위 0.5% 독차지 (빨강 = MAD<1 평평 채널) | 순위상관 {r:.2f}")
    fig.tight_layout(); fig.savefig(HERE / "coverage_groups" / "monopoly_vs_mad.png", dpi=90)


if __name__ == "__main__":
    main()
