"""
가정 9-2: DINOv2 버전 vs 숫자 버전 — 탈주기가 놓친 GT 이상 212개를 사례로 비교
  분류: DINOv2만 잡음 / 숫자만 잡음 / 둘 다 잡음 / 둘 다 놓침  (구간 50% 이상 맞히면 "잡음")
  그림: 그 이상을 잡은 채널(둘 다 놓쳤으면 z가 가장 크게 반응한 채널), GT = 빨강, 아래 띠 = DINOv2(파랑) / 숫자(초록) 예측
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import prf, f1, best_pred, segs, PH, SC, SMD, DQ, ZQ

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "compare_dino_vs_num"
RES.mkdir(exist_ok=True)
CATS = ["DINOv2만", "숫자만", "둘 다", "둘 다 놓침"]


def combine_thr(p1, S, Z, y):
    """eval_dsz_allch.combine과 같은 탐색, 채널별 발화 여부까지 반환"""
    fin = np.isfinite(S); Sv = np.where(fin, S, -np.inf)
    dts = np.unique(np.percentile(S[fin], DQ)); zts = np.unique(np.percentile(Z, ZQ))
    best, bp, fire = f1(*prf(y, p1)), p1, np.zeros_like(Z, bool)
    for dt in dts:
        Db = Sv > dt
        for zt in zts:
            fb = Db & (Z > zt); p = p1 | fb.any(0); v = f1(*prf(y, p))
            if v > best:
                best, bp, fire = v, p, fb
    return bp, fire


def main():
    rows, cache = [], {}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        d = np.load(SC / f"{e}.npz")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        pD, fD = combine_thr(p1, d["D"].astype(float), Z, y)
        pB, fB = combine_thr(p1, d["B"].astype(float), Z, y)
        cache[e] = (te, y, pD, pB)
        for a, b in segs(y):
            if p1[a:b].mean() >= .5:
                continue
            cD, cB = pD[a:b].mean() >= .5, pB[a:b].mean() >= .5
            cat = CATS[0] if cD and not cB else CATS[1] if cB and not cD else CATS[2] if cD else CATS[3]
            fire = fD if cD else fB if cB else None
            ch = int(fire[:, a:b].sum(1).argmax()) if fire is not None else int(Z[:, a:b].max(1).argmax())
            sud = float(np.abs(te[a:b, ch] - np.median(te[max(0, a - 100):a, ch])).max()
                        / max(1.4826 * np.median(np.abs(te[max(0, a - 100):a, ch] - np.median(te[max(0, a - 100):a, ch]))),
                              0.01 * (tr[:, ch].max() - tr[:, ch].min()), 1e-6)) if a > 10 else 0.0
            in_rng = bool(te[a:b, ch].max() <= tr[:, ch].max() and te[a:b, ch].min() >= tr[:, ch].min())
            rows.append(dict(e=e, a=int(a), b=int(b), L=int(b - a), cat=cat, ch=ch, sud=sud, in_range=in_rng))
    json.dump(rows, open(RES / "cases.json", "w"), ensure_ascii=False, indent=1)

    print(f"탈주기가 놓친 GT 이상 {len(rows)}개")
    print(f"{'':12s}{'개수':>5s}{'spike≤15':>10s}{'16~224':>8s}{'225+':>6s}{'길이 중앙값':>10s}{'갑작스러움 중앙값':>14s}{'값이 train 범위 안':>16s}")
    for c in CATS:
        s = [r for r in rows if r["cat"] == c]
        if not s:
            print(f"{c:12s}{0:5d}"); continue
        L = np.array([r["L"] for r in s])
        print(f"{c:12s}{len(s):5d}{(L <= 15).sum():10d}{((L > 15) & (L <= 224)).sum():8d}{(L > 224).sum():6d}"
              f"{int(np.median(L)):10d}{np.median([r['sud'] for r in s]):14.1f}{np.mean([r['in_range'] for r in s]) * 100:15.0f}%")

    for c in CATS:
        s = [r for r in rows if r["cat"] == c]
        pick = [s[i] for i in np.random.default_rng(0).choice(len(s), min(12, len(s)), replace=False)] if s else []
        for f0 in range(0, len(pick), 12):
            fig, ax = plt.subplots(4, 3, figsize=(18, 13))
            for A, r in zip(ax.flat, pick[f0:f0 + 12]):
                te, y, pD, pB = cache[r["e"]]
                pad = max(150, r["L"]); s0, t0 = max(0, r["a"] - pad), min(len(te), r["b"] + pad)
                x = te[s0:t0, r["ch"]]; lo, hi = x.min(), x.max(); rg = hi - lo + 1e-9
                A.plot(range(s0, t0), x, color="k", lw=.8)
                A.axvspan(r["a"], r["b"], color="r", alpha=.25)
                A.fill_between(range(s0, t0), lo - .10 * rg, lo - .05 * rg, where=pD[s0:t0], color="royalblue", step="mid")
                A.fill_between(range(s0, t0), lo - .17 * rg, lo - .12 * rg, where=pB[s0:t0], color="green", step="mid")
                A.set_ylim(lo - .2 * rg, hi + .05 * rg)
                A.set_title(f"{r['e']} ch{r['ch']} {r['L']}틱 갑작 {r['sud']:.0f}배 {'범위안' if r['in_range'] else '범위밖'}", fontsize=9)
            for A in ax.flat[len(pick[f0:f0 + 12]):]:
                A.axis("off")
            fig.suptitle(f"[{c}] 탈주기가 놓친 이상 — 빨강 = GT, 아래 띠: 파랑 = DINOv2 버전 예측, 초록 = 숫자 버전 예측", fontsize=13)
            fig.tight_layout(); fig.savefig(RES / f"{c.replace(' ', '')}.png", dpi=72); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
