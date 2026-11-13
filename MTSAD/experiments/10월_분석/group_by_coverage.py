"""
가정 10 → 새 흐름 ①: 현재 방법(Dinov2_deseson_col_z, GT 없이 38채널)이 아예 못 잡는 이상 모으기
GT 이상 327개를 최종 예측(탈주기 OR 2차)이 덮은 비율로 나눔. 두 버전(DINOv2, 숫자) 중 더 많이 덮은 쪽 기준.
  0틱 그룹: 두 버전 모두 한 틱도 표시 안 함 (핵심 분석 대상)
  일부 그룹: 1틱 이상 ~ 50% 미만
  잡음 그룹: 50% 이상
결과: coverage_groups/cases.json + 0틱 그룹 그림
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import best_pred, segs, combine, PH, SC, SMD
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
RES = HERE / "coverage_groups"
RES.mkdir(exist_ok=True)
GROUPS = ["0틱", "일부(1틱~50%미만)", "잡음(50%이상)"]
BINS = [(1, 15, "spike≤15"), (16, 224, "16~224"), (225, 10 ** 9, "225+")]


def main():
    rows, cache = [], {}
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=",")
        te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        Z = (np.abs(te - tr.mean(0)) / np.maximum(tr.std(0), 1e-3)).T
        d = np.load(SC / f"{e}.npz")
        p1 = best_pred(np.load(PH / f"{e}_phase_deseason.npz")["combined"], y)
        pD = combine(p1, d["D"].astype(float), Z, y)[0]
        pB = combine(p1, d["B"].astype(float), Z, y)[0]
        cache[e] = (tr, te, y, pD, pB)
        for a, b in segs(y):
            cD, cB, c1 = float(pD[a:b].mean()), float(pB[a:b].mean()), float(p1[a:b].mean())
            cov = max(cD, cB)
            g = GROUPS[0] if cov == 0 else (GROUPS[1] if cov < .5 else GROUPS[2])
            chs = interp_channels(e, a, b)
            zc = int(Z[:, a:b].max(1).argmax())
            rows.append(dict(e=e, a=int(a), b=int(b), L=int(b - a), group=g, cov_dino=cD, cov_num=cB, cov_deseason=c1,
                             zmax_ch=zc, zmax=float(Z[zc, a:b].max()), label_chs=chs))
    json.dump(rows, open(RES / "cases.json", "w"), ensure_ascii=False, indent=1)

    print(f"GT 이상 {len(rows)}개 (최종 예측 = 탈주기 OR 2차, 두 버전 중 더 많이 덮은 쪽 기준)\n")
    print(f"{'':20s}{'개수':>6s}" + "".join(f"{b[2]:>10s}" for b in BINS) + f"{'길이 중앙값':>10s}")
    for g in GROUPS:
        s = [r for r in rows if r["group"] == g]
        L = np.array([r["L"] for r in s])
        print(f"{g:20s}{len(s):6d}" + "".join(f"{((L >= lo) & (L <= hi)).sum():10d}" for lo, hi, _ in BINS) + f"{int(np.median(L)):10d}")

    # 0틱 그룹 그림: z가 가장 크게 반응한 채널 + (있으면) 라벨 채널 하나 더
    z0 = [r for r in rows if r["group"] == GROUPS[0]]
    for f0 in range(0, len(z0), 12):
        fig, ax = plt.subplots(4, 3, figsize=(18, 13))
        for A, r in zip(ax.flat, z0[f0:f0 + 12]):
            tr, te, y, pD, pB = cache[r["e"]]
            pad = max(200, r["L"]); s, t = max(0, r["a"] - pad), min(len(te), r["b"] + pad)
            c = r["zmax_ch"]
            A.plot(range(s, t), te[s:t, c], color="k", lw=.8, label=f"ch{c} (z 최대)")
            lc = [x for x in r["label_chs"] if x != c]
            if lc:
                c2 = lc[0]
                x2 = te[s:t, c2]; x1 = te[s:t, c]
                x2s = (x2 - x2.min()) / (x2.max() - x2.min() + 1e-9) * (x1.max() - x1.min() + 1e-9) + x1.min()
                A.plot(range(s, t), x2s, color="gray", lw=.6, alpha=.7, label=f"ch{c2} (라벨, 높이 맞춤)")
            dd = np.diff(np.r_[0, y[s:t], 0])
            for x0, x1_ in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
                A.axvspan(s + x0, s + x1_, color="r", alpha=.12)
            A.axvspan(r["a"], r["b"], color="r", alpha=.3)
            A.legend(fontsize=7, loc="upper right")
            A.set_title(f"{r['e']} [{r['a']},{r['b']}) {r['L']}틱  z최대 {r['zmax']:.0f}  라벨채널 {len(r['label_chs']) or '없음'}개", fontsize=8.5)
        for A in ax.flat[len(z0[f0:f0 + 12]):]:
            A.axis("off")
        fig.suptitle(f"0틱 그룹 ({f0 + 1}~{min(f0 + 12, len(z0))} / {len(z0)}) — 진한 빨강 = 이 이상, 연한 빨강 = 다른 GT", fontsize=13)
        fig.tight_layout(); fig.savefig(RES / f"0틱_{f0 // 12 + 1:02d}.png", dpi=70); plt.close(fig)
    print("saved to", RES)


if __name__ == "__main__":
    main()
