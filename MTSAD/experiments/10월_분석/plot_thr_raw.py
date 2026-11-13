"""
machine별 실제 값 (전체 test): spike 쪽만 (보정 spike 상위 DT% AND 갑작스러움 상위 ST%, 탈주기 없음)
  그리는 채널 = GT 채널 ∪ 한 번이라도 켜진 채널. 눈금 = train min~max (DINOv2 보는 것과 같음, 0~1로 자름)
  채널 색: 파랑 = GT 이상 안에서 맞춤(GT 채널이고 그 GT 안에서 켜짐) / 초록 = GT 채널인데 한 번도 못 맞춤 / 빨강 = GT 채널 아님(켜진 건 전부 오탐)
  점: 주황 = GT 구간 안에서 켜진 틱, 회색 = GT 밖에서 켜진 틱(오탐). 빨강 배경 = GT, ▼ = 놓친 GT
  이름 앞 "GT" = 그 machine 어딘가에서 GT 채널, 오른쪽 숫자 = 맞춘 GT 수 / 오탐 덩어리 수
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_dsz_allch import segs, SC, SMD
from eval_dt_fix import calibrate
from eval_spike_detector import suddenness
from exp_patch_groups import interp_channels

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
DT, ST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5, float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = HERE / "stage12_check" / f"thr_dt{DT:g}_st{ST:g}".replace(".", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for e in sorted(p.stem for p in SC.glob("*.npz")):
        y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
        tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
        S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); T = min(len(y), S.shape[1]); y, S, te = y[:T], S[:, :T], te[:T]
        G = np.nan_to_num(suddenness(te, tr), nan=0.0); fin = np.isfinite(S)
        m = (np.where(fin, S, -np.inf) > np.percentile(S[fin], 100 - DT)) & (G > np.percentile(G, 100 - ST))
        p = m.any(0); gt = segs(y); gch = [set(interp_channels(e, a, b)) for a, b in gt]
        allg = set().union(*gch) if gch else set()
        chs = sorted(allg | set(np.where(m.any(1))[0]))
        info = {}
        for c in chs:
            hitn = sum(c in g and m[c, a:b].any() for (a, b), g in zip(gt, gch))
            fpn = sum(1 for u, v in segs(m[c].astype(int)) if not y[u:v].any())
            col = "royalblue" if hitn else ("green" if c in allg else "red")
            info[c] = (hitn, fpn, col)
        fig, A = plt.subplots(figsize=(24, 0.55 * len(chs) + 1.5))
        xs = np.arange(T)
        for k, c in enumerate(chs):
            g0, g1 = tr[:, c].min(), tr[:, c].max()
            x = np.clip((te[:, c] - g0) / (g1 - g0 + 1e-8), 0, 1) * .8 + k
            hitn, fpn, col = info[c]
            A.plot(xs, x, color=col, lw=.4)
            on = np.where(m[c])[0]
            A.scatter(on, x[on], s=8, color=np.where(y[on] == 1, "orange", "gray"), zorder=3)
            A.text(-T * 0.003, k + .4, f"{'GT ' if c in allg else ''}ch{c}", ha="right", va="center", fontsize=8, color=col)
            A.text(T * 1.003, k + .4, f"맞춤 {hitn} / 오탐 {fpn}", va="center", fontsize=8, color=col)
        for (a, b), h in zip(gt, [p[a:b].any() for a, b in gt]):
            A.axvspan(a, max(b, a + 10), color="red", alpha=.12, lw=0)
            if not h:
                A.plot((a + b) / 2, len(chs) + .1, "kv", ms=7, clip_on=False)
        A.set_xlim(0, T); A.set_ylim(-.2, len(chs) + .3); A.set_yticks([])
        nfp = sum(1 for u, v in segs(p.astype(int)) if not y[u:v].any())
        A.set_title(f"{e} | spike 상위 {DT}% AND 갑작 상위 {ST}% | 걸친 GT {sum(p[a:b].any() for a, b in gt)}/{len(gt)}, 후보 {len(segs(p.astype(int)))} (오탐 {nfp})\n"
                    "파랑 = 맞춘 GT 채널, 초록 = 못 맞춘 GT 채널, 빨강 = GT 아닌 채널 | 주황 점 = GT 안에서 켜짐, 회색 점 = 오탐 | 빨강 배경 = GT, ▼ = 놓친 GT", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"raw_{e}.png", dpi=55); plt.close(fig)
        print(e, len(chs), "채널", flush=True)


if __name__ == "__main__":
    main()
