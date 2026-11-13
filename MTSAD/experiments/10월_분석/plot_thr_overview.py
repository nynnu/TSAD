"""
machine별 전체 시계열 한 줄씩: GT / 잡은 후보 (spike 쪽만: 보정 spike 상위 0.5% AND 갑작스러움 상위 1%, 탈주기 없음)
  빨강 막대 = GT (위 줄), 주황 막대 = 후보 (아래 줄), 검정 ▼ = 놓친 GT, 회색 막대 = 오탐 후보
  제목: 걸친 GT / 전체 GT, 후보 수 (그중 오탐)
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

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
DT, ST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5, float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = HERE / "stage12_check" / f"thr_dt{DT:g}_st{ST:g}".replace(".", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ents = sorted(p.stem for p in SC.glob("*.npz"))
    for page in range(2):
        chunk = ents[page * 14:(page + 1) * 14]
        fig, ax = plt.subplots(len(chunk), 1, figsize=(22, 1.6 * len(chunk)))
        for A, e in zip(ax, chunk):
            y = np.loadtxt(SMD / "test_label" / f"{e}.txt", delimiter=",").astype(int)
            tr = np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=",")
            S = calibrate(np.load(HERE / "spike_scores" / f"{e}.npz")["scores"].astype(float)); T = min(len(y), S.shape[1]); y, S = y[:T], S[:, :T]
            G = np.nan_to_num(suddenness(te[:T], tr), nan=0.0); fin = np.isfinite(S)
            p = ((np.where(fin, S, -np.inf) > np.percentile(S[fin], 100 - DT)) & (G > np.percentile(G, 100 - ST))).any(0)
            gt, bl = segs(y), segs(p.astype(int))
            hit = [p[a:b].any() for a, b in gt]
            fp = [(u, v) for u, v in bl if not y[u:v].any()]
            for a, b in gt:
                A.axvspan(a, max(b, a + 15), ymin=0.55, ymax=0.95, color="red", alpha=.8, lw=0)
            for u, v in bl:
                A.axvspan(u, max(v, u + 15), ymin=0.05, ymax=0.45, color="gray" if (u, v) in fp else "orange", alpha=.8, lw=0)
            for (a, b), h in zip(gt, hit):
                if not h:
                    A.plot((a + b) / 2, 1.08, "kv", ms=6, clip_on=False)
            A.set_xlim(0, T); A.set_ylim(0, 1); A.set_yticks([0.25, 0.75]); A.set_yticklabels(["후보", "GT"], fontsize=8)
            A.set_title(f"{e}  |  걸친 GT {sum(hit)}/{len(gt)}  |  후보 {len(bl)} (오탐 {len(fp)})", fontsize=10, loc="left")
        fig.suptitle(f"spike 쪽만: 보정 spike 상위 {DT}% AND 갑작스러움 상위 {ST}% (탈주기 없음)   빨강 = GT, 주황 = GT와 겹친 후보, 회색 = 오탐 후보, ▼ = 놓친 GT", fontsize=12)
        fig.tight_layout(); fig.savefig(OUT / f"overview_{page + 1}.png", dpi=60)


if __name__ == "__main__":
    main()
