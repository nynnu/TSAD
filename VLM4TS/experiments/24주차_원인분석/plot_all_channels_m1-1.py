"""
machine-1-1의 GT 세그먼트 8개 각각에 대해, 38채널 전부(+우리 점수, TimeRCD 점수)를
그려서 PDF 한 장(세그먼트당 한 페이지)으로 만든다. 새 연산 없음 -- 기존 캐시 재사용.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SMD_DIR = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD")
CACHE = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/results/VLM4TS_experiments_results_mv_v2/cache/SMD/machine-1-1")
TRCD_PATH = Path("/private/tmp/claude-501/-Users-na-yeonkim-Desktop----------------/0c4c5a17-093e-4909-be05-bf8667b30091/scratchpad/timercd_scores/machine-1-1.npz")
ENTITY = "machine-1-1"

segments = [
    (15849, 16395, 0.306, 0.082),
    (16963, 17517, 0.798, 0.477),
    (18071, 18528, 0.632, 0.044),
    (19367, 20088, 0.612, 0.146),
    (20786, 21195, 0.589, 0.528),
    (24679, 24682, 1.000, 1.000),
    (26114, 26116, 1.000, 1.000),
    (27554, 27556, 1.000, 1.000),
]

test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
n_ch = test.shape[1]

ours = np.stack([np.load(CACHE / f"overlay_g{g}_scores.npz")["ml_sum"] for g in range(8)]).max(axis=0)
trcd = np.load(TRCD_PATH)["scores"]


def norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


with PdfPages("machine-1-1_all_channels.pdf") as pdf:
    for cs, ce, r_ours, r_trcd in segments:
        seg_len = ce - cs
        pad = max(100, seg_len // 2)
        s = max(0, cs - pad)
        e = min(len(test), ce + pad)

        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(10, 5, hspace=0.7, wspace=0.3)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(range(s, e), norm(ours[s:e]), color="black", lw=1, label="우리 점수")
        ax0.plot(range(s, e), norm(trcd[s:e]), color="orange", lw=1, label="TimeRCD 점수", alpha=0.8)
        ax0.axvspan(cs, ce, color="red", alpha=0.2)
        ax0.legend(fontsize=8, loc="upper right")
        ax0.set_title(f"GT[{cs},{ce}) len={seg_len}  recall_ours={r_ours:.2f}  recall_trcd={r_trcd:.2f}", fontsize=11)

        for ch in range(n_ch):
            row = 1 + ch // 5
            col = ch % 5
            ax = fig.add_subplot(gs[row, col])
            v = test[s:e, ch]
            ax.plot(range(s, e), norm(v), color="steelblue", lw=0.7)
            ax.axvspan(cs, ce, color="red", alpha=0.2)
            ax.set_title(f"ch{ch}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{ENTITY}  GT[{cs},{ce})  -- 38채널 전부", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)

print("Saved: machine-1-1_all_channels.pdf")
