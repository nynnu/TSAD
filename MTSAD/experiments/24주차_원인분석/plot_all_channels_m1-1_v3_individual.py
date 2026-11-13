"""
v3: 새 방식(개별 채널 + 224틱 비중첩)의 점수로 machine-1-1 채널별 해석 PDF를 다시 만든다.
그룹 공유가 없으므로 모든 채널이 "개별" 점수를 가짐 -- group7류 오귀속 문제 자체가 사라짐.

새 DINOv2 연산 없음 -- results_stage1_individual_nonoverlap/machine-1-1_per_channel.npz 재사용.
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
NEW_SCORES = Path(__file__).resolve().parent / "results_stage1_individual_nonoverlap" / "machine-1-1_per_channel.npz"
TRCD_PATH = Path("/private/tmp/claude-501/-Users-na-yeonkim-Desktop----------------/0c4c5a17-093e-4909-be05-bf8667b30091/scratchpad/timercd_scores/machine-1-1.npz")
ENTITY = "machine-1-1"
BEST_THR = 5.4692  # combined score 기준 최적 임계값 (Precision=0.2149, Recall=0.7506, F1=0.3342)

gt_intervals = []
for line in (SMD_DIR / "interpretation_label" / f"{ENTITY}.txt").read_text().splitlines():
    rng, chs = line.split(":")
    s, e = (int(x) for x in rng.split("-"))
    gt_intervals.append((s, e, [int(c) - 1 for c in chs.split(",")]))


def gt_channels_for(cs, ce):
    for s, e, chs in gt_intervals:
        if s <= ce and cs <= e:
            return chs
    return []


test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
labels = np.loadtxt(SMD_DIR / "test_label" / f"{ENTITY}.txt", delimiter=",").astype(int)
n_ch = test.shape[1]

d = np.load(NEW_SCORES)
per_ch_scores = d["scores"]          # (38, T) -- 채널별 개별 점수 (z-score, 이미 정규화됨)
combined = d["combined"]             # (T,) -- 38채널 중 max
trcd = np.load(TRCD_PATH)["scores"]

diff = np.diff(np.concatenate([[0], labels, [0]]))
gt_starts = np.where(diff == 1)[0]
gt_ends = np.where(diff == -1)[0]
gt_segments = list(zip(gt_starts.tolist(), gt_ends.tolist()))

pred_ours = (combined > BEST_THR).astype(int)


def norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


with PdfPages("machine-1-1_all_channels_v3_individual.pdf") as pdf:
    for cs, ce in gt_segments:
        gt_chs = gt_channels_for(cs, ce)
        seg_len = ce - cs
        pad = max(100, seg_len // 2)
        s = max(0, cs - pad)
        e = min(len(test), ce + pad)

        r_ours = float(pred_ours[cs:ce].mean())
        r_trcd = None  # TimeRCD 자체 최적임계값은 별도 계산 필요하지만 결합점수 곡선으로 시각 비교만 함

        fig = plt.figure(figsize=(16, 22))
        gs = fig.add_gridspec(11, 5, height_ratios=[1.3] + [1] * 9 + [1.2], hspace=0.9, wspace=0.3)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(range(s, e), norm(combined[s:e]), color="black", lw=1, label="Stage1(개별+비중첩) 결합점수(38채널 max)")
        ax0.plot(range(s, e), norm(trcd[s:e]), color="orange", lw=1, label="TimeRCD 결합점수", alpha=0.85)
        ax0.axvspan(cs, ce, color="red", alpha=0.2)
        ax0.legend(fontsize=8, loc="upper right")
        ax0.set_title(f"GT[{cs},{ce}) len={seg_len}  recall_ours(개별방식, 임계값={BEST_THR:.2f})={r_ours:.2f}   "
                      f"GT원인채널(0-idx)={sorted(gt_chs)}", fontsize=10)

        z_records = []
        for ch in range(n_ch):
            row = 1 + ch // 5
            col = ch % 5
            ax = fig.add_subplot(gs[row, col])
            v = test[s:e, ch]
            ax.plot(range(s, e), norm(v), color="steelblue", lw=0.7)
            ax.axvspan(cs, ce, color="red", alpha=0.2)

            arr = per_ch_scores[ch]
            bg = arr[labels == 0]
            sigma = bg.std()
            if sigma < 1e-6:
                z_txt = "z=N/A(상수채널)"
                z = None
            else:
                z = (arr[cs:ce].mean() - bg.mean()) / sigma
                z_txt = f"z={z:.2f}"
                z_records.append((ch, z, ch in gt_chs))

            is_gt = ch in gt_chs
            title_color = "red" if is_gt else "black"
            mark = "[GT] " if is_gt else ""
            ax.set_title(f"{mark}ch{ch} {z_txt}\n(개별)", fontsize=7, color=title_color)
            ax.set_xticks([])
            ax.set_yticks([])

        gt_z = sorted([(c, z) for c, z, isgt in z_records if isgt], key=lambda x: -x[1])
        non_gt_z = [(c, z) for c, z, isgt in z_records if not isgt]
        gt_high = [c for c, z in gt_z if z > 2.326]
        gt_low = [c for c, z in gt_z if z <= 2.326]
        non_gt_high = sorted([c for c, z in non_gt_z if z > 2.326])

        lines = [
            f"[해석] (그룹 공유 없음 -- 전부 개별 채널 점수) recall_ours={r_ours:.2f} (임계값={BEST_THR:.2f} 고정)",
            f"GT 원인채널 중 z>2.326(공식 High 기준)로 뚜렷이 잡히는 것: {gt_high if gt_high else '없음'}",
            f"GT 원인채널인데 z<=2.326로 약하게만 보이는 것: {gt_low if gt_low else '없음'}",
            f"GT가 아닌데도 z>2.326로 튀는 채널(오탐 후보): {non_gt_high if non_gt_high else '없음'} (총 {len(non_gt_high)}개)",
            "이제 모든 채널이 자기 자신의 점수만 가짐 -- 예전처럼 죽은 채널이 옆 채널 점수를 물려받는 일 없음.",
        ]
        ax_text = fig.add_subplot(gs[10, :])
        ax_text.axis("off")
        ax_text.text(0.01, 0.9, "\n".join(lines), fontsize=9, va="top", wrap=True, transform=ax_text.transAxes)

        fig.suptitle(f"{ENTITY}  GT[{cs},{ce})  -- v3: 개별채널+비중첩 방식", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)

print("Saved: machine-1-1_all_channels_v3_individual.pdf")
