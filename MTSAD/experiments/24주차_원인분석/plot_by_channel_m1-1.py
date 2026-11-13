"""
machine-1-1: 채널 하나당 페이지 하나로, 그 채널이 8개 GT세그먼트 각각에서
어떻게 생겼는지 나란히 보여줌 (기존 plot_all_entities_channels.py는 반대로
세그먼트 하나당 페이지, 그 안에 38채널이었음 -- 이번엔 축을 뒤집음).

새 연산 없음 -- results_stage1_individual_nonoverlap/machine-1-1_per_channel.npz 재사용.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
SMD_DIR = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD")
ENTITY = "machine-1-1"


def pt_f1(labels, pred):
    tp = int(np.sum((pred == 1) & (labels == 1))); fp = int(np.sum((pred == 1) & (labels == 0))); fn = int(np.sum((pred == 0) & (labels == 1)))
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) > 0 else 0.0)


def best_prf(scores, labels, n=300):
    lo, hi = np.percentile(scores, 50), np.percentile(scores, 99.9)
    best = (0, 0, 0, lo)
    for thr in np.linspace(lo, hi, n):
        p, r, f1 = pt_f1(labels, (scores > thr).astype(int))
        if f1 > best[2]:
            best = (p, r, f1, thr)
    return best


def norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


def gt_channels_for(gt_intervals, cs, ce):
    for s, e, chs in gt_intervals:
        if s <= ce and cs <= e:
            return chs
    return []


test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
n_ch = test.shape[1]

d = np.load(BASE / "results_stage1_individual_nonoverlap" / f"{ENTITY}_per_channel.npz")
per_ch_scores, combined, labels = d["scores"], d["combined"], d["labels"]

p, r, f1, thr = best_prf(combined, labels)
pred_ours = (combined > thr).astype(int)

gt_intervals = []
for line in (SMD_DIR / "interpretation_label" / f"{ENTITY}.txt").read_text().splitlines():
    rng, chs = line.split(":")
    s, e = (int(x) for x in rng.split("-"))
    gt_intervals.append((s, e, [int(c) - 1 for c in chs.split(",")]))

diff = np.diff(np.concatenate([[0], labels, [0]]))
gt_segments = list(zip(np.where(diff == 1)[0].tolist(), np.where(diff == -1)[0].tolist()))
n_seg = len(gt_segments)

n_cols = 4
n_rows = int(np.ceil(n_seg / n_cols)) + 1  # +1은 채널 z-score 요약용 첫 줄

out_path = BASE / f"{ENTITY}_by_channel.pdf"
with PdfPages(out_path) as pdf:
    for ch in range(n_ch):
        arr = per_ch_scores[ch]
        bg = arr[labels == 0]
        sigma = bg.std()

        fig = plt.figure(figsize=(15, 3.2 * (n_rows)))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.3)

        # 첫 줄: 이 채널의 세그먼트별 z-score 막대그래프 요약
        ax_summary = fig.add_subplot(gs[0, :])
        zs, gt_flags, seg_labels = [], [], []
        for cs, ce in gt_segments:
            gt_chs = gt_channels_for(gt_intervals, cs, ce)
            if sigma < 1e-6:
                zs.append(0)
            else:
                zs.append((arr[cs:ce].mean() - bg.mean()) / sigma)
            gt_flags.append(ch in gt_chs)
            seg_labels.append(f"[{cs},{ce})")
        colors = ["red" if g else "gray" for g in gt_flags]
        ax_summary.bar(range(n_seg), zs, color=colors)
        ax_summary.axhline(2.326, color="black", linestyle="--", linewidth=1)
        ax_summary.set_xticks(range(n_seg))
        ax_summary.set_xticklabels(seg_labels, fontsize=7, rotation=20)
        ax_summary.set_title(f"ch{ch} -- 세그먼트별 z-score (빨강=이 세그먼트의 GT원인채널, 회색=아님)"
                              + ("  [상수채널]" if sigma < 1e-6 else ""), fontsize=10)

        for i, (cs, ce) in enumerate(gt_segments):
            gt_chs = gt_channels_for(gt_intervals, cs, ce)
            seg_len = ce - cs
            pad = max(100, seg_len // 2)
            s = max(0, cs - pad)
            e = min(len(test), ce + pad)

            row = 1 + i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            v = test[s:e, ch]
            ax.plot(range(s, e), norm(v), color="steelblue", lw=0.8)
            ax.axvspan(cs, ce, color="red", alpha=0.2)
            is_gt = ch in gt_chs
            r_ours = float(pred_ours[cs:ce].mean())
            ax.set_title(f"{'[GT] ' if is_gt else ''}[{cs},{ce}) len={seg_len}\nrecall={r_ours:.2f}",
                         fontsize=8, color=("red" if is_gt else "black"))
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{ENTITY} -- ch{ch}, 세그먼트 {n_seg}개 전체", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved: {out_path.name} ({n_ch}페이지)")
