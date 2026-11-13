"""
6개 entity(machine-1-1~1-5, machine-3-2) 전부에 대해, GT 세그먼트마다
38채널 + z-score + GT표시 + TimeRCD 비교 페이지를 만들어 entity당 PDF 1개씩 저장.

새 연산 없음 -- results_stage1_individual_nonoverlap/*.npz, timercd_scores/*.npz 재사용.
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

ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-3-2"]


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


def make_pdf(entity):
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    n_ch = test.shape[1]

    d = np.load(BASE / "results_stage1_individual_nonoverlap" / f"{entity}_per_channel.npz")
    per_ch_scores, combined, labels = d["scores"], d["combined"], d["labels"]
    trcd = np.load(BASE / "timercd_scores" / f"{entity}.npz")["scores"]

    p, r, f1, thr = best_prf(combined, labels)
    pred_ours = (combined > thr).astype(int)

    gt_intervals = []
    for line in (SMD_DIR / "interpretation_label" / f"{entity}.txt").read_text().splitlines():
        rng, chs = line.split(":")
        s, e = (int(x) for x in rng.split("-"))
        gt_intervals.append((s, e, [int(c) - 1 for c in chs.split(",")]))

    diff = np.diff(np.concatenate([[0], labels, [0]]))
    gt_segments = list(zip(np.where(diff == 1)[0].tolist(), np.where(diff == -1)[0].tolist()))

    out_path = BASE / f"{entity}_all_channels.pdf"
    with PdfPages(out_path) as pdf:
        for cs, ce in gt_segments:
            gt_chs = gt_channels_for(gt_intervals, cs, ce)
            seg_len = ce - cs
            pad = max(100, seg_len // 2)
            s = max(0, cs - pad)
            e = min(len(test), ce + pad)
            r_ours = float(pred_ours[cs:ce].mean())

            fig = plt.figure(figsize=(16, 22))
            gs = fig.add_gridspec(11, 5, height_ratios=[1.3] + [1] * 9 + [1.2], hspace=0.9, wspace=0.3)

            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(range(s, e), norm(combined[s:e]), color="black", lw=1, label="Stage1(개별+비중첩) 결합점수(38채널 max)")
            ax0.plot(range(s, e), norm(trcd[s:e]), color="orange", lw=1, label="TimeRCD 결합점수", alpha=0.85)
            ax0.axvspan(cs, ce, color="red", alpha=0.2)
            ax0.legend(fontsize=8, loc="upper right")
            ax0.set_title(f"{entity}  GT[{cs},{ce}) len={seg_len}  recall_ours(임계값={thr:.2f})={r_ours:.2f}   "
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
                    z_txt = "z=N/A(상수)"
                else:
                    z = (arr[cs:ce].mean() - bg.mean()) / sigma
                    z_txt = f"z={z:.2f}"
                    z_records.append((ch, z, ch in gt_chs))

                is_gt = ch in gt_chs
                mark = "[GT] " if is_gt else ""
                ax.set_title(f"{mark}ch{ch} {z_txt}", fontsize=7, color=("red" if is_gt else "black"))
                ax.set_xticks([])
                ax.set_yticks([])

            gt_z = [(c, z) for c, z, isgt in z_records if isgt]
            non_gt_z = [(c, z) for c, z, isgt in z_records if not isgt]
            gt_high = sorted([c for c, z in gt_z if z > 2.326])
            gt_low = sorted([c for c, z in gt_z if z <= 2.326])
            non_gt_high = sorted([c for c, z in non_gt_z if z > 2.326])

            lines = [
                f"[해석] recall_ours={r_ours:.2f} (entity 최적임계값={thr:.2f} 고정, entity 전체 P={p:.3f} R={r:.3f} F1={f1:.3f})",
                f"GT 원인채널 중 z>2.326로 뚜렷이 잡히는 것: {gt_high if gt_high else '없음'}",
                f"GT 원인채널인데 z<=2.326로 약하게만 보이는 것: {gt_low if gt_low else '없음'}",
                f"GT가 아닌데도 z>2.326로 튀는 채널(오탐 후보): {non_gt_high if non_gt_high else '없음'} (총 {len(non_gt_high)}개)",
            ]
            ax_text = fig.add_subplot(gs[10, :])
            ax_text.axis("off")
            ax_text.text(0.01, 0.9, "\n".join(lines), fontsize=9, va="top", wrap=True, transform=ax_text.transAxes)

            fig.suptitle(f"{entity}  GT[{cs},{ce})", fontsize=13)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[{entity}] {len(gt_segments)}페이지 -> {out_path.name}")


if __name__ == "__main__":
    for ent in ENTITIES:
        make_pdf(ent)
