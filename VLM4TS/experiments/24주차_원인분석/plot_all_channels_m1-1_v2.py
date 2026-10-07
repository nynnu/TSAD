"""
v2: 채널마다 실제 점수(z-score, 우리 방법 기준)를 계산해서 제목에 표시하고,
GT 원인채널 여부를 색으로 구분, 페이지 하단에 자동 생성 해석 텍스트를 추가.

TimeRCD는 공개 API가 채널별 점수를 안 주고(내부적으로 채널 평균을 낸 뒤
합쳐진 값만 반환) 전체 결합 점수만 있어서, 채널별 TimeRCD 숫자는 낼 수 없음
-- 이 사실 자체를 페이지에 명시.

새 DINOv2 연산 없음 -- 기존 캐시 재사용. build_channel_groups만 재호출(빠름, 상관계수 계산뿐).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/experiments/20주차실험")
import colab_multivariate_v2 as cm  # noqa: E402

SMD_DIR = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/mv_data/SMD")
CACHE = Path("/Users/na-yeonkim/Desktop/졸업프로젝트/실험_나영,나연/nynnu/TSAD/VLM4TS/results/VLM4TS_experiments_results_mv_v2/cache/SMD/machine-1-1")
TRCD_PATH = Path("/private/tmp/claude-501/-Users-na-yeonkim-Desktop----------------/0c4c5a17-093e-4909-be05-bf8667b30091/scratchpad/timercd_scores/machine-1-1.npz")
ENTITY = "machine-1-1"
INTRA_CHANNELS = [0, 1, 2, 3, 19, 20, 21, 22, 34, 35]

segments_meta = [
    (15849, 16395, 0.306, 0.082),
    (16963, 17517, 0.798, 0.477),
    (18071, 18528, 0.632, 0.044),
    (19367, 20088, 0.612, 0.146),
    (20786, 21195, 0.589, 0.528),
    (24679, 24682, 1.000, 1.000),
    (26114, 26116, 1.000, 1.000),
    (27554, 27556, 1.000, 1.000),
]

gt_intervals = []
for line in (SMD_DIR / "interpretation_label" / f"{ENTITY}.txt").read_text().splitlines():
    rng, chs = line.split(":")
    s, e = (int(x) for x in rng.split("-"))
    gt_intervals.append((s, e, [int(c) - 1 for c in chs.split(",")]))


def gt_channels_for(cs, ce):
    """interpretation_label 구간 경계가 test_label 구간 경계와 살짝 다를 수 있어(SMD 원본
    특성) 겹치는 걸 찾음, exact match 대신."""
    for s, e, chs in gt_intervals:
        if s <= ce and cs <= e:
            return chs
    return []

test = np.loadtxt(SMD_DIR / "test" / f"{ENTITY}.txt", delimiter=",")
train = np.loadtxt(SMD_DIR / "train" / f"{ENTITY}.txt", delimiter=",")
labels = np.loadtxt(SMD_DIR / "test_label" / f"{ENTITY}.txt", delimiter=",").astype(int)
n_ch = test.shape[1]

groups = cm.build_channel_groups(train)
ch_to_group = {}
for gi, g in enumerate(groups):
    for c in g:
        ch_to_group[c] = gi

ours_combined = np.stack([np.load(CACHE / f"overlay_g{g}_scores.npz")["ml_sum"] for g in range(8)]).max(axis=0)
trcd = np.load(TRCD_PATH)["scores"]

score_cache = {}
def channel_score_array(c):
    if c in score_cache:
        return score_cache[c]
    if c in INTRA_CHANNELS:
        arr = np.load(CACHE / f"ch{c}_scores.npz")["ml_sum"]
        src = "individual"
    elif c in ch_to_group:
        arr = np.load(CACHE / f"overlay_g{ch_to_group[c]}_scores.npz")["ml_sum"]
        src = f"group{ch_to_group[c]}(공유)"
    else:
        arr, src = None, "점수없음"
    score_cache[c] = (arr, src)
    return arr, src


def norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


with PdfPages("machine-1-1_all_channels_v2.pdf") as pdf:
    for cs, ce, r_ours, r_trcd in segments_meta:
        gt_chs = gt_channels_for(cs, ce)
        seg_len = ce - cs
        pad = max(100, seg_len // 2)
        s = max(0, cs - pad)
        e = min(len(test), ce + pad)

        fig = plt.figure(figsize=(16, 22))
        gs = fig.add_gridspec(11, 5, height_ratios=[1.3] + [1] * 9 + [1.2], hspace=0.9, wspace=0.3)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(range(s, e), norm(ours_combined[s:e]), color="black", lw=1, label="우리 결합점수(8그룹 max)")
        ax0.plot(range(s, e), norm(trcd[s:e]), color="orange", lw=1, label="TimeRCD 결합점수", alpha=0.85)
        ax0.axvspan(cs, ce, color="red", alpha=0.2)
        ax0.legend(fontsize=8, loc="upper right")
        ax0.set_title(f"GT[{cs},{ce}) len={seg_len}  recall_ours={r_ours:.2f}  recall_trcd={r_trcd:.2f}   "
                      f"GT원인채널(1-idx 라벨 기준, 0-idx로 변환)={sorted(c for c in gt_chs)}", fontsize=10)

        z_records = []
        for ch in range(n_ch):
            row = 1 + ch // 5
            col = ch % 5
            ax = fig.add_subplot(gs[row, col])
            v = test[s:e, ch]
            ax.plot(range(s, e), norm(v), color="steelblue", lw=0.7)
            ax.axvspan(cs, ce, color="red", alpha=0.2)

            arr, src = channel_score_array(ch)
            if arr is not None:
                bg = arr[labels == 0]
                mu, sigma = bg.mean(), bg.std() + 1e-8
                z = (arr[cs:ce].mean() - mu) / sigma
                z_records.append((ch, z, ch in gt_chs))
                z_txt = f"z={z:.2f}"
            else:
                z_txt = "z=N/A"

            is_gt = ch in gt_chs
            title_color = "red" if is_gt else "black"
            mark = "[GT] " if is_gt else ""
            ax.set_title(f"{mark}ch{ch} {z_txt}\n({src})", fontsize=7, color=title_color)
            ax.set_xticks([])
            ax.set_yticks([])

        # ---- 자동 해석 텍스트 ----
        gt_z = sorted([(c, z) for c, z, isgt in z_records if isgt], key=lambda x: -x[1])
        non_gt_z = [(c, z) for c, z, isgt in z_records if not isgt]
        gt_high = [c for c, z in gt_z if z > 1.5]
        gt_low = [c for c, z in gt_z if z <= 1.5]
        non_gt_high = sorted([c for c, z in non_gt_z if z > 1.5])

        lines = [
            f"[해석] recall_ours={r_ours:.2f}, recall_trcd={r_trcd:.2f}.",
            f"GT 원인채널 중 우리 점수(z>1.5)로 뚜렷이 잡히는 것: {gt_high if gt_high else '없음'}",
            f"GT 원인채널인데 우리 점수로도 약하게만 보이는 것(z<=1.5): {gt_low if gt_low else '없음'}",
            f"GT가 아닌데도 z>1.5로 비슷하게 튀는 채널: {non_gt_high if non_gt_high else '없음'} "
            f"(총 {len(non_gt_high)}개 -- 시스템 전반에 퍼진 증상일 가능성)",
            "TimeRCD는 공개 API가 채널별 점수를 반환하지 않음(모델 내부에서 채널 평균을 내고 하나로 합쳐서 출력) "
            "-- 그래서 채널별 TimeRCD 숫자는 위에 못 넣었고, 맨 위 결합점수(주황선)로만 비교 가능.",
        ]
        ax_text = fig.add_subplot(gs[10, :])
        ax_text.axis("off")
        ax_text.text(0.01, 0.9, "\n".join(lines), fontsize=9, va="top", wrap=True,
                     transform=ax_text.transAxes)

        fig.suptitle(f"{ENTITY}  GT[{cs},{ce})  -- 38채널 전부 + z-score", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)

print("Saved: machine-1-1_all_channels_v2.pdf")
