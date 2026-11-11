import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT = Path(r"C:\Users\김나영\Desktop\ny\tsad\report\attachments-18")
OUT.mkdir(parents=True, exist_ok=True)

# --- Fig 1: 3-way baseline comparison (precision/recall/F1) ---
summary_3way = json.loads((BASE / "experiments" / "results_smd_3way_baseline" / "summary.json").read_text(encoding="utf-8"))


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


labels = ["VLM4TS 자체\n(38채널 전체)", "canMMLM naive\n(4그룹 분할)", "우리 설계\n(Step0 top-4)"]
keys = ["vlm4ts_own", "canmmlm_naive", "our_design"]
precisions = [summary_3way[k]["precision"] for k in keys]
recalls = [summary_3way[k]["recall"] for k in keys]
f1s = [f1(summary_3way[k]["precision"], summary_3way[k]["recall"]) for k in keys]

x = np.arange(len(labels))
w = 0.25
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
ax.bar(x - w, precisions, w, label="Precision", color="#4C72B0")
ax.bar(x, recalls, w, label="Recall", color="#DD8452")
ax.bar(x + w, f1s, w, label="F1", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.75)
ax.set_ylabel("Score")
ax.set_title("3-way 채널 선택 방식 비교 (실제 SMD 9구간)")
ax.legend()
for i, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
    ax.text(i - w, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
    ax.text(i, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
    ax.text(i + w, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_3way_comparison.png")
plt.close(fig)

# --- Fig 2: Step0 개선 실험들 recall@k 비교 ---
step0_results = {
    "원본\n(단일참조)": 0.4762,
    "창 폭\n적응화": 0.4854,
    "전역 스케일\n정규화": 0.4894,
    "CLS 토큰": 0.5040,
    "z-score\n단독": 0.5086,
    "z-score\n융합": 0.5430,
    "다중참조\n앙상블": 0.5456,
    "배경\n마스킹": 0.5410,
}
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
names = list(step0_results.keys())
vals = list(step0_results.values())
colors = ["#888888"] + ["#C44E52" if v <= 0.4762 + 0.005 else "#55A868" for v in vals[1:]]
bars = ax.bar(names, vals, color=colors)
ax.axhline(0.4762, color="black", linestyle="--", linewidth=1, label="원본 baseline (0.476)")
ax.set_ylabel("recall@k")
ax.set_ylim(0.4, 0.58)
ax.set_title("Step0 재설계 8종 비교 (실제 SMD 9구간, recall@k)")
ax.legend(loc="lower right", fontsize=8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
plt.xticks(rotation=0, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig2_step0_variants.png")
plt.close(fig)

# --- Fig 3: 실제 SMD 세그먼트 시계열 예시 (overlay) ---
import numpy as np

SMD_DIR = BASE / "mv_data" / "SMD"
entity = "machine-1-5"
test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
cs, ce = 10620, 10637
gt_dims = [1, 2, 3, 4, 7, 24, 26, 32]
gt_channels = [d - 1 for d in gt_dims]
center = (cs + ce) // 2
WIN = 224
s = max(0, min(len(test) - WIN, center - WIN // 2))
e = s + WIN
window = test[s:e]

fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=130, sharex=True)
t = np.arange(s, e)
cmap = plt.get_cmap("tab10")
for i, c in enumerate(gt_channels[:5]):
    v = window[:, c]
    lo, hi = v.min(), v.max()
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
    axes[0].plot(t, norm, label=f"Ch{c} (GT 관련)", color=cmap(i), linewidth=1.1)
axes[0].axvspan(cs, ce, color="red", alpha=0.15, label="실제 이상 구간(라벨)")
axes[0].set_title(f"{entity}, 세그먼트 ({cs},{ce}) — GT 관련 채널 5개 (정규화)")
axes[0].legend(fontsize=7, ncol=3, loc="upper right")

other_channels = [c for c in range(38) if c not in gt_channels][:5]
for i, c in enumerate(other_channels):
    v = window[:, c]
    lo, hi = v.min(), v.max()
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
    axes[1].plot(t, norm, label=f"Ch{c} (무관)", color=cmap(i), linewidth=1.1, alpha=0.8)
axes[1].axvspan(cs, ce, color="red", alpha=0.15)
axes[1].set_title("무관 채널 5개 (비교용)")
axes[1].set_xlabel("timestep")
axes[1].legend(fontsize=7, ncol=3, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig3_smd_example_segment.png")
plt.close(fig)

print("saved figures to", OUT)
for p in sorted(OUT.iterdir()):
    print(" -", p.name)
