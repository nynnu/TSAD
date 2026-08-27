import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SANITY_DIR = Path(__file__).resolve().parent
OUT_DIR = SANITY_DIR / "results" / "channel_grouping9"
df = pd.read_csv(OUT_DIR / "pair_table.csv")

METHOD_COLS = {"Corr (full)": "corr_full", "Corr (post-onset)": "corr_post", "Attention (post-onset)": "attn_sim"}
COLORS = {"Corr (full)": "#1f77b4", "Corr (post-onset)": "#2ca02c", "Attention (post-onset)": "#d62728"}


def prf1(gt, pred):
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall and (precision + recall)) else (0.0 if (tp + fp + fn) else np.nan)
    return f1


# 1. AUC bar chart (the headline finding)
aucs = {}
for name, col in METHOD_COLS.items():
    pos, neg = df[df.gt_edge == 1][col], df[df.gt_edge == 0][col]
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    aucs[name] = stat / (len(pos) * len(neg))

fig, ax = plt.subplots(figsize=(7, 5))
names = list(aucs.keys())
vals = [aucs[n] for n in names]
bars = ax.bar(names, vals, color=[COLORS[n] for n in names], alpha=0.75)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="0.5 = 무작위(구분력 없음)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylabel("AUC (P(related pair score > unrelated pair score))")
ax.set_title("방법별 AUC — 0.5 미만이면 '거꾸로' 구분(불량)")
ax.set_ylim(0, 0.6)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_auc.png", dpi=130)
plt.close(fig)

# 2. Score distribution by gt_edge (why AUC < 0.5)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, col) in zip(axes, METHOD_COLS.items()):
    ax.hist(df[df.gt_edge == 0][col], bins=30, alpha=0.55, label="unrelated (gt=0)", color="#1f77b4", density=True)
    ax.hist(df[df.gt_edge == 1][col], bins=30, alpha=0.55, label="related (gt=1)", color="#d62728", density=True)
    ax.set_title(name)
    ax.set_xlabel("score")
    ax.legend(fontsize=8)
fig.suptitle("점수 분포: unrelated가 related보다 오른쪽(값이 큼)에 쏠려 있음 = 거꾸로", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_distributions.png", dpi=130)
plt.close(fig)

# 3. Finer F1-vs-threshold curve
thresholds = np.arange(0.05, 0.96, 0.05)
fig, ax = plt.subplots(figsize=(8, 5.5))
for name, col in METHOD_COLS.items():
    f1s = [prf1(df["gt_edge"].values, (df[col] > th).astype(int).values) for th in thresholds]
    ax.plot(thresholds, f1s, marker="o", markersize=3, label=name, color=COLORS[name])
ax.set_xlabel("threshold")
ax.set_ylabel("Edge F1")
ax.set_title("Threshold별 Edge F1 곡선 (A vs B)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_f1_vs_threshold.png", dpi=130)
plt.close(fig)

# 4. n_affected breakdown bar chart (best-threshold F1 per method, from cached CSV)
n_aff = pd.read_csv(OUT_DIR / "n_affected_breakdown.csv")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(n_aff))
width = 0.35
ax.bar(x - width / 2, n_aff["A_f1"], width, label="A (corr, best threshold)", color="#1f77b4", alpha=0.8)
ax.bar(x + width / 2, n_aff["B_f1"], width, label="B (attention, best threshold)", color="#d62728", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(n_aff["n_affected"])
ax.set_xlabel("n_affected")
ax.set_ylabel("Edge F1")
ax.set_title("n_affected별 Edge F1 (주의: n_affected가 클수록 positive 비율↑ -> F1이 base rate로 부풀려짐)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_n_affected.png", dpi=130)
plt.close(fig)

# 5. homogeneous/heterogeneous recall bar chart
homog = pd.read_csv(OUT_DIR / "homogeneous_breakdown.csv")
fig, ax = plt.subplots(figsize=(6.5, 5))
x = np.arange(len(homog))
ax.bar(x - width / 2, homog["A_recall"], width, label="A (corr, best threshold)", color="#1f77b4", alpha=0.8)
ax.bar(x + width / 2, homog["B_recall"], width, label="B (attention, best threshold)", color="#d62728", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["homogeneous" if v else "heterogeneous" for v in homog["type_match"]])
ax.set_ylabel("Recall (true edge 중 맞춘 비율)")
ax.set_title("homogeneous/heterogeneous별 Recall (참고: B는 거의 항상 1에 가까움 = 전부 edge라고 찍는 효과)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_homogeneous.png", dpi=130)
plt.close(fig)

print("AUCs:", aucs)
print("Saved charts to:", OUT_DIR)
