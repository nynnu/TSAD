import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI window)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 데이터 로드 및 전처리 ──────────────────────────────────────────────────────
df = pd.read_csv("comparison_results_gpt/results_summary.csv")
df = df[df["dataset"] != "artificialWithAnomaly"].reset_index(drop=True)

MODELS   = ["LLM-TSAD", "VLM4TS", "ViT4TS"]
DATASETS = df["dataset"].unique().tolist()
METRICS  = ["avg_f1_max", "avg_precision", "avg_recall", "avg_affi_f1"]
METRIC_LABELS = ["F1-Max", "Precision", "Recall", "Affi-F1"]

MODEL_COLORS = {
    "LLM-TSAD": "#4C72B0",
    "VLM4TS":   "#DD8452",
    "ViT4TS":   "#55A868",
}
DATASET_LABELS = {
    "SMAP":              "SMAP",
    "realAWSCloudwatch": "AWS Cloudwatch",
}

# ── Figure 1: Grouped Bar Chart (메트릭별 × 데이터셋별) ─────────────────────────
fig, axes = plt.subplots(len(DATASETS), len(METRICS),
                         figsize=(14, 5 * len(DATASETS)),
                         sharey=False)
if len(DATASETS) == 1:
    axes = [axes]

x = np.arange(len(MODELS))
bar_width = 0.55

for row, dataset in enumerate(DATASETS):
    sub = df[df["dataset"] == dataset].set_index("model")
    for col, (metric, mlabel) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes[row][col]
        vals = [sub.loc[m, metric] if m in sub.index else 0 for m in MODELS]
        bars = ax.bar(x, vals, width=bar_width,
                      color=[MODEL_COLORS[m] for m in MODELS],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {mlabel}", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

legend_patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
fig.legend(handles=legend_patches, loc="upper center", ncol=3,
           fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Anomaly Detection Performance Comparison\n(per Dataset & Metric)",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig("comparison_results_gpt/fig1_per_dataset_metric.png",
            dpi=150, bbox_inches="tight")
print("Saved: fig1_per_dataset_metric.png")

# ── Figure 2: Radar Chart (데이터셋별, 모델 오버레이) ─────────────────────────────
N = len(METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]          # 닫기

fig2, axes2 = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 6),
                            subplot_kw=dict(polar=True))
if len(DATASETS) == 1:
    axes2 = [axes2]

for ax, dataset in zip(axes2, DATASETS):
    sub = df[df["dataset"] == dataset].set_index("model")
    for model in MODELS:
        if model not in sub.index:
            continue
        vals = [sub.loc[model, m] for m in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=model, color=MODEL_COLORS[model])
        ax.fill(angles, vals, alpha=0.12, color=MODEL_COLORS[model])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

fig2.suptitle("Radar Chart: Model Comparison per Dataset",
              fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("comparison_results_gpt/fig2_radar.png",
            dpi=150, bbox_inches="tight")
print("Saved: fig2_radar.png")

# ── Figure 3: 메트릭별 전체 평균 (두 데이터셋 통합) ──────────────────────────────
avg_df = df.groupby("model")[METRICS].mean().reindex(MODELS)

fig3, ax3 = plt.subplots(figsize=(9, 5))
x3 = np.arange(len(METRICS))
n_models = len(MODELS)
total_w = 0.65
bw = total_w / n_models

for i, model in enumerate(MODELS):
    offsets = x3 - total_w / 2 + bw * i + bw / 2
    bars = ax3.bar(offsets, avg_df.loc[model, METRICS], width=bw * 0.9,
                   color=MODEL_COLORS[model], label=model,
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, avg_df.loc[model, METRICS]):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax3.set_xticks(x3)
ax3.set_xticklabels(METRIC_LABELS, fontsize=12)
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("Average Score", fontsize=12)
ax3.set_title("Overall Average Performance (SMAP + AWS Cloudwatch)",
              fontsize=13, fontweight="bold")
ax3.legend(fontsize=11, frameon=False)
ax3.spines[["top", "right"]].set_visible(False)
ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig("comparison_results_gpt/fig3_overall_avg.png",
            dpi=150, bbox_inches="tight")
print("Saved: fig3_overall_avg.png")

# ── 수치 요약 출력 ────────────────────────────────────────────────────────────
print("\n=== Results Summary (artificialWithAnomaly excluded) ===")
print(df.to_string(index=False))
print("\n=== Overall Average ===")
print(avg_df.round(4).to_string())
