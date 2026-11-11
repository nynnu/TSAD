import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
ANALYSIS_DIR = BASE / "experiments" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))
OUT = Path(r"C:\Users\김나영\Desktop\ny\tsad\report\attachments-19")
OUT.mkdir(parents=True, exist_ok=True)

I8_SUMMARY = json.loads((BASE / "experiments" / "results_heatmap_overlay_74" / "summary_I8.json").read_text(encoding="utf-8"))

# ─────────────────────────────────────────────
# Fig1: 파이프라인 구조 (Stage1 -> Stage2 핸드오프)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=140)
ax.axis("off")


def box(x, y, w, h, text, color="#E8F0FE", fontsize=9.5, edge="#4C72B0"):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3, linespacing=1.4)


def arrow(x0, y0, x1, y1, text=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6), zorder=1)
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, text, ha="center", fontsize=8, color="#555")


box(0.3, 1.3, 2.1, 1.8, "Stage1\nDINOv2\n채널별 이상 점수\n(38채널 전부)", color="#E8F0FE")
box(3.0, 1.3, 2.1, 1.8, "38채널 순위\n(자르지 않음)\nfront-loading", color="#FFF3CD")
box(5.7, 0.3, 2.6, 1.6, "상단: heatmap\n38채널 개요\n(점수순 정렬)", color="#E2F0D9", fontsize=8.7)
box(5.7, 2.3, 2.6, 1.6, "하단: overlay\n상위 8채널\n상세 파형", color="#E2F0D9", fontsize=8.7)
box(8.9, 1.3, 2.0, 1.8, "GPT-4o\n콜 1회\nJSON 응답", color="#FADBD8")

arrow(2.4, 2.2, 3.0, 2.2)
arrow(5.1, 2.2, 5.5, 1.1, "컴포짓\n이미지 1장")
arrow(5.1, 2.2, 5.5, 3.1)
arrow(8.3, 1.1, 8.9, 1.9)
arrow(8.3, 3.1, 8.9, 2.5)

ax.text(1.35, 3.25, "Stage1 (스코어링)", ha="center", fontsize=10, fontweight="bold", color="#2A5A8C")
ax.text(8.9, 3.25, "Stage2 (판단)", ha="center", fontsize=10, fontweight="bold", color="#943126")
ax.set_xlim(0, 11.2)
ax.set_ylim(0, 4.0)
fig.suptitle("Stage1 → Stage2 핸드오프 구조 (I8 방식)", fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig(OUT / "fig1_pipeline_diagram.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved fig1")

# ─────────────────────────────────────────────
# Fig2: 실제 heatmap+overlay 컴포짓 예시
# ─────────────────────────────────────────────
from smd_3way_baseline_comparison import load_smd_test
from step1v3_dino_graph_smd import WIN, _centered_window
from pilot_heatmap_overlay_nsweep import render_heatmap_overlay

score_cache = json.loads((BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json").read_text(encoding="utf-8"))
seg_id = "machine-1-1_18071_18528"
entity, cs, ce = "machine-1-1", 18071, 18528
test = load_smd_test(entity)
center = (cs + ce) // 2
s_, e_ = _centered_window(len(test), center, WIN)
window = test[s_:e_]
scores = {int(c): v for c, v in score_cache[seg_id].items()}
ranked = sorted(scores, key=lambda c: -scores[c])
img_b64 = render_heatmap_overlay(window, ranked, 8)
import base64
(OUT / "fig2_heatmap_overlay_example.png").write_bytes(base64.b64decode(img_b64))
print("saved fig2")

# ─────────────────────────────────────────────
# Fig3: F1 비교 (74개 규모, A vs B vs I8)
# ─────────────────────────────────────────────
labels = ["전체 채널 제시\n(vlm4ts_own, 1콜)", "단순 그룹 분할\n(canmmlm_naive, 4콜)", "heatmap+overlay\n(우리 새 방식, 1콜)"]
f1s = [I8_SUMMARY["mean_f1"]["A"], I8_SUMMARY["mean_f1"]["B"], I8_SUMMARY["mean_f1"]["I8"]]
colors = ["#AAB7B8", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=140)
bars = ax.bar(labels, f1s, color=colors, width=0.55)
for b, v in zip(bars, f1s):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylabel("F1")
ax.set_ylim(0, 0.40)
ax.set_title("채널 선택 방식별 성능 (SMD 74개 세그먼트)")
fig.tight_layout()
fig.savefig(OUT / "fig3_f1_comparison_74.png", dpi=140)
plt.close(fig)
print("saved fig3")

# ─────────────────────────────────────────────
# Fig4: 부트스트랩 신뢰구간 (통계적 유의성)
# ─────────────────────────────────────────────
seg_boot = I8_SUMMARY["segment_bootstrap"]
pairs = [("I8 vs 전체 채널 제시\n(I8 - A)", seg_boot["I8_minus_A"]), ("I8 vs 단순 그룹 분할\n(I8 - B)", seg_boot["I8_minus_B"])]

fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=140)
y_pos = [1, 0]
for i, (label, (mean, lo, hi)) in enumerate(pairs):
    ax.errorbar(mean, y_pos[i], xerr=[[mean - lo], [hi - mean]], fmt="o", color="#2A5A8C",
                capsize=6, markersize=9, elinewidth=2.2)
ax.axvline(0, color="#943126", linestyle="--", linewidth=1.3)
ax.set_yticks(y_pos)
ax.set_yticklabels([p[0] for p in pairs], fontsize=10)
ax.set_xlabel("F1 차이 (95% 신뢰구간, 세그먼트 단위 부트스트랩 1000회)")
ax.set_title("I8이 통계적으로 유의미하게 이기는지 (0을 안 지나면 유의)")
ax.set_ylim(-0.6, 1.6)
fig.tight_layout()
fig.savefig(OUT / "fig4_bootstrap_ci.png", dpi=140)
plt.close(fig)
print("saved fig4")

# ─────────────────────────────────────────────
# Fig5: gap 분위별 성능 비교
# ─────────────────────────────────────────────
q = I8_SUMMARY["by_quartile"]
q_order = ["Q1_밀집", "Q2Q3_중간", "Q4_뚜렷"]
q_labels = ["채널값 밀집\n구간", "중간\n구간", "채널값 뚜렷히\n갈리는 구간"]
a_vals = [q[k]["f1_A"] for k in q_order]
b_vals = [q[k]["f1_B"] for k in q_order]
i8_vals = [q[k]["f1_I8"] for k in q_order]

x = np.arange(len(q_labels))
w = 0.25
fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=140)
ax.bar(x - w, a_vals, w, label="전체 채널 제시", color="#AAB7B8")
ax.bar(x, b_vals, w, label="단순 그룹 분할(4콜)", color="#DD8452")
ax.bar(x + w, i8_vals, w, label="heatmap+overlay(1콜)", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(q_labels)
ax.set_ylabel("F1")
ax.set_title("구간 특성별 성능 비교")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig5_quartile_breakdown.png", dpi=140)
plt.close(fig)
print("saved fig5")

print(f"\nAll figures saved to: {OUT}")
