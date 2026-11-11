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
OUT = Path(r"C:\Users\김나영\Desktop\ny\tsad\report\attachments-20")
OUT.mkdir(parents=True, exist_ok=True)

I_PILOT = json.loads((BASE / "experiments" / "results_pilot_heatmap_nsweep" / "pilot_summary_I.json").read_text(encoding="utf-8"))
J_SUMMARY = json.loads((BASE / "experiments" / "results_pilot_top8_only" / "pilot_summary_J.json").read_text(encoding="utf-8"))
L_SUMMARY = json.loads((BASE / "experiments" / "results_pilot_score_annotation" / "pilot_summary_L.json").read_text(encoding="utf-8"))
K2_SUMMARY = json.loads((BASE / "experiments" / "results_pilot_reduced_index" / "pilot_summary_K2.json").read_text(encoding="utf-8"))
I8_74 = json.loads((BASE / "experiments" / "results_heatmap_overlay_74" / "summary_I8.json").read_text(encoding="utf-8"))

# ─────────────────────────────────────────────
# Fig1: 전체 구조도 (Stage1 -> Stage2, z-score 포인트 텍스트 포함)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
ax.axis("off")


def box(x, y, w, h, text, color="#E8F0FE", fontsize=8.8, edge="#4C72B0"):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=1.3, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3, linespacing=1.35)


def arrow(x0, y0, x1, y1, text="", fontsize=7.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5), zorder=1)
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, text, ha="center", fontsize=fontsize, color="#555")


box(0.2, 1.6, 1.9, 1.6, "Stage1\nDINOv2\n채널별 이상 점수\n(38채널 전부)", color="#E8F0FE")
box(2.6, 1.6, 1.9, 1.6, "38채널 순위\n(자르지 않음)", color="#FFF3CD")
box(5.0, 0.2, 2.5, 1.5, "① heatmap\n38채널 개요\n(점수순 정렬)", color="#E2F0D9", fontsize=8.2)
box(5.0, 1.85, 2.5, 1.5, "② overlay\n상위 8채널\n상세 파형", color="#E2F0D9", fontsize=8.2)
box(5.0, 3.5, 2.5, 1.5, "③ z-score 포인트\n텍스트 (신규)\n채널당 |z| 상위 25개", color="#F9E5C9", fontsize=8.0)
box(8.1, 1.6, 2.3, 1.6, "GPT-4o\n콜 1회\nJSON 응답", color="#FADBD8")

arrow(2.1, 2.4, 2.6, 2.4)
arrow(4.5, 2.4, 4.9, 1.0)
arrow(4.5, 2.4, 4.9, 2.6)
arrow(4.5, 2.4, 4.9, 4.2)
arrow(7.5, 1.0, 8.1, 2.1)
arrow(7.5, 2.6, 8.1, 2.4)
arrow(7.5, 4.2, 8.1, 2.7)

ax.text(1.15, 3.5, "Stage1 (스코어링)", ha="center", fontsize=10, fontweight="bold", color="#2A5A8C")
ax.text(9.25, 3.5, "Stage2 (판단)", ha="center", fontsize=10, fontweight="bold", color="#943126")
ax.text(6.25, 5.25, "① + ② + ③ = 이미지 1장 + 텍스트, 콜은 여전히 1개", ha="center", fontsize=9, color="#333")
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 5.6)
fig.suptitle("최종 구조: heatmap + overlay + z-score 포인트 텍스트 (K2)", fontsize=12, y=0.99)
fig.tight_layout()
fig.savefig(OUT / "fig1_full_pipeline_k2.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved fig1")

# ─────────────────────────────────────────────
# Fig2: z-score 포인트 선별 예시 (실제 채널 1개)
# ─────────────────────────────────────────────
from smd_3way_baseline_comparison import load_smd_test
from step1v3_dino_graph_smd import WIN, _centered_window, load_smd

score_cache = json.loads((BASE / "experiments" / "results_adaptive_vs_fixed" / "step0_scores_cache.json").read_text(encoding="utf-8"))
seg_id = "machine-1-1_18071_18528"
entity, cs, ce = "machine-1-1", 18071, 18528
test = load_smd_test(entity)
train, _ = load_smd(entity)
center = (cs + ce) // 2
s_, e_ = _centered_window(len(test), center, WIN)
window = test[s_:e_]
scores = {int(c): v for c, v in score_cache[seg_id].items()}
ranked = sorted(scores, key=lambda c: -scores[c])
c = ranked[0]  # 1등 채널로 예시
v = window[:, c]
mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
z = np.abs((v - mu) / sigma)
top_idx = np.sort(np.argsort(-z)[:25])
lo, hi = float(v.min()), float(v.max())
norm = (v - lo) / (hi - lo)

fig, ax = plt.subplots(figsize=(8, 3.2), dpi=140)
ax.plot(np.arange(len(norm)), norm, linewidth=1.0, color="#4C72B0", label=f"채널 {c} 전체 파형 (224개 지점)")
ax.scatter(top_idx, norm[top_idx], color="#C0392B", s=28, zorder=5, label="|z-score| 상위 25개 (텍스트로 추가되는 지점)")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("시간 인덱스")
ax.set_ylabel("정규화된 값")
ax.set_title(f"z-score 포인트 선별 예시 (채널 {c}, 1등 후보)")
fig.tight_layout()
fig.savefig(OUT / "fig2_zscore_point_selection.png", dpi=140)
plt.close(fig)
print("saved fig2")

# ─────────────────────────────────────────────
# Fig3: I8 구조 위 추가 시도 4가지 비교 (파일럿 n=18)
# ─────────────────────────────────────────────
labels = ["I8\n(heatmap+overlay)", "J\nheatmap 제거\n(top-8만)", "L\n전체 224pt/채널\n점수·값 텍스트", "K2\n상위 25pt/채널\nz-score 텍스트"]
f1s = [I_PILOT["mean_f1"]["I8"], J_SUMMARY["mean_f1"]["J"], L_SUMMARY["mean_f1"]["L"], K2_SUMMARY["mean_f1"]["K2"]]
colors = ["#55A868", "#AAB7B8", "#C0392B", "#2E86C1"]

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
bars = ax.bar(labels, f1s, color=colors, width=0.6)
for b, v in zip(bars, f1s):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.axhline(I_PILOT["mean_f1"]["I8"], color="#55A868", linestyle="--", linewidth=1, alpha=0.6)
ax.set_ylabel("F1 (파일럿, n=18)")
ax.set_ylim(0, 0.40)
ax.set_title("I8 구조 위에서 시도한 4가지 변형 비교")
fig.tight_layout()
fig.savefig(OUT / "fig3_addon_experiments.png", dpi=140)
plt.close(fig)
print("saved fig3")

# ─────────────────────────────────────────────
# Fig4: I8 vs K2 분위별 비교 + Q4 L 대조
# ─────────────────────────────────────────────
q_order = ["Q1_밀집", "Q2Q3_중간", "Q4_뚜렷"]
q_labels = ["채널값 밀집\n구간", "중간\n구간", "채널값 뚜렷히\n갈리는 구간"]
i8_vals = [I_PILOT["by_quartile"][k]["f1_I8"] for k in q_order]
l_vals = [L_SUMMARY["by_quartile"][k]["f1_L"] for k in q_order]
k2_vals = [K2_SUMMARY["by_quartile"][k]["f1_K2"] for k in q_order]

x = np.arange(len(q_labels))
w = 0.25
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
ax.bar(x - w, i8_vals, w, label="I8 (텍스트 없음)", color="#55A868")
ax.bar(x, l_vals, w, label="L (텍스트 과다, 224pt/채널)", color="#C0392B")
ax.bar(x + w, k2_vals, w, label="K2 (텍스트 축소, 25pt/채널)", color="#2E86C1")
ax.set_xticks(x)
ax.set_xticklabels(q_labels)
ax.set_ylabel("F1")
ax.set_title("구간별: 텍스트 정보량에 따른 효과 (L vs K2)")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig4_l_vs_k2_quartile.png", dpi=140)
plt.close(fig)
print("saved fig4")

print(f"\nAll figures saved to: {OUT}")
