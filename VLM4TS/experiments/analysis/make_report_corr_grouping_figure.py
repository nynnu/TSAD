import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT = Path(r"C:\Users\김나영\Desktop\ny\tsad\report\attachments-19")
OUT.mkdir(parents=True, exist_ok=True)

summary = json.loads((BASE / "experiments" / "results_corr_grouping" / "summary.json").read_text(encoding="utf-8"))
thresholds = summary["thresholds"]
overall = summary["overall"]
by_q = summary["by_quartile"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=140)

# 왼쪽: 임계값별 그룹 수 (canmmlm_naive의 4그룹과 비교)
n_groups = [overall[str(t)]["n_groups_mean"] for t in thresholds]
ax1.plot(thresholds, n_groups, marker="o", color="#2E86C1", linewidth=2, markersize=7)
ax1.axhline(4, color="#943126", linestyle="--", linewidth=1.3, label="canmmlm_naive의 그룹 수 (4개)")
ax1.set_xlabel("상관계수 임계값")
ax1.set_ylabel("세그먼트당 평균 그룹 수")
ax1.set_title("임계값을 바꿔도 4개 근처로 안 옴")
ax1.legend(fontsize=8.5)
for t, n in zip(thresholds, n_groups):
    ax1.annotate(f"{n:.1f}", (t, n), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

# 오른쪽: 분위별 "20채널 이상 큰 덩어리" 비율 (threshold=0.5)
q_labels = ["채널값 밀집\n구간(Q1)", "중간\n구간", "채널값 뚜렷히\n갈리는 구간(Q4)"]
q_keys = ["Q1_밀집", "Q2Q3_중간", "Q4_뚜렷"]
big_blob = [by_q[k]["0.5"]["big_blob_frac"] * 100 for k in q_keys]
colors = ["#C0392B", "#F39C12", "#27AE60"]
bars = ax2.bar(q_labels, big_blob, color=colors, width=0.55)
for b, v in zip(bars, big_blob):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10.5, fontweight="bold")
ax2.set_ylabel("20채널 이상 큰 덩어리가 나온 비율")
ax2.set_title("밀집 구간일수록 하나로 뭉쳐버림 (임계값 0.5)")
ax2.set_ylim(0, 65)

fig.suptitle("상관관계 임계값 그룹핑 시도 (74개 세그먼트) — canmmlm_naive 대안으로는 실패", fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig6_corr_grouping_failure.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved fig6_corr_grouping_failure.png")
