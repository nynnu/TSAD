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

from smd_3way_baseline_comparison import load_smd_test
from step1v3_dino_graph_smd import WIN, _centered_window, load_smd

N = 8
N_POINTS = 25

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
top8 = ranked[:N]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8.6), dpi=140, gridspec_kw={"height_ratios": [1.4, 1]})

# ── 상단: heatmap (38채널 전체, 점수 순 정렬) ──
heat = np.zeros((len(ranked), window.shape[0]))
for i, c in enumerate(ranked):
    v = window[:, c]
    lo, hi = float(v.min()), float(v.max())
    heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
ax1.imshow(heat, aspect="auto", cmap="viridis")
ax1.set_yticks(range(len(ranked)))
ax1.set_yticklabels([str(c) for c in ranked], fontsize=5)
ax1.set_xticks([])
ax1.set_title("① heatmap: 38채널 전체 (Step0 점수순 정렬, 위=가장 이상함)", fontsize=8.5)

# ── 하단: overlay (상위 8채널 상세 파형) + z-score 상위 25포인트 표시 ──
cmap = plt.get_cmap("tab10")
for i, c in enumerate(top8):
    v = window[:, c]
    lo, hi = float(v.min()), float(v.max())
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
    color = cmap(i % cmap.N)
    ax2.plot(np.arange(len(norm)), norm, linewidth=0.8, color=color, label=f"Ch{c}")

    mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
    z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
    top_idx = np.argsort(-z)[:N_POINTS]
    ax2.scatter(top_idx, norm[top_idx], color=color, s=16, zorder=5, edgecolors="black", linewidths=0.3)

ax2.legend(fontsize=6, ncol=min(6, len(top8)), loc="upper right")
ax2.set_title("② overlay: 상위 8채널 상세 파형 (● = ③ z-score 상위 25개, 텍스트로도 같이 전달됨)", fontsize=8.5)
ax2.set_xlabel("시간 인덱스")

fig.suptitle("최종 입력 이미지 예시: heatmap + overlay + z-score 포인트", fontsize=11, y=0.995)
fig.tight_layout()
fig.savefig(OUT / "fig7_heatmap_overlay_indexaware_example.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved fig7_heatmap_overlay_indexaware_example.png")
