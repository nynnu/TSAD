import base64
import sys
from io import BytesIO
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
OUT = Path(r"C:\Users\김나영\AppData\Local\Temp\claude\C--Windows-System32\cfa8c392-19e8-45eb-b9aa-6be6808b0caf\scratchpad")
OUT.mkdir(parents=True, exist_ok=True)

from smd_3way_baseline_comparison import load_smd_test, naive_groups, render_overlay
from step1v3_dino_graph_smd import WIN, _centered_window

entity, cs, ce = "machine-1-1", 18071, 18528
test = load_smd_test(entity)
center = (cs + ce) // 2
s_, e_ = _centered_window(len(test), center, WIN)
window = test[s_:e_]
N_CHANNELS = 38

# ─────────────────────────────────────────────
# 이미지 A: VLM4TS 다변량 확장 방식 -- subplot grid (채널마다 작은 칸)
# ─────────────────────────────────────────────
n_cols, n_rows = 8, 5  # 40칸, 38채널 사용
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6.5), dpi=140)
for c in range(N_CHANNELS):
    ax = axes[c // n_cols, c % n_cols]
    v = window[:, c]
    lo, hi = float(v.min()), float(v.max())
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
    ax.plot(norm, linewidth=0.6, color="black")
    ax.set_title(f"ch{c}", fontsize=6, pad=1)
    ax.set_xticks([])
    ax.set_yticks([])
for c in range(N_CHANNELS, n_rows * n_cols):
    axes[c // n_cols, c % n_cols].axis("off")
fig.suptitle("VLM4TS 다변량 확장(subplot grid): 38채널, 채널마다 작은 칸 하나씩", fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig(OUT / "presentation_A_vlm4ts_subplot_grid.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved A")

# ─────────────────────────────────────────────
# 이미지 B: canMMLM splitting(canmmlm_naive) -- 실제 4개 그룹, 각각 별도 콜로 들어가는 이미지
# ─────────────────────────────────────────────
groups = naive_groups(N_CHANNELS, 4)
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=140)
for gi, group in enumerate(groups):
    ax = axes[gi // 2, gi % 2]
    cmap = plt.get_cmap("tab10" if len(group) <= 10 else "tab20")
    for i, c in enumerate(group):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(norm, linewidth=0.8, color=cmap(i % cmap.N), label=f"Ch{c}")
    ax.legend(fontsize=5.5, ncol=min(5, len(group)), loc="upper right")
    ax.set_title(f"콜 {gi+1}/4 — 채널 {group[0]}~{group[-1]} ({len(group)}개)", fontsize=9)
fig.suptitle("canMMLM splitting(canmmlm_naive): 그룹마다 별도 콜로 4번 전송되는 실제 입력 이미지", fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "presentation_B_canmmlm_splitting.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("saved B")

# 개별 4장도 따로 저장 (그룹1개씩 실제 크기로 확인하고 싶을 때)
for gi, group in enumerate(groups):
    img_b64 = render_overlay(window, group)
    (OUT / f"presentation_B_group{gi+1}_actual_input.png").write_bytes(base64.b64decode(img_b64))
print("saved individual group images")
