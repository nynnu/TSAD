import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OUT = Path(r"C:\Users\김나영\Desktop\ny\tsad\report\attachments-18")
SMD_DIR = BASE / "mv_data" / "SMD"

entity = "machine-1-5"
train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
cs, ce = 10620, 10637
gt_dims = [1, 2, 3, 4, 7, 24, 26, 32]
gt_channels = [d - 1 for d in gt_dims]
WIN = 224
center = (cs + ce) // 2


def cwin(arr_len, c, win):
    s = max(0, min(arr_len - win, c - win // 2))
    return s, s + win


s, e = cwin(len(test), center, WIN)
window = test[s:e]

# ---------------------------------------------------------------
# Fig 4: overlay vs subplot 렌더링 실제 비교 (같은 데이터, §19)
# ---------------------------------------------------------------
show_channels = gt_channels[:4]
fig = plt.figure(figsize=(11, 5), dpi=130)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

ax_overlay = fig.add_subplot(gs[0, 0])
cmap = plt.get_cmap("tab10")
for i, c in enumerate(show_channels):
    v = window[:, c]
    lo, hi = v.min(), v.max()
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
    ax_overlay.plot(norm, color=cmap(i), linewidth=1.2, label=f"Ch{c}")
ax_overlay.set_title("Overlay 방식\n(root accuracy 55.5%)", fontsize=11)
ax_overlay.legend(fontsize=8, loc="upper right")
ax_overlay.set_xticks([])
ax_overlay.set_yticks([])

gs_sub = gs[0, 1].subgridspec(len(show_channels), 1, hspace=0.15)
for i, c in enumerate(show_channels):
    ax = fig.add_subplot(gs_sub[i, 0])
    v = window[:, c]
    lo, hi = v.min(), v.max()
    norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
    ax.plot(norm, color=cmap(i), linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(f"Ch{c}", fontsize=8, rotation=0, labelpad=18)
    if i == 0:
        ax.set_title("Stacked-subplot 방식\n(root accuracy 26.2%)", fontsize=11)

fig.suptitle("같은 4채널, 렌더링 방식만 다르게 (machine-1-5, 세그먼트 10620~10637)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_overlay_vs_subplot.png")
plt.close(fig)

# ---------------------------------------------------------------
# Fig 5: 단일 참조 vs 5개 참조 평균 — Step0 개선 이유 시각화
# ---------------------------------------------------------------
example_channel = 8  # machine-1-5, GT에 없는 채널 (무관 채널) - 단일참조 우연성 예시용
static_positions = np.linspace(WIN // 2, len(train) - WIN // 2, 5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)

ax = axes[0]
mid = len(train) // 2
s0, e0 = cwin(len(train), mid, WIN)
single_ref = train[s0:e0, example_channel]
lo, hi = single_ref.min(), single_ref.max()
norm_single = (single_ref - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(single_ref)
ax.plot(norm_single, color="#C44E52", linewidth=1.3)
ax.set_title("기존: train 중앙 지점 1개만 참조", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1]
cmap2 = plt.get_cmap("viridis")
for i, c in enumerate(static_positions):
    sp, ep = cwin(len(train), c, WIN)
    ref = train[sp:ep, example_channel]
    lo, hi = ref.min(), ref.max()
    norm = (ref - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(ref)
    ax.plot(norm, color=cmap2(i / 4), linewidth=1.0, alpha=0.8, label=f"참조{i+1}")
ax.set_title("개선: train 5개 지점 참조 (평균으로 사용)", fontsize=10)
ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.05))
ax.set_xticks([])
ax.set_yticks([])

fig.suptitle(f"Ch{example_channel}(정상 채널) — '정상'이 어떻게 생겼는지, 참조 지점마다 형태가 다름", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig5_singleref_vs_multiref.png")
plt.close(fig)

print("saved:", [p.name for p in sorted(OUT.iterdir()) if p.name.startswith("fig4") or p.name.startswith("fig5")])
