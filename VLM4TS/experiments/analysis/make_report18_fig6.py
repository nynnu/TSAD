import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(r"C:\Users\김나영\Desktop\ny\tsad\report\attachments-18")

positions = ["위치0\n(왼쪽 끝)", "위치1", "위치2\n(가운데)", "위치3", "위치4", "위치5\n(오른쪽 끝)"]
distinctiveness = [0.0867, 0.0505, 0.0497, 0.0505, 0.0448, 0.0903]
colors = ["#C44E52" if v > 0.07 else "#4C72B0" for v in distinctiveness]

fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
bars = ax.bar(positions, distinctiveness, color=colors)
ax.set_ylabel("위치 민감도 (다른 위치와의 임베딩 차이)")
ax.set_title("같은 신호를 이미지 내 위치만 바꿔 넣었을 때\nDINOv2 임베딩이 얼마나 달라지는가")
for b, v in zip(bars, distinctiveness):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig6_dino_position_bias.png")
print("saved fig6")
