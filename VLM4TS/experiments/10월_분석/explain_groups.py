"""실험 설명용 그림: 그래프 1장을 16x16 patch로 나누고, 선이 지나는 patch를 A/B/C 그룹으로 색칠 + 그 patch들의 DINOv2 벡터를 2D로 줄인 모습."""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_inside import load_model, extract, ts_to_image_global, SMD, OUT, WIN, GRID

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
E, CH, S = "machine-1-6", 12, 22306          # 창 [22306, 22530)
tr = np.loadtxt(SMD / "train" / f"{E}.txt", delimiter=",")
te = np.loadtxt(SMD / "test" / f"{E}.txt", delimiter=",")
y = np.loadtxt(SMD / "test_label" / f"{E}.txt", delimiter=",").astype(int)
img = ts_to_image_global(te[S:S + WIN, CH], tr[:, CH].min(), tr[:, CH].max())
px = np.asarray(img)[:, :, 0] < 128                              # 검은 픽셀 = 선
P = 224 // GRID                                                  # patch 한 변 14px = 14틱

group = np.full((GRID, GRID), "", dtype=object)
for r in range(GRID):
    for c in range(GRID):
        if not px[r*P:(r+1)*P, c*P:(c+1)*P].any():
            continue                                             # 배경
        if y[S + c*P:S + (c+1)*P].any():
            group[r, c] = "A"                                    # GT 이상 구간
        elif r < GRID - 2:
            group[r, c] = "B"                                    # 정상 구간인데 위로 튄 선
        else:
            group[r, c] = "C"                                    # 정상 구간의 바닥 선
COL = {"A": "red", "B": "orange", "C": "royalblue"}
NAME = {"A": "A. GT spike", "B": "B. 라벨 없는 spike", "C": "C. 평범한 선"}

_, patches, _ = extract(load_model(), [img])
V = patches[0].reshape(GRID, GRID, -1)
idx = [(r, c) for r in range(GRID) for c in range(GRID) if group[r, c]]
xy = PCA(2).fit_transform(np.stack([V[r, c] for r, c in idx]))

fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
ax[0].imshow(img); ax[0].set_title("① 그래프 1장 (224틱 → 224×224 이미지)", fontsize=12)
ax[1].imshow(img)
for k in range(GRID + 1):
    ax[1].axhline(k*P - .5, color="gray", lw=.4); ax[1].axvline(k*P - .5, color="gray", lw=.4)
for r, c in idx:
    ax[1].add_patch(Rectangle((c*P - .5, r*P - .5), P, P, color=COL[group[r, c]], alpha=.45))
ax[1].set_title("② 16×16 patch로 나누고, 선이 지나는 patch만 색칠\n(흰 배경 patch는 안 씀)", fontsize=12)
for g in "ABC":
    pts = [xy[i] for i, (r, c) in enumerate(idx) if group[r, c] == g]
    if pts:
        pts = np.array(pts); ax[2].scatter(pts[:, 0], pts[:, 1], c=COL[g], s=60, label=f"{NAME[g]} ({len(pts)}개)", edgecolors="k")
ax[2].legend(fontsize=10)
ax[2].set_title("③ 색칠한 patch 하나 = DINOv2 숫자 768개 → 2개로 줄여서 점 하나로\n(같은 색끼리 뭉치면 DINOv2가 구분한다는 뜻)", fontsize=12)
for a in ax[:2]:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"실험 설명 예시: {E} ch{CH}, 그래프 1장 기준", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "실험설명_patch그룹.png", dpi=90)
print("saved")
