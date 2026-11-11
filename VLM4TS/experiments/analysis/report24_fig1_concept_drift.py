"""
report24용 이미지1: "왜 train 기준 정상성이 안 맞는지" 실제 데이터로 보여주기.
machine-1-3 ch11, 세그먼트 machine-1-3_1257_1268 (z_train=1.172로 못 잡았는데
z_local=12.447로 강하게 잡힌 실제 사례).

3단 그래프: (a) train 전체 구간 ch11, (b) 후보 구간 근처 지역-문맥 구간 ch11,
(c) 후보 구간(GT) 자체 ch11 확대. train과 지역문맥의 "정상 수준" 자체가 다르다는 걸
시각적으로 보여줌.
"""
import sys
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH

entity, ch = "machine-1-3", 11
sid = "machine-1-3_1257_1268"

segs = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
seg = next(s for s in segs if s["entity"] == entity and s["start"] == 1257)
cs, ce = seg["start"], seg["end"]

train, test = load_smd(entity)
center = (cs + ce) // 2
s_, e_ = _centered_window(len(test), center, WIN)
window = test[s_:e_, ch]

# 지역 문맥: 후보 구간 근처에서 가장 "평범한"(표준편차 낮은) 600틱 구간을 골라서
# 실제 robust filtering이 지역 은행을 지을 때 하는 일(튀는 구간 제외)을 반영
candidates = [(cs - off, cs - off + 600) for off in range(1000, 7500, 500)] + \
             [(ce + off, ce + off + 600) for off in range(1000, 7500, 500)]
candidates = [(s, e) for s, e in candidates if 0 <= s and e < len(test)]
local_s, local_e = min(candidates, key=lambda se: test[se[0]:se[1], ch].std())

fig, axes = plt.subplots(3, 1, figsize=(9, 6), dpi=100)

axes[0].plot(train[:3000, ch], color="steelblue", linewidth=0.6)
axes[0].set_title(f"(a) TRAIN 구간 (ch{ch}, 앞 3000틱) — train z-score는 이 구간의 평균/분산을 '정상'으로 봄", fontsize=9)
axes[0].axhline(train[:, ch].mean(), color="gray", linestyle="--", linewidth=0.8)

axes[1].plot(test[local_s:local_e, ch], color="seagreen", linewidth=0.8)
axes[1].set_title(f"(b) 지역-문맥 구간 (test, 후보 구간에서 ~6000틱 떨어진 test 구간) — 지역문맥 z-score는 이 구간을 '정상'으로 봄", fontsize=9)
axes[1].axhline(test[local_s:local_e, ch].mean(), color="gray", linestyle="--", linewidth=0.8)

axes[2].plot(np.arange(s_, e_), window, color="black", linewidth=1.0)
axes[2].axvspan(cs, ce, color="red", alpha=0.15)
axes[2].set_title(f"(c) 후보 구간(GT, 빨간 음영) 포함 224틱 윈도우 — 실제 판단 대상", fontsize=9)

train_mu, train_std = train[:, ch].mean(), train[:, ch].std()
local_mu, local_std = test[local_s:local_e, ch].mean(), test[local_s:local_e, ch].std()
seg_val = test[cs:ce+1, ch].mean()
z_train = (seg_val - train_mu) / (train_std + 1e-9)
z_local_approx = (seg_val - local_mu) / (local_std + 1e-9)

fig.suptitle(f"machine-1-3 ch{ch}, 세그먼트 {sid} — train 기준 vs 지역문맥 기준 '정상 수준' 자체가 다름\n"
             f"(구간평균 대비 근사 z: train={z_train:.2f}, 지역문맥={z_local_approx:.2f} — "
             f"실제 patch-KNN z는 train={1.172:.2f}, 지역문맥={12.447:.2f})",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.90])

out = BASE / "experiments" / "analysis" / "report24_fig1_concept_drift.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved:", out)
