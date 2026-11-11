"""
report24용 이미지2: K25 실제 파이프라인 입력 이미지 (강한확신/약한확신 등급 표시).
segment machine-1-4_20193_20233 (k=2, pilot에서 precision 0.222->0.333 검증했던
바로 그 세그먼트) 기준. K4/local_z_cache 캐시를 재사용해서 DINOv2는 zoom-range
계산에만 씀(각 후보채널당 1회, 가볍게 끝남).
"""
import sys
from pathlib import Path
import json

import numpy as np
import torch
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
import colab_multivariate_v2 as cm
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, N_CHANNELS, constant_channels, compute_zscores,
)
from experiment_stage2_k16_patch_intensity import compute_time_profile
from experiment_stage2_k18_zoomed_subplot import ZOOM_WIDTH
import experiment_stage2_v16 as v16
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k25_local_context import (
    select_channels_train_or_local, render_zoomed_subplot_grid,
)

cm.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sid = "machine-1-2_15540_15605"
entity = "machine-1-2"

segs = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
seg = next(s for s in segs if s["entity"] == entity and s["start"] == 15540)
cs, ce = seg["start"], seg["end"]

train, test = load_smd(entity)
degenerate = constant_channels(train)
center = (cs + ce) // 2
s_, e_ = _centered_window(len(test), center, WIN)
window = test[s_:e_]

entity_channel_calib = {}
z_train = compute_zscores(entity, train, window, entity_channel_calib, degenerate)

local_cache = json.loads((BASE / "experiments" / "results_stage2_k25_local_context" / "local_z_cache.json").read_text(encoding="utf-8"))
z_local = {int(c): v for c, v in local_cache[sid].items()}
for c in range(N_CHANNELS):
    if c not in z_local:
        z_local[c] = -1e9

ranked, selected, strong, weak = select_channels_train_or_local(z_train, z_local)
print(f"selected={len(selected)}개 (강한확신={len(strong)}, 약한확신={len(weak)})")
print("강한확신:", sorted(strong))
print("약한확신:", sorted(weak))

cmin, cmax = v16.gn_train(train, range(N_CHANNELS))

zoom_ranges = {}
for c in selected:
    tr_cls, tr_patches, _ = entity_channel_calib[(entity, c)]
    profile = compute_time_profile(tr_cls, tr_patches, window[:, c])
    peak = int(np.argmax(profile))
    half = ZOOM_WIDTH // 2
    zs_ = max(0, min(WIN - ZOOM_WIDTH, peak - half))
    zoom_ranges[c] = (zs_, zs_ + ZOOM_WIDTH)

img_b64 = render_zoomed_subplot_grid(window, ranked, selected, cmin, cmax, zoom_ranges)

import base64
out = BASE / "experiments" / "analysis" / "report24_fig2_k25_pipeline.png"
out.write_bytes(base64.b64decode(img_b64))
print("saved:", out)
