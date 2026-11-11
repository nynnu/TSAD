"""
발표자료용 실측 스크립트: DINOv2 1회 추론 시간, VLM 1회 호출 시간/토큰 수.
기존 결과 JSON엔 타이밍이 안 남아있어서 직접 측정.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, N_CHANNELS, constant_channels, compute_zscores, select_channels_hysteresis,
    _bank_and_calib_idx,
)
import experiment_stage2_v16 as v16
from experiment_stage2_k16_patch_intensity import compute_time_profile
from experiment_stage2_k18_zoomed_subplot import ZOOM_WIDTH
from experiment_stage2_k26_reference_zoomed import (
    render_reference_zoomed_grid, build_prompt_reference, find_nearest_reference,
)

cm.DEVICE = torch.device("cpu")
print(f"DEVICE = {cm.DEVICE}")

# === 1. DINOv2 단일 추론 시간 ===
print("\n=== DINOv2 1회 추론(224x224, 배치=1) ===")
model = cm.get_model()
dummy_img = cm.ts_to_image_fast(np.random.randn(224))
batch = torch.stack([cm.dinov2_transform(dummy_img)]).to(cm.DEVICE)

with torch.no_grad():
    out = model.forward_features(batch)  # warmup

times = []
with torch.no_grad():
    for _ in range(10):
        t0 = time.time()
        out = model.forward_features(batch)
        times.append(time.time() - t0)
print(f"10회 평균 = {np.mean(times)*1000:.1f}ms, 표준편차 = {np.std(times)*1000:.1f}ms")
print(f"patch_tokens shape = {tuple(out['x_norm_patchtokens'].shape)}")
print(f"cls_token shape = {tuple(out['x_norm_clstoken'].shape)}")

# === 2. Stage1 실측: 세그먼트 1개, 38채널 전체(은행 캐시 없이 처음부터) ===
print("\n=== Stage1: 세그먼트 1개, 38채널 전체 스코어링(뱅크는 디스크 캐시 재사용, 최초 구축 비용 미포함) ===")
entity = "machine-1-4"
train, test = load_smd(entity)
degenerate = constant_channels(train)
segs = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
seg = next(s for s in segs if s["entity"] == entity)
cs, ce = seg["start"], seg["end"]
center = (cs + ce) // 2
s_, e_ = _centered_window(len(test), center, WIN)
window = test[s_:e_]
entity_channel_calib = {}

t0 = time.time()
zs = compute_zscores(entity, train, window, entity_channel_calib, degenerate)
t_stage1 = time.time() - t0
n_valid = sum(1 for c in range(N_CHANNELS) if c not in degenerate)
print(f"38채널({n_valid}개 유효) 전체 소요 = {t_stage1:.1f}초 (채널당 평균 {t_stage1/n_valid*1000:.0f}ms)")

corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
ranked, selected = select_channels_hysteresis(zs, corr)
print(f"선택된 후보 채널 수 = {len(selected)}")

cmin, cmax = v16.gn_train(train, range(N_CHANNELS))
zoom_ranges, ref_segs = {}, {}
bank_windows_cache = {}
for c in selected:
    tr_cls, tr_patches, _ = entity_channel_calib[(entity, c)]
    profile = compute_time_profile(tr_cls, tr_patches, window[:, c])
    peak = int(np.argmax(profile))
    half = ZOOM_WIDTH // 2
    zs_ = max(0, min(WIN - ZOOM_WIDTH, peak - half))
    zoom_ranges[c] = (zs_, zs_ + ZOOM_WIDTH)
    if c not in bank_windows_cache:
        w_all = cm.get_windows(train[:, c])
        bank_idx, _ = _bank_and_calib_idx(len(w_all))
        bank_windows_cache[c] = [w_all[i] for i in bank_idx]
    ref_full = find_nearest_reference(window[:, c], bank_windows_cache[c], cmin[c], cmax[c])
    ref_segs[c] = ref_full[zs_:zs_ + ZOOM_WIDTH]

img_b64 = render_reference_zoomed_grid(window, ranked, selected, cmin, cmax, zoom_ranges, ref_segs)
prompt = build_prompt_reference(list(range(N_CHANNELS)), selected, window, train, cmin, cmax, zoom_ranges)
print(f"\n이미지 크기(base64) = {len(img_b64)} bytes, 프롬프트 글자수 = {len(prompt)}")

# === 3. VLM 실제 콜 1회 - 시간/토큰 수 측정 ===
print("\n=== VLM 실제 1회 호출 (gpt-4o) ===")
from smd_3way_baseline_comparison import _client, MODEL_NAME

t0 = time.time()
resp = _client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
    ]}],
    temperature=0.0, max_tokens=1000,
)
t_vlm = time.time() - t0
print(f"VLM 1회 호출 소요 = {t_vlm:.1f}초")
print(f"prompt_tokens = {resp.usage.prompt_tokens}, completion_tokens = {resp.usage.completion_tokens}, "
      f"total_tokens = {resp.usage.total_tokens}")
print(f"응답 앞부분: {resp.choices[0].message.content[:200]}")
