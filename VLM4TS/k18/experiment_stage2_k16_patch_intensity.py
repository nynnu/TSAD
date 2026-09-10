"""
Stage2 K16: K12(calibrated-subplot, 지금까지 최고 F1=0.4731) 위에 DINOv2 patch-KNN의
"패치(=시간구간)별" 이상강도를 시간축 강도 띠(intensity band)로 얹는다.

배경
----
K15(z-score 숫자 + 정상참조윈도우 비교)는 오히려 크게 나빠졌다(0.343) -- VLM이
너무 보수적으로 변해서 recall이 무너졌다. 이번엔 다른 접근: 숫자나 비교 대상을
더 주는 게 아니라, DINOv2가 이미 계산한 "이 채널의 어느 시간대가 제일 이상한지"
위치 정보를 그래프 위에 시각적으로 얹어서 VLM이 스캔할 필요 없이 바로 보게 한다.

knn_patch_score(..., return_win=True)는 sum/topk10으로 뭉개기 전의 (1, 256) 패치별
거리(knn_win)를 준다. 이미지가 224x224/patch14=16x16 패치 그리드이고
ts_to_image_fast가 x축=시간, y축=값으로 그리므로, knn_win을 (16,16)로 reshape해서
각 시간열(column)의 최대 거리(그 열에서 실제 선이 지나간 패치가 제일 값이 큼)를
뽑으면 길이 16짜리 "시간대별 이상강도" 프로파일이 나온다. 이걸 WIN(224)로
보간해서, 각 채널 subplot 라인 아래에 색 강도 띠로 얹는다(빨간색 진할수록
DINOv2가 그 시간대를 이상하다고 봄).

같은 48세그먼트, 같은 hysteresis 채널선택. z-score 텍스트/정상참조 비교는
K15에서 역효과로 확인됐으므로 넣지 않음(K12 기반 위에 강도띠만 추가, 변수 하나만
바꿈).

사용법
------
  python experiment_stage2_k16_patch_intensity.py --stage1
  python experiment_stage2_k16_patch_intensity.py --run
"""
import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[1]  # k18 폴더는 repo 루트 바로 아래(원래는 parents[3])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
import colab_multivariate_v2 as cm
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, N_CHANNELS, OUT_DIR as K4_OUT_DIR,
    constant_channels, compute_zscores, select_channels_hysteresis, f1_of,
    N_POINTS_PER_CHANNEL,
)
import experiment_stage2_v16 as v16
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from smd_3way_baseline_comparison import call_vlm, parse_response

OUT_DIR = BASE / "experiments" / "results_stage2_k16_patch_intensity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_STRICT, ALPHA_LOOSE, CORR_THR = 0.01, 0.1, 0.5
K4_OVERLAY_CHECKPOINT = K4_OUT_DIR / f"checkpoint_hyst_s{ALPHA_STRICT}_l{ALPHA_LOOSE}_c{CORR_THR}.json"
K12_SUMMARY = BASE / "experiments" / "results_stage2_k12_calibrated_subplot" / "summary.json"


def compute_time_profile(tr_cls, tr_patches, window_channel):
    """채널 하나의 test 윈도우에 대해, DINOv2 patch-KNN 거리를 (16,16) 그리드로
    풀어서 시간열(column)별 최댓값을 뽑고 WIN 길이로 보간 -> 0~1 정규화된
    시간대별 이상강도 프로파일(길이 WIN) 반환."""
    test_img = cm.ts_to_image_fast(window_channel)
    te_cls, te_patches = cm.extract_dinov2([test_img], multilayer=False)
    sc = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, return_win=True)
    knn_win = sc["knn_win"][0]  # (256,)
    side = int(np.sqrt(len(knn_win)))
    grid = knn_win.reshape(side, side)
    col_profile = grid.max(axis=0)  # 길이 side(=16), 시간축(x)에 대응
    x_src = np.linspace(0, WIN - 1, side)
    x_dst = np.arange(WIN)
    profile = np.interp(x_dst, x_src, col_profile)
    lo, hi = profile.min(), profile.max()
    return (profile - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(profile)


def render_intensity_subplot_grid(window, ranked, selected, cmin, cmax, profiles, n_cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap

    n_sel = len(selected)
    n_rows = -(-n_sel // n_cols)
    fig = plt.figure(figsize=(6, 3.5 + n_rows * 1.1), dpi=100)
    gs = GridSpec(2, 1, height_ratios=[1.4, n_rows * 0.9], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    heat = np.zeros((len(ranked), window.shape[0]))
    for i, c in enumerate(ranked):
        v = window[:, c]
        lo, hi = float(v.min()), float(v.max())
        heat[i] = (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0
    ax1.imshow(heat, aspect="auto", cmap="viridis")
    ax1.set_yticks(range(len(ranked)))
    ax1.set_yticklabels([str(c) for c in ranked], fontsize=5)
    ax1.set_xticks([])
    ax1.set_title("Heatmap: 38 channels, sorted by adaptive z-score", fontsize=7)

    reds = LinearSegmentedColormap.from_list("reds_alpha", [(1, 1, 1, 0), (1, 0, 0, 0.55)])
    gs_bottom = gs[1].subgridspec(n_rows, n_cols, hspace=0.6, wspace=0.3)
    for i, c in enumerate(selected):
        ax = fig.add_subplot(gs_bottom[i // n_cols, i % n_cols])
        norm_v = v16._n(window[:, c], cmin[c], cmax[c])
        ax.imshow(profiles[c][None, :], aspect="auto", cmap=reds, extent=[0, len(norm_v), -0.3, 1.3],
                  origin="lower", zorder=0)
        ax.plot(norm_v, color="black", linewidth=0.7, zorder=2)
        ax.axhline(1.0, color="blue", linewidth=0.3, linestyle=":", zorder=1)
        ax.axhline(0.0, color="blue", linewidth=0.3, linestyle=":", zorder=1)
        ax.set_ylim(-0.3, 1.3)
        ax.set_title(f"ch{c}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Bottom: {n_sel} candidate channels. Black line = normalized value (blue dotted = training "
                 f"min/max). RED SHADING = DINOv2's own per-timestep anomaly intensity for that channel "
                 f"(darker red = DINOv2 flags that specific time region as more unusual pattern) -- use this "
                 f"to know WHERE to focus, not just whether the line crosses the blue lines.",
                 fontsize=6, y=0.5 - n_rows * 0.02)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_intensity(all_channels, selected, window, train, cmin, cmax):
    blocks = []
    for i, c in enumerate(selected):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        normv = v16._n(v, cmin[c], cmax[c])
        pts = ", ".join(f"({idx}, {normv[idx]:.3f})" for idx in top_idx)
        blocks.append(f"Channel {c} (rank {i+1}), top-{N_POINTS_PER_CHANNEL} most-deviating points "
                       f"(normalized to TRAINING min/max, so >1.0 or <0.0 means it exceeds the training range): {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels, one row per channel (row label = channel number, sorted by an adaptive anomaly z-score, most suspicious at top), color = normalized value over time. Use this for a full overview.

Bottom panel: {len(selected)} candidate channels ({selected}) that an adaptive per-channel threshold flagged as statistically unusual. Each candidate channel is shown in its OWN independent small subplot, all sharing the SAME training-calibrated scale (blue dotted lines = training min/max, so >1.0 or <0.0 means it exceeds the training range). Additionally, each panel has a RED SHADED BACKGROUND showing DINOv2's own per-timestep anomaly-intensity estimate for that channel -- darker red at a given time position means a separate learned model (not just the raw value) independently flagged that specific moment as having an unusual local shape/pattern. Use the red shading to know WHERE in the window to focus your attention, and combine it with whether the black line actually crosses outside the blue training-range lines at that same location.

For each of the {len(selected)} candidate channels, here are the (time index, normalized value) points that deviate most strongly from that channel's normal (training) range:

{history_text}

Use the panels (including the red intensity shading) and this point data together to identify which of these {len(selected)} candidate channels show genuinely anomalous behavior in this window (you may judge that ALL or only SOME of them are truly anomalous). No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {selected} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run(execute=False):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    overlay_ckpt = json.loads(K4_OVERLAY_CHECKPOINT.read_text(encoding="utf-8")) if K4_OVERLAY_CHECKPOINT.exists() else {}
    k12_summary = json.loads(K12_SUMMARY.read_text(encoding="utf-8")) if K12_SUMMARY.exists() else None
    k12_f1 = {r["seg_id"]: r["f1_calibrated_subplot"] for r in k12_summary["rows"]} if k12_summary else {}
    print(f"세그먼트 수 = {len(segments)}, overlay 조건은 K4 캐시 재사용(신규 콜 0개), "
          f"patch-intensity만 신규 1콜씩")

    all_channels = list(range(N_CHANNELS))
    checkpoint_path = OUT_DIR / "checkpoint_intensity.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}

    entity_data, entity_channel_calib, entity_degenerate, entity_corr = {}, {}, {}, {}
    rows = []

    for seg in segments:
        entity, cs, ce = seg["entity"], seg["start"], seg["end"]
        gt = set(d - 1 for d in seg["dims"])
        seg_id = f"{entity}_{cs}_{ce}"

        overlay_entry = overlay_ckpt.get(seg_id)
        if overlay_entry is None or overlay_entry.get("status") != "OK":
            print(f"  [SKIP] {seg_id}: K4 캐시에 없음(페어 불가)")
            continue

        if entity not in entity_data:
            train, test = load_smd(entity)
            entity_data[entity] = (train, test)
        train, test = entity_data[entity]
        if entity not in entity_degenerate:
            entity_degenerate[entity] = constant_channels(train)
        if entity not in entity_corr:
            corr = np.corrcoef(train.T)
            entity_corr[entity] = np.nan_to_num(corr, nan=0.0)
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        t0 = time.time()
        zs = compute_zscores(entity, train, window, entity_channel_calib, entity_degenerate[entity])
        ranked, selected = select_channels_hysteresis(zs, entity_corr[entity], ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
        cmin, cmax = v16.gn_train(train, range(N_CHANNELS))

        profiles = {}
        for c in selected:
            tr_cls, tr_patches, _ = entity_channel_calib[(entity, c)]
            profiles[c] = compute_time_profile(tr_cls, tr_patches, window[:, c])

        print(f"  {seg_id}: k(GT)={len(gt)} selected={len(selected)}개 (patch-intensity, 1콜) "
              f"({time.time()-t0:.1f}s)", flush=True)

        if not execute:
            continue

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_intensity_subplot_grid(window, ranked, selected, cmin, cmax, profiles)
            prompt = build_prompt_intensity(all_channels, selected, window, train, cmin, cmax)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    [{checkpoint[seg_id]['status']}] pred={pred}", flush=True)

        if pred is None:
            continue
        f1_overlay_all = f1_of(overlay_entry["pred"], gt)
        f1_intensity = f1_of(pred, gt)
        rows.append({"seg_id": seg_id, "k": len(gt), "n_selected": len(selected),
                     "f1_overlay_all": f1_overlay_all, "f1_calibrated_subplot": k12_f1.get(seg_id),
                     "f1_intensity": f1_intensity})

    if execute and rows:
        f1_all = np.array([r["f1_overlay_all"] for r in rows])
        f1_i = np.array([r["f1_intensity"] for r in rows])
        diff = f1_i - f1_all
        stat, p = wilcoxon(f1_i, f1_all) if np.any(diff != 0) else (0.0, 1.0)
        print(f"\nn={len(rows)}")
        print(f"overlay-전체(K4)     평균 F1 = {f1_all.mean():.4f}")
        print(f"patch-intensity(신규) 평균 F1 = {f1_i.mean():.4f}")
        print(f"차이(intensity-overlay) = {diff.mean():+.4f}, paired Wilcoxon p = {p:.4f}")
        if all(r["f1_calibrated_subplot"] is not None for r in rows):
            f1_c12 = np.array([r["f1_calibrated_subplot"] for r in rows])
            stat2, p2 = wilcoxon(f1_i, f1_c12) if np.any(f1_i - f1_c12 != 0) else (0.0, 1.0)
            print(f"calibrated-subplot(K12, 참고) 평균 F1 = {f1_c12.mean():.4f} (intensity 대비 p={p2:.4f})")
        (OUT_DIR / "summary.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    run(execute=args.run)
