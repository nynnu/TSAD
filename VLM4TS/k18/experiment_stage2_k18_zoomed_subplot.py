"""
Stage2 K18: K16(패치 단위 이상강도)에서 계산한 "이 채널의 어느 시간대가 제일
이상한지" 위치 정보를 색으로 살짝 칠하는 대신, 그 구간을 실제로 확대(zoom)해서
보여준다.

배경
----
지금까지 subplot 칸(6열 격자)은 WIN=224틱 전체를 작은 칸 하나에 욱여넣었다 --
진짜 이상 구간이 몇 픽셀짜리 미세한 삐죽임으로 뭉개졌을 수 있다. K16(색으로 강도
표시)은 위치 정보를 "옅게 칠하는" 수준이라 도움이 안 됐는데(K12와 동률), 이번엔
그 위치를 실제로 크롭해서 화면을 꽉 채우게 확대한다 -- 위치 정보를 "암시"가 아니라
"확대"로 활용.

상단 heatmap 패널은 여전히 WIN 전체(맥락 유지), 하단 각 채널 detail 패널만
peak 위치 중심 ZOOM_WIDTH(기본 60틱, 전체의 1/4 이하)로 크롭해서 확대.
같은 calibrated(train 고정) 정규화 유지.

같은 48세그먼트, 같은 hysteresis 채널선택.

사용법
------
  python experiment_stage2_k18_zoomed_subplot.py --stage1
  python experiment_stage2_k18_zoomed_subplot.py --run
"""
import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[1]  # k18 폴더는 repo 루트 바로 아래(원래는 parents[3])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, N_CHANNELS, OUT_DIR as K4_OUT_DIR,
    constant_channels, compute_zscores, select_channels_hysteresis, f1_of,
    N_POINTS_PER_CHANNEL,
)
from experiment_stage2_k16_patch_intensity import compute_time_profile
import experiment_stage2_v16 as v16
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from smd_3way_baseline_comparison import call_vlm, parse_response

OUT_DIR = BASE / "experiments" / "results_stage2_k18_zoomed_subplot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_STRICT, ALPHA_LOOSE, CORR_THR = 0.01, 0.1, 0.5
ZOOM_WIDTH = 60
K4_OVERLAY_CHECKPOINT = K4_OUT_DIR / f"checkpoint_hyst_s{ALPHA_STRICT}_l{ALPHA_LOOSE}_c{CORR_THR}.json"
K12_SUMMARY = BASE / "experiments" / "results_stage2_k12_calibrated_subplot" / "summary.json"


def render_zoomed_subplot_grid(window, ranked, selected, cmin, cmax, zoom_ranges, n_cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

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
    ax1.set_title(f"Heatmap: 38 channels, FULL {window.shape[0]}-tick window, sorted by adaptive z-score", fontsize=7)

    gs_bottom = gs[1].subgridspec(n_rows, n_cols, hspace=0.6, wspace=0.3)
    for i, c in enumerate(selected):
        ax = fig.add_subplot(gs_bottom[i // n_cols, i % n_cols])
        zs_, ze_ = zoom_ranges[c]
        seg = window[zs_:ze_, c]
        norm_v = v16._n(seg, cmin[c], cmax[c])
        ax.plot(np.arange(zs_, ze_), norm_v, color="black", linewidth=0.9)
        ax.axhline(1.0, color="red", linewidth=0.4, linestyle="--")
        ax.axhline(0.0, color="red", linewidth=0.4, linestyle="--")
        ax.set_ylim(-0.3, 1.3)
        ax.set_title(f"ch{c} [zoom {zs_}-{ze_}]", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Bottom: {n_sel} candidate channels, each panel ZOOMED IN to the {ZOOM_WIDTH}-tick sub-region "
                 f"where DINOv2 detected the strongest local pattern anomaly for that channel (out of the full "
                 f"{window.shape[0]}-tick window shown in the heatmap above) -- red dashed = training min/max, "
                 f"same calibrated scale as before.",
                 fontsize=6, y=0.5 - n_rows * 0.02)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_zoomed(all_channels, selected, window, train, cmin, cmax, zoom_ranges):
    blocks = []
    for i, c in enumerate(selected):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        normv = v16._n(v, cmin[c], cmax[c])
        pts = ", ".join(f"({idx}, {normv[idx]:.3f})" for idx in top_idx)
        zs_, ze_ = zoom_ranges[c]
        blocks.append(f"Channel {c} (rank {i+1}), image panel zoomed into ticks [{zs_}-{ze_}] "
                       f"(DINOv2's detected peak-anomaly sub-region), top-{N_POINTS_PER_CHANNEL} most-deviating "
                       f"points across the FULL window (normalized to TRAINING min/max, so >1.0 or <0.0 means "
                       f"it exceeds the training range): {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels over the FULL window, one row per channel, color = normalized value over time. Use this for overall context.

Bottom panel: {len(selected)} candidate channels ({selected}) that an adaptive per-channel threshold flagged as statistically unusual. Each is shown in its OWN independent small subplot, but IMPORTANTLY each panel is ZOOMED IN to a {ZOOM_WIDTH}-tick sub-region (labeled in the title as "[zoom start-end]") -- specifically the sub-region where a separate DINOv2 model detected the strongest local pattern anomaly for THAT channel. This means what you see filling each panel is already the most suspicious part of that channel's full window, magnified for clarity. All panels share the SAME training-calibrated scale (red dashed lines = training min/max).

For each of the {len(selected)} candidate channels, here is which sub-region is shown plus the most-deviating points across the full window:

{history_text}

Use the zoomed panels and this point data together to identify which of these {len(selected)} candidate channels show genuinely anomalous behavior (you may judge that ALL or only SOME of them are truly anomalous). No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {selected} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def run(execute=False):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    overlay_ckpt = json.loads(K4_OVERLAY_CHECKPOINT.read_text(encoding="utf-8")) if K4_OVERLAY_CHECKPOINT.exists() else {}
    k12_summary = json.loads(K12_SUMMARY.read_text(encoding="utf-8")) if K12_SUMMARY.exists() else None
    k12_f1 = {r["seg_id"]: r["f1_calibrated_subplot"] for r in k12_summary["rows"]} if k12_summary else {}
    print(f"세그먼트 수 = {len(segments)}, overlay 조건은 K4 캐시 재사용(신규 콜 0개), "
          f"zoomed-subplot만 신규 1콜씩")

    all_channels = list(range(N_CHANNELS))
    checkpoint_path = OUT_DIR / "checkpoint_zoomed.json"
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

        zoom_ranges = {}
        for c in selected:
            tr_cls, tr_patches, _ = entity_channel_calib[(entity, c)]
            profile = compute_time_profile(tr_cls, tr_patches, window[:, c])
            peak = int(np.argmax(profile))
            half = ZOOM_WIDTH // 2
            zs_ = max(0, min(WIN - ZOOM_WIDTH, peak - half))
            ze_ = zs_ + ZOOM_WIDTH
            zoom_ranges[c] = (zs_, ze_)

        print(f"  {seg_id}: k(GT)={len(gt)} selected={len(selected)}개 (zoomed, 1콜) ({time.time()-t0:.1f}s)", flush=True)

        if not execute:
            continue

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_zoomed_subplot_grid(window, ranked, selected, cmin, cmax, zoom_ranges)
            prompt = build_prompt_zoomed(all_channels, selected, window, train, cmin, cmax, zoom_ranges)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    [{checkpoint[seg_id]['status']}] pred={pred}", flush=True)

        if pred is None:
            continue
        f1_overlay_all = f1_of(overlay_entry["pred"], gt)
        f1_zoomed = f1_of(pred, gt)
        rows.append({"seg_id": seg_id, "k": len(gt), "n_selected": len(selected),
                     "f1_overlay_all": f1_overlay_all, "f1_calibrated_subplot": k12_f1.get(seg_id),
                     "f1_zoomed": f1_zoomed})

    if execute and rows:
        f1_all = np.array([r["f1_overlay_all"] for r in rows])
        f1_z = np.array([r["f1_zoomed"] for r in rows])
        diff = f1_z - f1_all
        stat, p = wilcoxon(f1_z, f1_all) if np.any(diff != 0) else (0.0, 1.0)
        print(f"\nn={len(rows)}")
        print(f"overlay-전체(K4)   평균 F1 = {f1_all.mean():.4f}")
        print(f"zoomed(신규)       평균 F1 = {f1_z.mean():.4f}")
        print(f"차이(zoomed-overlay) = {diff.mean():+.4f}, paired Wilcoxon p = {p:.4f}")
        if all(r["f1_calibrated_subplot"] is not None for r in rows):
            f1_c12 = np.array([r["f1_calibrated_subplot"] for r in rows])
            stat2, p2 = wilcoxon(f1_z, f1_c12) if np.any(f1_z - f1_c12 != 0) else (0.0, 1.0)
            print(f"calibrated-subplot(K12, 참고) 평균 F1 = {f1_c12.mean():.4f} (zoomed 대비 p={p2:.4f})")
        (OUT_DIR / "summary.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    run(execute=args.run)
