"""
Stage2 K26: RATFM 스타일 -- K18(zoomed-subplot, F1=0.5033) 이미지는 그대로 두고,
각 후보 채널 패널 옆에 "train bank에서 검색한, 모양이 제일 비슷한 정상(normal)
윈도우"를 나란히 보여준다. VLM이 "정상은 이렇게 생겼다"는 기준점 없이 후보만 보고
판단하던 것 -- self-consistency(K25, 부정적 결과 확정: 오류가 무작위 노이즈가
아니라 일관된 오판이었음)로는 못 고치는 문제를, 애초에 비교 기준을 줘서 풀 수
있는지 테스트.

배경
----
RATFM(arXiv:2506.02081): "판단할 구간에 그 도메인의 정상 참조 예시를 검색해서
같이 주면, 사전지식 없이도 이상 구분이 쉬워진다" -- zero-shot 유지, retrieval만
추가. K25에서 밝혀진 진짜 문제(모델이 5번 다시 물어도 매번 똑같이 틀림 --
agreement=1.00인데 f1=0.0)는 "판단 기준점 자체가 없어서 생기는 체계적 오판"일
가능성이 있고, 이건 참조 예시로 기준점을 주면 고칠 수 있는 종류의 문제.

같은 48세그먼트, 같은 hysteresis 채널선택, 같은 zoom 위치(K18과 동일). 유일한
추가: 채널마다 정상 참조 패널 1개.

참조 검색 방법
--------------
채널별 train bank(60개 윈도우, K4의 calib 캐시가 이미 갖고 있음)에서, 후보의
FULL window(224틱)와 정규화(z-score 아님, train min/max 기준 0~1) L2 거리가
가장 작은 bank 윈도우 1개를 "가장 비슷한 정상 예시"로 채택 -- DINOv2 재호출
없이 raw 값만으로 계산(공짜). 그 참조 윈도우도 후보와 동일한 zoom 구간
[zs_, ze_]으로 잘라서 같은 스케일로 나란히 그린다(시각적으로 직접 비교 가능).

사용법
------
  python experiment_stage2_k26_reference_zoomed.py --stage1
  python experiment_stage2_k26_reference_zoomed.py --smoke   # n=12
  python experiment_stage2_k26_reference_zoomed.py --run     # 전체 48
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

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
from experiment_stage2_k4_adaptive import (
    GT_SEGMENTS_PATH, N_CHANNELS, OUT_DIR as K4_OUT_DIR,
    constant_channels, compute_zscores, select_channels_hysteresis, f1_of,
    N_POINTS_PER_CHANNEL, _bank_and_calib_idx,
)
from experiment_stage2_k16_patch_intensity import compute_time_profile
from experiment_stage2_k18_zoomed_subplot import ZOOM_WIDTH
import experiment_stage2_v16 as v16
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from smd_3way_baseline_comparison import call_vlm, parse_response
import colab_multivariate_v2 as cm

OUT_DIR = BASE / "experiments" / "results_stage2_k26_reference_zoomed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_STRICT, ALPHA_LOOSE, CORR_THR = 0.01, 0.1, 0.5
K4_OVERLAY_CHECKPOINT = K4_OUT_DIR / f"checkpoint_hyst_s{ALPHA_STRICT}_l{ALPHA_LOOSE}_c{CORR_THR}.json"
K18_SUMMARY = BASE / "experiments" / "results_stage2_k18_zoomed_subplot" / "summary.json"


def find_nearest_reference(cand_full, bank_windows, cmin_c, cmax_c):
    """cand_full(224,) 정규화 후, bank_windows(각 224,) 중 L2 거리 최소인 것을 반환."""
    cand_n = v16._n(cand_full, cmin_c, cmax_c)
    best_i, best_d = None, None
    for i, w in enumerate(bank_windows):
        wn = v16._n(w, cmin_c, cmax_c)
        d = float(np.sum((wn - cand_n) ** 2))
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return bank_windows[best_i]


def render_reference_zoomed_grid(window, ranked, selected, cmin, cmax, zoom_ranges, ref_segs, n_per_row=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_sel = len(selected)
    n_rows = -(-n_sel // n_per_row)
    n_cols = n_per_row * 2  # 채널마다 [ref | candidate] 2칸
    fig = plt.figure(figsize=(8, 3.5 + n_rows * 1.1), dpi=100)
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

    gs_bottom = gs[1].subgridspec(n_rows, n_cols, hspace=0.6, wspace=0.4)
    for i, c in enumerate(selected):
        row, col_pair = i // n_per_row, (i % n_per_row) * 2
        zs_, ze_ = zoom_ranges[c]

        ax_ref = fig.add_subplot(gs_bottom[row, col_pair])
        ref_seg = ref_segs[c]
        ref_n = v16._n(ref_seg, cmin[c], cmax[c])
        ax_ref.plot(np.arange(zs_, ze_), ref_n, color="tab:blue", linewidth=0.9)
        ax_ref.axhline(1.0, color="red", linewidth=0.4, linestyle="--")
        ax_ref.axhline(0.0, color="red", linewidth=0.4, linestyle="--")
        ax_ref.set_ylim(-0.3, 1.3)
        ax_ref.set_title(f"ch{c} NORMAL ref", fontsize=5.5)
        ax_ref.set_xticks([]); ax_ref.set_yticks([])

        ax_cand = fig.add_subplot(gs_bottom[row, col_pair + 1])
        seg = window[zs_:ze_, c]
        norm_v = v16._n(seg, cmin[c], cmax[c])
        ax_cand.plot(np.arange(zs_, ze_), norm_v, color="black", linewidth=0.9)
        ax_cand.axhline(1.0, color="red", linewidth=0.4, linestyle="--")
        ax_cand.axhline(0.0, color="red", linewidth=0.4, linestyle="--")
        ax_cand.set_ylim(-0.3, 1.3)
        ax_cand.set_title(f"ch{c} CANDIDATE [{zs_}-{ze_}]", fontsize=5.5)
        ax_cand.set_xticks([]); ax_cand.set_yticks([])

    fig.suptitle(f"Bottom: {n_sel} candidate channels, each as a pair -- LEFT (blue) = a retrieved NORMAL "
                 f"reference window from training (same channel, same zoom range, nearest-shape match), "
                 f"RIGHT (black) = the actual candidate. Same calibrated scale (red dashed = training min/max). "
                 f"Compare shape/level directly against its own reference.",
                 fontsize=6, y=0.5 - n_rows * 0.02)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_prompt_reference(all_channels, selected, window, train, cmin, cmax, zoom_ranges):
    blocks = []
    for i, c in enumerate(selected):
        v = window[:, c]
        mu, sigma = float(train[:, c].mean()), float(train[:, c].std())
        z = np.abs((v - mu) / sigma) if sigma > 1e-9 else np.zeros_like(v)
        top_idx = np.sort(np.argsort(-z)[:N_POINTS_PER_CHANNEL])
        normv = v16._n(v, cmin[c], cmax[c])
        pts = ", ".join(f"({idx}, {normv[idx]:.3f})" for idx in top_idx)
        zs_, ze_ = zoom_ranges[c]
        blocks.append(f"Channel {c} (rank {i+1}), zoomed to ticks [{zs_}-{ze_}], top-{N_POINTS_PER_CHANNEL} "
                       f"most-deviating points across the FULL window (normalized to TRAINING min/max): {pts}")
    history_text = "\n".join(blocks)

    return f"""You are shown a composite image with two panels for a multivariate industrial system with {len(all_channels)} channels (numbered {all_channels}).

Top panel: a heatmap overview of ALL {len(all_channels)} channels over the FULL window. Use this for overall context.

Bottom panel: {len(selected)} candidate channels ({selected}) that an adaptive per-channel threshold flagged as statistically unusual. Each candidate is shown as a PAIR of small subplots side by side: the LEFT one (blue line, titled "NORMAL ref") is a real example of what this SAME channel looks like during confirmed-normal training data, retrieved because its shape is the closest match to the candidate -- this is your baseline for "what normal looks like" for that specific channel. The RIGHT one (black line, titled "CANDIDATE") is the actual window you must judge. Both are zoomed to the same tick range and plotted on the same training-calibrated scale (red dashed lines = training min/max), so they are directly comparable.

For each of the {len(selected)} candidate channels, here is the zoom range plus the most-deviating points across the full window:

{history_text}

Compare each candidate panel DIRECTLY against its own normal-reference panel (not just against the red dashed training range) -- look for a genuine shape/level difference between the two, not just whether the candidate looks "busy". Identify which of these {len(selected)} candidate channels show genuinely anomalous behavior (you may judge that ALL or only SOME of them are truly anomalous). No ground truth or hints are given.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"anomalous_channels": [list of channel numbers from {selected} that you judge anomalous], "confidence": "low" or "medium" or "high"}}"""


def render_example(example_index=0):
    """Render one genuine K26 VLM input without making a VLM call."""
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    if not 0 <= example_index < len(segments):
        raise ValueError(f"example index must be in [0, {len(segments) - 1}]")
    seg = segments[example_index]
    entity, cs, ce = seg["entity"], seg["start"], seg["end"]
    seg_id = f"{entity}_{cs}_{ce}"
    train, test = load_smd(entity)
    start, end = _centered_window(len(test), (cs + ce) // 2, WIN)
    window = test[start:end]
    degenerate = constant_channels(train)
    calib = {}
    zscores = compute_zscores(entity, train, window, calib, degenerate)
    corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
    ranked, selected = select_channels_hysteresis(zscores, corr, ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
    cmin, cmax = v16.gn_train(train, range(N_CHANNELS))

    bank_cache, zoom_ranges, ref_segs = {}, {}, {}
    for c in selected:
        tr_cls, tr_patches, _ = calib[(entity, c)]
        peak = int(np.argmax(compute_time_profile(tr_cls, tr_patches, window[:, c])))
        zoom_start = max(0, min(WIN - ZOOM_WIDTH, peak - ZOOM_WIDTH // 2))
        zoom_ranges[c] = (zoom_start, zoom_start + ZOOM_WIDTH)
        windows = cm.get_windows(train[:, c])
        bank_idx, _ = _bank_and_calib_idx(len(windows))
        bank_cache[c] = [windows[i] for i in bank_idx]
        reference = find_nearest_reference(window[:, c], bank_cache[c], cmin[c], cmax[c])
        ref_segs[c] = reference[zoom_start:zoom_start + ZOOM_WIDTH]

    encoded = render_reference_zoomed_grid(window, ranked, selected, cmin, cmax, zoom_ranges, ref_segs)
    output = OUT_DIR / f"example_reference_zoomed_{example_index:02d}_{seg_id}.png"
    output.write_bytes(base64.b64decode(encoded))
    print(f"{seg_id}: selected={selected}")
    print(f"Rendered K26 reference-zoomed input: {output}")


def run(n_limit=None, execute=False):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    if n_limit:
        segments = segments[:n_limit]
    overlay_ckpt = json.loads(K4_OVERLAY_CHECKPOINT.read_text(encoding="utf-8")) if K4_OVERLAY_CHECKPOINT.exists() else {}
    k18_summary = json.loads(K18_SUMMARY.read_text(encoding="utf-8")) if K18_SUMMARY.exists() else None
    k18_f1 = {r["seg_id"]: r["f1_zoomed"] for r in k18_summary["rows"]} if k18_summary else {}
    print(f"세그먼트 수 = {len(segments)}, overlay 조건은 K4 캐시 재사용(신규 콜 0개), "
          f"reference-zoomed만 신규 1콜씩")

    all_channels = list(range(N_CHANNELS))
    checkpoint_path = OUT_DIR / "checkpoint_ref.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}

    entity_data, entity_channel_calib, entity_degenerate, entity_corr, entity_bank_windows = {}, {}, {}, {}, {}
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
        if entity not in entity_bank_windows:
            entity_bank_windows[entity] = {}
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        t0 = time.time()
        zs = compute_zscores(entity, train, window, entity_channel_calib, entity_degenerate[entity])
        ranked, selected = select_channels_hysteresis(zs, entity_corr[entity], ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
        cmin, cmax = v16.gn_train(train, range(N_CHANNELS))

        zoom_ranges, ref_segs = {}, {}
        for c in selected:
            tr_cls, tr_patches, _ = entity_channel_calib[(entity, c)]
            profile = compute_time_profile(tr_cls, tr_patches, window[:, c])
            peak = int(np.argmax(profile))
            half = ZOOM_WIDTH // 2
            zs_ = max(0, min(WIN - ZOOM_WIDTH, peak - half))
            zoom_ranges[c] = (zs_, zs_ + ZOOM_WIDTH)

            if c not in entity_bank_windows[entity]:
                w_all = cm.get_windows(train[:, c])
                bank_idx, _ = _bank_and_calib_idx(len(w_all))
                entity_bank_windows[entity][c] = [w_all[i] for i in bank_idx]
            ref_full = find_nearest_reference(window[:, c], entity_bank_windows[entity][c], cmin[c], cmax[c])
            ref_segs[c] = ref_full[zs_:zs_ + ZOOM_WIDTH]

        print(f"  {seg_id}: k(GT)={len(gt)} selected={len(selected)}개 (reference-zoomed, 1콜) "
              f"({time.time()-t0:.1f}s)", flush=True)

        if not execute:
            continue

        if checkpoint.get(seg_id, {}).get("status") == "OK":
            pred = checkpoint[seg_id]["pred"]
        else:
            img = render_reference_zoomed_grid(window, ranked, selected, cmin, cmax, zoom_ranges, ref_segs)
            prompt = build_prompt_reference(all_channels, selected, window, train, cmin, cmax, zoom_ranges)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[seg_id] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    [{checkpoint[seg_id]['status']}] pred={pred}", flush=True)

        if pred is None:
            continue
        f1_overlay_all = f1_of(overlay_entry["pred"], gt)
        f1_ref = f1_of(pred, gt)
        rows.append({"seg_id": seg_id, "k": len(gt), "n_selected": len(selected),
                     "f1_overlay_all": f1_overlay_all, "f1_zoomed_direct": k18_f1.get(seg_id), "f1_reference": f1_ref})

    if execute and rows:
        f1_all = np.array([r["f1_overlay_all"] for r in rows])
        f1_r = np.array([r["f1_reference"] for r in rows])
        diff = f1_r - f1_all
        stat, p = wilcoxon(f1_r, f1_all) if np.any(diff != 0) else (0.0, 1.0)
        print(f"\nn={len(rows)}")
        print(f"overlay-전체(K4)      평균 F1 = {f1_all.mean():.4f}")
        print(f"reference-zoomed(신규) 평균 F1 = {f1_r.mean():.4f}")
        print(f"차이(reference-overlay) = {diff.mean():+.4f}, paired Wilcoxon p = {p:.4f}")
        if all(r["f1_zoomed_direct"] is not None for r in rows):
            f1_18 = np.array([r["f1_zoomed_direct"] for r in rows])
            stat2, p2 = wilcoxon(f1_r, f1_18) if np.any(f1_r - f1_18 != 0) else (0.0, 1.0)
            print(f"zoomed-direct(K18, 참조 없음) 평균 F1 = {f1_18.mean():.4f} (reference 대비 diff={float((f1_r-f1_18).mean()):+.4f}, p={p2:.4f})")
        (OUT_DIR / "summary.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2), encoding="utf-8")
        print(f"저장: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="n=12로 소규모 방향성 확인 (VLM 실행)")
    ap.add_argument("--run", action="store_true", help="전체 48세그먼트 VLM 실행")
    ap.add_argument("--render-example", action="store_true", help="render one K26 input image; no VLM call")
    ap.add_argument("--example-index", type=int, default=0, help="segment index for --render-example")
    args = ap.parse_args()
    if args.render_example:
        render_example(args.example_index)
    elif args.smoke:
        run(n_limit=12, execute=True)
    elif args.run:
        run(n_limit=None, execute=True)
    elif args.stage1:
        run(n_limit=3, execute=False)
    else:
        run(n_limit=None, execute=False)
