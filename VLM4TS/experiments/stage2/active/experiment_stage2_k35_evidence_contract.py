"""K35: Stage-1/Stage-2 evidence-contract ablation.

Stage 1 is a train-only, robustly calibrated *screening* module.  Stage 2 must
not silently redo candidate selection: it validates each Stage-1 proposal with
the same train-normalized scale and may reject every proposal.  Three visual
conditions hold selection, DINO localization, VLM, prompt schema and paging
constant:
  HZ   global heatmap + per-candidate local focus (K18-like)
  HF   global heatmap + aligned candidate full traces
  HFZ  global heatmap + aligned full traces + one local focus per candidate

Unlike K34, every rendered panel follows Stage 1's train-only wide-quantile normalization.
Normal-reference plots are deliberately excluded: they were not beneficial in
K33/K34 smoke tests and introduce a retrieval-mismatch failure mode.

Usage:
  python experiment_stage2_k35_evidence_contract.py --stage1 --smoke
  python experiment_stage2_k35_evidence_contract.py --run --smoke
  python experiment_stage2_k35_evidence_contract.py --run --confirm  # held-out 36
  python experiment_stage2_k35_evidence_contract.py --run
"""
import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))

import colab_multivariate_v2 as cm
from experiment_stage2_k4_adaptive import (
    CACHE_VERSION, GT_SEGMENTS_PATH, N_CHANNELS, constant_channels,
    _train_image_range, compute_zscores, f1_of, select_channels_hysteresis,
)
from smd_3way_baseline_comparison import call_vlm, parse_response
from step1v3_dino_graph_smd import WIN, _centered_window, load_smd

OUT_DIR = BASE / "experiments" / "results_stage2_k35_evidence_contract"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HFZR is K40: a controlled K33-style reference-band ablation on top of HFZ.
# It remains optional because the final pipeline is HFZ.
LAYOUTS = ("HZ", "HF", "HFZ")
ALPHA_STRICT, ALPHA_LOOSE, CORR_THR = 0.01, 0.1, 0.5
ZOOM_WIDTH = 60
MAX_CANDIDATES_PER_PAGE = 8


def normalize_train(values, vmin, vmax):
    return (values - vmin) / (vmax - vmin + 1e-8)


def fixed_time_profile(tr_cls, tr_patches, window_channel, stats):
    """DINO localization using exactly the fixed range used for Stage-1 scoring."""
    image = cm.ts_to_image_fast(window_channel, vmin=stats["vmin"], vmax=stats["vmax"])
    te_cls, te_patches = cm.extract_dinov2([image], multilayer=False)
    score = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, return_win=True)
    patch_scores = score["knn_win"][0]
    side = int(np.sqrt(len(patch_scores)))
    if side * side != len(patch_scores):
        raise ValueError(f"Expected square DINO patch grid, got {len(patch_scores)} patches.")
    profile = patch_scores.reshape(side, side).max(axis=0)
    return np.interp(np.arange(WIN), np.linspace(0, WIN - 1, side), profile)


def focus_interval(profile):
    peak = int(np.argmax(profile))
    start = max(0, min(WIN - ZOOM_WIDTH, peak - ZOOM_WIDTH // 2))
    return start, start + ZOOM_WIDTH


def render_heatmap(ax, window, ranked, selected, ranges):
    matrix = np.asarray([
        np.clip(normalize_train(window[:, c], *ranges[c]), -0.5, 1.5)
        for c in ranked
    ])
    ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=-0.5, vmax=1.5)
    selected_set = set(selected)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([f"ch {c}" if c in selected_set else str(c) for c in ranked], fontsize=5)
    for label, c in zip(ax.get_yticklabels(), ranked):
        if c in selected_set:
            label.set_color("crimson")
            label.set_fontweight("bold")
    ax.set_xlim(0, WIN - 1)
    ax.set_xlabel("time within common 224-tick window", fontsize=7)
    ax.set_title("All 38 channels on Stage-1 train-normalized scale; red labels = nominated", fontsize=8)


def draw_full(ax, values, channel, interval):
    ax.plot(np.arange(WIN), values, color="black", linewidth=1.0)
    ax.axvspan(interval[0], interval[1] - 1, color="#f59e0b", alpha=0.25)
    ax.axhline(0, color="#b91c1c", linewidth=0.6, linestyle="--")
    ax.axhline(1, color="#b91c1c", linewidth=0.6, linestyle="--")
    ax.set_xlim(0, WIN - 1)
    ax.set_ylim(-0.35, 1.35)
    ax.set_ylabel(f"ch {channel}", fontsize=7, rotation=0, labelpad=16)
    ax.tick_params(labelsize=6)


def draw_focus(ax, values, channel, interval):
    start, end = interval
    ax.plot(np.arange(start, end), values[start:end], color="black", linewidth=1.1)
    ax.axhline(0, color="#b91c1c", linewidth=0.6, linestyle="--")
    ax.axhline(1, color="#b91c1c", linewidth=0.6, linestyle="--")
    ax.set_xlim(start, end - 1)
    ax.set_ylim(-0.35, 1.35)
    ax.set_title(f"ch {channel} · local evidence [{start}, {end})", fontsize=7)
    ax.tick_params(labelsize=6)


def draw_focus_reference_band(ax, values, train, channel, interval, value_range):
    """K40: keep HFZ unchanged except for a train-only normal band in the zoom."""
    start, end = interval
    context_start = max(0, start - 48)
    # The fallback also matches K33: never use the focus interval for retrieval.
    if start - context_start < 16:
        context_start, start = 0, max(16, start)
    # colab_multivariate_v2 returns a Python list; convert once so that the
    # context/focus slices are applied across every reference window.
    bank = np.asarray(cm.get_windows(train[:, channel]))
    bank_context = normalize_train(bank[:, context_start:start], *value_range)
    candidate_context = values[channel][context_start:start]
    nearest = np.argsort(np.mean((bank_context - candidate_context) ** 2, axis=1))[:8]
    refs = normalize_train(bank[nearest, interval[0]:interval[1]], *value_range)
    q10, median, q90 = np.quantile(refs, [0.10, 0.50, 0.90], axis=0)
    x = np.arange(interval[0], interval[1])
    ax.fill_between(x, q10, q90, color="#60a5fa", alpha=0.32, label="normal 10–90%")
    ax.plot(x, median, color="#2563eb", linewidth=1.0, label="normal median")
    ax.plot(x, values[channel][interval[0]:interval[1]], color="black", linewidth=1.15, label="candidate")
    ax.axhline(0, color="#b91c1c", linewidth=0.6, linestyle="--")
    ax.axhline(1, color="#b91c1c", linewidth=0.6, linestyle="--")
    ax.set_xlim(interval[0], interval[1] - 1)
    ax.set_ylim(-0.35, 1.35)
    ax.set_title(f"ch {channel} · local evidence + normal band", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")


def make_image(layout, window, ranked, page, all_selected, intervals, ranges, train=None):
    if not page:
        raise ValueError("No Stage-1 candidates: VLM rendering must not be called.")
    values = {c: normalize_train(window[:, c], *ranges[c]) for c in page}
    rows = len(page)
    if layout == "HZ":
        cols = min(4, rows)
        grid_rows = int(np.ceil(rows / cols))
        fig = plt.figure(figsize=(12, 3.4 + grid_rows * 1.9), dpi=120)
        gs = GridSpec(2, 1, height_ratios=[1.5, max(1, grid_rows)], hspace=0.48)
        heat_ax = fig.add_subplot(gs[0])
        render_heatmap(heat_ax, window, ranked, all_selected, ranges)
        sub = gs[1].subgridspec(grid_rows, cols, hspace=0.8, wspace=0.4)
        for i, c in enumerate(page):
            draw_focus(fig.add_subplot(sub[i // cols, i % cols]), values[c], c, intervals[c])
    elif layout == "HF":
        fig = plt.figure(figsize=(12, 3.3 + rows * 1.15), dpi=120)
        gs = GridSpec(2, 1, height_ratios=[1.5, max(1, rows * 0.72)], hspace=0.48)
        heat_ax = fig.add_subplot(gs[0])
        render_heatmap(heat_ax, window, ranked, all_selected, ranges)
        sub = gs[1].subgridspec(rows, 1, hspace=0.48)
        for i, c in enumerate(page):
            draw_full(fig.add_subplot(sub[i, 0]), values[c], c, intervals[c])
    elif layout == "HFZ":
        fig = plt.figure(figsize=(13, 3.3 + rows * 1.4), dpi=120)
        gs = GridSpec(2, 1, height_ratios=[1.5, max(1, rows * 0.9)], hspace=0.48)
        heat_ax = fig.add_subplot(gs[0])
        render_heatmap(heat_ax, window, ranked, all_selected, ranges)
        sub = gs[1].subgridspec(rows, 2, width_ratios=[1.45, 1], hspace=0.72, wspace=0.25)
        for i, c in enumerate(page):
            draw_full(fig.add_subplot(sub[i, 0]), values[c], c, intervals[c])
            draw_focus(fig.add_subplot(sub[i, 1]), values[c], c, intervals[c])
    elif layout == "HFZR":
        if train is None:
            raise ValueError("HFZR requires train data for reference retrieval.")
        fig = plt.figure(figsize=(13, 3.3 + rows * 1.4), dpi=120)
        gs = GridSpec(2, 1, height_ratios=[1.5, max(1, rows * 0.9)], hspace=0.48)
        heat_ax = fig.add_subplot(gs[0])
        render_heatmap(heat_ax, window, ranked, all_selected, ranges)
        sub = gs[1].subgridspec(rows, 2, width_ratios=[1.45, 1], hspace=0.72, wspace=0.25)
        for i, c in enumerate(page):
            draw_full(fig.add_subplot(sub[i, 0]), values[c], c, intervals[c])
            draw_focus_reference_band(fig.add_subplot(sub[i, 1]), values, train, c, intervals[c], ranges[c])
    else:
        raise ValueError(f"Unknown layout {layout}")
    fig.suptitle(f"Stage-2 evidence contract · {layout} · candidates on this page: {page}", fontsize=10)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prompt(layout, page, all_selected, page_index, n_pages):
    description = {
        "HZ": "The lower plots are local 60-tick focus views for this page's candidates.",
        "HF": "The lower rows are full 224-tick traces on one common x-axis; amber marks each candidate's Stage-1 local peak interval.",
        "HFZ": "Each lower row contains the full 224-tick trace at left and its enlarged Stage-1 local peak interval at right.",
        "HFZR": "Each lower row contains the full 224-tick trace at left. At right, the same local interval overlays the candidate (black) with a train-only normal reference median (blue) and 10–90% band (light blue), retrieved using only the pre-interval context.",
    }[layout]
    return f"""You are validating Stage-1 anomaly proposals in a multivariate industrial time series.
Stage 1 nominated {all_selected}. This is page {page_index + 1} of {n_pages}; decide ONLY candidates {page}.

The top heatmap contains all 38 channels over the same 224-tick window. Every visual uses the same train-derived robust normalization: red dashed lines at 0 and 1 mark the normal training display range; a trace outside them is evidence, not an automatic verdict.
{description}

Stage-1 nomination is a screening prior, not ground truth. For each page candidate, require visible temporal evidence such as sustained change, meaningful shape departure, range exceedance, or coordinated system context. Reject isolated benign-looking fluctuations. It is valid to reject every candidate.

Respond ONLY as JSON:
{{"anomalous_channels": [a subset of {page}], "confidence": "low" | "medium" | "high"}}"""


def paged(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def safe_wilcoxon(values, baseline):
    delta = values - baseline
    return float(wilcoxon(values, baseline).pvalue) if np.any(delta != 0) else 1.0


def render_example(segment_index=0, layout="HFZ"):
    """Render one Stage-2 input page without making a VLM call."""
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    if not 0 <= segment_index < len(segments):
        raise ValueError(f"segment_index must be in [0, {len(segments) - 1}].")
    segment = segments[segment_index]
    entity, cs, ce = segment["entity"], segment["start"], segment["end"]
    train, test = load_smd(entity)
    start, end = _centered_window(len(test), (cs + ce) // 2, WIN)
    window = test[start:end]
    calibrations = {}
    degenerate = constant_channels(train)
    corr = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
    scores = compute_zscores(entity, train, window, calibrations, degenerate)
    ranked, selected = select_channels_hysteresis(scores, corr, ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
    ranges = {c: _train_image_range(train[:, c]) for c in range(N_CHANNELS)}
    intervals = {}
    for c in range(N_CHANNELS):
        if c not in degenerate:
            stats = calibrations[(entity, c)][2]
            ranges[c] = (stats["vmin"], stats["vmax"])
    for c in selected:
        tr_cls, tr_patches, stats = calibrations[(entity, c)]
        intervals[c] = focus_interval(fixed_time_profile(tr_cls, tr_patches, window[:, c], stats))
    if not selected:
        raise RuntimeError("This segment has no K4 candidates, so there is no Stage-2 image to render.")
    page = selected[:MAX_CANDIDATES_PER_PAGE]
    image = make_image(layout, window, ranked, page, selected, intervals, ranges, train=train)
    path = OUT_DIR / f"example_{layout}_page1_{segment_index:02d}_{entity}_{cs}_{ce}.png"
    path.write_bytes(base64.b64decode(image))
    print(json.dumps({"path": str(path), "seg_id": f"{entity}_{cs}_{ce}", "page": page, "selected": selected}, ensure_ascii=False))


def run(execute=False, limit=None, start_index=0, layouts=LAYOUTS, summary_name=None):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    if start_index:
        segments = segments[start_index:]
    if limit is not None:
        segments = segments[:limit]
    cache_path = OUT_DIR / f"checkpoint_{CACHE_VERSION}.json"
    checkpoint = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    data, calibrations, degenerate, corr = {}, {}, {}, {}
    rows = []

    for segment in segments:
        entity, cs, ce = segment["entity"], segment["start"], segment["end"]
        seg_id = f"{entity}_{cs}_{ce}"
        if entity not in data:
            train, test = load_smd(entity)
            data[entity] = (train, test)
            degenerate[entity] = constant_channels(train)
            corr[entity] = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
        train, test = data[entity]
        start, end = _centered_window(len(test), (cs + ce) // 2, WIN)
        window = test[start:end]
        scores = compute_zscores(entity, train, window, calibrations, degenerate[entity])
        ranked, selected = select_channels_hysteresis(scores, corr[entity], ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
        ranges = {c: _train_image_range(train[:, c]) for c in range(N_CHANNELS)}
        for c in range(N_CHANNELS):
            if c not in degenerate[entity]:
                stats = calibrations[(entity, c)][2]
                ranges[c] = (stats["vmin"], stats["vmax"])
        intervals = {}
        for c in selected:
            tr_cls, tr_patches, stats = calibrations[(entity, c)]
            intervals[c] = focus_interval(fixed_time_profile(tr_cls, tr_patches, window[:, c], stats))
        truth = set(d - 1 for d in segment["dims"])
        stage1_tp = len(set(selected) & truth)
        stage1_precision = stage1_tp / len(selected) if selected else 0.0
        stage1_recall = stage1_tp / len(truth) if truth else 0.0
        row = {"seg_id": seg_id, "k_gt": len(truth), "n_selected": len(selected),
               "stage1_oracle_f1": f1_of(selected, truth),
               "stage1_precision": stage1_precision, "stage1_recall": stage1_recall}
        print(f"  {seg_id}: selected={len(selected)} {selected}", flush=True)
        for layout in layouts:
            prediction = set()
            complete = True
            for page_index, page in enumerate(paged(selected, MAX_CANDIDATES_PER_PAGE)):
                key = f"{seg_id}:{layout}:page{page_index}"
                if checkpoint.get(key, {}).get("status") == "OK":
                    pred = checkpoint[key]["pred"]
                elif not execute:
                    complete = False
                    continue
                else:
                    image = make_image(layout, window, ranked, page, selected, intervals, ranges, train=train)
                    raw = call_vlm(build_prompt(layout, page, selected, page_index, len(paged(selected, MAX_CANDIDATES_PER_PAGE))), image)
                    parsed = parse_response(raw)
                    pred = sorted(set(parsed or []) & set(page)) if parsed is not None else None
                    checkpoint[key] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
                    cache_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
                if pred is None:
                    complete = False
                else:
                    prediction.update(pred)
            if not selected:
                prediction = set()
                complete = True
            row[f"pred_{layout}"] = sorted(prediction) if complete else None
            row[f"f1_{layout}"] = f1_of(prediction, truth) if complete else None
        rows.append(row)

    result = {"stage1_cache_version": CACHE_VERSION, "n": len(rows), "layouts": list(layouts), "rows": rows}
    result["mean_stage1_oracle_f1"] = float(np.mean([r["stage1_oracle_f1"] for r in rows])) if rows else None
    result["mean_stage1_precision"] = float(np.mean([r["stage1_precision"] for r in rows])) if rows else None
    result["mean_stage1_recall"] = float(np.mean([r["stage1_recall"] for r in rows])) if rows else None
    for layout in layouts:
        values = np.array([r[f"f1_{layout}"] for r in rows if r[f"f1_{layout}"] is not None])
        # In --stage1 mode there are deliberately no Stage-2 predictions.  Do
        # not turn the one legitimate empty-candidate decision into an apparent
        # zero-F1 visual-layout result.
        result[f"mean_{layout}"] = float(values.mean()) if execute and len(values) == len(rows) else None
    paired_rows = [r for r in rows if all(r[f"f1_{layout}"] is not None for layout in layouts)]
    if execute and paired_rows and "HZ" in layouts:
        base = np.array([r["f1_HZ"] for r in paired_rows])
        for layout in layouts:
            if layout == "HZ":
                continue
            values = np.array([r[f"f1_{layout}"] for r in paired_rows])
            result[f"diff_{layout}_minus_HZ"] = float((values - base).mean())
            result[f"p_{layout}_vs_HZ"] = safe_wilcoxon(values, base)
    if execute and paired_rows and "HFZ" in layouts and "HFZR" in layouts:
        hfz = np.array([r["f1_HFZ"] for r in paired_rows])
        hfzr = np.array([r["f1_HFZR"] for r in paired_rows])
        result["diff_HFZR_minus_HFZ"] = float((hfzr - hfz).mean())
        result["p_HFZR_vs_HFZ"] = safe_wilcoxon(hfzr, hfz)
    name = summary_name or ("summary_smoke.json" if limit is not None else "summary.json")
    (OUT_DIR / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true", help="inspect Stage-1 selections; no VLM calls")
    parser.add_argument("--run", action="store_true", help="call VLM for all three controlled layouts")
    parser.add_argument("--smoke", action="store_true", help="use first 12 paired segments")
    parser.add_argument("--confirm", action="store_true", help="use held-out 36 segments and compare HZ against smoke-selected HFZ")
    parser.add_argument("--render-example", action="store_true", help="render one Stage-2 input image; no VLM call")
    parser.add_argument("--example-index", type=int, default=0, help="segment index for --render-example")
    parser.add_argument("--render-layout", choices=LAYOUTS + ("HFZR",), default="HFZ", help="layout for --render-example")
    parser.add_argument("--reference-ablation", action="store_true", help="also run rejected K40 HFZR reference-band ablation")
    args = parser.parse_args()
    if args.smoke and args.confirm:
        parser.error("choose only one of --smoke or --confirm")
    if args.render_example:
        render_example(args.example_index, args.render_layout)
    elif args.confirm:
        layouts = ("HZ", "HFZ", "HFZR") if args.reference_ablation else ("HZ", "HFZ")
        run(execute=args.run, start_index=12, layouts=layouts, summary_name="summary_holdout36.json")
    else:
        layouts = LAYOUTS + (("HFZR",) if args.reference_ablation else ())
        run(execute=args.run, limit=12 if args.smoke else None, layouts=layouts)
