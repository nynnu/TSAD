"""K38: K35 HFZ plus compact index-aware text; all visual and Stage-1 conditions stay fixed."""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
import colab_multivariate_v2 as cm
import experiment_stage2_k35_evidence_contract as k35
from experiment_stage2_k4_adaptive import (
    CACHE_VERSION, GT_SEGMENTS_PATH, N_CHANNELS, _train_image_range,
    compute_zscores, constant_channels, f1_of, select_channels_hysteresis,
)
from smd_3way_baseline_comparison import call_vlm, parse_response
from step1v3_dino_graph_smd import WIN, _centered_window, load_smd

OUT_DIR = BASE / "experiments" / "results_stage2_k38_hfz_index_aware_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_STRICT, ALPHA_LOOSE, CORR_THR = 0.01, 0.1, 0.5
MAX_CANDIDATES_PER_PAGE, N_TEXT_POINTS = 8, 6
LAYOUT_NAME = "HFZ-I"
TEXT_INTRO = ("Index-aware local evidence (relative tick within the shared 224-tick window, "
              "train-normalized value). These are the six raw-z-most-deviant points inside each "
              "displayed zoom interval; use them only to read the plotted trace precisely:")


def paged(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def index_text(page, window, train, ranges, intervals):
    """Six train-z-ranked local points per channel, on the same relative x-axis as HFZ."""
    lines = []
    for c in page:
        start, end = intervals[c]
        mu, sigma = float(train[:, c].mean()), max(float(train[:, c].std()), 1e-3)
        local_z = np.abs((window[start:end, c] - mu) / sigma)
        local_idx = np.argsort(-local_z)[:N_TEXT_POINTS] + start
        local_idx.sort()
        lo, hi = ranges[c]
        values = (window[:, c] - lo) / (hi - lo + 1e-8)
        points = ", ".join(f"({int(i)}, {values[i]:.3f})" for i in local_idx)
        lines.append(f"ch {c}, zoom ticks [{start}, {end}): {points}")
    return "\n".join(lines)


def prompt(page, all_selected, window, train, ranges, intervals, page_index, n_pages):
    base = k35.build_prompt("HFZ", page, all_selected, page_index, n_pages)
    note = "\n" + TEXT_INTRO + "\n\n" + index_text(page, window, train, ranges, intervals) + "\n\n"
    return base.replace("Stage-1 nomination is a screening prior", note + "Stage-1 nomination is a screening prior")


def run(execute=False, limit=None, start_index=0, summary_name=None):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    if start_index:
        segments = segments[start_index:]
    if limit is not None:
        segments = segments[:limit]
    checkpoint_path = OUT_DIR / f"checkpoint_{CACHE_VERSION}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    data, calibrations, degenerate, corr, rows = {}, {}, {}, {}, []
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
            intervals[c] = k35.focus_interval(k35.fixed_time_profile(tr_cls, tr_patches, window[:, c], stats))
        truth = {d - 1 for d in segment["dims"]}
        tp = len(set(selected) & truth)
        row = {"seg_id": seg_id, "k_gt": len(truth), "selected": selected, "n_selected": len(selected),
               "stage1_oracle_f1": f1_of(selected, truth),
               "stage1_precision": tp / len(selected) if selected else 0.0,
               "stage1_recall": tp / len(truth) if truth else 0.0}
        print(f"  {seg_id}: selected={len(selected)} {selected}", flush=True)
        prediction, complete = set(), True
        for page_index, page in enumerate(paged(selected, MAX_CANDIDATES_PER_PAGE)):
            key = f"{seg_id}:{LAYOUT_NAME}:page{page_index}"
            if checkpoint.get(key, {}).get("status") == "OK":
                pred = checkpoint[key]["pred"]
            elif not execute:
                complete = False
                continue
            else:
                image = k35.make_image("HFZ", window, ranked, page, selected, intervals, ranges)
                raw = call_vlm(prompt(page, selected, window, train, ranges, intervals, page_index, len(paged(selected, MAX_CANDIDATES_PER_PAGE))), image)
                parsed = parse_response(raw)
                pred = sorted(set(parsed or []) & set(page)) if parsed is not None else None
                checkpoint[key] = {"status": "OK" if pred is not None else "PARSE_ERROR", "pred": pred, "raw": raw}
                checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
            if pred is None:
                complete = False
            else:
                prediction.update(pred)
        if not selected:
            complete = True
        row[f"pred_{LAYOUT_NAME}"] = sorted(prediction) if complete else None
        row[f"f1_{LAYOUT_NAME}"] = f1_of(prediction, truth) if complete else None
        rows.append(row)
    result = {"stage1_cache_version": CACHE_VERSION, "n": len(rows), "layout": LAYOUT_NAME, "rows": rows}
    for key in ("stage1_oracle_f1", "stage1_precision", "stage1_recall"):
        result[f"mean_{key}"] = float(np.mean([r[key] for r in rows])) if rows else None
    values = [r[f"f1_{LAYOUT_NAME}"] for r in rows if r[f"f1_{LAYOUT_NAME}"] is not None]
    result[f"mean_{LAYOUT_NAME}"] = float(np.mean(values)) if execute and len(values) == len(rows) else None
    name = summary_name or ("summary_smoke.json" if limit is not None else "summary.json")
    (OUT_DIR / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true", help="inspect fixed K4 candidates; no VLM calls")
    parser.add_argument("--run", action="store_true", help="render HFZ with index-aware text and call the VLM")
    parser.add_argument("--smoke", action="store_true", help="use first 12 paired segments")
    parser.add_argument("--confirm", action="store_true", help="use held-out 36 segments")
    args = parser.parse_args()
    if args.smoke and args.confirm:
        parser.error("choose only one of --smoke or --confirm")
    run(execute=args.run, limit=12 if args.smoke else None,
        start_index=12 if args.confirm else 0,
        summary_name="summary_holdout36.json" if args.confirm else None)
