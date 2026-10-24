"""K37: keep K4 screening, then localize each candidate with DINO-column x raw-z evidence.

K4 remains responsible for candidate recall.  This script never uses the local
joint signal to remove a K4 candidate: it only reorders candidates, chooses the
zoom interval, and reports a train-only evidence strength to the VLM.
"""
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
    compute_zscores, constant_channels, f1_of, get_or_build_channel_calib,
    select_channels_hysteresis,
)
from smd_3way_baseline_comparison import call_vlm, parse_response
from step1v3_dino_graph_smd import WIN, _centered_window, load_smd

OUT_DIR = BASE / "experiments" / "results_stage2_k37_local_joint_evidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_STRICT, ALPHA_LOOSE, CORR_THR = 0.01, 0.1, 0.5
ZOOM_WIDTH = 60
MAX_CANDIDATES_PER_PAGE = 8
# Each stream must be in its own train-derived upper 5% tail.  This is a
# support label, not a test-label-tuned decision boundary.
JOINT_TAIL_MIN = float(-np.log10(0.05))


def empirical_tail(values, reference):
    """-log10 upper-tail probability using only a normal/train reference."""
    ref = np.asarray(reference, dtype=float)
    return -np.log10((1 + np.count_nonzero(ref[:, None] >= values[None, :], axis=0)) / (len(ref) + 1))


def column_joint_evidence(train_channel, window_channel, tr_cls, tr_patches, stats):
    """Return 16-column train-calibrated DINO, raw-z, and conjunction evidence."""
    image = cm.ts_to_image_fast(window_channel, vmin=stats["vmin"], vmax=stats["vmax"])
    te_cls, te_patches = cm.extract_dinov2([image], multilayer=False)
    raw = cm.knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, return_win=True)["knn_win"][0]
    side = int(np.sqrt(raw.size))
    if side * side != raw.size:
        raise ValueError(f"Expected square DINO patch grid, got {raw.size} patches.")
    dino_columns = raw.reshape(side, side).max(axis=0)
    dino_tail = empirical_tail(dino_columns, np.asarray(stats["calib_columns"], dtype=float))

    mu, sigma = float(train_channel.mean()), max(float(train_channel.std()), 1e-3)
    test_z = np.abs((window_channel - mu) / sigma)
    # A column should be compared with normal *column peaks*, not individual
    # ticks; this prevents a 14-tick max from being artificially overconfident.
    test_z_columns = np.array([block.max() for block in np.array_split(test_z, side)])
    train_z = np.abs((train_channel - mu) / sigma)
    usable = len(train_z) // side * side
    train_z_columns = train_z[:usable].reshape(-1, side).max(axis=1)
    z_tail = empirical_tail(test_z_columns, train_z_columns)
    joint_tail = np.minimum(dino_tail, z_tail)
    return dino_tail, z_tail, joint_tail


def local_interval(joint_tail, dino_tail):
    """Prefer a corroborated column; fall back to DINO localization if absent."""
    profile = joint_tail if np.any(joint_tail >= JOINT_TAIL_MIN) else dino_tail
    side = len(profile)
    peak_tick = int(round(np.linspace(0, WIN - 1, side)[int(np.argmax(profile))]))
    start = max(0, min(WIN - ZOOM_WIDTH, peak_tick - ZOOM_WIDTH // 2))
    return start, start + ZOOM_WIDTH


def evidence_label(joint_tail):
    peak = float(np.max(joint_tail))
    if peak >= 2.0:
        return "strong", peak
    if peak >= JOINT_TAIL_MIN:
        return "moderate", peak
    return "weak", peak


def prompt(page, all_selected, evidence, page_index, n_pages):
    support = ", ".join(f"ch {c}: {evidence[c]['label']}" for c in page)
    return f"""You are validating Stage-1 anomaly proposals in a multivariate industrial time series.
Stage 1 nominated {all_selected}. This is page {page_index + 1} of {n_pages}; decide ONLY candidates {page}.

The top heatmap shows all channels over the same 224-tick window. Each candidate has a full trace and a local 60-tick evidence view. The local interval is chosen from a train-calibrated DINO patch-column signal, preferentially where an independent raw z-score is also unusually high. Candidate support labels are: {support}. These labels are supporting evidence, not ground truth.

Stage-1 nomination is a screening prior. Require visible temporal evidence such as sustained change, meaningful shape departure, range exceedance, or coordinated system context. Reject isolated benign-looking fluctuations. It is valid to reject every candidate.

Respond ONLY as JSON:
{{"anomalous_channels": [a subset of {page}], "confidence": "low" | "medium" | "high"}}"""


def paged(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def run(execute=False, limit=None, start_index=0, summary_name=None):
    segments = json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8"))
    if start_index:
        segments = segments[start_index:]
    if limit is not None:
        segments = segments[:limit]
    checkpoint_path = OUT_DIR / f"checkpoint_{CACHE_VERSION}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    data, calibrations, degenerate, corr = {}, {}, {}, {}
    rows = []

    for segment in segments:
        entity, cs, ce = segment["entity"], segment["start"], segment["end"]
        seg_id = f"{entity}_{cs}_{ce}"
        if entity not in data:
            train, test = load_smd(entity)
            data[entity] = train, test
            degenerate[entity] = constant_channels(train)
            corr[entity] = np.nan_to_num(np.corrcoef(train.T), nan=0.0)
        train, test = data[entity]
        start, end = _centered_window(len(test), (cs + ce) // 2, WIN)
        window = test[start:end]
        scores = compute_zscores(entity, train, window, calibrations, degenerate[entity])
        ranked, selected = select_channels_hysteresis(scores, corr[entity], ALPHA_STRICT, ALPHA_LOOSE, CORR_THR)
        evidence, intervals = {}, {}
        for c in selected:
            tr_cls, tr_patches, stats = calibrations[(entity, c)]
            d_tail, z_tail, j_tail = column_joint_evidence(train[:, c], window[:, c], tr_cls, tr_patches, stats)
            label, peak = evidence_label(j_tail)
            evidence[c] = {"label": label, "peak_joint_tail": peak,
                           "n_joint_columns": int(np.count_nonzero(j_tail >= JOINT_TAIL_MIN)),
                           "dino_tail": d_tail.tolist(), "z_tail": z_tail.tolist(), "joint_tail": j_tail.tolist()}
            intervals[c] = local_interval(j_tail, d_tail)
        selected = sorted(selected, key=lambda c: (-evidence[c]["peak_joint_tail"], -scores[c], c))
        truth = {d - 1 for d in segment["dims"]}
        tp = len(set(selected) & truth)
        row = {"seg_id": seg_id, "k_gt": len(truth), "selected": selected, "n_selected": len(selected),
               "stage1_oracle_f1": f1_of(selected, truth),
               "stage1_precision": tp / len(selected) if selected else 0.0,
               "stage1_recall": tp / len(truth) if truth else 0.0,
               "evidence": evidence}
        print(f"  {seg_id}: selected={len(selected)} {selected}", flush=True)

        ranges = {c: _train_image_range(train[:, c]) for c in range(N_CHANNELS)}
        for c in range(N_CHANNELS):
            if c not in degenerate[entity]:
                ranges[c] = (calibrations[(entity, c)][2]["vmin"], calibrations[(entity, c)][2]["vmax"])
        prediction, complete = set(), True
        for page_index, page in enumerate(paged(selected, MAX_CANDIDATES_PER_PAGE)):
            key = f"{seg_id}:HFZ-J:page{page_index}"
            if checkpoint.get(key, {}).get("status") == "OK":
                pred = checkpoint[key]["pred"]
            elif not execute:
                complete = False
                continue
            else:
                image = k35.make_image("HFZ", window, ranked, page, selected, intervals, ranges)
                raw = call_vlm(prompt(page, selected, evidence, page_index, len(paged(selected, MAX_CANDIDATES_PER_PAGE))), image)
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
        row["pred_HFZ_J"] = sorted(prediction) if complete else None
        row["f1_HFZ_J"] = f1_of(prediction, truth) if complete else None
        rows.append(row)

    result = {"stage1_cache_version": CACHE_VERSION, "n": len(rows), "layout": "HFZ-J", "rows": rows}
    for key in ("stage1_oracle_f1", "stage1_precision", "stage1_recall"):
        result[f"mean_{key}"] = float(np.mean([r[key] for r in rows])) if rows else None
    values = [r["f1_HFZ_J"] for r in rows if r["f1_HFZ_J"] is not None]
    result["mean_HFZ_J"] = float(np.mean(values)) if execute and len(values) == len(rows) else None
    name = summary_name or ("summary_smoke.json" if limit is not None else "summary.json")
    (OUT_DIR / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true", help="inspect K4 candidates and local evidence; no VLM calls")
    parser.add_argument("--run", action="store_true", help="render HFZ-J and call the VLM")
    parser.add_argument("--smoke", action="store_true", help="use first 12 paired segments")
    parser.add_argument("--confirm", action="store_true", help="use held-out 36 segments")
    args = parser.parse_args()
    if args.smoke and args.confirm:
        parser.error("choose only one of --smoke or --confirm")
    run(execute=args.run, limit=12 if args.smoke else None,
        start_index=12 if args.confirm else 0,
        summary_name="summary_holdout36.json" if args.confirm else None)
