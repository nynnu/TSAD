"""
top-k sweep for our_design's channel-selection stage (design doc §29 follow-up).

§29 found our_design (Step0 top-4) has high precision (0.611) but low recall
(0.118) vs baselines, dragging F1 to the worst of the three. Before concluding
Step0 itself needs a redesign (expensive), first check the cheap fix: does
just raising k recover recall? Reuses the SAME cached Step0 rankings
(step1v3 checkpoint's ranked_step0_top10, no DINOv2 recomputation) and the
same overlay render / prompt / parser as the 3-way script -- only k changes.
"""

import json

import numpy as np

from smd_3way_baseline_comparison import (
    BASE, SEGMENTS, STEP1V3_CHECKPOINT, WIN,
    _centered_window, build_prompt, call_vlm, load_smd_test,
    parse_response, prf, render_overlay,
)

RESULTS_DIR = BASE / "experiments" / "results_smd_topk_sweep"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
K_VALUES = [6, 8]


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    step1v3 = json.loads(STEP1V3_CHECKPOINT.read_text(encoding="utf-8"))
    entity_test = {}

    for k in K_VALUES:
        for entity, (cs, ce), gt_dims in SEGMENTS:
            seg_id = f"{entity}_{cs}_{ce}"
            case_id = f"{seg_id}_k{k}"
            if checkpoint.get(case_id, {}).get("status") == "OK":
                print(f"[SKIP] {case_id}")
                continue

            if entity not in entity_test:
                entity_test[entity] = load_smd_test(entity)
            test = entity_test[entity]
            center = (cs + ce) // 2
            s, e = _centered_window(len(test), center, WIN)
            window = test[s:e]

            topk = step1v3.get(seg_id, {}).get("ranked_step0_top10", [])[:k]
            img = render_overlay(window, topk)
            prompt = build_prompt(topk)
            raw = call_vlm(prompt, img)
            pred = parse_response(raw)
            checkpoint[case_id] = {"status": "OK" if pred is not None else "PARSE_ERROR",
                                    "pred": pred, "raw": raw, "seg_id": seg_id, "k": k, "topk": topk}
            save_checkpoint(checkpoint)
            print(f"[{checkpoint[case_id]['status']}] {case_id}: topk={topk} pred={pred}", flush=True)

    # ---- Scoring pass ----
    summary = {}
    for k in K_VALUES:
        vals = []
        for entity, (cs, ce), gt_dims in SEGMENTS:
            seg_id = f"{entity}_{cs}_{ce}"
            gt_channels = set(d - 1 for d in gt_dims)
            entry = checkpoint.get(f"{seg_id}_k{k}", {})
            if entry.get("status") == "OK":
                p, r = prf(set(entry["pred"] or []), gt_channels)
                vals.append((p, r))
        if vals:
            precision = float(np.mean([v[0] for v in vals]))
            recall = float(np.mean([v[1] for v in vals]))
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            summary[f"k={k}"] = {"n": len(vals), "precision": precision, "recall": recall, "f1": f1}
        else:
            summary[f"k={k}"] = {"n": 0}

    print("\n" + "=" * 60)
    print("TOP-K SWEEP SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run()
