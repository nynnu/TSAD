"""
Step0 redesign: adaptive window width instead of fixed WIN=224 (design doc §31).

§30 confirmed top-k tuning can't fix our_design's recall problem -- the real
cause (per §25) must be Step0's scoring itself. Rereading [[Resolution-Context
Dilemma]] and the DINOv2 note against the current implementation
(step1v3_dino_graph_smd.py) found a concrete mechanism: many of our 9 real SMD
segments have anomaly spans far shorter than the fixed WIN=224 window used for
both static and dynamic embeddings (e.g. span=3 samples = 1.3% of the window).
A 224-sample window centered on a 3-sample anomaly is >98% normal padding, so
static and dynamic embeddings end up nearly identical regardless of channel.

Fix tested here: window width adaptive to span (width = clip(span*4, 40, 224)),
same width used for BOTH static and dynamic windows (fair comparison -- avoids
confounding "shape difference" with "sample-density difference"). No API cost
-- local DINOv2 only, same 9 segments, same recall@k metric as step1v3.
"""

import json
from pathlib import Path

import numpy as np

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, _centered_window, cosine_dist,
    embed_channel_window, load_smd,
)

RESULTS_DIR = BASE / "experiments" / "results_step0_adaptive_window"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

MIN_WIDTH = 40
MAX_WIDTH = 224
SPAN_MULT = 4


def adaptive_width(span: int) -> int:
    return int(np.clip(span * SPAN_MULT, MIN_WIDTH, MAX_WIDTH))


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
    entity_data = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        span = ce - cs
        width = adaptive_width(span)
        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, width)
        s_static, e_static = _centered_window(len(train), len(train) // 2, width)

        print(f"[{seg_id}] span={span} width={width} ...", flush=True)
        gt_channels = [d - 1 for d in gt_dims]
        step0_score = {}
        for c in range(N_CHANNELS):
            static_emb = embed_channel_window(train[:, c], s_static, e_static)
            dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            step0_score[c] = cosine_dist(static_emb, dynamic_emb)

        ranked = sorted(step0_score, key=lambda c: -step0_score[c])
        k = len(gt_channels)
        gt_set = set(gt_channels)
        recall = len(set(ranked[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "span": span, "width": width,
            "gt_channels_0idx": gt_channels, "k": k,
            "recall_adaptive": recall, "ranked_top10": ranked[:10],
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} gt={gt_channels} recall_adaptive={recall:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    r_adaptive = np.mean([r["recall_adaptive"] for r in rows])

    # compare against original fixed-224 Step0 recall (already computed, step1v3 checkpoint)
    step1v3_ckpt_path = BASE / "experiments" / "results_step1v3_dino_graph_smd" / "checkpoint.json"
    r_fixed = None
    if step1v3_ckpt_path.exists():
        step1v3 = json.loads(step1v3_ckpt_path.read_text(encoding="utf-8"))
        fixed_vals = [step1v3[r["entity"] + f"_{r['cs']}_{r['ce']}"]["recall_step0"]
                      for r in rows if r["entity"] + f"_{r['cs']}_{r['ce']}" in step1v3]
        if fixed_vals:
            r_fixed = float(np.mean(fixed_vals))

    print(f"\n{'='*60}\nSTEP0 ADAPTIVE-WINDOW SUMMARY (n={len(rows)})\n{'='*60}")
    print(f"Fixed WIN=224 recall@k    = {r_fixed}")
    print(f"Adaptive-width recall@k   = {r_adaptive:.4f}")
    if r_fixed is not None:
        print(f"Delta                     = {r_adaptive - r_fixed:+.4f}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "recall_fixed224": r_fixed, "recall_adaptive": r_adaptive,
        "delta": (r_adaptive - r_fixed) if r_fixed is not None else None,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
