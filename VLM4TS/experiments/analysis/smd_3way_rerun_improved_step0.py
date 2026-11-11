"""
Re-verify §29's our_design condition with the improved Step0 (design doc §36).

§34 (multi-ref ensemble, N=5) and §35-b (DINOv2+z-score max-fusion) each
independently improved Step0 recall@k (+0.069, +0.067) over the original
single-static-reference DINOv2 score used when §29's 3-way comparison was
run. This script combines BOTH winning mechanisms -- multi-ref DINOv2 score
fused (max, rank-normalized) with z-score -- to pick a new top-4 per segment,
then reruns ONLY condition C (our_design) with these improved rankings
through the same VLM channel-selection call as §29, to see whether
precision/recall/F1 actually improve in the full pipeline (not just Step0's
own recall@k in isolation). Conditions A/B are NOT rerun (unchanged, reuse
§29's numbers for comparison). 9 API calls only.
"""

import json

import numpy as np

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist, load_smd,
)
from smd_step0_multiref_ensemble import static_ref_windows
from smd_step0_statistical_ensemble import zscore_score, _rank_norm
from step1v3_dino_graph_smd import embed_channel_window
from smd_3way_baseline_comparison import render_overlay, build_prompt, call_vlm, parse_response, prf

RESULTS_DIR = BASE / "experiments" / "results_smd_3way_rerun_improved_step0"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_STATIC_REFS = 5


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
    entity_static_embeds = {}  # entity -> {c: [emb, ...]} (multi-ref, N=5)

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint and checkpoint[seg_id].get("status") == "OK":
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        if entity not in entity_static_embeds:
            print(f"  Computing {N_STATIC_REFS} static ref embeddings for {entity} ...", flush=True)
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            entity_static_embeds[entity] = {
                c: [embed_channel_window(train[:, c], s, e) for s, e in ref_windows] for c in range(N_CHANNELS)
            }
        static = entity_static_embeds[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] scoring (multi-ref + z-score fusion) ...", flush=True)
        dino_score, z_score = {}, {}
        for c in range(N_CHANNELS):
            dyn_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            dists = [cosine_dist(ref_emb, dyn_emb) for ref_emb in static[c]]
            dino_score[c] = float(np.mean(dists))
            z_score[c] = zscore_score(train[:, c], test[s_dyn:e_dyn, c])

        dino_rank = _rank_norm(dino_score)
        z_rank = _rank_norm(z_score)
        combined = {c: max(dino_rank[c], z_rank[c]) for c in range(N_CHANNELS)}
        top4 = sorted(combined, key=lambda c: -combined[c])[:4]

        gt_channels = set(d - 1 for d in gt_dims)
        window = test[s_dyn:e_dyn]
        img = render_overlay(window, top4)
        prompt = build_prompt(top4)
        raw = call_vlm(prompt, img)
        pred = parse_response(raw)

        checkpoint[seg_id] = {
            "status": "OK" if pred is not None else "PARSE_ERROR",
            "top4": top4, "pred": pred, "raw": raw, "gt_channels": sorted(gt_channels),
        }
        save_checkpoint(checkpoint)
        print(f"[{checkpoint[seg_id]['status']}] {seg_id}: top4={top4} pred={pred}", flush=True)

    # ---- Scoring ----
    vals = []
    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        entry = checkpoint.get(seg_id, {})
        if entry.get("status") == "OK":
            gt_channels = set(d - 1 for d in gt_dims)
            p, r = prf(set(entry["pred"] or []), gt_channels)
            vals.append((p, r))

    if vals:
        precision = float(np.mean([v[0] for v in vals]))
        recall = float(np.mean([v[1] for v in vals]))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        summary = {"n": len(vals), "precision": precision, "recall": recall, "f1": f1}
    else:
        summary = {"n": 0}

    print(f"\n{'='*60}\nIMPROVED our_design RE-VERIFY SUMMARY\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run()
