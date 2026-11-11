"""
N_STATIC_REFS sweep for multi-ref ensemble (design doc §34 follow-up).
Tests N in {3, 5, 10} -- does more references keep helping, or does §34's
N=5 already capture most of the benefit? Same 9 segments, same fixed WIN=224,
local DINOv2 only.
"""

import json

import numpy as np

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist,
    embed_channel_window, load_smd,
)
from smd_step0_multiref_ensemble import static_ref_windows

RESULTS_DIR = BASE / "experiments" / "results_step0_multiref_sweep"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_VALUES = [3, 5, 10]


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
    # entity -> N -> {c: [embeddings]}
    entity_static_embeds = {}

    for n_refs in N_VALUES:
        for entity, (cs, ce), gt_dims in SEGMENTS:
            seg_id = f"{entity}_{cs}_{ce}"
            case_id = f"{seg_id}_N{n_refs}"
            if case_id in checkpoint:
                print(f"[SKIP] {case_id}", flush=True)
                continue

            if entity not in entity_data:
                entity_data[entity] = load_smd(entity)
            train, test = entity_data[entity]

            entity_static_embeds.setdefault(entity, {})
            if n_refs not in entity_static_embeds[entity]:
                print(f"  Computing {n_refs} static reference embeddings for {entity} ...", flush=True)
                ref_windows = static_ref_windows(len(train), WIN, n_refs)
                entity_static_embeds[entity][n_refs] = {
                    c: [embed_channel_window(train[:, c], s, e) for s, e in ref_windows] for c in range(N_CHANNELS)
                }
            static = entity_static_embeds[entity][n_refs]

            center = (cs + ce) // 2
            s_dyn, e_dyn = _centered_window(len(test), center, WIN)

            gt_channels = [d - 1 for d in gt_dims]
            gt_set = set(gt_channels)
            k = len(gt_channels)

            step0_score = {}
            for c in range(N_CHANNELS):
                dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
                dists = [cosine_dist(ref_emb, dynamic_emb) for ref_emb in static[c]]
                step0_score[c] = float(np.mean(dists))

            ranked = sorted(step0_score, key=lambda c: -step0_score[c])
            recall = len(set(ranked[:k]) & gt_set) / k

            checkpoint[case_id] = {"entity": entity, "cs": cs, "ce": ce, "n_refs": n_refs, "k": k, "recall": recall}
            save_checkpoint(checkpoint)
            print(f"[OK] {case_id}: k={k} recall={recall:.2f}", flush=True)

    rows = list(checkpoint.values())
    summary = {}
    for n_refs in N_VALUES:
        vals = [r["recall"] for r in rows if r["n_refs"] == n_refs]
        if vals:
            summary[f"N={n_refs}"] = float(np.mean(vals))

    print(f"\n{'='*60}\nMULTI-REF N-SWEEP SUMMARY\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
