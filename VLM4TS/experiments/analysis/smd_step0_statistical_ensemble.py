"""
Step0 redesign, hypothesis 5: ensemble DINOv2 with a plain statistical signal
(design doc §33 follow-up, candidate #3).

DINOv2 reads window SHAPE (visual pattern). SMD is server-metric data where
many real anomalies are magnitude/level shifts (CPU/memory/queue spikes) --
a signal a trivial statistical z-score can catch directly and cheaply,
independent of any vision-encoding noise. Combine via max-fusion of
rank-normalized scores (same combiner already validated in Step1 v2, design
doc §17): combined(c) = max(rank_norm(step0_dino(c)), rank_norm(zscore(c))).

step0_dino here = the ORIGINAL single-static-reference DINOv2 score (isolate
this one variable against the same baseline as §31-33, not the multi-ref
version from §34 -- combining candidates is the next step only if this one
also helps standalone). zscore(c) = |mean(dynamic window) - mean(train
channel)| / std(train channel) -- no image rendering, no DINOv2, near-zero
extra compute.
"""

import json

import numpy as np

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist,
    embed_channel_window, load_smd,
)

RESULTS_DIR = BASE / "experiments" / "results_step0_statistical_ensemble"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"


def _rank_norm(d: dict) -> dict:
    ranked = sorted(d, key=lambda k: -d[k])
    n = len(ranked)
    return {c: 1.0 - (ranked.index(c) / max(1, n - 1)) for c in d}


def zscore_score(train_channel: np.ndarray, dynamic_window: np.ndarray) -> float:
    mu, sigma = float(train_channel.mean()), float(train_channel.std())
    if sigma < 1e-9:
        return 0.0
    return abs(float(dynamic_window.mean()) - mu) / sigma


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
    entity_static_embeds = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        if entity not in entity_static_embeds:
            print(f"  Computing static reference embeddings for {entity} ...", flush=True)
            s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
            entity_static_embeds[entity] = {c: embed_channel_window(train[:, c], s_static, e_static) for c in range(N_CHANNELS)}
        static = entity_static_embeds[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] ...", flush=True)
        gt_channels = [d - 1 for d in gt_dims]
        gt_set = set(gt_channels)
        k = len(gt_channels)

        step0_score, z_score = {}, {}
        for c in range(N_CHANNELS):
            dyn_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            step0_score[c] = cosine_dist(static[c], dyn_emb)
            z_score[c] = zscore_score(train[:, c], test[s_dyn:e_dyn, c])

        step0_rank = _rank_norm(step0_score)
        z_rank = _rank_norm(z_score)
        combined = {c: max(step0_rank[c], z_rank[c]) for c in range(N_CHANNELS)}

        recall_step0 = len(set(sorted(step0_score, key=lambda c: -step0_score[c])[:k]) & gt_set) / k
        recall_z = len(set(sorted(z_score, key=lambda c: -z_score[c])[:k]) & gt_set) / k
        recall_combined = len(set(sorted(combined, key=lambda c: -combined[c])[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce,
            "gt_channels_0idx": gt_channels, "k": k,
            "recall_step0": recall_step0, "recall_zscore": recall_z, "recall_combined": recall_combined,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} step0={recall_step0:.2f} zscore={recall_z:.2f} combined={recall_combined:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return

    summary = {
        "recall_step0": float(np.mean([r["recall_step0"] for r in rows])),
        "recall_zscore": float(np.mean([r["recall_zscore"] for r in rows])),
        "recall_combined": float(np.mean([r["recall_combined"] for r in rows])),
    }
    print(f"\n{'='*60}\nSTEP0 STATISTICAL ENSEMBLE SUMMARY (n={len(rows)})\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"n": len(rows), **summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
