"""
Dynamic top-k selection instead of fixed top-4 (design doc §36 follow-up).

§36 found the real bottleneck isn't Step0 score quality (already at 83% of
its theoretical ceiling under fixed k=4) but the fixed-4 cutoff itself: GT
channel-set sizes range 4-8, so showing only 4 structurally caps recall at
~0.656 even with perfect ranking. Using the TRUE GT size to pick k would be
oracle/cheating -- not deployable. Instead: gap-based elbow detection on the
sorted combined Step0 scores (multi-ref DINOv2 + z-score fusion, §34/§35-b) --
a standard threshold-free cutoff method. Cut where the score drops the most
between consecutive ranked channels (the largest "gap" = where genuinely
anomalous channels likely end and noise begins), clamped to [MIN_K, MAX_K].

Step 1 (this script): local-only check -- does the dynamically-sized
candidate SET even contain enough GT channels (recall@selected, no VLM yet)?
If the set-level ceiling improves over fixed-4's ~0.656, proceed to rerun
through the VLM (separate script) for real precision/recall/F1.
"""

import json

import numpy as np

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist, load_smd,
    embed_channel_window,
)
from smd_step0_multiref_ensemble import static_ref_windows
from smd_step0_statistical_ensemble import zscore_score, _rank_norm

RESULTS_DIR = BASE / "experiments" / "results_step0_dynamic_k"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_STATIC_REFS = 5
MIN_K, MAX_K = 2, 10


def gap_based_k(ranked_scores: list[float]) -> int:
    """Pick cutoff at the largest gap between consecutive sorted (desc) scores, within [MIN_K, MAX_K]."""
    search_upper = min(MAX_K, len(ranked_scores) - 1)
    gaps = [ranked_scores[i] - ranked_scores[i + 1] for i in range(search_upper)]
    best_i = int(np.argmax(gaps[MIN_K - 1:])) + (MIN_K - 1)  # only consider cutoffs >= MIN_K
    return best_i + 1  # number of channels to keep (index+1)


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
            print(f"  Computing {N_STATIC_REFS} static ref embeddings for {entity} ...", flush=True)
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            entity_static_embeds[entity] = {
                c: [embed_channel_window(train[:, c], s, e) for s, e in ref_windows] for c in range(N_CHANNELS)
            }
        static = entity_static_embeds[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] ...", flush=True)
        dino_score, z_score = {}, {}
        for c in range(N_CHANNELS):
            dyn_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            dists = [cosine_dist(ref_emb, dyn_emb) for ref_emb in static[c]]
            dino_score[c] = float(np.mean(dists))
            z_score[c] = zscore_score(train[:, c], test[s_dyn:e_dyn, c])

        dino_rank = _rank_norm(dino_score)
        z_rank = _rank_norm(z_score)
        combined = {c: max(dino_rank[c], z_rank[c]) for c in range(N_CHANNELS)}
        ranked_channels = sorted(combined, key=lambda c: -combined[c])
        ranked_scores = [combined[c] for c in ranked_channels]

        dyn_k = gap_based_k(ranked_scores)
        selected = ranked_channels[:dyn_k]

        gt_channels = [d - 1 for d in gt_dims]
        gt_set = set(gt_channels)
        true_k = len(gt_channels)
        recall_selected = len(set(selected) & gt_set) / true_k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "true_k": true_k, "dynamic_k": dyn_k,
            "selected": selected, "gt_channels_0idx": gt_channels, "recall_selected": recall_selected,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: true_k={true_k} dynamic_k={dyn_k} recall_selected={recall_selected:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    r_mean = np.mean([r["recall_selected"] for r in rows])
    k_mean = np.mean([r["dynamic_k"] for r in rows])
    print(f"\n{'='*60}\nDYNAMIC-K SUMMARY (n={len(rows)})\n{'='*60}")
    print(f"Mean dynamic_k        = {k_mean:.2f}  (fixed was 4)")
    print(f"Mean recall@selected  = {r_mean:.4f}  (fixed-4 recall@k was 0.5456)")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "mean_dynamic_k": float(k_mean), "recall_selected": float(r_mean),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
