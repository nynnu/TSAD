"""
Step0 redesign, hypothesis 3: multi-reference ensemble (design doc §33 follow-up).

§31-33 found that single-parameter tweaks (window width, normalization scale)
each swing individual segments a lot but net out to ~0 on average -- pointing
at a structural noise problem rather than a parameter problem: comparing a
channel's dynamic window against exactly ONE static reference window (a
single point sample of "normal" from the middle of train) is a noisy,
high-variance estimate. If that one static window happens to be atypical for
any reason, the whole per-channel score is thrown off.

Fix tested here: replace the single static reference with an ensemble of
N_STATIC_REFS windows spread across train, and score = mean cosine distance
from the dynamic window to each reference (a more robust estimate of "how
different is this channel from its normal range", reduces reliance on any
one reference window's idiosyncrasies). Same 9 segments, same fixed WIN=224,
same normalization as the ORIGINAL step1v3 script (isolate this one variable).
No API cost -- local DINOv2 only, static embeddings cached per entity (shared
across that entity's 3 segments).
"""

import json

import numpy as np

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist,
    embed_channel_window, load_smd,
)

RESULTS_DIR = BASE / "experiments" / "results_step0_multiref_ensemble"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_STATIC_REFS = 5


def static_ref_windows(train_len: int, win: int, n_refs: int) -> list[tuple[int, int]]:
    centers = np.linspace(win // 2, train_len - win // 2, n_refs).astype(int)
    return [_centered_window(train_len, c, win) for c in centers]


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
    entity_static_embeds = {}  # entity -> {c: [emb_ref1, emb_ref2, ...]}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        if entity not in entity_static_embeds:
            print(f"  Computing {N_STATIC_REFS} static reference embeddings for {entity} (38 channels) ...", flush=True)
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            static = {c: [embed_channel_window(train[:, c], s, e) for s, e in ref_windows] for c in range(N_CHANNELS)}
            entity_static_embeds[entity] = static
        static = entity_static_embeds[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] ...", flush=True)
        gt_channels = [d - 1 for d in gt_dims]
        step0_score = {}
        for c in range(N_CHANNELS):
            dynamic_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            dists = [cosine_dist(ref_emb, dynamic_emb) for ref_emb in static[c]]
            step0_score[c] = float(np.mean(dists))

        ranked = sorted(step0_score, key=lambda c: -step0_score[c])
        k = len(gt_channels)
        gt_set = set(gt_channels)
        recall = len(set(ranked[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce,
            "gt_channels_0idx": gt_channels, "k": k,
            "recall_multiref": recall, "ranked_top10": ranked[:10],
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} gt={gt_channels} recall_multiref={recall:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    r_multi = np.mean([r["recall_multiref"] for r in rows])

    step1v3_ckpt_path = BASE / "experiments" / "results_step1v3_dino_graph_smd" / "checkpoint.json"
    r_fixed = None
    if step1v3_ckpt_path.exists():
        step1v3 = json.loads(step1v3_ckpt_path.read_text(encoding="utf-8"))
        fixed_vals = [step1v3[r["entity"] + f"_{r['cs']}_{r['ce']}"]["recall_step0"]
                      for r in rows if r["entity"] + f"_{r['cs']}_{r['ce']}" in step1v3]
        if fixed_vals:
            r_fixed = float(np.mean(fixed_vals))

    print(f"\n{'='*60}\nSTEP0 MULTI-REF ENSEMBLE SUMMARY (n={len(rows)})\n{'='*60}")
    print(f"Single static-ref recall@k = {r_fixed}")
    print(f"Multi-ref ({N_STATIC_REFS}) ensemble recall@k = {r_multi:.4f}")
    if r_fixed is not None:
        print(f"Delta                       = {r_multi - r_fixed:+.4f}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "recall_single_ref": r_fixed, "recall_multiref": r_multi,
        "n_static_refs": N_STATIC_REFS,
        "delta": (r_multi - r_fixed) if r_fixed is not None else None,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
