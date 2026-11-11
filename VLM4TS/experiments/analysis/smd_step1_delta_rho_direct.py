"""
Step1 v4: delta_rho (correlation-shift) used DIRECTLY as a per-channel score
(design doc §41 follow-up).

§41 confirmed (using the ORIGINAL paper's own diagnostic code, unmodified)
that our real SMD segments show a genuinely strong correlation-structure
shift (Δρ 6-13x over the paper's own corr_thr=0.1), not just a univariate
one. All 5 prior Step1 mechanisms (v1/v3/overlay-INTER/FFT-band) used a
CUSTOM whole-row MEAN-deviation formula of our own design. This script
instead reuses the diagnostic paper's OWN per-pair statistic directly:
Ra, Rc = correlation matrices (Pearson/Spearman/dcor) of the anomaly window
vs its preceding normal context (ctx_size=300, exactly as
`utils/tools.py:max_delta_rho_lagged`, lag=0/pairs_info=() case) -- then,
per channel c, score(c) = max_{d != c} |Ra[c,d] - Rc[c,d]| (row-max of the
delta matrix), rather than our own mean-based formula. This is the first
Step1 attempt using the exact statistic + exact context definition (real
preceding TEST points via the official label mask, not a fixed TRAIN
reference window) that the paper itself validated -- not a redesigned
analog of it.

Standalone recall@k, then fused (max, rank-normalized) with Step0's best
mechanism (§34 multi-ref ensemble). Same 9 real SMD segments. No API cost.
"""

import json
import sys
from pathlib import Path

import numpy as np

AT_UNIV_REPO = Path(r"C:\Users\김나영\Desktop\TSAD\mtsad-benchmarks-are-mostly-univariate")
sys.path.insert(0, str(AT_UNIV_REPO))
from utils.tools import spearman_matrix, dcor_matrix, preceding_normal_context, DEFAULT_CTX_SIZE  # noqa: E402

from step1v3_dino_graph_smd import BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist, load_smd, embed_channel_window
from smd_step0_multiref_ensemble import static_ref_windows

RESULTS_DIR = BASE / "experiments" / "results_step1_delta_rho_direct"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_STATIC_REFS = 5
METHODS = ["pearson", "spearman", "dcor"]


def load_smd_with_labels(entity):
    train = np.loadtxt(BASE / "mv_data" / "SMD" / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(BASE / "mv_data" / "SMD" / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(BASE / "mv_data" / "SMD" / "test_label" / f"{entity}.txt")
    return train, test, labels


def corr_matrix(X, method):
    if method == "pearson":
        return np.corrcoef(X.T)
    if method == "spearman":
        return spearman_matrix(X)
    return dcor_matrix(X)


def per_channel_delta_rho(anom, ctx, method):
    v = np.nonzero((anom.std(0) > 1e-4) & (ctx.std(0) > 1e-4))[0]
    score = {c: 0.0 for c in range(anom.shape[1])}
    if len(v) < 3:
        return score
    with np.errstate(invalid="ignore", divide="ignore"):
        Ra = corr_matrix(anom[:, v], method)
        Rc = corr_matrix(ctx[:, v], method)
    D = np.abs(Ra - Rc)
    np.fill_diagonal(D, 0.0)
    D = np.nan_to_num(D, nan=0.0)
    for pos, c in enumerate(v):
        score[c] = float(np.max(D[pos]))
    return score


def _rank_norm(d: dict) -> dict:
    ranked = sorted(d, key=lambda k: -d[k])
    n = len(ranked)
    return {c: 1.0 - (ranked.index(c) / max(1, n - 1)) for c in d}


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
    entity_dino_static = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            train, test, labels = load_smd_with_labels(entity)
            entity_data[entity] = (train, test, labels)
        train, test, labels = entity_data[entity]
        normal_mask = labels == 0

        if entity not in entity_dino_static:
            print(f"  Computing {N_STATIC_REFS} DINOv2 static ref embeddings for {entity} ...", flush=True)
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            entity_dino_static[entity] = {
                c: [embed_channel_window(train[:, c], s, e) for s, e in ref_windows] for c in range(N_CHANNELS)
            }
        dino_static = entity_dino_static[entity]

        gt_channels = [d - 1 for d in gt_dims]
        gt_set = set(gt_channels)
        k = len(gt_channels)

        anom = test[cs:ce]
        ctx = preceding_normal_context(cs, test, normal_mask, ctx_size=DEFAULT_CTX_SIZE)
        if ctx is None:
            print(f"[SKIP-NOCTX] {seg_id}: insufficient preceding normal context", flush=True)
            continue

        print(f"[{seg_id}] scoring (delta_rho direct + DINOv2 multi-ref fusion) ...", flush=True)

        method_recalls = {}
        method_combined_recalls = {}
        # Step0 (for fusion, computed once)
        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)
        dino_score = {}
        for c in range(N_CHANNELS):
            dyn_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            dists = [cosine_dist(ref_emb, dyn_emb) for ref_emb in dino_static[c]]
            dino_score[c] = float(np.mean(dists))
        recall_dino_alone = len(set(sorted(dino_score, key=lambda c: -dino_score[c])[:k]) & gt_set) / k
        dino_rank = _rank_norm(dino_score)

        for method in METHODS:
            rho_score = per_channel_delta_rho(anom, ctx, method)
            recall_rho_alone = len(set(sorted(rho_score, key=lambda c: -rho_score[c])[:k]) & gt_set) / k
            rho_rank = _rank_norm(rho_score)
            combined = {c: max(dino_rank[c], rho_rank[c]) for c in range(N_CHANNELS)}
            recall_combined = len(set(sorted(combined, key=lambda c: -combined[c])[:k]) & gt_set) / k
            method_recalls[method] = recall_rho_alone
            method_combined_recalls[method] = recall_combined

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "gt_channels_0idx": gt_channels, "k": k,
            "recall_dino_alone": recall_dino_alone,
            "recall_rho_alone": method_recalls, "recall_combined": method_combined_recalls,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} dino={recall_dino_alone:.2f} rho_alone={method_recalls} combined={method_combined_recalls}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    summary = {"recall_dino_alone": float(np.mean([r["recall_dino_alone"] for r in rows]))}
    for method in METHODS:
        summary[f"recall_rho_alone_{method}"] = float(np.mean([r["recall_rho_alone"][method] for r in rows]))
        summary[f"recall_combined_{method}"] = float(np.mean([r["recall_combined"][method] for r in rows]))

    print(f"\n{'='*60}\nSTEP1 v4 DELTA-RHO DIRECT SUMMARY (n={len(rows)})\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"n": len(rows), **summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
