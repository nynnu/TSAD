"""
LASSO-based channel relevance (CALAD, arXiv:2605.23139) -- zero-shot
adaptation (design doc candidate #3, audited 2026-08-05).

Read in full (raw-text, ingested prior to this session -- tsad/raw-text/CALAD
(raw-text).md). Official code repo linked in the paper (github.com/hirundo1218/
CALAD) currently returns 404 (not published / private) -- confirmed by direct
clone attempt. Implemented directly from the paper's own exact formula
(Section 3.3, Eq. 1-2, "Implementation Details" lambda=0.001) since no
executable reference exists to check against; `sklearn.linear_model.Lasso`
is the standard, unambiguous implementation of that exact optimization
problem, not a reinterpretation.

CALAD's original recipe (fully trained, NOT reusable zero-shot per the wiki
page's own noted limitation):
  1. Train a transformer autoencoder to reconstruct all channels.
  2. e_{i,c} = |x_{i,c} - xhat_{i,c}| (per-timestep, per-channel recon error).
  3. y_i = mean_c(e_{i,c})  (Eq. 2: per-timestep GLOBAL anomaly signal).
  4. LASSO: X = raw multivariate window (predictors), y = y_i (response).
     Non-zero-coefficient channels = "anomaly-relevant".

Zero-shot substitution tested here (this project's own idea, not the paper's):
replace the trained-autoencoder reconstruction error with the SAME per-timestep
z-score signal already validated in §41/§42 (official code from "Anomalies in
MTS Benchmarks Are Mostly Univariate", preceding_normal_context, ctx_size=300)
-- y_i = mean_c |z_{i,c}| against that context, no training required. X = the
same WIN=224 centered window used throughout today's Step0 experiments,
standardized against the same context's per-channel mean/std (so X and y
share one consistent scale).

Reports: (1) LASSO channel-coefficient ranking standalone recall@k, (2) fused
(max, rank-normalized) with Step0's §34 multi-ref ensemble, (3) the sparsity
check CALAD itself highlighted (MSL: 9/55 non-zero) -- mean number of non-zero
channels per segment here, as direct evidence for/against the k<=4 design
choice. No API cost -- local only.
"""

import json
import sys
from pathlib import Path

import numpy as np

# NOTE: torch (imported transitively via step1v3_dino_graph_smd) must load
# BEFORE numba/llvmlite (imported transitively via the univariate-diagnostic
# repo's utils.tools -> dcor). Importing dcor first causes a DLL init clash
# with torch's c10.dll on this Windows/CPU setup (OSError WinError 1114).
from step1v3_dino_graph_smd import BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist, embed_channel_window
from smd_step0_multiref_ensemble import static_ref_windows

from sklearn.linear_model import Lasso

AT_UNIV_REPO = Path(r"C:\Users\김나영\Desktop\TSAD\mtsad-benchmarks-are-mostly-univariate")
sys.path.insert(0, str(AT_UNIV_REPO))
from utils.tools import preceding_normal_context, DEFAULT_CTX_SIZE  # noqa: E402

RESULTS_DIR = BASE / "experiments" / "results_smd_lasso_channel_relevance"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_STATIC_REFS = 5
LASSO_ALPHA = 0.001  # exactly as CALAD's paper (Implementation Details)


def load_smd_with_labels(entity):
    train = np.loadtxt(BASE / "mv_data" / "SMD" / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(BASE / "mv_data" / "SMD" / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(BASE / "mv_data" / "SMD" / "test_label" / f"{entity}.txt")
    return train, test, labels


def lasso_channel_scores(window: np.ndarray, ctx: np.ndarray) -> dict:
    mu, sigma = ctx.mean(axis=0), ctx.std(axis=0)
    sigma = np.where(sigma < 1e-4, 1.0, sigma)
    X = (window - mu) / sigma  # standardized predictors, same scale as z
    z = np.abs(X)
    y = z.mean(axis=1)  # Eq.2 analog: per-timestep mean |z| across channels
    model = Lasso(alpha=LASSO_ALPHA, max_iter=20000)
    model.fit(X, y)
    coefs = np.abs(model.coef_)
    return {c: float(coefs[c]) for c in range(N_CHANNELS)}, int(np.sum(coefs > 1e-10))


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
            entity_data[entity] = load_smd_with_labels(entity)
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

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)
        window = test[s_dyn:e_dyn]
        ctx = preceding_normal_context(cs, test, normal_mask, ctx_size=DEFAULT_CTX_SIZE)
        if ctx is None:
            print(f"[SKIP-NOCTX] {seg_id}", flush=True)
            continue

        print(f"[{seg_id}] LASSO channel relevance ...", flush=True)
        lasso_score, n_nonzero = lasso_channel_scores(window, ctx)

        dino_score = {}
        for c in range(N_CHANNELS):
            dyn_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            dists = [cosine_dist(ref_emb, dyn_emb) for ref_emb in dino_static[c]]
            dino_score[c] = float(np.mean(dists))

        recall_lasso_alone = len(set(sorted(lasso_score, key=lambda c: -lasso_score[c])[:k]) & gt_set) / k
        recall_dino_alone = len(set(sorted(dino_score, key=lambda c: -dino_score[c])[:k]) & gt_set) / k

        lasso_rank = _rank_norm(lasso_score)
        dino_rank = _rank_norm(dino_score)
        combined = {c: max(dino_rank[c], lasso_rank[c]) for c in range(N_CHANNELS)}
        recall_combined = len(set(sorted(combined, key=lambda c: -combined[c])[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "gt_channels_0idx": gt_channels, "k": k,
            "n_nonzero_lasso": n_nonzero,
            "recall_lasso_alone": recall_lasso_alone, "recall_dino_alone": recall_dino_alone,
            "recall_combined": recall_combined,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} n_nonzero={n_nonzero}/38 lasso={recall_lasso_alone:.2f} "
              f"dino={recall_dino_alone:.2f} combined={recall_combined:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    summary = {
        "n": len(rows),
        "mean_n_nonzero_lasso": float(np.mean([r["n_nonzero_lasso"] for r in rows])),
        "recall_lasso_alone": float(np.mean([r["recall_lasso_alone"] for r in rows])),
        "recall_dino_alone": float(np.mean([r["recall_dino_alone"] for r in rows])),
        "recall_combined": float(np.mean([r["recall_combined"] for r in rows])),
    }
    print(f"\n{'='*60}\nLASSO CHANNEL RELEVANCE SUMMARY (n={len(rows)})\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
