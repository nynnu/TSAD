"""
Step1 redesign, FFT-band-wise channel correlation (CATCH, ICLR 2025).

Read in full (paper's raw-text, 9914 lines, ingested prior to this session --
tsad/raw-text/CATCH (raw-text).md) and the method section (lines 340-410,
"FFT&Patching Layer": FFT of each channel -> patches along the FREQUENCY
axis, {P1..PL} = Patching(FFT(X)), channel correlations discovered
SEPARATELY per frequency patch/band -- not a single global correlation).
Official code (thuml org has no public repo for this one; the paper's core
CFM/Mask-Generator machinery is a TRAINED component, not reusable zero-shot --
ported here is only the underlying INSIGHT, matching this project's own
"[아이디어 사시점]" note on the wiki page: "FFT 기반 대역별 상관관계 스크리닝을
Step1에 보강하는 것을 검토할 가치".

Prior Step1 attempts (all single time-domain correlation/graph mechanisms):
  v1 (naive correlation, synthetic): failed, -0.074
  v2 (whole-row graph deviation, synthetic): succeeded, +0.120
  v3 (same mechanism as v2, REAL SMD): failed, -0.054  <- direct predecessor
  overlay-INTER (report12's method, REAL SMD): failed, -0.138
v3's mechanism is exactly what this script replaces: instead of ONE
time-domain graph-deviation score per channel, split each window's FFT
spectrum into B=5 frequency bands (matching CATCH's own Figure 1 illustrative
split) and compute the graph-deviation score PER BAND, then take the MAX
across bands per channel -- CATCH's core claim is that an anomaly may only
show up as a channel-relationship shift in ONE specific band, so averaging
across bands (as v3 implicitly does by using only the time domain) would
dilute exactly the signal CATCH says matters.

Fused (max, rank-normalized) with Step0's best mechanism so far (§34 multi-ref
ensemble). Same 9 real SMD segments, same recall@k metric. No API cost --
local FFT + the existing local DINOv2 multi-ref scoring only.
"""

import json

import numpy as np

from step1v3_dino_graph_smd import BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, cosine_dist, load_smd, embed_channel_window
from smd_step0_multiref_ensemble import static_ref_windows

RESULTS_DIR = BASE / "experiments" / "results_step1_fft_band_graph"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_STATIC_REFS = 5  # reuse §34's winning Step0 config
N_BANDS = 5  # matches CATCH Figure 1's illustrative 5-band split


def band_magnitude_spectra(window: np.ndarray) -> list[np.ndarray]:
    """window: (T, N_CHANNELS) -> list of N_BANDS arrays, each (n_freqs_in_band, N_CHANNELS) magnitude."""
    spec = np.abs(np.fft.rfft(window, axis=0))  # (T//2+1, N_CHANNELS)
    n_freqs = spec.shape[0]
    edges = np.linspace(0, n_freqs, N_BANDS + 1).astype(int)
    return [spec[edges[i]:edges[i + 1]] for i in range(N_BANDS)]


def band_channel_corr(band_spec: np.ndarray, c: int, d: int) -> float:
    """cosine similarity between channel c's and d's magnitude spectrum within one band."""
    a, b = band_spec[:, c], band_spec[:, d]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def fft_band_graph_score(static_window: np.ndarray, dynamic_window: np.ndarray) -> dict:
    static_bands = band_magnitude_spectra(static_window)
    dynamic_bands = band_magnitude_spectra(dynamic_window)
    score = {}
    for c in range(N_CHANNELS):
        band_devs = []
        for static_b, dynamic_b in zip(static_bands, dynamic_bands):
            devs = [abs(band_channel_corr(dynamic_b, c, d) - band_channel_corr(static_b, c, d))
                    for d in range(N_CHANNELS) if d != c]
            band_devs.append(float(np.mean(devs)))
        score[c] = max(band_devs)  # CATCH's insight: take the band where the shift is largest
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
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        # Step0 multi-ref DINOv2 static embeddings (§34 config, reused for fusion)
        if entity not in entity_dino_static:
            print(f"  Computing {N_STATIC_REFS} DINOv2 static ref embeddings for {entity} ...", flush=True)
            ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
            entity_dino_static[entity] = {
                c: [embed_channel_window(train[:, c], s, e) for s, e in ref_windows] for c in range(N_CHANNELS)
            }
        dino_static = entity_dino_static[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)
        s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)

        print(f"[{seg_id}] scoring (FFT band graph + DINOv2 multi-ref fusion) ...", flush=True)
        gt_channels = [d - 1 for d in gt_dims]
        gt_set = set(gt_channels)
        k = len(gt_channels)

        # Step0: DINOv2 multi-ref (§34)
        dino_score = {}
        for c in range(N_CHANNELS):
            dyn_emb = embed_channel_window(test[:, c], s_dyn, e_dyn)
            dists = [cosine_dist(ref_emb, dyn_emb) for ref_emb in dino_static[c]]
            dino_score[c] = float(np.mean(dists))

        # Step1: FFT band-wise channel-graph deviation
        fft_score = fft_band_graph_score(train[s_static:e_static], test[s_dyn:e_dyn])

        dino_rank = _rank_norm(dino_score)
        fft_rank = _rank_norm(fft_score)
        combined = {c: max(dino_rank[c], fft_rank[c]) for c in range(N_CHANNELS)}

        recall_dino_alone = len(set(sorted(dino_score, key=lambda c: -dino_score[c])[:k]) & gt_set) / k
        recall_fft_alone = len(set(sorted(fft_score, key=lambda c: -fft_score[c])[:k]) & gt_set) / k
        recall_combined = len(set(sorted(combined, key=lambda c: -combined[c])[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "gt_channels_0idx": gt_channels, "k": k,
            "recall_dino_alone": recall_dino_alone, "recall_fft_alone": recall_fft_alone,
            "recall_combined": recall_combined,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} dino={recall_dino_alone:.2f} fft={recall_fft_alone:.2f} combined={recall_combined:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    summary = {
        "recall_dino_alone": float(np.mean([r["recall_dino_alone"] for r in rows])),
        "recall_fft_alone": float(np.mean([r["recall_fft_alone"] for r in rows])),
        "recall_combined": float(np.mean([r["recall_combined"] for r in rows])),
    }
    print(f"\n{'='*60}\nSTEP1 FFT-BAND GRAPH SUMMARY (n={len(rows)})\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"n": len(rows), **summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
