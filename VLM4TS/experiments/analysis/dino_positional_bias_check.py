"""
DINOv2's own intra-image positional bias (design doc candidate #4, audited
2026-08-05): "Positional Bias in Multimodal Embedding Models" (AAAI 2026,
arXiv:2511.11216) found CLIP-family image encoders favor BOTH ends
(U-shaped) of an image, structurally, across model size/resolution/patch
size/loss function -- independent of design choices. This is a check on
DINOv2 (our committed Step0 backbone) itself, not the VLM's judgment-level
order bias already tested in Step3.

Read in full (raw-text, ingested prior to this session -- Section 3.2's exact
protocol: isolate ONE content segment, move it across positions while masking
the rest with the model's neutral/mean RGB, evaluate how a downstream signal
varies by position, holding content fixed). Official code repo
(github.com/tiiuae/PosBias) cloned and checked: contains only the paper PDF
and a README stating "the code used in our experiments will be released
soon" -- no executable code available. Implemented directly from the paper's
own documented protocol.

Adaptation for DINOv2 (no paired text encoder, so no retrieval-accuracy
metric available): a FIXED content crop (a real SMD channel's line-plot
render) is placed into ONE of N_POSITIONS equal-width horizontal bands of an
otherwise neutral-gray canvas (gray = DINOv2's own ImageNet normalization
mean, the direct analog of the paper's CLIP-mean masking value) -- same
content, different position, nothing else changes. Embed each of the
N_POSITIONS resulting images with DINOv2 (patch-mean, our Step0 mechanism).
If DINOv2 has no positional bias, all N_POSITIONS embeddings of the SAME
content should be highly mutually similar (content-driven, not
position-driven); if bias exists (echoing CLIP's U-shape), similarity should
vary systematically by position, e.g. edges (0, N-1) differing more from the
rest than center positions do.

Repeated over K distinct real SMD content crops (different channels/entities/
windows) to average out single-sample noise. Local only, no API cost.
"""

import json
from pathlib import Path

import numpy as np

from step1v3_dino_graph_smd import BASE, N_CHANNELS, load_smd, _embed, _centered_window

RESULTS_DIR = BASE / "experiments" / "results_dino_positional_bias"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

N_POSITIONS = 6  # matches the paper's Figure 1/Figure 2 6-segment convention
NEUTRAL_GRAY = (0.485, 0.456, 0.406)  # DINOv2's own ImageNet normalization mean -- our analog of the paper's CLIP-mean masking fill
K_SAMPLES = 10

# K_SAMPLES distinct (entity, channel, window-center) content crops, spanning
# the 3 entities used throughout this session for consistency.
CONTENT_SPECS = [
    ("machine-1-1", 0, 5000), ("machine-1-1", 5, 12000), ("machine-1-1", 12, 20000),
    ("machine-1-2", 2, 6000), ("machine-1-2", 9, 11000), ("machine-1-2", 20, 18000),
    ("machine-1-5", 1, 4000), ("machine-1-5", 15, 9000), ("machine-1-5", 24, 15000),
    ("machine-1-5", 31, 21000),
]
CROP_LEN = 100


def render_positioned(content: np.ndarray, position: int) -> str:
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = float(content.min()), float(content.max())
    norm = (content - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(content)

    fig, ax = plt.subplots(figsize=(9, 1.5), dpi=100)
    ax.set_xlim(0, N_POSITIONS)
    ax.set_ylim(0, 1)
    for i in range(N_POSITIONS):
        if i != position:
            ax.axvspan(i, i + 1, color=NEUTRAL_GRAY, zorder=0)
    x = np.linspace(position, position + 1, len(norm))
    ax.plot(x, norm, color="black", linewidth=1.2, zorder=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


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
    entity_test = {}

    for si, (entity, channel, center) in enumerate(CONTENT_SPECS):
        sample_id = f"sample{si}_{entity}_ch{channel}_c{center}"
        if sample_id in checkpoint:
            print(f"[SKIP] {sample_id}", flush=True)
            continue
        if entity not in entity_test:
            _, test = load_smd(entity)
            entity_test[entity] = test
        test = entity_test[entity]
        s, e = _centered_window(len(test), center, CROP_LEN)
        content = test[s:e, channel]

        print(f"[{sample_id}] embedding at {N_POSITIONS} positions ...", flush=True)
        embeds = []
        for p in range(N_POSITIONS):
            img_b64 = render_positioned(content, p)
            embeds.append(_embed(img_b64))

        sim_matrix = [[cosine_sim(embeds[i], embeds[j]) for j in range(N_POSITIONS)] for i in range(N_POSITIONS)]
        # distinctiveness(p) = 1 - mean similarity of position p's embedding to all OTHER positions
        distinctiveness = []
        for p in range(N_POSITIONS):
            others = [sim_matrix[p][q] for q in range(N_POSITIONS) if q != p]
            distinctiveness.append(1.0 - float(np.mean(others)))

        checkpoint[sample_id] = {"entity": entity, "channel": channel, "center": center,
                                  "sim_matrix": sim_matrix, "distinctiveness": distinctiveness}
        save_checkpoint(checkpoint)
        print(f"[OK] {sample_id}: distinctiveness={['%.4f' % d for d in distinctiveness]}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    mean_distinct = np.mean([r["distinctiveness"] for r in rows], axis=0)
    mean_pairwise_sim = np.mean([np.mean(r["sim_matrix"]) for r in rows])

    print(f"\n{'='*60}\nDINOv2 POSITIONAL BIAS SUMMARY (n={len(rows)} content samples, {N_POSITIONS} positions)\n{'='*60}")
    print(f"Mean overall pairwise similarity (1.0 = fully translation-invariant): {mean_pairwise_sim:.4f}")
    print("Mean distinctiveness by position (higher = more affected by being at that position):")
    for p, d in enumerate(mean_distinct):
        print(f"  position {p}: {d:.4f}")
    edge_mean = float(np.mean([mean_distinct[0], mean_distinct[-1]]))
    center_mean = float(np.mean(mean_distinct[1:-1]))
    print(f"Edge positions (0,{N_POSITIONS-1}) mean distinctiveness = {edge_mean:.4f}")
    print(f"Center positions mean distinctiveness = {center_mean:.4f}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n_samples": len(rows), "mean_pairwise_similarity": float(mean_pairwise_sim),
        "mean_distinctiveness_by_position": mean_distinct.tolist(),
        "edge_mean_distinctiveness": edge_mean, "center_mean_distinctiveness": center_mean,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
