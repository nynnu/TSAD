"""
Step0 redesign, hypothesis 2: shared-scale normalization (design doc §32 follow-up).

§32 ruled out window-width dilution as the dominant Step0 bottleneck (net
effect ~0, gains and losses cancel out). Second candidate from §32's own
"next hypothesis" note: embed_channel_window currently min-max normalizes
EACH window (static, dynamic) independently to its own local [0,1] range.
This destroys amplitude information -- a channel that spikes 5x its normal
range gets rescaled to fill the same [0,1] frame as a channel that barely
moves, so a level-shift/magnitude anomaly looks like "a normal-sized wiggle"
after renormalization. Real anomalies are often magnitude anomalies (SMD is
server-metric data: CPU/memory/queue spikes), so this could be erasing the
exact signal Step0 needs.

Fix tested here: normalize both static and dynamic windows of a channel using
the SAME reference scale -- that channel's global train min/max -- and use a
FIXED y-axis view (ax.set_ylim) so a dynamic value that exceeds the normal
range is visibly clipped/pushed to the frame edge instead of being silently
rescaled to fit. Same 9 real SMD segments, same fixed WIN=224 (only the
normalization changes -- isolate this one variable, per the same methodology
as §31/§32), no API cost (local DINOv2 only).
"""

import json

import numpy as np

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, _embed, _render_single,
    cosine_dist, load_smd,
)

RESULTS_DIR = BASE / "experiments" / "results_step0_global_scale"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

VIEW_LO, VIEW_HI = -0.3, 1.3  # fixed view window around the [0,1] global-normalized range


def _render_single_fixed_ylim(values: np.ndarray) -> str:
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    ax.plot(np.arange(len(values)), values, color="black", linewidth=1.2)
    ax.set_ylim(VIEW_LO, VIEW_HI)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def embed_channel_window_global(values: np.ndarray, s: int, e: int, global_lo: float, global_hi: float) -> np.ndarray:
    seg = values[s:e]
    rng = global_hi - global_lo
    norm = (seg - global_lo) / rng if rng > 1e-9 else np.zeros_like(seg)
    return _embed(_render_single_fixed_ylim(norm))


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
    entity_global_stats = {}  # entity -> {c: (lo, hi)}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        if entity not in entity_global_stats:
            entity_global_stats[entity] = {
                c: (float(train[:, c].min()), float(train[:, c].max())) for c in range(N_CHANNELS)
            }
        global_stats = entity_global_stats[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)
        s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)

        print(f"[{seg_id}] ...", flush=True)
        gt_channels = [d - 1 for d in gt_dims]
        step0_score = {}
        for c in range(N_CHANNELS):
            lo, hi = global_stats[c]
            static_emb = embed_channel_window_global(train[:, c], s_static, e_static, lo, hi)
            dynamic_emb = embed_channel_window_global(test[:, c], s_dyn, e_dyn, lo, hi)
            step0_score[c] = cosine_dist(static_emb, dynamic_emb)

        ranked = sorted(step0_score, key=lambda c: -step0_score[c])
        k = len(gt_channels)
        gt_set = set(gt_channels)
        recall = len(set(ranked[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce,
            "gt_channels_0idx": gt_channels, "k": k,
            "recall_global_scale": recall, "ranked_top10": ranked[:10],
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} gt={gt_channels} recall_global_scale={recall:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    r_global = np.mean([r["recall_global_scale"] for r in rows])

    step1v3_ckpt_path = BASE / "experiments" / "results_step1v3_dino_graph_smd" / "checkpoint.json"
    r_fixed = None
    if step1v3_ckpt_path.exists():
        step1v3 = json.loads(step1v3_ckpt_path.read_text(encoding="utf-8"))
        fixed_vals = [step1v3[r["entity"] + f"_{r['cs']}_{r['ce']}"]["recall_step0"]
                      for r in rows if r["entity"] + f"_{r['cs']}_{r['ce']}" in step1v3]
        if fixed_vals:
            r_fixed = float(np.mean(fixed_vals))

    print(f"\n{'='*60}\nSTEP0 GLOBAL-SCALE NORMALIZATION SUMMARY (n={len(rows)})\n{'='*60}")
    print(f"Local per-window normalization recall@k = {r_fixed}")
    print(f"Global-scale normalization recall@k     = {r_global:.4f}")
    if r_fixed is not None:
        print(f"Delta                                    = {r_global - r_fixed:+.4f}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "recall_local_norm": r_fixed, "recall_global_scale": r_global,
        "delta": (r_global - r_fixed) if r_fixed is not None else None,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
