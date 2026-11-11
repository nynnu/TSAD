"""
Background/whitespace masking before DINOv2 patch-mean pooling (AnomalyDINO,
WACV 2025, design doc candidate #5, audited 2026-08-05).

Read in full (raw-text, ingested prior to this session) and official GitHub
code (github.com/dammsi/AnomalyDINO, cloned). AnomalyDINO's masking
(`src/backbones.py:DINOv2Wrapper.compute_background_mask`) thresholds the
FIRST PCA component of DINOv2's own patch features to separate a
foreground OBJECT from background in natural industrial-inspection photos
(MVTec-AD) -- built for images with a centered object on a distinct
background. That exact mechanism does not transfer: our line-plot renders
have no PCA-separable object/background structure (`ax.axis("off")` already
strips gridlines/axes/legend in every render function used all session --
there was never a "grid/axis" to remove). What DOES transfer is the paper's
underlying PRINCIPLE (patch-mean pooling over uninformative background
patches dilutes the embedding -- restrict comparison to informative patches
only): most of our 224x224 rendered canvas is blank white space except where
the line trace passes through, yet embed_channel_window's current
`.mean(dim=1)` pools over ALL patch tokens uniformly, including patches that
are pure white background.

Adaptation: for each rendered window (static and dynamic), build a per-patch
boolean mask directly from PIXEL content (does this 14x14 patch cell contain
any non-white pixel, i.e. does the line trace pass through it?) on the actual
224x224 image DINOv2 sees, then mean-pool ONLY over line-containing patches
instead of all patches. Same single-static-reference baseline as §31-33
(isolate this one variable), same 9 real SMD segments, no API cost.
"""

import json

import numpy as np
import torch

from step1v3_dino_graph_smd import BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, _get_model, _render_single, _device, load_smd

RESULTS_DIR = BASE / "experiments" / "results_step0_background_masking"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
PATCH_SIZE = 14
IMG_SIZE = 224
GRID = IMG_SIZE // PATCH_SIZE  # 16


def _embed_masked(img_b64: str) -> tuple:
    """Returns (full_mean_embedding, masked_mean_embedding)."""
    import base64
    from io import BytesIO
    import torchvision.transforms as T
    from PIL import Image

    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
    tfm_resize = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    x_raw = tfm_resize(img)  # (3, 224, 224), values in [0,1], NOT normalized yet

    # per-patch mask: does this 14x14 cell contain any non-white pixel? (line present)
    is_nonwhite = (x_raw.min(dim=0).values < 0.95)  # (224, 224) bool -- white ~= (1,1,1)
    patch_has_signal = np.zeros((GRID, GRID), dtype=bool)
    for i in range(GRID):
        for j in range(GRID):
            cell = is_nonwhite[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, j * PATCH_SIZE:(j + 1) * PATCH_SIZE]
            patch_has_signal[i, j] = bool(cell.any())
    mask_flat = patch_has_signal.reshape(-1)  # (256,)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = normalize(x_raw).unsqueeze(0).to(_device)
    model = _get_model()
    with torch.no_grad():
        tokens = model.forward_features(x)["x_norm_patchtokens"].squeeze(0)  # (256, dim)

    full_mean = tokens.mean(dim=0).cpu().numpy()
    if mask_flat.any():
        masked_mean = tokens[torch.from_numpy(mask_flat)].mean(dim=0).cpu().numpy()
    else:
        masked_mean = full_mean  # fallback: no signal patches found (shouldn't happen)
    return full_mean, masked_mean


def embed_channel_window_masked(values: np.ndarray, s: int, e: int) -> tuple:
    seg = values[s:e]
    lo, hi = float(seg.min()), float(seg.max())
    norm = (seg - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(seg)
    return _embed_masked(_render_single(norm))


def cosine_dist(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


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
    entity_static = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        if entity not in entity_static:
            print(f"  Computing static reference embeddings (full+masked) for {entity} ...", flush=True)
            s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
            entity_static[entity] = {c: embed_channel_window_masked(train[:, c], s_static, e_static) for c in range(N_CHANNELS)}
        static = entity_static[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] ...", flush=True)
        gt_channels = [d - 1 for d in gt_dims]
        gt_set = set(gt_channels)
        k = len(gt_channels)

        score_full, score_masked = {}, {}
        for c in range(N_CHANNELS):
            dyn_full, dyn_masked = embed_channel_window_masked(test[:, c], s_dyn, e_dyn)
            score_full[c] = cosine_dist(static[c][0], dyn_full)
            score_masked[c] = cosine_dist(static[c][1], dyn_masked)

        recall_full = len(set(sorted(score_full, key=lambda c: -score_full[c])[:k]) & gt_set) / k
        recall_masked = len(set(sorted(score_masked, key=lambda c: -score_masked[c])[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "gt_channels_0idx": gt_channels, "k": k,
            "recall_full": recall_full, "recall_masked": recall_masked,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} full={recall_full:.2f} masked={recall_masked:.2f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    summary = {
        "recall_full_patch_mean": float(np.mean([r["recall_full"] for r in rows])),
        "recall_masked_patch_mean": float(np.mean([r["recall_masked"] for r in rows])),
    }
    print(f"\n{'='*60}\nBACKGROUND-MASKED PATCH-MEAN SUMMARY (n={len(rows)})\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"n": len(rows), **summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
