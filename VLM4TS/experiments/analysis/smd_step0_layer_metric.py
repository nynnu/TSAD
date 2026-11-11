"""
Step0 redesign, hypothesis 4: embedding source (CLS vs patch-mean) x distance
metric (cosine vs L2) (design doc §33 follow-up, candidate #2).

Same single-static-reference baseline as step1v3 (isolate this one variable,
same methodology as §31-33) -- only what we extract from DINOv2's forward
pass and how we compare it changes. Patch-token mean-pool may blur a single
anomalous channel's local signal into a global average; CLS token is trained
as DINOv2's own global semantic summary and might be more decisive. Cosine
distance ignores magnitude; L2 keeps it. Both features come from ONE forward
pass per window (no extra compute cost over the original).
"""

import json
from io import BytesIO

import numpy as np
import torch

from step1v3_dino_graph_smd import (
    BASE, SEGMENTS, N_CHANNELS, WIN, _centered_window, _get_model, _render_single, load_smd,
)

RESULTS_DIR = BASE / "experiments" / "results_step0_layer_metric"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"


def _embed_both(img_b64: str) -> dict:
    import base64
    import torchvision.transforms as T
    from PIL import Image
    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    from step1v3_dino_graph_smd import _device
    x = tfm(img).unsqueeze(0).to(_device)
    model = _get_model()
    with torch.no_grad():
        feats = model.forward_features(x)
        patch_mean = feats["x_norm_patchtokens"].mean(dim=1).squeeze(0).cpu().numpy()
        cls = feats["x_norm_clstoken"].squeeze(0).cpu().numpy()
    return {"patch_mean": patch_mean, "cls": cls}


def embed_channel_window_both(values: np.ndarray, s: int, e: int) -> dict:
    seg = values[s:e]
    lo, hi = float(seg.min()), float(seg.max())
    norm = (seg - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(seg)
    return _embed_both(_render_single(norm))


def cosine_dist(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def l2_dist(a, b):
    return float(np.linalg.norm(a - b))


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


CONDITIONS = ["patch_cosine", "patch_l2", "cls_cosine", "cls_l2"]


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    entity_data = {}
    entity_static = {}  # entity -> {c: {"patch_mean":.., "cls":..}}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        if entity not in entity_static:
            print(f"  Computing static reference embeddings for {entity} ...", flush=True)
            s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
            entity_static[entity] = {c: embed_channel_window_both(train[:, c], s_static, e_static) for c in range(N_CHANNELS)}
        static = entity_static[entity]

        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] ...", flush=True)
        gt_channels = [d - 1 for d in gt_dims]
        gt_set = set(gt_channels)
        k = len(gt_channels)

        scores = {cond: {} for cond in CONDITIONS}
        for c in range(N_CHANNELS):
            dyn = embed_channel_window_both(test[:, c], s_dyn, e_dyn)
            scores["patch_cosine"][c] = cosine_dist(static[c]["patch_mean"], dyn["patch_mean"])
            scores["patch_l2"][c] = l2_dist(static[c]["patch_mean"], dyn["patch_mean"])
            scores["cls_cosine"][c] = cosine_dist(static[c]["cls"], dyn["cls"])
            scores["cls_l2"][c] = l2_dist(static[c]["cls"], dyn["cls"])

        recalls = {}
        for cond in CONDITIONS:
            ranked = sorted(scores[cond], key=lambda c: -scores[cond][c])
            recalls[cond] = len(set(ranked[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce,
            "gt_channels_0idx": gt_channels, "k": k, "recalls": recalls,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} recalls={recalls}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return

    summary = {cond: float(np.mean([r["recalls"][cond] for r in rows])) for cond in CONDITIONS}
    print(f"\n{'='*60}\nSTEP0 LAYER/METRIC SUMMARY (n={len(rows)})\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"n": len(rows), **summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
