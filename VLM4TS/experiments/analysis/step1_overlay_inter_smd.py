"""
report 12 §5's actual validated "INTER" mechanism, tested with our recall@k
metric on real SMD data (2026-08-05). This directly closes the loop opened
by Step1 v3's failure (step1v3_dino_graph_smd.py, per-channel relational
graph -- negative result). Unlike v3, this is NOT a new hypothesis: it is
report 12's own "상관 높은 4채널 overlay plot → DINOv2 patch scoring → 그룹
집계" method (INTER, avg 0.727 across 5 SMD entities in their F1 evaluation),
re-run here with our recall@k(GT root-cause channels) metric instead of
their F1, on the same 9 segments used throughout today's SMD experiments.

Method
------
1. Static reference (train) correlation matrix (38x38, Pearson) -> greedy
   group channels into ~4-channel clusters by correlation (closest match to
   report 12's "상관 높은 4채널" grouping, done simply/deterministically here).
2. For each segment: render each group's channels OVERLAID on one axes
   (matches report 12's INTER rendering, and today's established finding
   that overlay > stacked-subplot, wiki concept "Time Series to Image
   Encoding" 2026-08-05 update) for both the static and dynamic windows.
3. Group score = DINOv2 cosine distance between static-overlay and
   dynamic-overlay embeddings (same Step0 mechanism, applied to the group
   image instead of a single-channel image).
4. Each channel inherits its group's score (channels in the same group are
   tied). Rank all 38 channels by this score, compute recall@k against GT.
5. Compare to Step0-alone (per-channel, independent rendering) on the SAME
   segments -- both already available from step1v3_dino_graph_smd.py's
   checkpoint (recall_step0 field), reused here without recomputation.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
RESULTS_DIR = BASE / "experiments" / "results_step1_overlay_inter_smd"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
V3_CHECKPOINT = BASE / "experiments" / "results_step1v3_dino_graph_smd" / "checkpoint.json"
N_CHANNELS = 38
WIN = 224
GROUP_SIZE = 4
DINO_MODEL_NAME = "dinov2_vits14"

SEGMENTS = [
    ("machine-1-1", (15849, 16368), [1, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (18071, 18528), [1, 2, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (24679, 24682), [9, 13, 14, 15]),
    ("machine-1-2", (4629, 4688), [9, 10, 11, 13, 15, 18]),
    ("machine-1-2", (15925, 15973), [6, 7, 10, 11, 13, 14, 20, 30]),
    ("machine-1-2", (22264, 22336), [1, 2, 3, 4]),
    ("machine-1-5", (10620, 10637), [1, 2, 3, 4, 7, 24, 26, 32]),
    ("machine-1-5", (14068, 14072), [19, 20, 21, 22, 28, 31]),
    ("machine-1-5", (21287, 21298), [1, 2, 3, 4, 6, 7, 24, 26]),
]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Loading DINOv2 ({DINO_MODEL_NAME}) on {_device} ...", flush=True)
        _model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL_NAME, pretrained=True)
        _model = _model.to(_device).eval()
    return _model


def load_smd(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    return train, test


def _centered_window(arr_len, center, win):
    s = max(0, min(arr_len - win, center - win // 2))
    return s, s + win


def greedy_correlation_groups(train: np.ndarray, group_size: int = GROUP_SIZE) -> list[list[int]]:
    """Group channels by Pearson correlation on a stable long sample of train."""
    sample = train[:: max(1, len(train) // 5000)]  # subsample for speed, keep up to ~5000 rows
    corr = np.corrcoef(sample.T)
    corr = np.nan_to_num(corr, nan=0.0)
    remaining = list(range(N_CHANNELS))
    groups = []
    while remaining:
        seed = remaining.pop(0)
        if not remaining:
            groups.append([seed])
            break
        sims = [(c, abs(corr[seed, c])) for c in remaining]
        sims.sort(key=lambda x: -x[1])
        take = [c for c, _ in sims[: group_size - 1]]
        for c in take:
            remaining.remove(c)
        groups.append([seed] + take)
    return groups


def render_overlay(arr: np.ndarray, s: int, e: int, channels: list[int]) -> str:
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    seg = arr[s:e]
    for c in channels:
        v = seg[:, c]
        lo, hi = float(v.min()), float(v.max())
        norm = (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)
        ax.plot(np.arange(len(norm)), norm, linewidth=0.8)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _embed(img_b64: str) -> np.ndarray:
    import torchvision.transforms as T
    from PIL import Image
    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0).to(_device)
    model = _get_model()
    with torch.no_grad():
        feat = model.forward_features(x)["x_norm_patchtokens"].mean(dim=1)
    return feat.squeeze(0).cpu().numpy()


def cosine_dist(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def load_checkpoint(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(CHECKPOINT)
    v3_checkpoint = load_checkpoint(V3_CHECKPOINT)  # to reuse recall_step0 baseline

    entity_data = {}
    entity_groups = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]

        if entity not in entity_groups:
            print(f"  Computing correlation groups for {entity} ...", flush=True)
            entity_groups[entity] = greedy_correlation_groups(train)
            print(f"  {len(entity_groups[entity])} groups", flush=True)
        groups = entity_groups[entity]

        gt_channels = [d - 1 for d in gt_dims]
        s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] scoring {len(groups)} groups ...", flush=True)
        channel_score = {}
        for g in groups:
            static_img = render_overlay(train, s_static, e_static, g)
            dynamic_img = render_overlay(test, s_dyn, e_dyn, g)
            e_static_emb = _embed(static_img)
            e_dyn_emb = _embed(dynamic_img)
            score = cosine_dist(e_static_emb, e_dyn_emb)
            for c in g:
                channel_score[c] = score

        ranked = sorted(channel_score, key=lambda c: -channel_score[c])
        k = len(gt_channels)
        gt_set = set(gt_channels)
        recall_inter = len(set(ranked[:k]) & gt_set) / k

        recall_step0 = v3_checkpoint.get(seg_id, {}).get("recall_step0")

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "gt_channels_0idx": gt_channels, "k": k,
            "n_groups": len(groups), "recall_inter": recall_inter, "recall_step0_reused": recall_step0,
            "ranked_top10": ranked[:10],
        }
        save_checkpoint(CHECKPOINT, checkpoint)
        print(f"[OK] {seg_id}: k={k} recall_inter={recall_inter:.2f} recall_step0={recall_step0}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    r_inter = np.mean([r["recall_inter"] for r in rows])
    valid_step0 = [r["recall_step0_reused"] for r in rows if r.get("recall_step0_reused") is not None]
    r_step0 = np.mean(valid_step0) if valid_step0 else None

    print(f"\n{'='*60}\nSTEP1 OVERLAY-INTER SUMMARY (n={len(rows)})\n{'='*60}")
    print(f"Step0 alone (reused from v3) recall@k = {r_step0}")
    print(f"Overlay-INTER recall@k                = {r_inter:.4f}")
    if r_step0 is not None:
        print(f"Delta                                  = {r_inter - r_step0:+.4f}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "recall_step0_reused": r_step0, "recall_inter": r_inter,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
