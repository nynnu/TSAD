"""
Step1 v3: DINOv2 channel-relationship graph on REAL SMD data (2026-08-05)
==========================================================================
Report 16 item 4 ("multi-scale을 그래프로 그려서 관계 표현한 후 입력... PMGC
논문에서 나왔듯 Graph를 통해 다변량 관계를 파악하는 것이 도움이 되는지 증명"),
executed on real production data instead of synthetic data, using the same
Step0 pipeline already validated today (final-layer DINOv2 patch tokens,
no residual/CLS-vs-patch complication -- that debate was old weekly-log
history, not a settled result worth re-deriving; see wiki log 2026-08-05).

Question: does adding an explicit channel-relationship graph (report 12 §5's
"INTER > INTRA" finding, done here as an explicit mechanism rather than just
overlaying correlated channels into one image) improve recall@k over Step0
(per-channel DINOv2 distance) alone, on real SMD data with real interpretation
labels as ground truth?

Method
------
For each of 9 hand-picked SMD interpretation_label segments (3 per entity,
machine-1-1/2/5, moderate GT-channel-set sizes):
  1. Static reference window: a fixed WIN=224 window from the middle of TRAIN
     (confirmed normal, shared across all segments of that entity).
  2. Dynamic window: the candidate segment itself, padded/centered to WIN=224.
  3. For all 38 channels, render each window as an individual line plot and
     extract DINOv2 (vits14, final layer) mean-pooled patch tokens.
  4. Step0 score(c) = cosine distance between static(c) and dynamic(c) --
     the same mechanism as today's synthetic Step0 sanity check.
  5. Graph score(c) = mean_d |cos_sim(static(c),static(d)) -
     cos_sim(dynamic(c),dynamic(d))| over all other channels d -- whole-row
     relational deviation, same mechanism as today's synthetic Step1 v2.
  6. Combined = max(rank_norm(Step0), rank_norm(Graph)) -- max-fusion,
     established as the better combiner in today's synthetic Step1 v2 test.
  7. recall@k (k = |GT channels|) for Step0-alone vs Combined.

No API calls -- local DINOv2 only. Checkpointed per segment so an
interruption can resume.
"""

import base64
import json
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# 스크립트 파일 위치 기준 상대경로 -- 로컬(Windows)/Colab(Drive 마운트) 어디서 돌려도 그대로 동작
BASE = Path(__file__).resolve().parents[2]
SMD_DIR = BASE / "mv_data" / "SMD"
RESULTS_DIR = BASE / "experiments" / "results_step1v3_dino_graph_smd"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"
N_CHANNELS = 38
WIN = 224
DINO_MODEL_NAME = "dinov2_vits14"

# 9 hand-picked segments (3 per entity), moderate GT-set size (4-8 channels),
# from SMD's official interpretation_label (fetched 2026-08-04, dims 1-indexed).
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


def _render_single(values: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    ax.plot(np.arange(len(values)), values, color="black", linewidth=1.2)
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


def embed_channel_window(values: np.ndarray, s: int, e: int) -> np.ndarray:
    seg = values[s:e]
    lo, hi = float(seg.min()), float(seg.max())
    norm = (seg - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(seg)
    return _embed(_render_single(norm))


def cosine_dist(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def cosine_sim(a, b):
    return 1.0 - cosine_dist(a, b)


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

    entity_static_embeds = {}  # entity -> {ch: embedding}
    entity_data = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_data:
            train, test = load_smd(entity)
            entity_data[entity] = (train, test)
        train, test = entity_data[entity]

        if entity not in entity_static_embeds:
            print(f"  Computing static reference embeddings for {entity} (38 channels) ...", flush=True)
            t0 = time.time()
            s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
            static = {}
            for c in range(N_CHANNELS):
                static[c] = embed_channel_window(train[:, c], s_static, e_static)
            entity_static_embeds[entity] = static
            print(f"  done in {time.time()-t0:.1f}s", flush=True)
        static = entity_static_embeds[entity]

        gt_channels = [d - 1 for d in gt_dims]  # -> 0-indexed
        center = (cs + ce) // 2
        s_dyn, e_dyn = _centered_window(len(test), center, WIN)

        print(f"[{seg_id}] computing dynamic embeddings (38 channels) ...", flush=True)
        t0 = time.time()
        dynamic = {}
        for c in range(N_CHANNELS):
            dynamic[c] = embed_channel_window(test[:, c], s_dyn, e_dyn)
        print(f"  done in {time.time()-t0:.1f}s", flush=True)

        # Step0 score: static-vs-dynamic distance per channel
        step0_score = {c: cosine_dist(static[c], dynamic[c]) for c in range(N_CHANNELS)}

        # Graph score: whole-row relational deviation (static graph vs dynamic graph)
        graph_score = {}
        for c in range(N_CHANNELS):
            devs = []
            for d in range(N_CHANNELS):
                if d == c:
                    continue
                sim_static = cosine_sim(static[c], static[d])
                sim_dynamic = cosine_sim(dynamic[c], dynamic[d])
                devs.append(abs(sim_dynamic - sim_static))
            graph_score[c] = float(np.mean(devs))

        step0_rank = _rank_norm(step0_score)
        graph_rank = _rank_norm(graph_score)
        combined = {c: max(step0_rank[c], graph_rank[c]) for c in range(N_CHANNELS)}

        ranked_step0 = sorted(step0_score, key=lambda c: -step0_score[c])
        ranked_combined = sorted(combined, key=lambda c: -combined[c])

        k = len(gt_channels)
        gt_set = set(gt_channels)
        recall_step0 = len(set(ranked_step0[:k]) & gt_set) / k
        recall_combined = len(set(ranked_combined[:k]) & gt_set) / k

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "gt_channels_0idx": gt_channels, "k": k,
            "recall_step0": recall_step0, "recall_combined": recall_combined,
            "ranked_step0_top10": ranked_step0[:10], "ranked_combined_top10": ranked_combined[:10],
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: k={k} gt={gt_channels} "
              f"recall_step0={recall_step0:.2f} recall_combined={recall_combined:.2f}", flush=True)

    # Summary
    rows = [v for v in checkpoint.values()]
    if not rows:
        print("No results.")
        return
    r0 = np.mean([r["recall_step0"] for r in rows])
    rc = np.mean([r["recall_combined"] for r in rows])
    print(f"\n{'='*60}\nSTEP1 v3 SUMMARY (n={len(rows)})\n{'='*60}")
    print(f"Step0 alone recall@k    = {r0:.4f}")
    print(f"Step0+Graph recall@k    = {rc:.4f}  ({rc-r0:+.4f})")
    n_improved = sum(1 for r in rows if r["recall_combined"] > r["recall_step0"])
    n_worsened = sum(1 for r in rows if r["recall_combined"] < r["recall_step0"])
    print(f"Improved: {n_improved}  Worsened: {n_worsened}  Unchanged: {len(rows)-n_improved-n_worsened}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "n": len(rows), "recall_step0": r0, "recall_combined": rc,
        "n_improved": n_improved, "n_worsened": n_worsened,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
