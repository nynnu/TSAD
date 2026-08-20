"""
F1 track, Stage 1 variant (C): INTRA (per-channel, no grouping) with the §34
multi-reference fix, sliding-window scoring over a FULL entity test series,
report12-exact post-processing + interval-F1.

Why this exists: (A) max-over-channels used patch-KNN+multilayer, which
already has an implicit multi-reference bank (100 train windows via KNN) --
not the single-static-reference issue. (B) INTER (correlation-group overlay)
DOES use a single static reference window (one embedding, cosine distance) --
exactly the mechanism that recall@k's §34 found +0.069 improvement on by
averaging 5 static reference windows instead of 1. This script applies that
exact §34 fix (see smd_step0_multiref_ensemble.py: static_ref_windows,
N_STATIC_REFS=5, score = mean of cosine_dist(ref_i, dynamic) across the 5
refs -- NOT distance-of-means) to the SIMPLEST baseline first: INTRA
(per-channel, no grouping at all -- render_single/embed_channel_window from
step1v3_dino_graph_smd.py), before re-testing INTER with the same fix.

Also excludes zero-variance (constant) channels -- detected per-entity from
train data (std < 1e-8), not hardcoded -- from both scoring and the
max-over-channels aggregation, since machine-3-4 had 8 such channels and they
carry no signal (their correlation-grouping was already shown meaningless,
NaN->0 fallback).

--entities: run this same fixed setup (multiref-5, constant-channel exclusion,
INTRA, report12 post-processing) across multiple entities to check whether
machine-3-4's F1 jump (0.1765 -> 0.4598) generalizes or was a fluke.

No VLM/GPT calls anywhere in this script.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = BASE / "experiments" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

OUT_DIR = BASE / "experiments" / "results_f1_track_pilot"
CACHE_DIR = OUT_DIR / "cache_intra_multiref"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENTITY = "machine-3-4"
DEFAULT_ENTITIES = ["machine-3-4", "machine-1-1", "machine-1-4", "machine-2-1", "machine-3-7", "machine-1-8"]
WIN = 224
STRIDE = 56
N_STATIC_REFS = 5  # exact port of §34 (smd_step0_multiref_ensemble.py)
LOOSE_PCT = 90.0
MERGE_GAP = WIN // 2
MIN_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def constant_channels(train, n_channels, eps=1e-8):
    """Per-entity zero-variance channel detection (not hardcoded -- different
    entities have different constant channels)."""
    stds = train.std(axis=0)
    return {c for c in range(n_channels) if stds[c] < eps}


class HFDinov2Wrapper:
    def __init__(self, hf_model):
        self.model = hf_model

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def forward_features(self, batch):
        out = self.model(pixel_values=batch)
        return {"x_norm_patchtokens": out.last_hidden_state[:, 1:]}


def _load_dinov2_via_pip(device):
    try:
        from transformers import Dinov2Model
    except ImportError:
        print("  transformers 미설치 -- pip install 중...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
        from transformers import Dinov2Model
    hf_model = Dinov2Model.from_pretrained("facebook/dinov2-small")
    return HFDinov2Wrapper(hf_model).to(device).eval()


def render_single(values):
    """Exact port of step1v3_dino_graph_smd._render_single."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    ax.plot(np.arange(len(values)), values, color="black", linewidth=1.2)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def embed_batch(imgs, model, device):
    import torchvision.transforms as T
    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = torch.stack([tfm(img) for img in imgs]).to(device)
    with torch.no_grad():
        feat = model.forward_features(x)["x_norm_patchtokens"].mean(dim=1)
    return feat.cpu().numpy()


def normed_window(values, s, e):
    seg = values[s:e]
    lo, hi = float(seg.min()), float(seg.max())
    return (seg - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(seg)


def cosine_dist_batch(dyn_embs, ref_emb):
    denom = np.linalg.norm(dyn_embs, axis=1) * np.linalg.norm(ref_emb)
    dot = dyn_embs @ ref_emb
    return 1.0 - np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-12)


def static_ref_windows(train_len, win, n_refs):
    """Exact port of smd_step0_multiref_ensemble.static_ref_windows."""
    centers = np.linspace(win // 2, train_len - win // 2, n_refs).astype(int)
    out = []
    for c in centers:
        s = max(0, c - win // 2)
        e = min(train_len, s + win)
        s = max(0, e - win)
        out.append((s, e))
    return out


def sliding_windows(n, win, step):
    starts = list(range(0, n - win + 1, step))
    if not starts or starts[-1] + win < n:
        starts.append(n - win)
    return starts


def get_gt_intervals(entity):
    from step1v3_dino_graph_smd import SMD_DIR
    labels = np.loadtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",").astype(int)
    diff = np.diff(np.concatenate(([0], labels, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist())), len(labels)


def scores_to_intervals_report12(win_scores, T_test):
    thr = float(np.percentile(win_scores, LOOSE_PCT))
    binary = np.zeros(T_test, dtype=int)
    for i, s_win in enumerate(win_scores):
        s = i * STRIDE
        if s_win >= thr:
            e = min(s + WIN, T_test)
            binary[s:e] = 1
    raw, in_seg, ss = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            ss, in_seg = i, True
        elif not v and in_seg:
            raw.append((ss, i - 1))
            in_seg = False
    if in_seg:
        raw.append((ss, T_test - 1))
    merged = []
    for iv in raw:
        if merged and iv[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(list(iv))
    return [(s, e) for s, e in merged if e - s + 1 >= MIN_LEN]


def interval_f1_fixed(pred_ivs, gt_ivs):
    if not gt_ivs:
        return (1.0 if not pred_ivs else 0.0), 1.0, 1.0
    if not pred_ivs:
        return 0.0, 0.0, 0.0
    TP_pred = sum(1 for p in pred_ivs if any(not (g[1] < p[0] or p[1] < g[0]) for g in gt_ivs))
    TP_gt = sum(1 for g in gt_ivs if any(not (g[1] < p[0] or p[1] < g[0]) for p in pred_ivs))
    P = TP_pred / len(pred_ivs)
    R = TP_gt / len(gt_ivs)
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return F, P, R


def run_test_timing(entity, model):
    from step1v3_dino_graph_smd import load_smd, N_CHANNELS

    train, test = load_smd(entity)
    const = constant_channels(train, N_CHANNELS)
    active_channels = [c for c in range(N_CHANNELS) if c not in const]
    print(f"Entity: {entity}  active channels: {len(active_channels)}/{N_CHANNELS} "
          f"(excluded constant: {sorted(const)})", flush=True)

    starts_all = sliding_windows(len(test), WIN, STRIDE)
    n_all = len(starts_all)
    print(f"windows/channel={n_all}  N_STATIC_REFS={N_STATIC_REFS}", flush=True)

    n_sub = max(1, int(n_all * 0.15))
    t0 = time.time()
    c = active_channels[0]
    ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
    ref_imgs = [render_single(normed_window(train[:, c], s, e)) for s, e in ref_windows]
    ref_embs = embed_batch(ref_imgs, model, DEVICE)
    dyn_imgs = [render_single(normed_window(test[:, c], s, s + WIN)) for s in starts_all[:n_sub]]
    dyn_embs = embed_batch(dyn_imgs, model, DEVICE)
    _ = np.mean([cosine_dist_batch(dyn_embs, ref) for ref in ref_embs], axis=0)
    elapsed = time.time() - t0
    per_window = elapsed / n_sub
    est_1ch = per_window * n_all
    est_all_ch = est_1ch * len(active_channels)
    print(f"\n{n_sub}/{n_all} 윈도우(채널 1개) 소요: {elapsed:.1f}s ({per_window*1000:.1f}ms/window)", flush=True)
    print(f"채널 1개 전체: {est_1ch:.1f}s  {len(active_channels)}개 활성채널 전체 추정: "
          f"{est_all_ch:.1f}s ({est_all_ch/60:.1f}분)", flush=True)


def run_full_one_entity(entity, model, batch_size=64):
    from step1v3_dino_graph_smd import load_smd, N_CHANNELS

    train, test = load_smd(entity)
    const = constant_channels(train, N_CHANNELS)
    active_channels = [c for c in range(N_CHANNELS) if c not in const]
    starts_all = sliding_windows(len(test), WIN, STRIDE)
    n_all = len(starts_all)
    print(f"\n=== {entity} === active={len(active_channels)}/{N_CHANNELS} "
          f"(excluded constant: {sorted(const)})  windows/ch={n_all}", flush=True)

    t0 = time.time()
    channel_scores = {}
    for c in active_channels:
        cache_path = CACHE_DIR / entity / f"ch{c}_winscores.npy"
        if cache_path.exists():
            channel_scores[c] = np.load(cache_path)
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ref_windows = static_ref_windows(len(train), WIN, N_STATIC_REFS)
        ref_imgs = [render_single(normed_window(train[:, c], s, e)) for s, e in ref_windows]
        ref_embs = embed_batch(ref_imgs, model, DEVICE)

        scores = np.zeros(n_all)
        for b in range(0, n_all, batch_size):
            batch_starts = starts_all[b:b + batch_size]
            dyn_imgs = [render_single(normed_window(test[:, c], s, s + WIN)) for s in batch_starts]
            dyn_embs = embed_batch(dyn_imgs, model, DEVICE)
            dists_per_ref = np.stack([cosine_dist_batch(dyn_embs, ref) for ref in ref_embs])  # (n_refs, batch)
            scores[b:b + len(batch_starts)] = dists_per_ref.mean(axis=0)  # mean-of-distances, not distance-of-means
        channel_scores[c] = scores
        np.save(cache_path, scores)
        print(f"  ch{c} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    entity_win_score = np.stack(list(channel_scores.values())).max(axis=0)  # max-over-active-channels
    gt_intervals, T = get_gt_intervals(entity)
    pred_intervals = scores_to_intervals_report12(entity_win_score, T)
    F, P, R = interval_f1_fixed(pred_intervals, gt_intervals)
    total_elapsed = time.time() - t0

    print(f"GT intervals: {len(gt_intervals)}  Pred intervals: {len(pred_intervals)}", flush=True)
    print(f"{entity}: F1={F:.4f}  P={P:.4f}  R={R:.4f}  ({total_elapsed:.0f}s)", flush=True)

    return {
        "entity": entity, "n_active_channels": len(active_channels), "excluded_constant_channels": sorted(const),
        "n_windows": n_all, "n_static_refs": N_STATIC_REFS,
        "n_gt_intervals": len(gt_intervals), "n_pred_intervals": len(pred_intervals),
        "gt_intervals": gt_intervals, "pred_intervals": pred_intervals,
        "F1": F, "precision": P, "recall": R, "total_elapsed_s": total_elapsed,
    }


def run(mode, entities, batch_size=64):
    print(f"Device: {DEVICE}", flush=True)
    print("Loading DINOv2 ViT-S/14 via transformers/HuggingFace Hub...", flush=True)
    model = _load_dinov2_via_pip(DEVICE)

    if mode == "test":
        run_test_timing(entities[0], model)
        print("\n[STOP] 테스트 모드 완료 -- 승인 후 --full로 재실행해라.", flush=True)
        return

    results = [run_full_one_entity(e, model, batch_size) for e in entities]

    f1s = [r["F1"] for r in results]
    print(f"\n{'='*60}\nSUMMARY ({len(results)} entities)\n{'='*60}")
    for r in results:
        print(f"  {r['entity']:<15} F1={r['F1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
              f"n_gt={r['n_gt_intervals']}  n_pred={r['n_pred_intervals']}  n_active_ch={r['n_active_channels']}")
    print(f"\nMean F1={np.mean(f1s):.4f}  Std={np.std(f1s):.4f}  Min={np.min(f1s):.4f}  Max={np.max(f1s):.4f}")

    (OUT_DIR / "intra_multiref_multi_entity_result.json").write_text(json.dumps({
        "entities": entities, "results": results,
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "min_f1": float(np.min(f1s)), "max_f1": float(np.max(f1s)),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'intra_multiref_multi_entity_result.json'}")


def demo():
    starts = sliding_windows(1000, 224, 56)
    assert starts[0] == 0 and starts[-1] + 224 == 1000
    refs = static_ref_windows(2000, 224, 5)
    assert len(refs) == 5
    for s, e in refs:
        assert e - s == 224 and 0 <= s and e <= 2000
    v = np.random.randn(500)
    n = normed_window(v, 10, 234)
    assert n.min() >= 0.0 - 1e-9 and n.max() <= 1.0 + 1e-9
    T_test = 2000
    win_starts = sliding_windows(T_test, WIN, STRIDE)
    scores = np.zeros(len(win_starts))
    spike_idx = [i for i, s in enumerate(win_starts) if 800 <= s <= 900]
    scores[spike_idx] = 1.0
    ivs = scores_to_intervals_report12(scores, T_test)
    assert any(s <= 850 <= e for s, e in ivs), f"expected spike detected, got {ivs}"
    F, P, R = interval_f1_fixed(ivs, [(800, 900 + WIN)])
    assert F > 0
    train = np.random.randn(1000, 10)
    train[:, 3] = 5.0  # constant channel
    const = constant_channels(train, 10)
    assert const == {3}, f"expected only channel 3 flagged constant, got {const}"
    print("demo OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--entities", type=str, default=None,
                    help="comma-separated entity list, default: " + ",".join(DEFAULT_ENTITIES))
    args = p.parse_args()
    entities = args.entities.split(",") if args.entities else DEFAULT_ENTITIES
    if args.demo:
        demo()
    elif args.full:
        run("full", entities)
    else:
        run("test", entities)
