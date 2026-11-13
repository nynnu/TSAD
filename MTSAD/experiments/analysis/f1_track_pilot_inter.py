"""
F1 track, Stage 1 variant (B): INTER (correlation-group overlay) sliding-window
scoring over a FULL entity test series, then report12-exact post-processing +
interval-F1 -- compared against (A) max-over-channels patch-KNN
(f1_track_pilot_stage1.py).

Why INTER here: report12 already validated INTER (correlated channels grouped
~4-per-group, overlay image, single DINOv2 embedding cosine distance vs a
static train reference) as F1=0.727 on 5 SMD entities -- but that was measured
as a *time-detection* F1 (this same question), even though it failed at
*channel attribution* (recall@k). Since the F1 track only asks "is there an
anomaly at this time", INTER is exactly the right thing to re-try here, not a
new idea.

Grouping/scoring logic ported unchanged from step1_overlay_inter_smd.py
(greedy_correlation_groups, render_overlay, embed, cosine_dist) -- only the
model loading (pip/transformers instead of torch.hub, same reason as the other
Colab scripts this session) and the sliding-window / post-processing /
F1-metric parts are new, copied from report12's experiment_dinov2_ml_scoring.py
(90th-pct threshold, merge_gap=WIN//2, min_len=10, interval_f1_fixed) --
NOT the official-VLM4TS-repo Gaussian-alpha-sweep metric used in
f1_track_pilot_stage1.py, per the finding that report12's own metric is the
one whose numbers we're trying to reproduce/compare against.

Aggregation: max-over-groups at each window (same "collapse to one entity-
level score" principle as (A)'s max-over-channels).

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
CACHE_DIR = OUT_DIR / "cache_inter"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENTITY = "machine-3-4"
WIN = 224
STRIDE = 56
GROUP_SIZE = 4
LOOSE_PCT = 90.0
MERGE_GAP = WIN // 2
MIN_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HFDinov2Wrapper:
    """Same wrapper as the other Colab scripts this session -- see
    patchknn_absolute_threshold_pilot.py for rationale."""

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
        return {"x_norm_clstoken": out.last_hidden_state[:, 0], "x_norm_patchtokens": out.last_hidden_state[:, 1:]}


def _load_dinov2_via_pip(device):
    try:
        from transformers import Dinov2Model
    except ImportError:
        print("  transformers 미설치 -- pip install 중...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
        from transformers import Dinov2Model
    hf_model = Dinov2Model.from_pretrained("facebook/dinov2-small")
    return HFDinov2Wrapper(hf_model).to(device).eval()


def greedy_correlation_groups(train, n_channels, group_size=GROUP_SIZE):
    """Exact port of step1_overlay_inter_smd.greedy_correlation_groups."""
    sample = train[:: max(1, len(train) // 5000)]
    corr = np.corrcoef(sample.T)
    corr = np.nan_to_num(corr, nan=0.0)
    remaining = list(range(n_channels))
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


def render_overlay(arr, s, e, channels):
    """Exact port of step1_overlay_inter_smd.render_overlay (matplotlib, kept
    as-is -- this is a small multi-line overlay, not the per-pixel PIL loop
    that was the bottleneck in colab_multivariate_v2.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

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
    from PIL import Image
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


def cosine_dist_batch(dyn_embs, static_emb):
    denom = np.linalg.norm(dyn_embs, axis=1) * np.linalg.norm(static_emb)
    dot = dyn_embs @ static_emb
    return 1.0 - np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-12)


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


def scores_to_intervals_report12(win_scores, offset, T_test):
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
            raw.append((ss + offset, i - 1 + offset))
            in_seg = False
    if in_seg:
        raw.append((ss + offset, T_test - 1 + offset))
    merged = []
    for iv in raw:
        if merged and iv[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(list(iv))
    return [(s, e) for s, e in merged if e - s + 1 >= MIN_LEN]


def interval_f1_fixed(pred_ivs, gt_ivs):
    """Exact port of experiment_dinov2_ml_scoring.interval_f1_fixed (report12)."""
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


def run(mode, batch_size=64):
    from step1v3_dino_graph_smd import load_smd, N_CHANNELS, _centered_window

    print(f"Device: {DEVICE}", flush=True)
    print("Loading DINOv2 ViT-S/14 via transformers/HuggingFace Hub...", flush=True)
    model = _load_dinov2_via_pip(DEVICE)

    train, test = load_smd(ENTITY)
    groups = greedy_correlation_groups(train, N_CHANNELS)
    print(f"Entity: {ENTITY}  groups: {len(groups)}  sizes: {[len(g) for g in groups]}", flush=True)

    starts_all = sliding_windows(len(test), WIN, STRIDE)
    n_all = len(starts_all)
    print(f"windows/group={n_all}", flush=True)

    if mode == "test":
        n_sub = max(1, int(n_all * 0.15))
        t0 = time.time()
        s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
        g = groups[0]
        static_img = render_overlay(train, s_static, e_static, g)
        static_emb = embed_batch([static_img], model, DEVICE)[0]
        dyn_imgs = [render_overlay(test, s, s + WIN, g) for s in starts_all[:n_sub]]
        dyn_embs = embed_batch(dyn_imgs, model, DEVICE)
        _ = cosine_dist_batch(dyn_embs, static_emb)
        elapsed = time.time() - t0
        per_window = elapsed / n_sub
        est_full_1g = per_window * n_all
        est_full_all_g = est_full_1g * len(groups)
        print(f"\n{n_sub}/{n_all} 윈도우(그룹 1개) 소요: {elapsed:.1f}s ({per_window*1000:.1f}ms/window)", flush=True)
        print(f"그룹 1개 전체: {est_full_1g:.1f}s  {len(groups)}개 그룹 전체 추정: {est_full_all_g:.1f}s ({est_full_all_g/60:.1f}분)", flush=True)
        print("\n[STOP] 테스트 모드 완료 -- 승인 후 --full로 재실행해라.", flush=True)
        return

    # ---- full mode ----
    t0 = time.time()
    group_scores = np.zeros((len(groups), n_all))
    for gi, g in enumerate(groups):
        cache_path = CACHE_DIR / ENTITY / f"group{gi}_winscores.npy"
        if cache_path.exists():
            group_scores[gi] = np.load(cache_path)
            print(f"  [SKIP cached] group{gi} {g}", flush=True)
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        s_static, e_static = _centered_window(len(train), len(train) // 2, WIN)
        static_img = render_overlay(train, s_static, e_static, g)
        static_emb = embed_batch([static_img], model, DEVICE)[0]

        scores = np.zeros(n_all)
        for b in range(0, n_all, batch_size):
            batch_starts = starts_all[b:b + batch_size]
            dyn_imgs = [render_overlay(test, s, s + WIN, g) for s in batch_starts]
            dyn_embs = embed_batch(dyn_imgs, model, DEVICE)
            scores[b:b + len(batch_starts)] = cosine_dist_batch(dyn_embs, static_emb)
        group_scores[gi] = scores
        np.save(cache_path, scores)
        print(f"  group{gi} {g} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    entity_win_score = group_scores.max(axis=0)  # (B) max-over-groups
    gt_intervals, T = get_gt_intervals(ENTITY)
    pred_intervals = scores_to_intervals_report12(entity_win_score, 0, T)
    F, P, R = interval_f1_fixed(pred_intervals, gt_intervals)

    total_elapsed = time.time() - t0
    print(f"\nGT intervals: {len(gt_intervals)}  Pred intervals: {len(pred_intervals)}", flush=True)
    print(f"(B) INTER  F1={F:.4f}  P={P:.4f}  R={R:.4f}", flush=True)
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)

    (OUT_DIR / "inter_full_result.json").write_text(json.dumps({
        "entity": ENTITY, "n_groups": len(groups), "group_sizes": [len(g) for g in groups],
        "n_windows": n_all, "n_gt_intervals": len(gt_intervals), "n_pred_intervals": len(pred_intervals),
        "gt_intervals": gt_intervals, "pred_intervals": pred_intervals,
        "F1": F, "precision": P, "recall": R, "total_elapsed_s": total_elapsed,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'inter_full_result.json'}")


def demo():
    starts = sliding_windows(1000, 224, 56)
    assert starts[0] == 0 and starts[-1] + 224 == 1000
    train = np.random.randn(2000, 38)
    train[:, 0] = train[:, 1] + np.random.randn(2000) * 0.01  # force ch0,ch1 correlated
    groups = greedy_correlation_groups(train, 38)
    assert sum(len(g) for g in groups) == 38
    assert any(0 in g and 1 in g for g in groups), "correlated channels should end up in the same group"
    T_test = 2000  # >> WIN=224, realistic scale (real entities are ~23000+)
    win_starts = sliding_windows(T_test, WIN, STRIDE)
    scores = np.zeros(len(win_starts))
    spike_idx = [i for i, s in enumerate(win_starts) if 800 <= s <= 900]
    scores[spike_idx] = 1.0
    ivs = scores_to_intervals_report12(scores, 0, T_test)
    assert any(s <= 850 <= e for s, e in ivs), f"expected spike detected, got {ivs}"
    F, P, R = interval_f1_fixed(ivs, [(800, 900 + WIN)])
    assert F > 0
    print("demo OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    if args.demo:
        demo()
    elif args.full:
        run("full")
    else:
        run("test")
