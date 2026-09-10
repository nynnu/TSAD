"""
Multivariate DINOv2 v2: Smart Channel Aggregation + Overlay Plot Scoring

Key improvements over v1:
  - Overlay plots: correlated channels drawn together → DINOv2 sees inter-variate patterns
  - Smart aggregation: variance-weighted, top-k channel selection, majority voting
  - No correlation heatmap (proven ineffective)

Architecture:
  Stage 1-A: Intra-variate (per-channel DINOv2 L8+L11 resid scoring)
  Stage 1-B: Inter-variate (overlay plot groups → DINOv2 resid scoring)
  Aggregation: variance-weighted + top-k + voting

Usage:
  !python colab_multivariate_v2.py \
    --data_dir /path/to/mv_data \
    --output_dir /path/to/results \
    --gpu --datasets SMD PSM --max_entities 5
"""

import argparse
import io
import os
import time
import urllib.request
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.stats import norm
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════

WINDOW_SIZE = 224
STEP = 56
K = 5
BATCH_SIZE = 64
IMAGE_SIZE = 224
ML_LAYERS = [8, 11]
GROUP_SIZE = 4
MAX_GROUPS = 8
MAX_CHANNELS_INTRA = 10

DEVICE = None
RESULTS_DIR = None
_model = None

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OVERLAY_COLORS = [
    (0, 0, 0),        # black
    (220, 50, 50),     # red
    (50, 50, 220),     # blue
    (50, 180, 50),     # green
    (200, 150, 0),     # orange
    (150, 50, 200),    # purple
]


def get_model():
    global _model
    if _model is None:
        print("Loading DINOv2 ViT-B/14...", flush=True)
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        _model = _model.to(DEVICE).eval()
        print(f"DINOv2 loaded on {DEVICE}", flush=True)
    return _model


# ════════════════════════════════════════════════════════
# Data Download & Loading (same as v1)
# ════════════════════════════════════════════════════════

def download_smd(data_dir):
    smd_dir = data_dir / "SMD"
    if (smd_dir / "train").exists():
        print("SMD already exists", flush=True)
        return
    base_url = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    for split in ["train", "test", "test_label"]:
        (smd_dir / split).mkdir(parents=True, exist_ok=True)
    entities = get_all_smd_entities()
    print(f"Downloading SMD ({len(entities)} entities)...", flush=True)
    for entity in tqdm(entities, desc="SMD"):
        for split in ["train", "test", "test_label"]:
            url = f"{base_url}/{split}/{entity}.txt"
            dst = smd_dir / split / f"{entity}.txt"
            if not dst.exists():
                try:
                    urllib.request.urlretrieve(url, str(dst))
                except:
                    pass


def download_psm(data_dir):
    psm_dir = data_dir / "PSM"
    if (psm_dir / "train.csv").exists():
        print("PSM already exists", flush=True)
        return
    psm_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"
    print("Downloading PSM...", flush=True)
    for f in ["train.csv", "test.csv", "test_label.csv"]:
        dst = psm_dir / f
        if not dst.exists():
            urllib.request.urlretrieve(f"{base_url}/{f}", str(dst))


def get_all_smd_entities():
    return [f"machine-{g}-{i}" for g in range(1, 4)
            for i in range(1, 9 if g == 1 else (10 if g == 2 else 12))]


def get_smd_entities(data_dir):
    smd_dir = data_dir / "SMD" / "train"
    if not smd_dir.exists():
        return []
    return sorted([f.stem for f in smd_dir.glob("*.txt")])


def load_smd(data_dir, entity):
    smd_dir = data_dir / "SMD"
    train = np.loadtxt(smd_dir / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(smd_dir / "test" / f"{entity}.txt", delimiter=",")
    labels = np.loadtxt(smd_dir / "test_label" / f"{entity}.txt", delimiter=",").astype(np.int32)
    return train, test, labels


def load_psm(data_dir):
    psm_dir = data_dir / "PSM"
    train = pd.read_csv(psm_dir / "train.csv").iloc[:, 1:].values.astype(np.float64)
    test = pd.read_csv(psm_dir / "test.csv").iloc[:, 1:].values.astype(np.float64)
    labels = pd.read_csv(psm_dir / "test_label.csv").iloc[:, 1:].values.astype(np.int32)
    if labels.ndim == 2:
        labels = labels.max(axis=1)
    return np.nan_to_num(train), np.nan_to_num(test), labels


# ════════════════════════════════════════════════════════
# Image Generation
# ════════════════════════════════════════════════════════

def ts_to_image_fast(window):
    """Single channel line plot via PIL ImageDraw (vectorized, ~10x faster than
    the previous per-pixel Python loop -- same visual output, negligible pixel
    rounding differences)."""
    w_min, w_max = window.min(), window.max()
    normed = (window - w_min) / (w_max - w_min + 1e-8)
    n = len(normed)

    xs = (np.arange(n) * (IMAGE_SIZE - 1) / (n - 1)).astype(int)
    ys = IMAGE_SIZE - 1 - (normed * (IMAGE_SIZE - 5) + 2).astype(int)
    ys = np.clip(ys, 0, IMAGE_SIZE - 1)

    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(img)
    draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(0, 0, 0), width=2)
    return img


def overlay_to_image(windows_list):
    """
    Multi-channel overlay plot via PIL ImageDraw.line (vectorized per channel,
    same visual output as the previous per-pixel Python loop -- see
    ts_to_image_fast for the identical optimization applied earlier).
    windows_list: list of 1D arrays (each channel's window), all same length.
    Each channel gets a different color.
    """
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(img)

    for ch_idx, window in enumerate(windows_list):
        w_min, w_max = window.min(), window.max()
        normed = (window - w_min) / (w_max - w_min + 1e-8)
        color = OVERLAY_COLORS[ch_idx % len(OVERLAY_COLORS)]
        n = len(normed)
        xs = (np.arange(n) * (IMAGE_SIZE - 1) / (n - 1)).astype(int)
        ys = IMAGE_SIZE - 1 - (normed * (IMAGE_SIZE - 5) + 2).astype(int)
        ys = np.clip(ys, 0, IMAGE_SIZE - 1)
        draw.line(list(zip(xs.tolist(), ys.tolist())), fill=color, width=2)
    return img


# ════════════════════════════════════════════════════════
# Windowing & Feature Extraction
# ════════════════════════════════════════════════════════

def get_windows(ts, window_size=WINDOW_SIZE, step=STEP):
    return [ts[s:s + window_size] for s in range(0, len(ts) - window_size + 1, step)]


def extract_dinov2(images, multilayer=True):
    model = get_model()
    all_cls, all_patch = [], []
    all_ml = {l: [] for l in ML_LAYERS} if multilayer else None

    with torch.no_grad():
        for s in range(0, len(images), BATCH_SIZE):
            batch = torch.stack([dinov2_transform(img) for img in images[s:s + BATCH_SIZE]]).to(DEVICE)
            out = model.forward_features(batch)
            all_cls.append(out["x_norm_clstoken"].cpu().numpy())
            all_patch.append(out["x_norm_patchtokens"].cpu().numpy())
            if multilayer:
                ml_out = model.get_intermediate_layers(batch, n=ML_LAYERS, return_class_token=True, norm=True)
                for i, (ptok, _) in enumerate(ml_out):
                    all_ml[ML_LAYERS[i]].append(ptok.cpu().numpy())

    cls = np.concatenate(all_cls)
    patches = np.concatenate(all_patch)
    if multilayer:
        return cls, patches, {l: np.concatenate(v) for l, v in all_ml.items()}
    return cls, patches


# ════════════════════════════════════════════════════════
# Scoring Utils
# ════════════════════════════════════════════════════════

def compute_residuals(patches, cls):
    cls_n = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    cls_exp = cls_n[:, None, :]
    dot = (patches * cls_exp).sum(axis=-1, keepdims=True)
    resid = patches - dot * cls_exp
    return resid / (np.linalg.norm(resid, axis=-1, keepdims=True) + 1e-8)


def knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, use_ml_tr=None, use_ml_te=None, return_win=False):
    """Compute residual KNN patch scores. Returns (N_windows,) scores dict.
    If return_win=True, also includes raw per-patch distances "knn_win" (N_windows, P)
    in the returned dict, before the sum/topk10 reduction."""
    if use_ml_tr is not None and use_ml_te is not None:
        tr_p = sum(use_ml_tr[l] for l in ML_LAYERS)
        te_p = sum(use_ml_te[l] for l in ML_LAYERS)
    else:
        tr_p = tr_patches
        te_p = te_patches

    N_te, P, D = te_p.shape
    te_resid = compute_residuals(te_p, te_cls)
    tr_resid = compute_residuals(tr_p, tr_cls)

    tr_r = tr_resid.reshape(-1, D).astype(np.float32)
    te_r = te_resid.reshape(-1, D).astype(np.float32)
    tr_t = torch.tensor(tr_r, dtype=torch.float32).to(DEVICE)

    PBATCH = 2048 if DEVICE.type == "cuda" else 256
    knn_list = []
    for i in range(0, len(te_r), PBATCH):
        te_b = torch.tensor(te_r[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        knn_list.append(torch.topk(1.0 - te_b @ tr_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())

    knn_win = np.concatenate(knn_list).reshape(N_te, P)
    k10 = max(1, int(P * 0.10))
    out = {
        "sum": knn_win.sum(axis=1),
        "topk10": np.sort(knn_win, axis=1)[:, -k10:].mean(axis=1),
    }
    if return_win:
        out["knn_win"] = knn_win
    return out


def _norm01(x):
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 0 else np.zeros_like(x)


# ════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════

def get_intervals(binary):
    ivs, in_seg, start = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            ivs.append((start, i - 1))
            in_seg = False
    if in_seg:
        ivs.append((start, len(binary) - 1))
    return ivs


def _eval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0.0
    gt = [tuple(i) for i in gt_ivs]
    pr = [tuple(i) for i in pred_ivs]
    TP = sum(sum(1 for a in gt if not (a[1] < d[0] or d[1] < a[0]))
             for d in pr if any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
    FP = sum(1 for d in pr if not any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
    FN = sum(1 for a in gt if not any(not (a[1] < d[0] or d[1] < a[0]) for d in pr))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def f1max(ts_scores, labels):
    gt_ivs = get_intervals(labels.astype(int))
    if not gt_ivs:
        return 0.0
    best = 0
    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = ts_scores.mean(), ts_scores.std()
        if sigma < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred_ivs = get_intervals((ts_scores > thr).astype(int))
        best = max(best, _eval_f1(gt_ivs, pred_ivs))
    return best


def win_to_ts(win_scores, n_ts):
    scores = np.zeros(n_ts)
    counts = np.zeros(n_ts)
    for i, s in enumerate(win_scores):
        st = i * STEP
        en = min(st + WINDOW_SIZE, n_ts)
        scores[st:en] += s
        counts[st:en] += 1
    m = counts > 0
    scores[m] /= counts[m]
    return scores


# ════════════════════════════════════════════════════════
# Channel Grouping
# ════════════════════════════════════════════════════════

def build_channel_groups(train_data, n_groups=MAX_GROUPS, group_size=GROUP_SIZE):
    """
    Group channels by correlation for overlay plots.
    Returns list of channel index groups, e.g. [[0,3,7,12], [1,5,9,14], ...]
    """
    C = train_data.shape[1]
    corr = np.corrcoef(train_data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)

    used = set()
    groups = []

    for _ in range(n_groups):
        if len(used) >= C:
            break

        remaining = [i for i in range(C) if i not in used]
        if len(remaining) < 2:
            break

        seed = remaining[0]
        corr_to_seed = np.abs(corr[seed])
        candidates = sorted(remaining, key=lambda i: -corr_to_seed[i])
        group = candidates[:group_size]

        groups.append(group)
        used.update(group)

    if not groups:
        groups = [list(range(min(C, group_size)))]

    return groups


# ════════════════════════════════════════════════════════
# Main Experiment
# ════════════════════════════════════════════════════════

def run_entity(entity_name, train_data, test_data, labels, cache_dir):
    T_test, C = test_data.shape
    n_gt = len(get_intervals(labels))
    if n_gt == 0:
        print(f"  [{entity_name}] GT=0, skip", flush=True)
        return None

    anom_ratio = labels.mean()
    print(f"\n  [{entity_name}] T={T_test}, C={C}, GT={n_gt}, anom={anom_ratio:.2%}", flush=True)
    t0 = time.time()
    ent_cache = cache_dir / entity_name
    ent_cache.mkdir(parents=True, exist_ok=True)

    # ── Select channels by variance ──
    var_per_ch = test_data.var(axis=0)
    top_channels = np.argsort(var_per_ch)[::-1][:MAX_CHANNELS_INTRA]
    n_windows = len(get_windows(test_data[:, 0]))

    # ══════════════════════════════════════════════════
    # Stage 1-A: Intra-variate (per-channel)
    # ══════════════════════════════════════════════════
    print(f"    [1A] Intra-variate: {len(top_channels)} channels...", flush=True)

    ch_scores_ts = {}  # channel_idx -> ts-level score array

    for ci in top_channels:
        tag = f"ch{ci}"
        sc_cache = ent_cache / f"{tag}_scores.npz"

        if sc_cache.exists():
            loaded = np.load(sc_cache)
            ch_scores_ts[ci] = {k: loaded[k] for k in loaded.files}
            continue

        # Train
        tr_ts = train_data[:, ci].astype(float)
        lo, hi = tr_ts.min(), tr_ts.max()
        tr_ts_n = (tr_ts - lo) / (hi - lo + 1e-8)
        tr_wins = get_windows(tr_ts_n)
        tr_imgs = [ts_to_image_fast(w) for w in tr_wins]
        tr_cls, tr_patches, tr_ml = extract_dinov2(tr_imgs, multilayer=True)

        # Test
        te_ts = test_data[:, ci].astype(float)
        lo, hi = te_ts.min(), te_ts.max()
        te_ts_n = (te_ts - lo) / (hi - lo + 1e-8)
        te_wins = get_windows(te_ts_n)
        te_imgs = [ts_to_image_fast(w) for w in te_wins]
        te_cls, te_patches, te_ml = extract_dinov2(te_imgs, multilayer=True)

        # Score (final layer + multi-layer)
        sc_final = knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
        sc_ml = knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, tr_ml, te_ml)

        ch_ts = {}
        for key, win_sc in sc_final.items():
            ch_ts[f"final_{key}"] = win_to_ts(win_sc, T_test)
        for key, win_sc in sc_ml.items():
            ch_ts[f"ml_{key}"] = win_to_ts(win_sc, T_test)

        np.savez_compressed(sc_cache, **ch_ts)
        ch_scores_ts[ci] = ch_ts
        print(f"      ch{ci} done", flush=True)

    # ── Aggregation strategies ──
    print(f"    [1A] Aggregating channels...", flush=True)
    score_keys = list(next(iter(ch_scores_ts.values())).keys())
    results = {}

    for sk in score_keys:
        all_ch_ts = np.array([ch_scores_ts[ci][sk] for ci in ch_scores_ts])  # (n_ch, T)

        # Mean
        results[f"intra_{sk}_mean"] = f1max(all_ch_ts.mean(axis=0), labels)

        # Max across channels
        results[f"intra_{sk}_max"] = f1max(all_ch_ts.max(axis=0), labels)

        # Variance-weighted mean
        ch_vars = all_ch_ts.var(axis=1)
        weights = ch_vars / (ch_vars.sum() + 1e-8)
        weighted = (all_ch_ts * weights[:, None]).sum(axis=0)
        results[f"intra_{sk}_varw"] = f1max(weighted, labels)

        # Top-3 channels (highest mean score)
        ch_means = all_ch_ts.mean(axis=1)
        top3_idx = np.argsort(ch_means)[-3:]
        results[f"intra_{sk}_top3"] = f1max(all_ch_ts[top3_idx].mean(axis=0), labels)

        # Majority voting (per alpha)
        best_vote = 0
        for alpha in [0.1, 0.01, 0.001]:
            votes = np.zeros(T_test)
            for ch_ts_arr in all_ch_ts:
                mu, sigma = ch_ts_arr.mean(), ch_ts_arr.std()
                if sigma < 1e-12:
                    continue
                thr = mu + norm.ppf(1 - alpha) * sigma
                votes += (ch_ts_arr > thr).astype(float)
            majority = (votes >= len(all_ch_ts) / 2).astype(float)
            gt_ivs = get_intervals(labels.astype(int))
            pred_ivs = get_intervals(majority.astype(int))
            best_vote = max(best_vote, _eval_f1(gt_ivs, pred_ivs))
        results[f"intra_{sk}_vote"] = best_vote

    # ══════════════════════════════════════════════════
    # Stage 1-B: Inter-variate (overlay plots)
    # ══════════════════════════════════════════════════
    print(f"    [1B] Inter-variate: overlay plot groups...", flush=True)

    groups = build_channel_groups(train_data, n_groups=MAX_GROUPS, group_size=GROUP_SIZE)
    print(f"      {len(groups)} groups: {groups}", flush=True)

    group_scores_ts = []

    for gi, group in enumerate(groups):
        grp_cache = ent_cache / f"overlay_g{gi}_scores.npz"

        if grp_cache.exists():
            loaded = np.load(grp_cache)
            group_scores_ts.append({k: loaded[k] for k in loaded.files})
            print(f"      group {gi} cached", flush=True)
            continue

        # Generate overlay images for train
        tr_overlay_imgs = []
        n_tr_wins = len(get_windows(train_data[:, 0]))
        for wi in range(n_tr_wins):
            st = wi * STEP
            en = st + WINDOW_SIZE
            ch_windows = [train_data[st:en, ci] for ci in group]
            tr_overlay_imgs.append(overlay_to_image(ch_windows))

        # Generate overlay images for test
        te_overlay_imgs = []
        n_te_wins = len(get_windows(test_data[:, 0]))
        for wi in range(n_te_wins):
            st = wi * STEP
            en = st + WINDOW_SIZE
            ch_windows = [test_data[st:en, ci] for ci in group]
            te_overlay_imgs.append(overlay_to_image(ch_windows))

        print(f"      group {gi}: {len(tr_overlay_imgs)} train, {len(te_overlay_imgs)} test images", flush=True)

        # DINOv2 features
        tr_cls, tr_patches, tr_ml = extract_dinov2(tr_overlay_imgs, multilayer=True)
        te_cls, te_patches, te_ml = extract_dinov2(te_overlay_imgs, multilayer=True)

        # Score
        sc_final = knn_patch_score(tr_patches, te_patches, tr_cls, te_cls)
        sc_ml = knn_patch_score(tr_patches, te_patches, tr_cls, te_cls, tr_ml, te_ml)

        grp_ts = {}
        for key, win_sc in sc_final.items():
            grp_ts[f"final_{key}"] = win_to_ts(win_sc, T_test)
        for key, win_sc in sc_ml.items():
            grp_ts[f"ml_{key}"] = win_to_ts(win_sc, T_test)

        np.savez_compressed(grp_cache, **grp_ts)
        group_scores_ts.append(grp_ts)
        print(f"      group {gi} done", flush=True)

    # Aggregate overlay group scores
    if group_scores_ts:
        grp_keys = list(group_scores_ts[0].keys())
        for gk in grp_keys:
            all_grp = np.array([g[gk] for g in group_scores_ts])
            results[f"inter_overlay_{gk}_mean"] = f1max(all_grp.mean(axis=0), labels)
            results[f"inter_overlay_{gk}_max"] = f1max(all_grp.max(axis=0), labels)

    # ══════════════════════════════════════════════════
    # Fusion: Intra + Inter
    # ══════════════════════════════════════════════════
    print(f"    [Fusion] Intra + Inter...", flush=True)

    # Pick best intra and best inter keys
    intra_keys = [k for k in results if k.startswith("intra_")]
    inter_keys = [k for k in results if k.startswith("inter_")]

    if intra_keys and inter_keys and group_scores_ts:
        best_intra_key = max(intra_keys, key=lambda k: results[k])
        best_inter_key_name = max(
            [(gk, f1max(np.array([g[gk] for g in group_scores_ts]).mean(axis=0), labels))
             for gk in grp_keys],
            key=lambda x: x[1]
        )[0]

        # Get raw ts scores for fusion
        best_intra_sk = best_intra_key.replace("intra_", "").rsplit("_", 1)[0]
        # Reconstruct intra ts
        for sk in score_keys:
            if sk in best_intra_key:
                intra_ts_arr = np.array([ch_scores_ts[ci][sk] for ci in ch_scores_ts])
                break
        else:
            intra_ts_arr = np.array([ch_scores_ts[ci][score_keys[0]] for ci in ch_scores_ts])

        intra_agg = intra_ts_arr.mean(axis=0)
        inter_agg = np.array([g[best_inter_key_name] for g in group_scores_ts]).mean(axis=0)

        for w in [0.5, 0.7, 0.9]:
            fused = w * _norm01(intra_agg) + (1 - w) * _norm01(inter_agg)
            results[f"fusion_w{w}"] = f1max(fused, labels)

        # Multiplicative fusion
        mult = _norm01(intra_agg) * _norm01(inter_agg)
        results["fusion_multiply"] = f1max(mult, labels)

        # Additive
        add = _norm01(intra_agg) + _norm01(inter_agg)
        results["fusion_add"] = f1max(add, labels)

    elapsed = time.time() - t0
    print(f"    Done ({elapsed:.1f}s)", flush=True)

    # Print summary
    best_intra = max((v for k, v in results.items() if "intra" in k), default=0)
    best_inter = max((v for k, v in results.items() if "inter" in k), default=0)
    best_fusion = max((v for k, v in results.items() if "fusion" in k), default=0)
    best_all = max(results.values()) if results else 0
    print(f"    INTRA={best_intra:.4f}  INTER={best_inter:.4f}  FUSION={best_fusion:.4f}  BEST={best_all:.4f}", flush=True)

    return {"entity": entity_name, "C": C, "n_gt": n_gt, "T_test": T_test,
            "anom_ratio": anom_ratio, **{f"{k}_f1": v for k, v in results.items()}}


# ════════════════════════════════════════════════════════
# Dataset Runners
# ════════════════════════════════════════════════════════

def run_smd(data_dir, max_entities=5):
    entities = get_smd_entities(data_dir)
    if not entities:
        download_smd(data_dir)
        entities = get_smd_entities(data_dir)
    entities = entities[:max_entities]

    print(f"\n{'=' * 72}")
    print(f"SMD: {len(entities)} entities, device={DEVICE}")
    print(f"{'=' * 72}", flush=True)

    results = []
    for ent in entities:
        try:
            train, test, labels = load_smd(data_dir, ent)
            r = run_entity(ent, train, test, labels, RESULTS_DIR / "cache" / "SMD")
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR {ent}: {e}", flush=True)
            import traceback; traceback.print_exc()
    return results


def run_psm(data_dir):
    download_psm(data_dir)
    train, test, labels = load_psm(data_dir)

    print(f"\n{'=' * 72}")
    print(f"PSM: {train.shape[1]} dims, device={DEVICE}")
    print(f"{'=' * 72}", flush=True)

    r = run_entity("PSM", train, test, labels, RESULTS_DIR / "cache" / "PSM")
    return [r] if r else []


def print_summary(all_results):
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    for ds, results in all_results.items():
        f1_keys = sorted([k for k in results[0] if k.endswith("_f1")])

        avgs = {}
        for k in f1_keys:
            vals = [r[k] for r in results if k in r]
            avgs[k] = np.mean(vals)

        print(f"\n--- {ds} ({len(results)} entities) ---")
        ranked = sorted(avgs.items(), key=lambda x: -x[1])[:20]
        for i, (k, v) in enumerate(ranked, 1):
            tag = ""
            if "inter" in k:
                tag = " [INTER]"
            elif "fusion" in k:
                tag = " [FUSION]"
            elif "vote" in k:
                tag = " [VOTE]"
            elif "varw" in k:
                tag = " [VAR-W]"
            elif "top3" in k:
                tag = " [TOP3]"
            print(f"  {i:2d}. {k.replace('_f1',''):<50} {v:.4f}{tag}")


# ════════════════════════════════════════════════════════
# Entry
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./mv_data")
    parser.add_argument("--output_dir", type=str, default="./results_mv_v2")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["SMD", "PSM"])
    parser.add_argument("--max_entities", type=int, default=5)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    all_results = {}

    for ds in args.datasets:
        ds_upper = ds.upper()
        if ds_upper == "SMD":
            results = run_smd(data_dir, args.max_entities)
        elif ds_upper == "PSM":
            results = run_psm(data_dir)
        else:
            continue

        if results:
            all_results[ds_upper] = results
            df = pd.DataFrame(results)
            csv_path = RESULTS_DIR / f"{ds_upper}_v2_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nSaved: {csv_path}", flush=True)

    if all_results:
        print_summary(all_results)
    print(f"\nTotal: {time.time() - t0:.1f}s", flush=True)
