"""
Multivariate DINOv2 Experiment for Google Colab (GPU)

Step 1: Per-channel DINOv2 scoring (intra-variate baseline)
Step 2: Inter-variate correlation diff map scoring (NEW - core contribution)
Step 3: Intra x Inter fusion

Datasets: SMD (38-dim, 28 entities), PSM (25-dim)

Usage in Colab:
  # Cell 1: Mount drive
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2: Install
  !pip install torch torchvision matplotlib scipy pandas tqdm pillow

  # Cell 3: Run
  !python "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/colab_multivariate.py" \
    --output_dir "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/results_multivariate" \
    --gpu --datasets SMD PSM
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
from PIL import Image
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
_DPI = 100
_FIG_INCHES = IMAGE_SIZE / _DPI
ML_LAYERS = [8, 11]

DEVICE = None
RESULTS_DIR = None
_model = None

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_model():
    global _model
    if _model is None:
        print("Loading DINOv2 ViT-B/14...")
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        _model = _model.to(DEVICE).eval()
        print(f"DINOv2 loaded on {DEVICE}")
    return _model


# ════════════════════════════════════════════════════════
# Data Download & Loading
# ════════════════════════════════════════════════════════

def download_smd(data_dir):
    """Download SMD from OmniAnomaly GitHub."""
    smd_dir = data_dir / "SMD"
    if (smd_dir / "train").exists():
        print("SMD already downloaded")
        return

    base_url = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    for split in ["train", "test", "test_label"]:
        (smd_dir / split).mkdir(parents=True, exist_ok=True)

    entities = [f"machine-{g}-{i}" for g in range(1, 4) for i in range(1, 9 if g == 1 else (10 if g == 2 else 12))]

    print(f"Downloading SMD ({len(entities)} entities)...")
    for entity in tqdm(entities, desc="SMD download"):
        for split in ["train", "test", "test_label"]:
            url = f"{base_url}/{split}/{entity}.txt"
            dst = smd_dir / split / f"{entity}.txt"
            if not dst.exists():
                try:
                    urllib.request.urlretrieve(url, str(dst))
                except Exception as e:
                    print(f"  Failed: {entity}/{split}: {e}")
    print(f"SMD downloaded to {smd_dir}")


def download_psm(data_dir):
    """Download PSM from eBay RANSynCoders GitHub."""
    psm_dir = data_dir / "PSM"
    if (psm_dir / "train.csv").exists():
        print("PSM already downloaded")
        return

    psm_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"

    print("Downloading PSM...")
    for fname in ["train.csv", "test.csv", "test_label.csv"]:
        url = f"{base_url}/{fname}"
        dst = psm_dir / fname
        if not dst.exists():
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(url, str(dst))
    print(f"PSM downloaded to {psm_dir}")


def load_smd(data_dir, entity_name):
    smd_dir = data_dir / "SMD"
    train = np.loadtxt(smd_dir / "train" / f"{entity_name}.txt", delimiter=",")
    test = np.loadtxt(smd_dir / "test" / f"{entity_name}.txt", delimiter=",")
    labels = np.loadtxt(smd_dir / "test_label" / f"{entity_name}.txt", delimiter=",").astype(np.int32)
    return train, test, labels


def load_psm(data_dir):
    psm_dir = data_dir / "PSM"
    train_df = pd.read_csv(psm_dir / "train.csv")
    test_df = pd.read_csv(psm_dir / "test.csv")
    labels_df = pd.read_csv(psm_dir / "test_label.csv")

    train = train_df.iloc[:, 1:].values.astype(np.float64)
    test = test_df.iloc[:, 1:].values.astype(np.float64)
    labels = labels_df.iloc[:, 1:].values.astype(np.int32)
    if labels.ndim == 2:
        labels = labels.max(axis=1)

    train = np.nan_to_num(train, nan=0.0)
    test = np.nan_to_num(test, nan=0.0)
    return train, test, labels


def get_smd_entities(data_dir):
    smd_dir = data_dir / "SMD" / "train"
    if not smd_dir.exists():
        return []
    return sorted([f.stem for f in smd_dir.glob("*.txt")])


# ════════════════════════════════════════════════════════
# Image & Feature Utils (same as colab_dinov2_full.py)
# ════════════════════════════════════════════════════════

def ts_to_image(window):
    """Fast line plot: draw directly to PIL instead of matplotlib."""
    w_min, w_max = window.min(), window.max()
    normed = (window - w_min) / (w_max - w_min + 1e-8)

    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    pixels = img.load()

    n = len(normed)
    for i in range(n - 1):
        x0 = int(i * (IMAGE_SIZE - 1) / (n - 1))
        x1 = int((i + 1) * (IMAGE_SIZE - 1) / (n - 1))
        y0 = IMAGE_SIZE - 1 - int(normed[i] * (IMAGE_SIZE - 3) + 1)
        y1 = IMAGE_SIZE - 1 - int(normed[i + 1] * (IMAGE_SIZE - 3) + 1)
        y0 = max(0, min(IMAGE_SIZE - 1, y0))
        y1 = max(0, min(IMAGE_SIZE - 1, y1))

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steps = max(dx, dy, 1)
        for s in range(steps + 1):
            t = s / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                pixels[x, y] = (0, 0, 0)
    return img


_RdBu_r_lut = None
def _get_rdbu_lut():
    global _RdBu_r_lut
    if _RdBu_r_lut is None:
        cm = plt.cm.RdBu_r
        _RdBu_r_lut = np.array([cm(i / 255.0)[:3] for i in range(256)])
        _RdBu_r_lut = (255 * _RdBu_r_lut).astype(np.uint8)
    return _RdBu_r_lut


def corr_map_to_image(corr_matrix):
    """Fast correlation heatmap: numpy + PIL resize, no matplotlib per-call."""
    lut = _get_rdbu_lut()
    normed = ((corr_matrix + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    rgb = lut[normed]
    img = Image.fromarray(rgb, "RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return img


def get_windows_1d(ts, window_size=WINDOW_SIZE, step=STEP):
    return [ts[s:s + window_size] for s in range(0, len(ts) - window_size + 1, step)]


def get_windows_nd(data, window_size=WINDOW_SIZE, step=STEP):
    """Multivariate sliding window -> list of (window_size, C) arrays."""
    T = data.shape[0]
    return [data[s:s + window_size] for s in range(0, T - window_size + 1, step)]


def extract_dinov2(images, multilayer=True):
    model = get_model()
    all_cls, all_patch, all_ml = [], [], {l: [] for l in ML_LAYERS}

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
        ml = {l: np.concatenate(v) for l, v in all_ml.items()}
        return cls, patches, ml
    return cls, patches


# ════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════

def compute_residuals(patches, cls):
    cls_n = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    cls_exp = cls_n[:, None, :]
    dot = (patches * cls_exp).sum(axis=-1, keepdims=True)
    resid = patches - dot * cls_exp
    return resid / (np.linalg.norm(resid, axis=-1, keepdims=True) + 1e-8)


def _topk_avg(scores_2d, pct):
    P = scores_2d.shape[1]
    k = max(1, int(P * pct))
    return np.sort(scores_2d, axis=1)[:, -k:].mean(axis=1)


def _norm01(x):
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 0 else np.zeros_like(x)


def score_channel(tr_patches, te_patches, tr_cls, te_cls, tr_ml=None, te_ml=None):
    """Score a single channel using LP + ML methods."""
    N_te, P, D = te_patches.shape

    te_resid = compute_residuals(te_patches, te_cls)
    tr_resid = compute_residuals(tr_patches, tr_cls)
    tr_r_flat = tr_resid.reshape(-1, D).astype(np.float32)
    te_r_flat = te_resid.reshape(-1, D).astype(np.float32)

    tr_r_t = torch.tensor(tr_r_flat, dtype=torch.float32).to(DEVICE)
    PBATCH = 1024 if DEVICE.type == "cuda" else 256
    knn_res = []
    for i in range(0, len(te_r_flat), PBATCH):
        te_rb = torch.tensor(te_r_flat[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        knn_res.append(torch.topk(1.0 - te_rb @ tr_r_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())
    knn_r_win = np.concatenate(knn_res).reshape(N_te, P)

    results = {
        "resid_sum": knn_r_win.sum(axis=1),
        "resid_topk10": _topk_avg(knn_r_win, 0.10),
    }

    # Multi-layer
    if tr_ml and te_ml:
        tr_sum = sum(tr_ml[l] for l in ML_LAYERS)
        te_sum = sum(te_ml[l] for l in ML_LAYERS)
        te_ml_resid = compute_residuals(te_sum, te_cls)
        tr_ml_resid = compute_residuals(tr_sum, tr_cls)
        tr_mlr = tr_ml_resid.reshape(-1, D).astype(np.float32)
        te_mlr = te_ml_resid.reshape(-1, D).astype(np.float32)
        tr_mlr_t = torch.tensor(tr_mlr, dtype=torch.float32).to(DEVICE)
        knn_ml = []
        for i in range(0, len(te_mlr), PBATCH):
            te_b = torch.tensor(te_mlr[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
            knn_ml.append(torch.topk(1.0 - te_b @ tr_mlr_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())
        knn_ml_win = np.concatenate(knn_ml).reshape(N_te, P)
        results["ml_resid_sum"] = knn_ml_win.sum(axis=1)
        results["ml_resid_topk10"] = _topk_avg(knn_ml_win, 0.10)

    return results


def score_correlation(train_windows, test_windows):
    """
    Inter-variate scoring via correlation diff map + DINOv2.

    For each test window, compute correlation matrix (C x C),
    compare with train baseline correlation via DINOv2 patch features.
    """
    n_train = len(train_windows)
    n_test = len(test_windows)

    # Compute train baseline correlation (average over all train windows)
    train_corrs = []
    for w in train_windows:
        c = np.corrcoef(w.T)
        c = np.nan_to_num(c, nan=0.0)
        train_corrs.append(c)
    baseline_corr = np.mean(train_corrs, axis=0)

    # Generate diff images
    print("    Generating correlation diff maps...")
    diff_images = []
    for w in test_windows:
        test_corr = np.corrcoef(w.T)
        test_corr = np.nan_to_num(test_corr, nan=0.0)
        diff = test_corr - baseline_corr
        diff_images.append(corr_map_to_image(diff))

    baseline_images = [corr_map_to_image(baseline_corr)] * min(50, n_train)

    # Extract DINOv2 features
    print("    Extracting DINOv2 features for correlation maps...")
    _, tr_patches = extract_dinov2(baseline_images, multilayer=False)
    _, te_patches = extract_dinov2(diff_images, multilayer=False)

    # Simple scoring: L2 distance of patch features from baseline
    N_te, P, D = te_patches.shape
    tr_flat = tr_patches.reshape(-1, D).astype(np.float32)
    te_flat = te_patches.reshape(-1, D).astype(np.float32)
    tr_n = tr_flat / (np.linalg.norm(tr_flat, axis=1, keepdims=True) + 1e-8)
    te_n = te_flat / (np.linalg.norm(te_flat, axis=1, keepdims=True) + 1e-8)

    tr_t = torch.tensor(tr_n, dtype=torch.float32).to(DEVICE)
    PBATCH = 1024 if DEVICE.type == "cuda" else 256
    knn_list = []
    for i in range(0, len(te_n), PBATCH):
        te_b = torch.tensor(te_n[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        knn_list.append(torch.topk(1.0 - te_b @ tr_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())
    knn_win = np.concatenate(knn_list).reshape(N_te, P)

    # Also: simple Frobenius norm of diff (non-DINOv2 baseline)
    frob_scores = np.zeros(n_test)
    for i, w in enumerate(test_windows):
        test_corr = np.corrcoef(w.T)
        test_corr = np.nan_to_num(test_corr, nan=0.0)
        frob_scores[i] = np.linalg.norm(test_corr - baseline_corr, "fro")

    return {
        "inter_dino_sum": knn_win.sum(axis=1),
        "inter_dino_topk10": _topk_avg(knn_win, 0.10),
        "inter_frob": frob_scores,
    }


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


def f1max(ts_scores, labels):
    best_f1 = 0
    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = ts_scores.mean(), ts_scores.std()
        if sigma < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred = (ts_scores > thr).astype(int)
        gt_ivs = get_intervals(labels.astype(int))
        pred_ivs = get_intervals(pred)
        if not gt_ivs:
            continue
        gt = [tuple(i) for i in gt_ivs]
        pr = [tuple(i) for i in pred_ivs]
        TP = sum(sum(1 for a in gt if not (a[1] < d[0] or d[1] < a[0])) for d in pr if any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
        FP = sum(1 for d in pr if not any(not (a[1] < d[0] or d[1] < a[0]) for a in gt))
        FN = sum(1 for a in gt if not any(not (a[1] < d[0] or d[1] < a[0]) for d in pr))
        p = TP / (TP + FP) if (TP + FP) > 0 else 0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


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
# Main Experiment
# ════════════════════════════════════════════════════════

def run_multivariate_entity(entity_name, train_data, test_data, labels, cache_dir,
                            max_channels=10):
    """Run full multivariate experiment on one entity."""
    T_test, C = test_data.shape
    n_gt = len(get_intervals(labels))
    if n_gt == 0:
        print(f"  [{entity_name}] GT=0, skipping")
        return None

    print(f"\n  [{entity_name}] T_test={T_test}, C={C}, GT={n_gt}, anom={labels.mean():.2%}")
    t0 = time.time()

    # ── Step 1: Per-channel intra-variate scoring ──
    print(f"    Step 1: Per-channel scoring (top {min(C, max_channels)} of {C} channels)...")
    channel_scores = {}
    channels_to_use = min(C, max_channels)

    # Select channels with highest variance (most informative)
    var_per_channel = test_data.var(axis=0)
    top_channels = np.argsort(var_per_channel)[::-1][:channels_to_use]

    for ci in top_channels:
        ch_cache = cache_dir / entity_name
        ch_name = f"ch{ci}"

        tr_1d = train_data[:, ci:ci+1]
        te_1d = test_data[:, ci:ci+1]

        cls_c = ch_cache / f"{ch_name}_cls.npy"
        patch_c = ch_cache / f"{ch_name}_patches.npy"
        ml_c = ch_cache / f"{ch_name}_ml_L8.npy"

        if cls_c.exists() and patch_c.exists() and ml_c.exists():
            tr_cls = np.load(ch_cache / f"{ch_name}_train_cls.npy")
            tr_patches = np.load(ch_cache / f"{ch_name}_train_patches.npy")
            te_cls = np.load(cls_c)
            te_patches = np.load(patch_c)
            tr_ml = {l: np.load(ch_cache / f"{ch_name}_train_ml_L{l}.npy") for l in ML_LAYERS}
            te_ml = {l: np.load(ch_cache / f"{ch_name}_ml_L{l}.npy") for l in ML_LAYERS}
        else:
            ch_cache.mkdir(parents=True, exist_ok=True)

            ts_tr = tr_1d[:, 0].astype(float)
            lo, hi = ts_tr.min(), ts_tr.max()
            ts_tr = (ts_tr - lo) / (hi - lo + 1e-8)
            wins_tr = get_windows_1d(ts_tr)
            print(f"      ch{ci} train: {len(wins_tr)} windows, generating images...", flush=True)
            imgs_tr = [ts_to_image(w) for w in wins_tr]
            print(f"      ch{ci} train: extracting DINOv2...", flush=True)
            tr_cls, tr_patches, tr_ml = extract_dinov2(imgs_tr, multilayer=True)

            ts_te = te_1d[:, 0].astype(float)
            lo, hi = ts_te.min(), ts_te.max()
            ts_te = (ts_te - lo) / (hi - lo + 1e-8)
            wins_te = get_windows_1d(ts_te)
            print(f"      ch{ci} test: {len(wins_te)} windows, generating images...", flush=True)
            imgs_te = [ts_to_image(w) for w in wins_te]
            print(f"      ch{ci} test: extracting DINOv2...", flush=True)
            te_cls, te_patches, te_ml = extract_dinov2(imgs_te, multilayer=True)

            np.save(ch_cache / f"{ch_name}_train_cls.npy", tr_cls)
            np.save(ch_cache / f"{ch_name}_train_patches.npy", tr_patches)
            np.save(cls_c, te_cls)
            np.save(patch_c, te_patches)
            for l in ML_LAYERS:
                np.save(ch_cache / f"{ch_name}_train_ml_L{l}.npy", tr_ml[l])
                np.save(ch_cache / f"{ch_name}_ml_L{l}.npy", te_ml[l])

        ch_sc = score_channel(tr_patches, te_patches, tr_cls, te_cls, tr_ml, te_ml)
        for key, win_sc in ch_sc.items():
            if key not in channel_scores:
                channel_scores[key] = []
            channel_scores[key].append(win_to_ts(win_sc, T_test))

    # Aggregate per-channel scores: mean across channels
    intra_results = {}
    for key, ts_list in channel_scores.items():
        agg = np.mean(ts_list, axis=0)
        f1 = f1max(agg, labels)
        intra_results[f"intra_{key}_mean"] = f1

        agg_max = np.max(ts_list, axis=0)
        f1_max = f1max(agg_max, labels)
        intra_results[f"intra_{key}_max"] = f1_max

    # ── Step 2: Inter-variate correlation scoring ──
    print(f"    Step 2: Inter-variate correlation scoring...")
    train_wins = get_windows_nd(train_data)
    test_wins = get_windows_nd(test_data)

    inter_sc = score_correlation(train_wins, test_wins)
    inter_results = {}
    for key, win_sc in inter_sc.items():
        ts = win_to_ts(win_sc, T_test)
        inter_results[key] = f1max(ts, labels)

    # ── Step 3: Fusion (intra x inter) ──
    print(f"    Step 3: Intra x Inter fusion...")
    fusion_results = {}

    best_intra_key = max(channel_scores.keys(), key=lambda k: intra_results.get(f"intra_{k}_mean", 0))
    intra_ts = np.mean(channel_scores[best_intra_key], axis=0)
    inter_ts = win_to_ts(inter_sc["inter_dino_sum"], T_test)

    for w in [0.5, 0.7, 0.9]:
        fused = w * _norm01(intra_ts) + (1 - w) * _norm01(inter_ts)
        fusion_results[f"fusion_w{w}"] = f1max(fused, labels)

    mult_fused = _norm01(intra_ts) * _norm01(inter_ts)
    fusion_results["fusion_multiply"] = f1max(mult_fused, labels)

    elapsed = time.time() - t0
    print(f"    Done ({elapsed:.1f}s)")

    all_results = {"entity": entity_name, "n_channels": C, "n_gt": n_gt,
                   "T_test": T_test, "anom_ratio": labels.mean()}
    all_results.update({f"{k}_f1": v for k, v in intra_results.items()})
    all_results.update({f"{k}_f1": v for k, v in inter_results.items()})
    all_results.update({f"{k}_f1": v for k, v in fusion_results.items()})

    # Print summary
    best_intra = max(intra_results.values()) if intra_results else 0
    best_inter = max(inter_results.values()) if inter_results else 0
    best_fusion = max(fusion_results.values()) if fusion_results else 0
    print(f"    INTRA best={best_intra:.4f}  INTER best={best_inter:.4f}  FUSION best={best_fusion:.4f}")

    return all_results


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
    print(f"SMD: {len(entities)} entities")
    print(f"{'=' * 72}")

    results = []
    for ent in entities:
        try:
            train, test, labels = load_smd(data_dir, ent)
            cache_dir = RESULTS_DIR / "cache" / "SMD"
            r = run_multivariate_entity(ent, train, test, labels, cache_dir, max_channels=10)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR {ent}: {e}")
            import traceback; traceback.print_exc()

    return results


def run_psm(data_dir):
    download_psm(data_dir)
    train, test, labels = load_psm(data_dir)

    print(f"\n{'=' * 72}")
    print(f"PSM: {train.shape[1]} dimensions")
    print(f"{'=' * 72}")

    cache_dir = RESULTS_DIR / "cache" / "PSM"
    r = run_multivariate_entity("PSM", train, test, labels, cache_dir, max_channels=10)
    return [r] if r else []


# ════════════════════════════════════════════════════════
# Entry
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./mv_data")
    parser.add_argument("--output_dir", type=str, default="./results_multivariate")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["SMD", "PSM"])
    parser.add_argument("--max_entities", type=int, default=5, help="Max SMD entities to run")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data: {data_dir}")
    print(f"Output: {RESULTS_DIR}")

    t0 = time.time()
    all_results = {}

    for ds in args.datasets:
        ds_upper = ds.upper()
        if ds_upper == "SMD":
            results = run_smd(data_dir, max_entities=args.max_entities)
        elif ds_upper == "PSM":
            results = run_psm(data_dir)
        else:
            print(f"Unknown dataset: {ds}")
            continue

        if results:
            all_results[ds_upper] = results
            df = pd.DataFrame(results)
            csv_path = RESULTS_DIR / f"{ds_upper}_multivariate_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nSaved: {csv_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("MULTIVARIATE EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    for ds, results in all_results.items():
        f1_keys = sorted([k for k in results[0] if k.endswith("_f1")])
        print(f"\n--- {ds} ({len(results)} entities) ---")

        avgs = {}
        for k in f1_keys:
            vals = [r[k] for r in results]
            avgs[k] = np.mean(vals)

        ranked = sorted(avgs.items(), key=lambda x: -x[1])[:15]
        for i, (k, v) in enumerate(ranked, 1):
            tag = ""
            if "inter" in k:
                tag = " [INTER]"
            elif "fusion" in k:
                tag = " [FUSION]"
            elif "intra" in k:
                tag = " [INTRA]"
            print(f"  {i:2d}. {k.replace('_f1',''):<45} {v:.4f}{tag}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
