"""
DINOv2 Full Experiment Suite for Google Colab (GPU)
Upload this file to Colab and run: !python colab_dinov2_full.py

Experiments:
  1. LP Scoring (train-based): knn_sum, resid_sum, resid_topk variants
  2. ZS Scoring (zero-shot): gmd, testbank, ltr variants
  3. Ensemble: LP+ZS weighted combination
  4. Multi-Layer: layer 2,5,8,11 patch tokens (NEW - needs GPU)

Prerequisites (run these cells first in Colab):
  !pip install torch torchvision matplotlib scipy pandas tqdm pillow

  # Mount Google Drive (upload VLM4TS data folder)
  from google.colab import drive
  drive.mount('/content/drive')

  # Or upload data directly:
  # !gdown <your_gdrive_link> -O data.zip && unzip data.zip

Usage:
  !python colab_dinov2_full.py --data_dir /content/drive/MyDrive/VLM4TS/data --gpu
  !python colab_dinov2_full.py --data_dir ./data --gpu --datasets SMAP MSL NAB
"""

import argparse
import ast
import io
import json
import os
import time
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

MULTI_LAYERS = [2, 5, 8, 11]  # 0-indexed, Dinomaly uses middle layers

MSL_CHANNELS = [
    "P-11", "T-12", "D-15", "C-1", "F-8",
    "F-7", "T-13", "D-16", "T-8", "P-14", "D-14",
]
SMAP_CHANNELS = [
    "P-1", "P-3", "P-4", "P-7",
    "D-1", "D-2", "D-3",
    "F-1", "F-2", "F-3",
    "T-1", "T-2", "T-3",
    "R-1",
]
NAB_CHANNELS = [
    "ec2_cpu_utilization_5f5533", "ec2_cpu_utilization_24ae8d",
    "ec2_cpu_utilization_53ea38", "ec2_cpu_utilization_77c1ca",
    "ec2_cpu_utilization_825cc2", "ec2_cpu_utilization_ac20cd",
    "ec2_cpu_utilization_fe7f93", "ec2_disk_write_bytes_1ef3de",
    "ec2_disk_write_bytes_c0d644", "ec2_network_in_257a54",
    "ec2_network_in_5abac7", "elb_request_count_8c0756",
    "grok_asg_anomaly", "iio_us-east-1_i-a2eb1cd9_NetworkIn",
    "rds_cpu_utilization_cc0c53", "rds_cpu_utilization_e47b3b",
]

# ════════════════════════════════════════════════════════
# Globals
# ════════════════════════════════════════════════════════

DEVICE = None
DATA_DIR = None
RESULTS_DIR = None
_model = None
_anomalies = None

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def init_globals(args):
    global DEVICE, DATA_DIR, RESULTS_DIR
    DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(args.data_dir)
    RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")
    print(f"Output: {RESULTS_DIR}")
    if DEVICE.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")


def get_model():
    global _model
    if _model is None:
        print("Loading DINOv2 ViT-B/14...")
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        _model = _model.to(DEVICE).eval()
        print("DINOv2 loaded.")
    return _model


# ════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════

def get_anomalies():
    global _anomalies
    if _anomalies is None:
        df = pd.read_csv(DATA_DIR / "anomalies.csv")
        _anomalies = {}
        for _, row in df.iterrows():
            sig = row["signal"]
            events = ast.literal_eval(row["events"]) if isinstance(row["events"], str) else row["events"]
            _anomalies[sig] = events
    return _anomalies


def load_data(dataset_name, channel_name, train_ratio=0.5):
    ds = dataset_name.upper()
    if ds in ("MSL", "SMAP"):
        csv_path = DATA_DIR / ds / f"{channel_name}.csv"
    else:
        csv_path = DATA_DIR / "realAWSCloudwatch" / f"{channel_name}.csv"

    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    values = df["value"].values.astype(np.float64)

    anom = get_anomalies()
    labels_full = np.zeros(len(values), dtype=np.int32)
    if channel_name in anom:
        for s, e in anom[channel_name]:
            mask = (timestamps >= s) & (timestamps <= e)
            labels_full[mask] = 1

    split = int(len(values) * train_ratio)
    return (values[:split].reshape(-1, 1),
            values[split:].reshape(-1, 1),
            labels_full[split:])


# ════════════════════════════════════════════════════════
# Image Generation
# ════════════════════════════════════════════════════════

def ts_to_image(window):
    w_min, w_max = window.min(), window.max()
    normed = (window - w_min) / (w_max - w_min + 1e-8)
    fig = plt.figure(figsize=(_FIG_INCHES, _FIG_INCHES), dpi=_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(normed, color="black", linewidth=1.0, antialiased=True)
    ax.set_xlim(0, len(window) - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return img


def get_windows(ts):
    return [ts[s:s + WINDOW_SIZE] for s in range(0, len(ts) - WINDOW_SIZE + 1, STEP)]


# ════════════════════════════════════════════════════════
# Feature Extraction
# ════════════════════════════════════════════════════════

def extract_features(images, multilayer=False):
    """Extract DINOv2 features. If multilayer, also returns intermediate layer patches."""
    model = get_model()
    all_cls, all_patch = [], []
    all_ml_patches = {l: [] for l in MULTI_LAYERS} if multilayer else None

    with torch.no_grad():
        for s in tqdm(range(0, len(images), BATCH_SIZE), desc="DINOv2", unit="batch"):
            batch = torch.stack(
                [dinov2_transform(img) for img in images[s:s + BATCH_SIZE]]
            ).to(DEVICE)

            out = model.forward_features(batch)
            all_cls.append(out["x_norm_clstoken"].cpu().numpy())
            all_patch.append(out["x_norm_patchtokens"].cpu().numpy())

            if multilayer:
                ml_out = model.get_intermediate_layers(
                    batch, n=MULTI_LAYERS, return_class_token=True, norm=True)
                for i, (patch_tok, _cls_tok) in enumerate(ml_out):
                    all_ml_patches[MULTI_LAYERS[i]].append(patch_tok.cpu().numpy())

    cls = np.concatenate(all_cls)
    patches = np.concatenate(all_patch)

    if multilayer:
        ml_patches = {l: np.concatenate(v) for l, v in all_ml_patches.items()}
        return cls, patches, ml_patches
    return cls, patches


def extract_or_cache(data, channel, split, cache_dir, multilayer=False):
    cls_cache = cache_dir / f"{channel}_{split}_cls.npy"
    patch_cache = cache_dir / f"{channel}_{split}_patches.npy"
    ml_prefix = cache_dir / f"{channel}_{split}_ml"

    has_ml = all((cache_dir / f"{channel}_{split}_ml_L{l}.npy").exists() for l in MULTI_LAYERS) if multilayer else False

    if cls_cache.exists() and patch_cache.exists() and (not multilayer or has_ml):
        cls = np.load(cls_cache)
        patches = np.load(patch_cache)
        print(f"    Cache: {channel}/{split} cls={cls.shape} patches={patches.shape}")
        if multilayer:
            ml = {l: np.load(cache_dir / f"{channel}_{split}_ml_L{l}.npy") for l in MULTI_LAYERS}
            return cls, patches, ml
        return cls, patches

    ts = data[:, 0].astype(float)
    lo, hi = ts.min(), ts.max()
    ts = (ts - lo) / (hi - lo + 1e-8)
    wins = get_windows(ts)
    print(f"    {channel}/{split}: {len(wins)} windows")
    imgs = [ts_to_image(w) for w in wins]

    result = extract_features(imgs, multilayer=multilayer)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if multilayer:
        cls, patches, ml_patches = result
        np.save(cls_cache, cls)
        np.save(patch_cache, patches)
        for l, arr in ml_patches.items():
            np.save(cache_dir / f"{channel}_{split}_ml_L{l}.npy", arr)
        print(f"    Saved: cls={cls.shape}, patches={patches.shape}, ML layers={list(ml_patches.keys())}")
        return cls, patches, ml_patches
    else:
        cls, patches = result
        np.save(cls_cache, cls)
        np.save(patch_cache, patches)
        print(f"    Saved: cls={cls.shape}, patches={patches.shape}")
        return cls, patches


# ════════════════════════════════════════════════════════
# Scoring Functions
# ════════════════════════════════════════════════════════

def _topk_avg(scores_2d, pct):
    P = scores_2d.shape[1]
    k = max(1, int(P * pct))
    return np.sort(scores_2d, axis=1)[:, -k:].mean(axis=1)


def _norm01(x):
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 0 else np.zeros_like(x)


def compute_residuals(patches, cls):
    cls_n = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    cls_exp = cls_n[:, None, :]
    dot = (patches * cls_exp).sum(axis=-1, keepdims=True)
    resid = patches - dot * cls_exp
    return resid / (np.linalg.norm(resid, axis=-1, keepdims=True) + 1e-8)


def lp_scores(tr_patches, te_patches, tr_cls, te_cls):
    N_te, P, D = te_patches.shape
    tr_flat = tr_patches.reshape(-1, D).astype(np.float32)
    te_flat = te_patches.reshape(-1, D).astype(np.float32)
    tr_n = tr_flat / (np.linalg.norm(tr_flat, axis=1, keepdims=True) + 1e-8)
    te_n = te_flat / (np.linalg.norm(te_flat, axis=1, keepdims=True) + 1e-8)

    te_resid_n = compute_residuals(te_patches, te_cls)
    tr_resid_n = compute_residuals(tr_patches, tr_cls)
    tr_r_flat = tr_resid_n.reshape(-1, D).astype(np.float32)
    te_r_flat = te_resid_n.reshape(-1, D).astype(np.float32)

    tr_t = torch.tensor(tr_n, dtype=torch.float32).to(DEVICE)
    tr_r_t = torch.tensor(tr_r_flat, dtype=torch.float32).to(DEVICE)

    PBATCH = 1024 if DEVICE.type == "cuda" else 256
    knn_raw, knn_res = [], []
    for i in range(0, len(te_n), PBATCH):
        te_b = torch.tensor(te_n[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        te_rb = torch.tensor(te_r_flat[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        knn_raw.append(torch.topk(1.0 - te_b @ tr_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())
        knn_res.append(torch.topk(1.0 - te_rb @ tr_r_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())

    knn_win = np.concatenate(knn_raw).reshape(N_te, P)
    knn_r_win = np.concatenate(knn_res).reshape(N_te, P)

    return {
        "lp_knn_sum": knn_win.sum(axis=1),
        "lp_knn_max": knn_win.max(axis=1),
        "lp_resid_sum": knn_r_win.sum(axis=1),
        "lp_resid_max": knn_r_win.max(axis=1),
        "lp_resid_topk5": _topk_avg(knn_r_win, 0.05),
        "lp_resid_topk10": _topk_avg(knn_r_win, 0.10),
        "lp_topk10": _topk_avg(knn_win, 0.10),
    }


def zs_scores(te_patches, te_cls):
    N, P, D = te_patches.shape
    resid_n = compute_residuals(te_patches, te_cls)

    # GMD
    g_mean = resid_n.mean(axis=0)
    g_n = g_mean / (np.linalg.norm(g_mean, axis=-1, keepdims=True) + 1e-8)
    gmd_dist = 1.0 - (resid_n * g_n[None]).sum(axis=-1)

    # Testbank
    all_r = resid_n.reshape(-1, D).astype(np.float32)
    all_t = torch.tensor(all_r, dtype=torch.float32).to(DEVICE)
    tb_sum, tb_topk10 = np.zeros(N), np.zeros(N)
    for i in range(N):
        r_i = torch.tensor(resid_n[i].astype(np.float32)).to(DEVICE)
        dist = 1.0 - r_i @ all_t.T
        for j in range(P):
            dist[j, i * P:(i + 1) * P] = float("inf")
        knn = torch.topk(dist, K, dim=1, largest=False).values.mean(dim=1).cpu().numpy()
        tb_sum[i] = knn.sum()
        k10 = max(1, int(P * 0.10))
        tb_topk10[i] = np.sort(knn)[-k10:].mean()

    return {
        "zs_gmd_sum": gmd_dist.sum(axis=1),
        "zs_gmd_topk10": _topk_avg(gmd_dist, 0.10),
        "zs_testbank_sum": tb_sum,
        "zs_testbank_topk10": tb_topk10,
    }


def ml_scores(tr_ml, te_ml, tr_cls, te_cls):
    """Multi-layer scoring: sum patch features from multiple layers, then score."""
    results = {}
    for layer_set_name, layers in [("L2_5_8_11", [2, 5, 8, 11]), ("L5_8_11", [5, 8, 11]), ("L8_11", [8, 11])]:
        tr_sum = sum(tr_ml[l] for l in layers if l in tr_ml)
        te_sum = sum(te_ml[l] for l in layers if l in te_ml)
        N_te, P, D = te_sum.shape

        te_resid = compute_residuals(te_sum, te_cls)
        tr_resid = compute_residuals(tr_sum, tr_cls)

        tr_r_flat = tr_resid.reshape(-1, D).astype(np.float32)
        te_r_flat = te_resid.reshape(-1, D).astype(np.float32)

        tr_r_t = torch.tensor(tr_r_flat, dtype=torch.float32).to(DEVICE)
        PBATCH = 1024 if DEVICE.type == "cuda" else 256
        knn_res = []
        for i in range(0, len(te_r_flat), PBATCH):
            te_rb = torch.tensor(te_r_flat[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
            knn_res.append(torch.topk(1.0 - te_rb @ tr_r_t.T, K, dim=1, largest=False).values.mean(1).cpu().numpy())

        knn_r_win = np.concatenate(knn_res).reshape(N_te, P)
        results[f"ml_{layer_set_name}_resid_sum"] = knn_r_win.sum(axis=1)
        results[f"ml_{layer_set_name}_resid_topk10"] = _topk_avg(knn_r_win, 0.10)

    return results


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
    best_f1, best_p, best_r, best_a = 0, 0, 0, 0
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
            best_f1, best_p, best_r, best_a = f1, p, r, alpha
    return best_f1, best_p, best_r, best_a


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
# Main Runner
# ════════════════════════════════════════════════════════

def run_channel(dataset_name, channel_name, do_multilayer=True):
    cache_dir = RESULTS_DIR / "cache" / dataset_name
    train_data, test_data, labels = load_data(dataset_name, channel_name)
    n_ts = len(test_data)
    n_gt = len(get_intervals(labels.astype(int)))

    if n_gt == 0:
        print(f"  [{dataset_name}/{channel_name}] GT=0, skipping")
        return None

    print(f"\n  [{dataset_name}/{channel_name}] T={n_ts}, GT={n_gt}, anom={labels.mean():.2%}")

    if do_multilayer:
        tr_cls, tr_patches, tr_ml = extract_or_cache(train_data, channel_name, "train", cache_dir, multilayer=True)
        te_cls, te_patches, te_ml = extract_or_cache(test_data, channel_name, "test", cache_dir, multilayer=True)
    else:
        tr_cls, tr_patches = extract_or_cache(train_data, channel_name, "train", cache_dir, multilayer=False)
        te_cls, te_patches = extract_or_cache(test_data, channel_name, "test", cache_dir, multilayer=False)
        tr_ml, te_ml = None, None

    all_scores = {}

    # LP scores
    lp = lp_scores(tr_patches, te_patches, tr_cls, te_cls)
    all_scores.update(lp)

    # ZS scores
    zs = zs_scores(te_patches, te_cls)
    all_scores.update(zs)

    # Multi-layer scores
    if do_multilayer and tr_ml and te_ml:
        ml = ml_scores(tr_ml, te_ml, tr_cls, te_cls)
        all_scores.update(ml)

    # Ensemble
    lp_rt10_ts = win_to_ts(lp["lp_resid_topk10"], n_ts)
    zs_tb10_ts = win_to_ts(zs["zs_testbank_topk10"], n_ts)
    ens_ts = 0.7 * _norm01(lp_rt10_ts) + 0.3 * _norm01(zs_tb10_ts)
    all_scores["ensemble_0.7"] = None  # placeholder

    # Evaluate all
    results = {"channel": channel_name, "n_gt": n_gt}
    for key, win_sc in all_scores.items():
        if win_sc is None:
            ts = ens_ts
        else:
            ts = win_to_ts(win_sc, n_ts)
        f1, p, r, a = f1max(ts, labels)
        results[f"{key}_f1"] = f1

    # Print key results
    keys_show = ["lp_resid_sum", "lp_resid_topk10", "zs_testbank_topk10", "ensemble_0.7"]
    if do_multilayer:
        keys_show.append("ml_L2_5_8_11_resid_topk10")
    vals = " ".join(f"{k.split('_', 1)[1][:18]}={results.get(k+'_f1', 0):.4f}" for k in keys_show)
    print(f"    {vals}")

    return results


def run_all(datasets, do_multilayer=True):
    all_results = []

    for ds in datasets:
        ds_upper = ds.upper()
        channels = {"SMAP": SMAP_CHANNELS, "MSL": MSL_CHANNELS, "NAB": NAB_CHANNELS}.get(ds_upper, [])

        print(f"\n{'=' * 72}")
        print(f"Dataset: {ds_upper} ({len(channels)} channels), device={DEVICE}")
        print(f"{'=' * 72}")

        ds_results = []
        for ch in channels:
            try:
                r = run_channel(ds_upper, ch, do_multilayer=do_multilayer)
                if r:
                    ds_results.append(r)
            except Exception as e:
                print(f"  ERROR {ch}: {e}")
                import traceback; traceback.print_exc()

        if ds_results:
            all_results.append((ds_upper, ds_results))
            df = pd.DataFrame(ds_results)
            csv_path = RESULTS_DIR / f"{ds_upper}_full_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n  Saved: {csv_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    f1_keys = set()
    for ds, results in all_results:
        for r in results:
            f1_keys.update(k for k in r if k.endswith("_f1"))
    f1_keys = sorted(f1_keys)

    ds_avgs = {}
    for ds, results in all_results:
        avgs = {}
        for k in f1_keys:
            vals = [r[k] for r in results if k in r]
            avgs[k] = np.mean(vals) if vals else 0
        ds_avgs[ds] = avgs

    # ALL average
    all_avg = {}
    for k in f1_keys:
        vals = [ds_avgs[ds][k] for ds in ds_avgs if ds_avgs[ds].get(k, 0) > 0]
        all_avg[k] = np.mean(vals) if vals else 0

    ranked = sorted(all_avg.items(), key=lambda x: -x[1])

    print(f"\n{'Method':<40}", end="")
    for ds in ds_avgs:
        print(f" {ds:>8}", end="")
    print(f" {'ALL':>8}")
    print("-" * (42 + 9 * (len(ds_avgs) + 1)))

    for i, (k, avg) in enumerate(ranked[:20], 1):
        name = k.replace("_f1", "")
        print(f"{i:2d}. {name:<37}", end="")
        for ds in ds_avgs:
            print(f" {ds_avgs[ds].get(k, 0):>8.4f}", end="")
        print(f" {avg:>8.4f}")


# ════════════════════════════════════════════════════════
# Entry
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to VLM4TS/data folder")
    parser.add_argument("--output_dir", type=str, default="./results_colab", help="Output directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--datasets", nargs="+", default=["SMAP", "MSL", "NAB"])
    parser.add_argument("--no_multilayer", action="store_true", help="Skip multi-layer experiment")
    args = parser.parse_args()

    init_globals(args)
    t0 = time.time()
    run_all(args.datasets, do_multilayer=not args.no_multilayer)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
