"""
DINOv2 Patch Scoring: resid_sum + top-k avg (AnomalyDINO-inspired)

Based on 나연이's experiment_lineplt_patch.py logic, adapted for VLM4TS data format.
Adds top-k% average scoring (AnomalyDINO WACV'25 / GW Glitch Jun'26).

Scoring methods compared:
  - knn_max:    max patch KNN distance (나연이 baseline)
  - knn_sum:    sum of all patch KNN distances
  - resid_sum:  residual (patch - CLS direction) KNN sum (나연이 best)
  - resid_max:  residual KNN max
  - topk_1pct:  top-1% patch avg (AnomalyDINO TVaR)
  - topk_5pct:  top-5% patch avg
  - topk_10pct: top-10% patch avg
  - resid_topk_1pct:  residual + top-1% avg (our new combo)
  - resid_topk_5pct:  residual + top-5% avg
  - resid_topk_10pct: residual + top-10% avg

Data: VLM4TS CSV format (timestamp, value) + anomalies.csv
"""

import ast
import io
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

WINDOW_SIZE = 224
STEP = 56
K = 5
BATCH_SIZE = 16
IMAGE_SIZE = 224
_DPI = 100
_FIG_INCHES = IMAGE_SIZE / _DPI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results_dinov2_topk"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
def _get_model():
    global _model
    if _model is None:
        print("Loading DINOv2 ViT-B/14...")
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        _model = _model.to(DEVICE).eval()
        print(f"DINOv2 loaded on {DEVICE}")
    return _model


# ════════════════════════════════════════════════════════
# Data Loading (VLM4TS format)
# ════════════════════════════════════════════════════════

def _load_anomalies():
    df = pd.read_csv(DATA_DIR / "anomalies.csv")
    anom = {}
    for _, row in df.iterrows():
        sig = row["signal"]
        events = ast.literal_eval(row["events"]) if isinstance(row["events"], str) else row["events"]
        anom[sig] = events
    return anom

_ANOMALIES = None
def _get_anomalies():
    global _ANOMALIES
    if _ANOMALIES is None:
        _ANOMALIES = _load_anomalies()
    return _ANOMALIES

def load_vlm4ts_data(dataset_name, channel_name, train_ratio=0.5):
    ds = dataset_name.upper()
    if ds in ("MSL", "SMAP"):
        csv_path = DATA_DIR / ds / f"{channel_name}.csv"
    else:
        csv_path = DATA_DIR / "realAWSCloudwatch" / f"{channel_name}.csv"

    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    values = df["value"].values.astype(np.float64)

    anom = _get_anomalies()
    labels_full = np.zeros(len(values), dtype=np.int32)

    if channel_name in anom:
        for start_ts, end_ts in anom[channel_name]:
            mask = (timestamps >= start_ts) & (timestamps <= end_ts)
            labels_full[mask] = 1

    split = int(len(values) * train_ratio)
    train_vals = values[:split].reshape(-1, 1)
    test_vals = values[split:].reshape(-1, 1)
    labels = labels_full[split:]

    return train_vals, test_vals, labels


# ════════════════════════════════════════════════════════
# Time Series → Image → DINOv2
# ════════════════════════════════════════════════════════

def time_series_to_image(window):
    w_min, w_max = window.min(), window.max()
    normalized = (window - w_min) / (w_max - w_min + 1e-8)
    fig = plt.figure(figsize=(_FIG_INCHES, _FIG_INCHES), dpi=_DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.plot(normalized, color="black", linewidth=1.0, antialiased=True)
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

def get_windows(ts, window_size=WINDOW_SIZE, step=STEP):
    return [ts[s:s + window_size] for s in range(0, len(ts) - window_size + 1, step)]

def extract_features(images):
    model = _get_model()
    all_cls, all_patch = [], []
    with torch.no_grad():
        for s in tqdm(range(0, len(images), BATCH_SIZE), desc="DINOv2", unit="batch"):
            batch = torch.stack(
                [dinov2_transform(img) for img in images[s:s + BATCH_SIZE]]
            ).to(DEVICE)
            out = model.forward_features(batch)
            all_cls.append(out["x_norm_clstoken"].cpu().numpy())
            all_patch.append(out["x_norm_patchtokens"].cpu().numpy())
    return np.concatenate(all_cls), np.concatenate(all_patch)


def extract_or_cache(data, channel, split, cache_dir):
    cls_cache = cache_dir / f"{channel}_{split}_cls.npy"
    patch_cache = cache_dir / f"{channel}_{split}_patches.npy"

    if cls_cache.exists() and patch_cache.exists():
        cls = np.load(cls_cache)
        patches = np.load(patch_cache)
        print(f"    Cache: {channel}/{split} cls={cls.shape} patches={patches.shape}")
        return cls, patches

    ts = data[:, 0].astype(float)
    lo, hi = ts.min(), ts.max()
    ts = (ts - lo) / (hi - lo + 1e-8)
    wins = get_windows(ts)
    print(f"    {channel}/{split}: {len(wins)} windows, generating images...")
    imgs = [time_series_to_image(w) for w in wins]
    print(f"    Extracting DINOv2 features...")
    cls, patches = extract_features(imgs)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cls_cache, cls)
    np.save(patch_cache, patches)
    print(f"    Saved: cls={cls.shape}, patches={patches.shape}")
    return cls, patches


# ════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════

def compute_all_scores(tr_patches, te_patches, tr_cls, te_cls):
    N_te, P, D = te_patches.shape

    # Normalize raw patches
    tr_flat = tr_patches.reshape(-1, D).astype(np.float32)
    te_flat = te_patches.reshape(-1, D).astype(np.float32)
    tr_n = tr_flat / (np.linalg.norm(tr_flat, axis=1, keepdims=True) + 1e-8)
    te_n = te_flat / (np.linalg.norm(te_flat, axis=1, keepdims=True) + 1e-8)

    # Residual: patch - CLS direction
    cls_te_n = te_cls / (np.linalg.norm(te_cls, axis=1, keepdims=True) + 1e-8)
    cls_tr_n = tr_cls / (np.linalg.norm(tr_cls, axis=1, keepdims=True) + 1e-8)

    cls_te_exp = cls_te_n[:, None, :]
    dot_te = (te_patches * cls_te_exp).sum(axis=-1, keepdims=True)
    te_resid = te_patches - dot_te * cls_te_exp

    cls_tr_exp = cls_tr_n[:, None, :]
    dot_tr = (tr_patches * cls_tr_exp).sum(axis=-1, keepdims=True)
    tr_resid = tr_patches - dot_tr * cls_tr_exp

    tr_r_flat = tr_resid.reshape(-1, D).astype(np.float32)
    te_r_flat = te_resid.reshape(-1, D).astype(np.float32)
    tr_r_n = tr_r_flat / (np.linalg.norm(tr_r_flat, axis=1, keepdims=True) + 1e-8)
    te_r_n = te_r_flat / (np.linalg.norm(te_r_flat, axis=1, keepdims=True) + 1e-8)

    # KNN distances
    tr_t = torch.tensor(tr_n, dtype=torch.float32).to(DEVICE)
    tr_r_t = torch.tensor(tr_r_n, dtype=torch.float32).to(DEVICE)

    PBATCH = 256
    knn_raw_list, knn_res_list = [], []
    for i in range(0, len(te_n), PBATCH):
        te_b = torch.tensor(te_n[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        te_rb = torch.tensor(te_r_n[i:i + PBATCH], dtype=torch.float32).to(DEVICE)
        dist_r = 1.0 - te_b @ tr_t.T
        dist_res = 1.0 - te_rb @ tr_r_t.T
        knn_raw_list.append(torch.topk(dist_r, K, dim=1, largest=False).values.mean(1).cpu().numpy())
        knn_res_list.append(torch.topk(dist_res, K, dim=1, largest=False).values.mean(1).cpu().numpy())

    knn_flat = np.concatenate(knn_raw_list)
    knn_r_flat = np.concatenate(knn_res_list)

    knn_win = knn_flat.reshape(N_te, P)
    knn_r_win = knn_r_flat.reshape(N_te, P)

    results = {}

    # 나연이 baselines
    results["knn_max"] = knn_win.max(axis=1)
    results["knn_sum"] = knn_win.sum(axis=1)
    results["resid_sum"] = knn_r_win.sum(axis=1)
    results["resid_max"] = knn_r_win.max(axis=1)

    # NEW: top-k% avg (AnomalyDINO TVaR)
    for pct_name, pct in [("1pct", 0.01), ("5pct", 0.05), ("10pct", 0.10)]:
        k_val = max(1, int(P * pct))
        topk_raw = np.sort(knn_win, axis=1)[:, -k_val:].mean(axis=1)
        topk_res = np.sort(knn_r_win, axis=1)[:, -k_val:].mean(axis=1)
        results[f"topk_{pct_name}"] = topk_raw
        results[f"resid_topk_{pct_name}"] = topk_res

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

def evaluate_intervals(gt_intervals, pred_intervals):
    gt = [tuple(i) for i in gt_intervals]
    pred = [tuple(i) for i in pred_intervals]
    TP, FP = 0, 0
    for d in pred:
        cnt = sum(1 for a in gt if not (a[1] < d[0] or d[1] < a[0]))
        if cnt > 0:
            TP += cnt
        else:
            FP += 1
    FN = sum(1 for a in gt if not any(not (a[1] < d[0] or d[1] < a[0]) for d in pred))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

def f1max(ts_scores, labels):
    best_f1, best_p, best_r, best_a = 0, 0, 0, 0
    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = ts_scores.mean(), ts_scores.std()
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred = (ts_scores > thr).astype(int)
        gt_ivs = get_intervals(labels.astype(int))
        pred_ivs = get_intervals(pred)
        f1, p, r = evaluate_intervals(gt_ivs, pred_ivs)
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
# Run
# ════════════════════════════════════════════════════════

def run_channel(dataset_name, channel_name):
    cache_dir = RESULTS_DIR / dataset_name
    t0 = time.time()

    train_data, test_data, labels = load_vlm4ts_data(dataset_name, channel_name)
    n_ts = len(test_data)
    n_gt = len(get_intervals(labels.astype(int)))
    anom_ratio = labels.sum() / len(labels) if len(labels) > 0 else 0

    print(f"\n  [{dataset_name}/{channel_name}] T_test={n_ts}, GT={n_gt}, anom={anom_ratio:.2%}")

    tr_cls, tr_patches = extract_or_cache(train_data, channel_name, "train", cache_dir)
    te_cls, te_patches = extract_or_cache(test_data, channel_name, "test", cache_dir)

    print(f"    Computing scores...")
    sc = compute_all_scores(tr_patches, te_patches, tr_cls, te_cls)

    results = {}
    for key, win_scores in sc.items():
        ts = win_to_ts(win_scores, n_ts)
        f1, p, r, a = f1max(ts, labels)
        results[key] = {"f1max": f1, "p": p, "r": r, "alpha": a}

    elapsed = time.time() - t0
    print(f"    Done ({elapsed:.1f}s)")
    print(f"    knn_max={results['knn_max']['f1max']:.4f}  "
          f"resid_sum={results['resid_sum']['f1max']:.4f}  "
          f"topk_1pct={results['topk_1pct']['f1max']:.4f}  "
          f"resid_topk_5pct={results['resid_topk_5pct']['f1max']:.4f}")

    return {"channel": channel_name, "n_gt": n_gt, **results}


def run_dataset(dataset_name, channels):
    print(f"\n{'=' * 72}")
    print(f"DINOv2 Patch Scoring (resid + top-k) -- {dataset_name}  ({len(channels)} channels)")
    print(f"  window={WINDOW_SIZE}  step={STEP}  K={K}  device={DEVICE}")
    print(f"{'=' * 72}")

    results = []
    for ch in channels:
        try:
            r = run_channel(dataset_name, ch)
            results.append(r)
        except Exception as e:
            print(f"  ERROR {ch}: {e}")
            import traceback; traceback.print_exc()
    return results


def print_results(dataset_name, results):
    keys = ["knn_max", "knn_sum", "resid_sum", "resid_max",
            "topk_1pct", "topk_5pct", "topk_10pct",
            "resid_topk_1pct", "resid_topk_5pct", "resid_topk_10pct"]

    print(f"\n{'=' * 100}")
    print(f"Results: {dataset_name}")
    print(f"{'=' * 100}")

    header = f"{'Channel':<20} {'GT':>3}"
    for k in keys:
        header += f" {k[:12]:>12}"
    print(header)
    print("-" * len(header))

    avgs = {k: [] for k in keys}
    for r in results:
        line = f"{r['channel']:<20} {r['n_gt']:>3}"
        for k in keys:
            v = r[k]["f1max"]
            line += f" {v:>12.4f}"
            avgs[k].append(v)
        print(line)

    print("-" * len(header))
    avg_line = f"{'AVERAGE':<20} {'':>3}"
    for k in keys:
        avg_line += f" {np.mean(avgs[k]):>12.4f}"
    print(avg_line)

    print(f"\n--- Ranking (by avg F1max) ---")
    ranked = sorted([(k, np.mean(avgs[k])) for k in keys], key=lambda x: -x[1])
    for i, (k, v) in enumerate(ranked, 1):
        print(f"  {i}. {k:<25} {v:.4f}")


if __name__ == "__main__":
    import sys

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["SMAP"]

    for ds in datasets:
        ds_upper = ds.upper()
        if ds_upper == "MSL":
            channels = MSL_CHANNELS
        elif ds_upper == "SMAP":
            channels = SMAP_CHANNELS
        elif ds_upper == "NAB":
            channels = NAB_CHANNELS
        else:
            print(f"Unknown dataset: {ds}")
            continue

        results = run_dataset(ds_upper, channels)
        if results:
            print_results(ds_upper, results)

            csv_path = RESULTS_DIR / f"{ds_upper}_results.csv"
            rows = []
            for r in results:
                row = {"channel": r["channel"], "n_gt": r["n_gt"]}
                for k in r:
                    if isinstance(r[k], dict) and "f1max" in r[k]:
                        row[f"{k}_f1"] = r[k]["f1max"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"\nSaved: {csv_path}")
