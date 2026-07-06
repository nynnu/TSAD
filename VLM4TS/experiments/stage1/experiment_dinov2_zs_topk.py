"""
DINOv2 Zero-Shot Scoring: No train data needed.

Based on experiment_lineplt_resid_zs.py (나연이 ZS methods) + top-k avg.

Zero-Shot methods (test data only):
  - zs_resid_gmd_sum:  residual patch → global mean distance → sum
  - zs_resid_gmd_max:  residual patch → global mean distance → max
  - zs_resid_gmd_topk_5:  residual GMD → top-5% avg
  - zs_resid_gmd_topk_10: residual GMD → top-10% avg
  - zs_resid_intra_sum: residual intra-window KNN → sum
  - zs_resid_intra_topk_10: residual intra-window KNN → top-10% avg
  - zs_resid_testbank_sum: test patches as memory bank → sum
  - zs_resid_testbank_topk_10: test patches as memory bank → top-10% avg

Comparison baselines (train-based, loaded from previous experiment):
  - LP resid_sum, LP resid_topk_10pct (from experiment_dinov2_topk)
"""

import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

warnings.filterwarnings("ignore")

from experiment_dinov2_topk import (
    load_vlm4ts_data, extract_or_cache, win_to_ts, f1max,
    get_intervals, WINDOW_SIZE, STEP, DEVICE, RESULTS_DIR,
    MSL_CHANNELS, SMAP_CHANNELS, NAB_CHANNELS, K,
)

ZS_RESULTS_DIR = RESULTS_DIR.parent / "results_dinov2_zs_topk"
K_TEMPORAL = 3


def _compute_residuals(patches, cls):
    cls_n = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    cls_exp = cls_n[:, None, :]
    dot = (patches * cls_exp).sum(axis=-1, keepdims=True)
    resid = patches - dot * cls_exp
    resid_n = resid / (np.linalg.norm(resid, axis=-1, keepdims=True) + 1e-8)
    return resid_n


def _topk_avg(scores_2d, pct):
    P = scores_2d.shape[1]
    k = max(1, int(P * pct))
    return np.sort(scores_2d, axis=1)[:, -k:].mean(axis=1)


# ════════════════════════════════════════════════════════
# ZS Method 1: Global Mean Distance (GMD)
# ════════════════════════════════════════════════════════

def zs_resid_gmd(resid_n):
    global_mean = resid_n.mean(axis=0)
    g_n = global_mean / (np.linalg.norm(global_mean, axis=-1, keepdims=True) + 1e-8)
    cos_sim = (resid_n * g_n[None]).sum(axis=-1)
    dist = 1.0 - cos_sim
    return {
        "zs_gmd_sum": dist.sum(axis=1),
        "zs_gmd_max": dist.max(axis=1),
        "zs_gmd_topk5": _topk_avg(dist, 0.05),
        "zs_gmd_topk10": _topk_avg(dist, 0.10),
    }


# ════════════════════════════════════════════════════════
# ZS Method 2: Intra-window KNN
# ════════════════════════════════════════════════════════

def zs_resid_intra(resid_n):
    N, P, D = resid_n.shape
    sum_scores = np.zeros(N)
    max_scores = np.zeros(N)
    topk10_scores = np.zeros(N)

    for i in range(N):
        r = torch.tensor(resid_n[i], dtype=torch.float32).to(DEVICE)
        sim = r @ r.T
        dist = 1.0 - sim
        dist.fill_diagonal_(float("inf"))
        knn = torch.topk(dist, min(K, P - 1), dim=1, largest=False).values.mean(dim=1)
        knn_np = knn.cpu().numpy()
        sum_scores[i] = knn_np.sum()
        max_scores[i] = knn_np.max()
        k10 = max(1, int(P * 0.10))
        topk10_scores[i] = np.sort(knn_np)[-k10:].mean()

    return {
        "zs_intra_sum": sum_scores,
        "zs_intra_max": max_scores,
        "zs_intra_topk10": topk10_scores,
    }


# ════════════════════════════════════════════════════════
# ZS Method 3: Test-side Memory Bank
# ════════════════════════════════════════════════════════

def zs_resid_testbank(resid_n):
    N, P, D = resid_n.shape
    all_r = resid_n.reshape(-1, D).astype(np.float32)
    all_t = torch.tensor(all_r, dtype=torch.float32).to(DEVICE)

    sum_scores = np.zeros(N)
    max_scores = np.zeros(N)
    topk10_scores = np.zeros(N)

    for i in range(N):
        r_i = torch.tensor(resid_n[i].astype(np.float32)).to(DEVICE)
        dist = 1.0 - r_i @ all_t.T
        for j in range(P):
            dist[j, i * P:(i + 1) * P] = float("inf")
        knn = torch.topk(dist, K, dim=1, largest=False).values.mean(dim=1)
        knn_np = knn.cpu().numpy()
        sum_scores[i] = knn_np.sum()
        max_scores[i] = knn_np.max()
        k10 = max(1, int(P * 0.10))
        topk10_scores[i] = np.sort(knn_np)[-k10:].mean()

    return {
        "zs_testbank_sum": sum_scores,
        "zs_testbank_max": max_scores,
        "zs_testbank_topk10": topk10_scores,
    }


# ════════════════════════════════════════════════════════
# ZS Method 4: Temporal Neighbor (LTR)
# ════════════════════════════════════════════════════════

def zs_resid_ltr(resid_n, k=K_TEMPORAL):
    N, P, D = resid_n.shape
    scores = np.zeros((N, P), dtype=np.float32)
    for i in range(N):
        nbr_idx = [j for j in range(max(0, i - k), min(N, i + k + 1)) if j != i]
        if not nbr_idx:
            continue
        nbr_mean = resid_n[nbr_idx].mean(axis=0)
        nbr_n = nbr_mean / (np.linalg.norm(nbr_mean, axis=-1, keepdims=True) + 1e-8)
        cos_sim = (resid_n[i] * nbr_n).sum(axis=-1)
        scores[i] = 1.0 - cos_sim
    return {
        "zs_ltr_sum": scores.sum(axis=1),
        "zs_ltr_max": scores.max(axis=1),
        "zs_ltr_topk10": _topk_avg(scores, 0.10),
    }


# ════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════

def run_channel(dataset_name, channel_name):
    cache_dir = RESULTS_DIR / dataset_name
    t0 = time.time()

    train_data, test_data, labels = load_vlm4ts_data(dataset_name, channel_name)
    n_ts = len(test_data)
    n_gt = len(get_intervals(labels.astype(int)))

    if n_gt == 0:
        print(f"  [{dataset_name}/{channel_name}] GT=0, skipping")
        return None

    print(f"\n  [{dataset_name}/{channel_name}] T_test={n_ts}, GT={n_gt}")

    te_cls, te_patches = extract_or_cache(test_data, channel_name, "test", cache_dir)
    resid_n = _compute_residuals(te_patches, te_cls)

    print(f"    Computing ZS scores...")
    all_scores = {}
    all_scores.update(zs_resid_gmd(resid_n))
    all_scores.update(zs_resid_intra(resid_n))
    all_scores.update(zs_resid_ltr(resid_n))
    all_scores.update(zs_resid_testbank(resid_n))

    results = {}
    for key, win_scores in all_scores.items():
        ts = win_to_ts(win_scores, n_ts)
        f1, p, r, a = f1max(ts, labels)
        results[key] = {"f1max": f1, "p": p, "r": r, "alpha": a}

    elapsed = time.time() - t0
    gmd_s = results["zs_gmd_sum"]["f1max"]
    gmd_t10 = results["zs_gmd_topk10"]["f1max"]
    tb_s = results["zs_testbank_sum"]["f1max"]
    tb_t10 = results["zs_testbank_topk10"]["f1max"]
    print(f"    Done ({elapsed:.1f}s)")
    print(f"    gmd_sum={gmd_s:.4f}  gmd_topk10={gmd_t10:.4f}  "
          f"testbank_sum={tb_s:.4f}  testbank_topk10={tb_t10:.4f}")

    return {"channel": channel_name, "n_gt": n_gt, **results}


def run_dataset(dataset_name, channels):
    print(f"\n{'=' * 72}")
    print(f"DINOv2 Zero-Shot Scoring -- {dataset_name}  ({len(channels)} channels)")
    print(f"  window={WINDOW_SIZE}  step={STEP}  K={K}  device={DEVICE}")
    print(f"{'=' * 72}")

    results = []
    for ch in channels:
        try:
            r = run_channel(dataset_name, ch)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR {ch}: {e}")
            import traceback; traceback.print_exc()
    return results


def print_results(dataset_name, results):
    keys = ["zs_gmd_sum", "zs_gmd_max", "zs_gmd_topk5", "zs_gmd_topk10",
            "zs_intra_sum", "zs_intra_topk10",
            "zs_ltr_sum", "zs_ltr_topk10",
            "zs_testbank_sum", "zs_testbank_topk10"]

    print(f"\n{'=' * 120}")
    print(f"ZS Results: {dataset_name} ({len(results)} channels, GT>0)")
    print(f"{'=' * 120}")

    avgs = {k: [] for k in keys}
    for r in results:
        for k in keys:
            avgs[k].append(r[k]["f1max"])

    print(f"\n--- Ranking (by avg F1max) ---")
    ranked = sorted([(k, np.mean(avgs[k])) for k in keys], key=lambda x: -x[1])
    for i, (k, v) in enumerate(ranked, 1):
        print(f"  {i:2d}. {k:<30} {v:.4f}")

    print(f"\n--- Per-channel ---")
    header = f"{'Channel':<30} {'GT':>3}"
    show_keys = ["zs_gmd_sum", "zs_gmd_topk10", "zs_intra_sum", "zs_ltr_sum", "zs_testbank_sum", "zs_testbank_topk10"]
    for k in show_keys:
        header += f" {k[3:]:>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['channel']:<30} {r['n_gt']:>3}"
        for k in show_keys:
            line += f" {r[k]['f1max']:>14.4f}"
        print(line)


import pandas as pd

if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["SMAP", "MSL", "NAB"]

    for ds in datasets:
        ds_upper = ds.upper()
        if ds_upper == "MSL":
            channels = MSL_CHANNELS
        elif ds_upper == "SMAP":
            channels = SMAP_CHANNELS
        elif ds_upper == "NAB":
            channels = NAB_CHANNELS
        else:
            continue

        results = run_dataset(ds_upper, channels)
        if results:
            print_results(ds_upper, results)

            ZS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = ZS_RESULTS_DIR / f"{ds_upper}_zs_results.csv"
            rows = []
            for r in results:
                row = {"channel": r["channel"], "n_gt": r["n_gt"]}
                for k in r:
                    if isinstance(r[k], dict) and "f1max" in r[k]:
                        row[f"{k}_f1"] = r[k]["f1max"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"\nSaved: {csv_path}")
