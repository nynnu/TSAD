"""
Adaptive LP/ZS Selection: automatically choose between LP and ZS scoring.

Strategy: measure train/test distribution distance.
- If distance > threshold: use ZS (distribution shift detected)
- If distance <= threshold: use LP (stable distribution)

Distribution distance metrics:
  1. CLS cosine distance: mean cosine dist between train/test CLS tokens
  2. Patch distribution divergence: KL-like divergence of patch norms
  3. Score variance ratio: variance of ZS scores vs LP scores

Also computes ensemble: weighted combination of LP and ZS scores.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

from experiment_dinov2_topk import (
    load_vlm4ts_data, extract_or_cache, win_to_ts, f1max,
    get_intervals, compute_all_scores, WINDOW_SIZE, STEP, RESULTS_DIR,
    MSL_CHANNELS, SMAP_CHANNELS, NAB_CHANNELS,
)
from experiment_dinov2_zs_topk import (
    _compute_residuals, zs_resid_gmd, zs_resid_testbank,
)


def _norm01(x):
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 0 else np.zeros_like(x)


def compute_distribution_distance(tr_cls, te_cls, tr_patches, te_patches):
    tr_cls_n = tr_cls / (np.linalg.norm(tr_cls, axis=1, keepdims=True) + 1e-8)
    te_cls_n = te_cls / (np.linalg.norm(te_cls, axis=1, keepdims=True) + 1e-8)

    tr_mean = tr_cls_n.mean(axis=0)
    te_mean = te_cls_n.mean(axis=0)
    tr_mean_n = tr_mean / (np.linalg.norm(tr_mean) + 1e-8)
    te_mean_n = te_mean / (np.linalg.norm(te_mean) + 1e-8)
    cls_dist = 1.0 - np.dot(tr_mean_n, te_mean_n)

    tr_norms = np.linalg.norm(tr_patches.reshape(-1, tr_patches.shape[-1]), axis=1)
    te_norms = np.linalg.norm(te_patches.reshape(-1, te_patches.shape[-1]), axis=1)
    norm_diff = abs(tr_norms.mean() - te_norms.mean()) / (tr_norms.std() + 1e-8)

    return cls_dist, norm_diff


def run_channel(dataset_name, channel_name):
    cache_dir = RESULTS_DIR / dataset_name

    train_data, test_data, labels = load_vlm4ts_data(dataset_name, channel_name)
    n_ts = len(test_data)
    n_gt = len(get_intervals(labels.astype(int)))
    if n_gt == 0:
        return None

    tr_cls, tr_patches = extract_or_cache(train_data, channel_name, "train", cache_dir)
    te_cls, te_patches = extract_or_cache(test_data, channel_name, "test", cache_dir)

    cls_dist, norm_diff = compute_distribution_distance(tr_cls, te_cls, tr_patches, te_patches)

    # LP scores
    lp_sc = compute_all_scores(tr_patches, te_patches, tr_cls, te_cls)
    lp_resid_topk10 = win_to_ts(lp_sc["resid_topk_10pct"], n_ts)
    lp_knn_sum = win_to_ts(lp_sc["knn_sum"], n_ts)

    # ZS scores
    resid_n = _compute_residuals(te_patches, te_cls)
    zs_gmd = zs_resid_gmd(resid_n)
    zs_tb = zs_resid_testbank(resid_n)
    zs_gmd_sum_ts = win_to_ts(zs_gmd["zs_gmd_sum"], n_ts)
    zs_tb_topk10_ts = win_to_ts(zs_tb["zs_testbank_topk10"], n_ts)

    # Individual F1s
    lp_rt10_f1 = f1max(lp_resid_topk10, labels)[0]
    lp_ks_f1 = f1max(lp_knn_sum, labels)[0]
    zs_gmd_f1 = f1max(zs_gmd_sum_ts, labels)[0]
    zs_tb10_f1 = f1max(zs_tb_topk10_ts, labels)[0]

    # Adaptive strategies
    results = {
        "channel": channel_name, "n_gt": n_gt,
        "cls_dist": cls_dist, "norm_diff": norm_diff,
        "lp_resid_topk10": lp_rt10_f1,
        "lp_knn_sum": lp_ks_f1,
        "zs_gmd_sum": zs_gmd_f1,
        "zs_testbank_topk10": zs_tb10_f1,
    }

    # Strategy 1: Threshold on cls_dist
    for thr in [0.001, 0.005, 0.01, 0.02, 0.05]:
        if cls_dist > thr:
            chosen = zs_tb_topk10_ts
            results[f"adapt_cls_{thr}"] = zs_tb10_f1
        else:
            chosen = lp_resid_topk10
            results[f"adapt_cls_{thr}"] = lp_rt10_f1

    # Strategy 2: Ensemble (LP + ZS normalized scores, then threshold)
    for w in [0.3, 0.5, 0.7]:
        ens = w * _norm01(lp_resid_topk10) + (1 - w) * _norm01(zs_tb_topk10_ts)
        ens_f1 = f1max(ens, labels)[0]
        results[f"ensemble_w{w}"] = ens_f1

    # Strategy 3: MAX fusion (take higher normalized score at each point)
    max_fuse = np.maximum(_norm01(lp_resid_topk10), _norm01(zs_tb_topk10_ts))
    results["max_fusion"] = f1max(max_fuse, labels)[0]

    # Strategy 4: Additive fusion
    add_fuse = _norm01(lp_resid_topk10) + _norm01(zs_tb_topk10_ts)
    results["add_fusion"] = f1max(add_fuse, labels)[0]

    return results


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["SMAP", "MSL", "NAB"]

    all_results = []

    for ds in datasets:
        ds_upper = ds.upper()
        channels = {"SMAP": SMAP_CHANNELS, "MSL": MSL_CHANNELS, "NAB": NAB_CHANNELS}.get(ds_upper, [])

        print(f"\n{'=' * 72}")
        print(f"Adaptive Scoring -- {ds_upper}")
        print(f"{'=' * 72}")

        ds_results = []
        for ch in channels:
            try:
                r = run_channel(ds_upper, ch)
                if r:
                    ds_results.append(r)
                    print(f"  {ch:<35} cls_d={r['cls_dist']:.6f} norm_d={r['norm_diff']:.4f} "
                          f"LP_rt10={r['lp_resid_topk10']:.4f} ZS_tb10={r['zs_testbank_topk10']:.4f} "
                          f"ens50={r['ensemble_w0.5']:.4f} max={r['max_fusion']:.4f}")
            except Exception as e:
                print(f"  ERROR {ch}: {e}")

        if ds_results:
            all_results.extend([(ds_upper, r) for r in ds_results])

            keys = ["lp_resid_topk10", "lp_knn_sum", "zs_gmd_sum", "zs_testbank_topk10",
                    "adapt_cls_0.001", "adapt_cls_0.005", "adapt_cls_0.01", "adapt_cls_0.02", "adapt_cls_0.05",
                    "ensemble_w0.3", "ensemble_w0.5", "ensemble_w0.7",
                    "max_fusion", "add_fusion"]

            print(f"\n  --- {ds_upper} Averages ---")
            for k in keys:
                vals = [r[k] for r in ds_results if k in r]
                if vals:
                    print(f"    {k:<30} {np.mean(vals):.4f}")

    # ALL datasets combined
    if all_results:
        print(f"\n{'=' * 72}")
        print(f"ALL DATASETS COMBINED")
        print(f"{'=' * 72}")
        keys = ["lp_resid_topk10", "lp_knn_sum", "zs_gmd_sum", "zs_testbank_topk10",
                "adapt_cls_0.001", "adapt_cls_0.005", "adapt_cls_0.01", "adapt_cls_0.02", "adapt_cls_0.05",
                "ensemble_w0.3", "ensemble_w0.5", "ensemble_w0.7",
                "max_fusion", "add_fusion"]

        ranked = []
        for k in keys:
            vals = [r[k] for _, r in all_results if k in r]
            if vals:
                ranked.append((k, np.mean(vals)))

        ranked.sort(key=lambda x: -x[1])
        for i, (k, v) in enumerate(ranked, 1):
            print(f"  {i:2d}. {k:<30} {v:.4f}")
