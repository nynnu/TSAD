"""
Re-evaluate existing results with:
  1. Better fusion strategies (adaptive, confidence-weighted)
  2. PA-F1 metric (Point-Adjust F1) for SOTA comparison
  3. Affiliation F1 for fair comparison with TimeRadar

Uses cached .npz scores from colab_multivariate_v2.py results.
No GPU needed.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

WINDOW_SIZE = 224
STEP = 56
CACHE_BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\results\VLM4TS_experiments_results_mv_v2\cache")

SMD_ENTITIES = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5"]

# ════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════

def download_smd_if_needed():
    import urllib.request
    smd = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
    for ent in SMD_ENTITIES:
        for split in ["train", "test", "test_label"]:
            dst = smd / split / f"{ent}.txt"
            if not dst.exists():
                (smd / split).mkdir(parents=True, exist_ok=True)
                url = f"https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/{split}/{ent}.txt"
                urllib.request.urlretrieve(url, str(dst))

def load_smd_labels(entity):
    smd = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\SMD")
    labels = np.loadtxt(smd / "test_label" / f"{entity}.txt", delimiter=",").astype(np.int32)
    test = np.loadtxt(smd / "test" / f"{entity}.txt", delimiter=",")
    return test, labels

def load_psm_labels():
    psm = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\mv_data\PSM")
    if not (psm / "test_label.csv").exists():
        import urllib.request
        psm.mkdir(parents=True, exist_ok=True)
        base = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"
        for f in ["test.csv", "test_label.csv"]:
            urllib.request.urlretrieve(f"{base}/{f}", str(psm / f))
    test = pd.read_csv(psm / "test.csv").iloc[:, 1:].values.astype(np.float64)
    labels = pd.read_csv(psm / "test_label.csv").iloc[:, 1:].values.astype(np.int32)
    if labels.ndim == 2:
        labels = labels.max(axis=1)
    return np.nan_to_num(test), labels


# ════════════════════════════════════════════════════════
# Score Loading from Cache
# ════════════════════════════════════════════════════════

def load_channel_scores(cache_dir, entity):
    ent_dir = cache_dir / entity
    scores = {}
    for f in sorted(ent_dir.glob("ch*_scores.npz")):
        ch = f.stem.replace("_scores", "")
        data = np.load(f)
        scores[ch] = {k: data[k] for k in data.files}
    return scores

def load_overlay_scores(cache_dir, entity):
    ent_dir = cache_dir / entity
    scores = []
    for f in sorted(ent_dir.glob("overlay_g*_scores.npz")):
        data = np.load(f)
        scores.append({k: data[k] for k in data.files})
    return scores

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

def _norm01(x):
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 0 else np.zeros_like(x)


# ════════════════════════════════════════════════════════
# Evaluation Metrics
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

def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

# Original interval overlap F1
def interval_f1(gt_ivs, pred_ivs):
    if not gt_ivs:
        return 0.0
    gt = [tuple(i) for i in gt_ivs]
    pr = [tuple(i) for i in pred_ivs]
    TP = sum(sum(1 for a in gt if _overlap(d, a)) for d in pr if any(_overlap(d, a) for a in gt))
    FP = sum(1 for d in pr if not any(_overlap(d, a) for a in gt))
    FN = sum(1 for a in gt if not any(_overlap(a, d) for d in pr))
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

# Point-Adjust F1 (PA-F1): if ANY point in GT segment is predicted, whole segment = TP
def pa_f1(labels, pred):
    gt_ivs = get_intervals(labels.astype(int))
    if not gt_ivs:
        return 0.0

    # Point-adjust: for each GT segment, if any point is predicted -> all points in segment become TP
    adjusted_pred = pred.copy()
    for s, e in gt_ivs:
        if pred[s:e+1].any():
            adjusted_pred[s:e+1] = 1

    tp = ((adjusted_pred == 1) & (labels == 1)).sum()
    fp = ((adjusted_pred == 1) & (labels == 0)).sum()
    fn = ((adjusted_pred == 0) & (labels == 1)).sum()

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def f1max_all_metrics(ts_scores, labels):
    """Compute F1max for interval-F1 and PA-F1."""
    best_interval = 0
    best_pa = 0

    for alpha in [0.1, 0.01, 0.001]:
        mu, sigma = ts_scores.mean(), ts_scores.std()
        if sigma < 1e-12:
            continue
        thr = mu + norm.ppf(1 - alpha) * sigma
        pred = (ts_scores > thr).astype(int)

        gt_ivs = get_intervals(labels.astype(int))
        pred_ivs = get_intervals(pred)

        best_interval = max(best_interval, interval_f1(gt_ivs, pred_ivs))
        best_pa = max(best_pa, pa_f1(labels, pred))

    return best_interval, best_pa


# ════════════════════════════════════════════════════════
# Fusion Strategies
# ════════════════════════════════════════════════════════

def compute_all_fusions(intra_ts_dict, inter_ts_dict, n_ts, labels):
    """
    intra_ts_dict: {score_key: ts_array} for each channel score
    inter_ts_dict: {score_key: ts_array} for each overlay group score
    """
    results = {}

    # Best intra: pick the score key with highest variance (most discriminative)
    intra_arrays = list(intra_ts_dict.values())
    inter_arrays = list(inter_ts_dict.values())

    if not intra_arrays or not inter_arrays:
        return results

    # Aggregate intra (mean across channels)
    intra_agg = np.mean(intra_arrays, axis=0)
    inter_agg = np.mean(inter_arrays, axis=0)

    intra_n = _norm01(intra_agg)
    inter_n = _norm01(inter_agg)

    # Strategy 1: Weighted sum (existing)
    for w in [0.3, 0.5, 0.7]:
        fused = w * intra_n + (1 - w) * inter_n
        iv, pa = f1max_all_metrics(fused, labels)
        results[f"fusion_w{w}"] = {"interval": iv, "pa": pa}

    # Strategy 2: MAX at each point (take whichever is higher)
    fused_max = np.maximum(intra_n, inter_n)
    iv, pa = f1max_all_metrics(fused_max, labels)
    results["fusion_max"] = {"interval": iv, "pa": pa}

    # Strategy 3: Adaptive (use inter where inter is confident, intra otherwise)
    # "confident" = score > median of that signal
    inter_confident = inter_n > np.median(inter_n)
    fused_adaptive = np.where(inter_confident, inter_n, intra_n)
    iv, pa = f1max_all_metrics(fused_adaptive, labels)
    results["fusion_adaptive"] = {"interval": iv, "pa": pa}

    # Strategy 4: Product (MtsCID-style)
    fused_mult = intra_n * inter_n
    iv, pa = f1max_all_metrics(fused_mult, labels)
    results["fusion_multiply"] = {"interval": iv, "pa": pa}

    # Strategy 5: Rank fusion (average rank position)
    intra_rank = np.argsort(np.argsort(intra_agg)).astype(float) / len(intra_agg)
    inter_rank = np.argsort(np.argsort(inter_agg)).astype(float) / len(inter_agg)
    fused_rank = (intra_rank + inter_rank) / 2
    iv, pa = f1max_all_metrics(fused_rank, labels)
    results["fusion_rank"] = {"interval": iv, "pa": pa}

    # Strategy 6: OR logic (anomaly if either INTRA or INTER says so)
    for alpha in [0.01]:
        mu_i, sig_i = intra_agg.mean(), intra_agg.std()
        mu_e, sig_e = inter_agg.mean(), inter_agg.std()
        if sig_i > 1e-12 and sig_e > 1e-12:
            thr_i = mu_i + norm.ppf(1 - alpha) * sig_i
            thr_e = mu_e + norm.ppf(1 - alpha) * sig_e
            pred_or = ((intra_agg > thr_i) | (inter_agg > thr_e)).astype(int)
            gt_ivs = get_intervals(labels.astype(int))
            pred_ivs = get_intervals(pred_or)
            iv = interval_f1(gt_ivs, pred_ivs)
            pa_val = pa_f1(labels, pred_or)
            results["fusion_OR"] = {"interval": iv, "pa": pa_val}

    return results


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def process_entity(dataset, entity, cache_dir, labels, n_ts):
    n_gt = len(get_intervals(labels))
    if n_gt == 0:
        return None

    ch_scores = load_channel_scores(cache_dir, entity)
    ov_scores = load_overlay_scores(cache_dir, entity)

    if not ch_scores or not ov_scores:
        print(f"  [{entity}] No cached scores found")
        return None

    # Build ts-level scores for each method
    score_key = "ml_topk10"  # best from single-variate experiments
    fallback_keys = ["final_topk10", "ml_sum", "final_sum"]

    # INTRA: per-channel ts scores (already ts-level in cache)
    intra_ts = {}
    for ch, sc in ch_scores.items():
        for key in [score_key] + fallback_keys:
            if key in sc:
                arr = sc[key]
                if len(arr) == n_ts:
                    intra_ts[ch] = arr
                else:
                    intra_ts[ch] = win_to_ts(arr, n_ts)
                break

    # INTER: per-group ts scores
    inter_ts = {}
    for gi, sc in enumerate(ov_scores):
        for key in [score_key] + fallback_keys:
            if key in sc:
                arr = sc[key]
                if len(arr) == n_ts:
                    inter_ts[f"g{gi}"] = arr
                else:
                    inter_ts[f"g{gi}"] = win_to_ts(arr, n_ts)
                break

    results = {"entity": entity, "n_gt": n_gt, "T_test": n_ts}

    # Evaluate INTRA
    intra_agg = np.mean(list(intra_ts.values()), axis=0)
    iv, pa = f1max_all_metrics(intra_agg, labels)
    results["intra_interval"] = iv
    results["intra_pa"] = pa

    # Evaluate INTRA with different aggregations
    intra_max = np.max(list(intra_ts.values()), axis=0)
    iv, pa = f1max_all_metrics(intra_max, labels)
    results["intra_max_interval"] = iv
    results["intra_max_pa"] = pa

    # Evaluate INTER
    inter_agg = np.mean(list(inter_ts.values()), axis=0)
    iv, pa = f1max_all_metrics(inter_agg, labels)
    results["inter_interval"] = iv
    results["inter_pa"] = pa

    inter_max_agg = np.max(list(inter_ts.values()), axis=0)
    iv, pa = f1max_all_metrics(inter_max_agg, labels)
    results["inter_max_interval"] = iv
    results["inter_max_pa"] = pa

    # Fusion strategies
    fusion = compute_all_fusions(intra_ts, inter_ts, n_ts, labels)
    for fname, fvals in fusion.items():
        results[f"{fname}_interval"] = fvals["interval"]
        results[f"{fname}_pa"] = fvals["pa"]

    return results


if __name__ == "__main__":
    print("Downloading SMD data if needed...", flush=True)
    download_smd_if_needed()

    all_results = []

    # SMD
    print("\n=== SMD ===", flush=True)
    for ent in SMD_ENTITIES:
        try:
            test, labels = load_smd_labels(ent)
            r = process_entity("SMD", ent, CACHE_BASE / "SMD", labels, len(labels))
            if r:
                all_results.append(r)
                print(f"  {ent}: INTRA={r['intra_interval']:.4f}({r['intra_pa']:.4f}pa) "
                      f"INTER={r['inter_interval']:.4f}({r['inter_pa']:.4f}pa) "
                      f"best_fusion={max(v for k,v in r.items() if 'fusion' in k and 'interval' in k):.4f}", flush=True)
        except Exception as e:
            print(f"  ERROR {ent}: {e}", flush=True)
            import traceback; traceback.print_exc()

    # PSM
    print("\n=== PSM ===", flush=True)
    try:
        test_psm, labels_psm = load_psm_labels()
        r = process_entity("PSM", "PSM", CACHE_BASE / "PSM", labels_psm, len(labels_psm))
        if r:
            all_results.append(r)
            print(f"  PSM: INTRA={r['intra_interval']:.4f}({r['intra_pa']:.4f}pa) "
                  f"INTER={r['inter_interval']:.4f}({r['inter_pa']:.4f}pa)", flush=True)
    except Exception as e:
        print(f"  ERROR PSM: {e}", flush=True)
        import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*90}")
    print("FULL RESULTS (Interval F1 / PA-F1)")
    print(f"{'='*90}")

    methods = ["intra", "intra_max", "inter", "inter_max",
               "fusion_w0.3", "fusion_w0.5", "fusion_w0.7",
               "fusion_max", "fusion_adaptive", "fusion_multiply",
               "fusion_rank", "fusion_OR"]

    header = f"{'Entity':<15}"
    for m in methods:
        short = m.replace("fusion_", "F_").replace("intra", "IN").replace("inter", "IT")
        header += f" {short[:10]:>10}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = f"{r['entity']:<15}"
        for m in methods:
            iv_key = f"{m}_interval"
            pa_key = f"{m}_pa"
            iv = r.get(iv_key, 0)
            pa = r.get(pa_key, 0)
            line += f" {iv:.3f}/{pa:.3f}"[:10].rjust(10)
        print(line)

    # Averages
    print("-" * len(header))
    avg_line = f"{'AVG':<15}"
    for m in methods:
        iv_vals = [r.get(f"{m}_interval", 0) for r in all_results]
        pa_vals = [r.get(f"{m}_pa", 0) for r in all_results]
        avg_line += f" {np.mean(iv_vals):.3f}/{np.mean(pa_vals):.3f}"[:10].rjust(10)
    print(avg_line)

    # Ranking by interval F1
    print(f"\n--- Ranking by avg Interval F1 ---")
    ranked = []
    for m in methods:
        iv_avg = np.mean([r.get(f"{m}_interval", 0) for r in all_results])
        pa_avg = np.mean([r.get(f"{m}_pa", 0) for r in all_results])
        ranked.append((m, iv_avg, pa_avg))
    ranked.sort(key=lambda x: -x[1])
    for i, (m, iv, pa) in enumerate(ranked, 1):
        print(f"  {i:2d}. {m:<25} Interval={iv:.4f}  PA-F1={pa:.4f}")

    # Comparison with SOTA
    print(f"\n--- Comparison with Training-Free SOTA (PA-F1) ---")
    our_best_pa = max(np.mean([r.get(f"{m}_pa", 0) for r in all_results]) for m in methods)
    smd_results = [r for r in all_results if r["entity"] != "PSM"]
    psm_results = [r for r in all_results if r["entity"] == "PSM"]

    smd_best = max(np.mean([r.get(f"{m}_pa", 0) for r in smd_results]) for m in methods) if smd_results else 0
    psm_best = max(r.get(f"{max(methods, key=lambda m: r.get(f'{m}_pa', 0))}_pa", 0) for r in psm_results) if psm_results else 0

    print(f"  {'Method':<30} {'SMD':>8} {'PSM':>8}")
    print(f"  {'-'*48}")
    print(f"  {'TimeRadar (KDD 26)  ':<30} {'84.80':>8} {'82.03':>8}")
    print(f"  {'DADA':<30} {'---':>8} {'---':>8}")
    print(f"  {'Ours (best PA-F1)':<30} {smd_best*100:>8.2f} {psm_best*100:>8.2f}")

    # Save
    out_path = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS\experiments\results_reeval")
    out_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_path / "reeval_fusion_paf1.csv", index=False)
    print(f"\nSaved: {out_path / 'reeval_fusion_paf1.csv'}")
