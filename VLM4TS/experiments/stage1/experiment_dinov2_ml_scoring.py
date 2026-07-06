"""
DINOv2 Multi-Layer Patch Scoring — 40-signal evaluation
=========================================================
Compares LP scoring methods on our exact 40 signals (NAB16+SMAP13+MSL11)
using the same fixed interval-F1 metric and 90th-pct threshold as dino_k5 LTR.

Scoring methods:
  knn_sum          : final-layer patch KNN sum
  resid_sum        : final-layer residual KNN sum (나연 baseline)
  resid_topk10     : final-layer residual top-10% avg
  ml_L8_11_resid_sum   : layer8+11 concatenated residual KNN sum
  ml_L8_11_resid_topk10: layer8+11 concatenated residual top-10% avg

Important design choices:
  - Memory bank  : first 50% of signal (train portion)
  - Scoring domain: second 50% only (test portion)
  - Threshold    : 90th percentile of test window scores
  - Metric       : fixed interval-F1 (TP_pred + TP_gt separated)

Limitation: 3 NAB signals (825cc2, 257a54, iio) have anomalies in the FIRST
50% only. LP methods will get F1=0 on these — a genuine limitation vs LTR.

Runtime estimate (CPU): ~40-70 min for 40 signals.

Usage:
  python experiment_dinov2_ml_scoring.py [--dataset NAB|SMAP|MSL|ALL]
"""

import argparse, ast, io, json, sys, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ──────────────────────────────────────────────────────────────────
WIN    = 224
STRIDE = 56
K      = 5
BATCH  = 16
LOOSE_PCT = 90.0   # threshold percentile (matches dino_k5 LTR Stage1)

BASE      = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE / "data"
OUT_DIR   = BASE / "experiments" / "results_dinov2_ml_scoring"
CACHE_DIR = OUT_DIR / "feature_cache"

DATASETS = {
    "MSL":  ["C-1","D-14","D-15","D-16","F-7","F-8","P-11","P-14","T-12","T-13","T-8"],
    "NAB":  ["ec2_cpu_utilization_24ae8d","ec2_cpu_utilization_53ea38",
             "ec2_cpu_utilization_5f5533","ec2_cpu_utilization_77c1ca",
             "ec2_cpu_utilization_825cc2","ec2_cpu_utilization_ac20cd",
             "ec2_cpu_utilization_fe7f93","ec2_disk_write_bytes_1ef3de",
             "ec2_disk_write_bytes_c0d644","ec2_network_in_257a54",
             "ec2_network_in_5abac7","elb_request_count_8c0756",
             "grok_asg_anomaly","iio_us-east-1_i-a2eb1cd9_NetworkIn",
             "rds_cpu_utilization_cc0c53","rds_cpu_utilization_e47b3b"],
    "SMAP": ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
             "F-1","F-2","F-3","P-1","T-1"],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
def get_model():
    global _model
    if _model is None:
        print("Loading DINOv2 ViT-B/14 ...")
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        _model = _model.to(DEVICE).eval()
        print(f"  Loaded on {DEVICE}")
    return _model


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data(ds, sig):
    if ds == "NAB":
        csv = DATA_DIR / "realAWSCloudwatch" / f"{sig}.csv"
    else:
        csv = DATA_DIR / ds / f"{sig}.csv"
    df = pd.read_csv(csv)
    return df["timestamp"].values, df["value"].values.astype(np.float64)

def load_gt_labels(sig, timestamps):
    df = pd.read_csv(DATA_DIR / "anomalies.csv")
    anom = {}
    for _, row in df.iterrows():
        anom[row["signal"]] = ast.literal_eval(row["events"]) if isinstance(row["events"], str) else row["events"]
    labels = np.zeros(len(timestamps), dtype=int)
    for s, e in anom.get(sig, []):
        labels[(timestamps >= s) & (timestamps <= e)] = 1
    return labels

def make_windows(values):
    return [values[s:s+WIN] for s in range(0, len(values) - WIN + 1, STRIDE)]

def render_image(window):
    w = window.copy().astype(float)
    lo, hi = w.min(), w.max()
    w = (w - lo) / (hi - lo + 1e-8)
    fig = plt.figure(figsize=(224/100, 224/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(w, color="black", linewidth=1.0)
    ax.set_xlim(0, len(w)-1); ax.set_ylim(-0.02, 1.02); ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.size != (224, 224): img = img.resize((224, 224), Image.LANCZOS)
    return img


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_with_intermediate(windows, cache_prefix):
    """Extract final + L8 + L11 features. Cache to disk. Returns (cls_f, patches_f, L8, L11)."""
    cls_path     = Path(f"{cache_prefix}_cls_f32.npy")
    final_path   = Path(f"{cache_prefix}_patches_f16.npy")
    L8_path      = Path(f"{cache_prefix}_L8_f16.npy")
    L11_path     = Path(f"{cache_prefix}_L11_f16.npy")

    if cls_path.exists() and final_path.exists() and L8_path.exists() and L11_path.exists():
        print(f"      cache hit: {cache_prefix.name}")
        return (np.load(cls_path),
                np.load(final_path).astype(np.float32),
                np.load(L8_path).astype(np.float32),
                np.load(L11_path).astype(np.float32))

    model = get_model()
    n = len(windows)
    print(f"      rendering {n} images ...")
    t0 = time.time()
    imgs = [render_image(w) for w in windows]
    print(f"      DINOv2 forward ({n} windows, batch={BATCH}) ...")

    cls_list, f_list, L8_list, L11_list = [], [], [], []
    with torch.no_grad():
        for s in range(0, n, BATCH):
            batch = torch.stack([_dino_transform(im) for im in imgs[s:s+BATCH]]).to(DEVICE)

            # Final layer
            out = model.forward_features(batch)
            cls_list.append(out["x_norm_clstoken"].cpu().numpy())
            f_list.append(out["x_norm_patchtokens"].cpu().numpy())

            # Intermediate layers 8 and 11
            inters = model.get_intermediate_layers(batch, n=[8, 11], return_class_token=False)
            # inters[0] = L8 patch tokens (B, 256, 768), inters[1] = L11 patch tokens
            L8_list.append(inters[0].cpu().numpy())
            L11_list.append(inters[1].cpu().numpy())

    cls_f   = np.concatenate(cls_list)         # (N, 768) float32
    f_patch = np.concatenate(f_list)           # (N, 256, 768) float32
    L8      = np.concatenate(L8_list)          # (N, 256, 768) float32
    L11     = np.concatenate(L11_list)         # (N, 256, 768) float32

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cls_path,   cls_f)
    np.save(final_path, f_patch.astype(np.float16))
    np.save(L8_path,    L8.astype(np.float16))
    np.save(L11_path,   L11.astype(np.float16))
    print(f"      cached ({time.time()-t0:.1f}s)")
    return cls_f, f_patch, L8, L11


# ── KNN scoring ───────────────────────────────────────────────────────────────

def _knn_dist(query_flat, bank_flat, k=K):
    """Cosine KNN distance: (N_q, D) vs (N_b, D). Returns (N_q,) mean of top-k."""
    q = torch.tensor(query_flat, dtype=torch.float32).to(DEVICE)
    b = torch.tensor(bank_flat,  dtype=torch.float32).to(DEVICE)
    # normalize
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    PBATCH = 512
    result = []
    for i in range(0, len(q), PBATCH):
        d = 1.0 - (q[i:i+PBATCH] @ b.T)
        result.append(torch.topk(d, k, dim=1, largest=False).values.mean(1).cpu().numpy())
    return np.concatenate(result)

def residual(patches, cls):
    """Remove CLS direction from each patch. patches: (N,P,D), cls: (N,D) -> (N,P,D)."""
    cls_n = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)  # (N,D)
    cls_exp = cls_n[:, None, :]                                           # (N,1,D)
    dot = (patches * cls_exp).sum(axis=-1, keepdims=True)                # (N,P,1)
    return patches - dot * cls_exp                                        # (N,P,D)

def compute_scores(tr_cls, tr_patches, te_cls, te_patches, tr_L8, te_L8, tr_L11, te_L11):
    """Compute all scoring methods. Returns dict {method_name: (N_te,) window scores}."""
    N_te, P, D = te_patches.shape
    results = {}

    # ── Final layer ──────────────────────────────────────────────────────────
    tr_resid = residual(tr_patches, tr_cls)     # (N_tr, 256, D)
    te_resid = residual(te_patches, te_cls)     # (N_te, 256, D)

    # knn_sum
    dist_f = _knn_dist(te_patches.reshape(-1, D), tr_patches.reshape(-1, D))
    results["knn_sum"] = dist_f.reshape(N_te, P).sum(axis=1)

    # resid_sum
    dist_r = _knn_dist(te_resid.reshape(-1, D), tr_resid.reshape(-1, D))
    knn_r = dist_r.reshape(N_te, P)
    results["resid_sum"] = knn_r.sum(axis=1)

    # resid_topk10
    k10 = max(1, int(P * 0.10))
    results["resid_topk10"] = np.sort(knn_r, axis=1)[:, -k10:].mean(axis=1)

    # ── Multi-layer L8+L11 concatenated ──────────────────────────────────────
    # Use the CLS from the FINAL layer for residual projection (most informative)
    tr_ml = np.concatenate([tr_L8, tr_L11], axis=-1)   # (N_tr, 256, 1536)
    te_ml = np.concatenate([te_L8, te_L11], axis=-1)   # (N_te, 256, 1536)
    D_ml = tr_ml.shape[-1]

    # Extend CLS to match concatenated dim by repeating (proxy)
    tr_cls_ml = np.concatenate([tr_cls, tr_cls], axis=-1)  # (N_tr, 1536)
    te_cls_ml = np.concatenate([te_cls, te_cls], axis=-1)  # (N_te, 1536)

    tr_ml_resid = residual(tr_ml, tr_cls_ml)
    te_ml_resid = residual(te_ml, te_cls_ml)

    dist_ml_r = _knn_dist(te_ml_resid.reshape(-1, D_ml), tr_ml_resid.reshape(-1, D_ml))
    knn_ml_r = dist_ml_r.reshape(N_te, P)
    results["ml_L8_11_resid_sum"]   = knn_ml_r.sum(axis=1)
    results["ml_L8_11_resid_topk10"] = np.sort(knn_ml_r, axis=1)[:, -k10:].mean(axis=1)

    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def interval_f1_fixed(pred_ivs, gt_ivs):
    """Fixed interval-F1: TP_pred + TP_gt separated (our metric)."""
    if not gt_ivs:
        return 1.0 if not pred_ivs else 0.0, 1.0, 1.0
    if not pred_ivs:
        return 0.0, 0.0, 0.0
    TP_pred = sum(1 for p in pred_ivs if any(not (g[1]<p[0] or p[1]<g[0]) for g in gt_ivs))
    TP_gt   = sum(1 for g in gt_ivs  if any(not (g[1]<p[0] or p[1]<g[0]) for p in pred_ivs))
    P = TP_pred / len(pred_ivs)
    R = TP_gt   / len(gt_ivs)
    F = 2*P*R/(P+R) if (P+R) > 0 else 0.0
    return F, P, R

def scores_to_intervals(win_scores, offset, T_test, merge_gap=WIN//2, min_len=10):
    """Convert window scores to intervals using 90th percentile threshold."""
    thr = float(np.percentile(win_scores, LOOSE_PCT))
    binary = np.zeros(T_test, dtype=int)
    for i, s_win in enumerate(win_scores):
        s = i * STRIDE
        if s_win >= thr:
            e = min(s + WIN, T_test)
            binary[s:e] = 1
    # segment → intervals
    raw, in_seg, ss = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_seg: ss, in_seg = i, True
        elif not v and in_seg: raw.append((ss+offset, i-1+offset)); in_seg = False
    if in_seg: raw.append((ss+offset, T_test-1+offset))
    # merge close intervals
    merged = []
    for iv in raw:
        if merged and iv[0]-merged[-1][1] <= merge_gap: merged[-1] = (merged[-1][0], iv[1])
        else: merged.append(list(iv))
    return [(s, e) for s, e in merged if e-s+1 >= min_len]

def get_gt_intervals(labels):
    ivs, in_seg, ss = [], False, 0
    for i, v in enumerate(labels):
        if v and not in_seg: ss, in_seg = i, True
        elif not v and in_seg: ivs.append((ss, i-1)); in_seg = False
    if in_seg: ivs.append((ss, len(labels)-1))
    return ivs


# ── Per-signal runner ─────────────────────────────────────────────────────────

def run_signal(ds, sig):
    ts, values = load_data(ds, sig)
    T = len(ts)
    split = T // 2
    labels = load_gt_labels(sig, ts)

    # Train / test values (index-based)
    train_vals = values[:split]
    test_vals  = values[split:]
    test_labels = labels[split:]
    T_test = len(test_vals)

    # GT intervals (relative to test start)
    gt_ivs_rel = get_gt_intervals(test_labels)
    n_gt = len(gt_ivs_rel)

    anom_in_test = gt_ivs_rel != []
    has_anom_test = bool(gt_ivs_rel)

    print(f"\n  [{ds}/{sig}] T={T} split={split} T_test={T_test} GT_in_test={n_gt}")
    if not has_anom_test:
        print(f"    WARNING: no GT in test portion — LP will get F1=0 on this signal")

    # Windows for train / test
    tr_wins = make_windows(train_vals)
    te_wins = make_windows(test_vals)
    n_tr = len(tr_wins); n_te = len(te_wins)
    print(f"    windows: train={n_tr}  test={n_te}")

    cache_pre = CACHE_DIR / f"{ds}__{sig}"
    tr_cls, tr_p, tr_L8, tr_L11 = extract_features_with_intermediate(tr_wins, Path(f"{cache_pre}__train"))
    te_cls, te_p, te_L8, te_L11 = extract_features_with_intermediate(te_wins, Path(f"{cache_pre}__test"))

    print(f"    scoring ...")
    sc = compute_scores(tr_cls, tr_p, te_cls, te_p, tr_L8, te_L8, tr_L11, te_L11)

    method_results = {}
    for name, win_scores in sc.items():
        pred_ivs = scores_to_intervals(win_scores, 0, T_test)
        f1, p, r = interval_f1_fixed(pred_ivs, gt_ivs_rel)
        method_results[name] = {"f1": round(f1, 4), "p": round(p, 4), "r": round(r, 4)}

    print(f"    resid_sum={method_results['resid_sum']['f1']:.4f}  "
          f"ml_L8_11_resid_sum={method_results['ml_L8_11_resid_sum']['f1']:.4f}")

    return {
        "ds": ds, "sig": sig, "T": T, "T_test": T_test, "n_gt": n_gt,
        "has_anom_test": has_anom_test,
        **{f"{k}_f1": v["f1"] for k, v in method_results.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ALL", choices=["NAB","SMAP","MSL","ALL"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "partial_results.jsonl"

    # Load already-done signals
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["ds"], r["sig"]))
                except: pass

    ds_list = [args.dataset] if args.dataset != "ALL" else ["MSL","NAB","SMAP"]
    signals = [(ds, sig) for ds in ds_list for sig in DATASETS[ds]]
    todo = [(ds, sig) for ds, sig in signals if (ds, sig) not in done]
    print(f"Signals to run: {len(todo)}/{len(signals)} (done={len(done)})")
    print(f"Device: {DEVICE}")
    print("NOTE: 3 NAB signals (825cc2, 257a54, iio) have anomalies in first 50% only → LP F1=0")

    t_total = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as fout:
        for ds, sig in todo:
            t0 = time.time()
            try:
                r = run_signal(ds, sig)
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                print(f"    saved ({time.time()-t0:.1f}s)")
            except Exception as e:
                import traceback
                print(f"  ERROR {ds}/{sig}: {e}")
                traceback.print_exc()

    # Print summary
    results = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try: results.append(json.loads(line))
            except: pass

    methods = ["knn_sum", "resid_sum", "resid_topk10",
               "ml_L8_11_resid_sum", "ml_L8_11_resid_topk10"]
    key_map = {m: f"{m}_f1" for m in methods}

    print(f"\n{'='*75}")
    print("RESULTS SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Method':30} {'NAB':>8} {'SMAP':>8} {'MSL':>8} {'ALL':>8}")
    print(f"  {'-'*60}")
    for m in methods:
        k = key_map[m]
        nab  = [r[k] for r in results if r["ds"]=="NAB" and r.get(k) is not None]
        smap = [r[k] for r in results if r["ds"]=="SMAP" and r.get(k) is not None]
        msl  = [r[k] for r in results if r["ds"]=="MSL" and r.get(k) is not None]
        all_ = nab + smap + msl
        print(f"  {m:30} {sum(nab)/len(nab) if nab else 0:>8.4f} "
              f"{sum(smap)/len(smap) if smap else 0:>8.4f} "
              f"{sum(msl)/len(msl) if msl else 0:>8.4f} "
              f"{sum(all_)/len(all_) if all_ else 0:>8.4f}")

    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Results: {jsonl_path}")

if __name__ == "__main__":
    main()
