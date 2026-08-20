"""
Alternative Scoring Methods for MAE Patch Features in Zero-Shot TSAD
====================================================================
Compares 6 scoring methods on MAE L10 patch features:
  1. Cosine distance (baseline, reuse cached scores)
  2. k-NN distance (k=3,5,10)
  3. Mahalanobis distance with PCA (d=32,64,128)
  4. LOF (k=5,10,20)
  5. Isolation Forest
  6. Reconstruction Feature Discrepancy (NAB only, 10 windows/signal)

Fixes vs spec:
- SMAP P-2 removed (file does not exist)
- Method 6 limited to NAB, 10 windows/signal (CPU constraint)
- Raw features re-extracted (cache has scores only, not features)
- MSL channels match previous experiment (C-2,M-2,M-3,M-7,P-14)
"""

import os, sys, time, ast, warnings, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from preprocessing.preprocess import draw_image, preprocess_time_series

try:
    from transformers import ViTMAEModel, ViTMAEForPreTraining
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, average_precision_score
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import gaussian_kde
except ImportError as e:
    print(f"[ERROR] {e}\npip install transformers scikit-learn matplotlib seaborn")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR    = os.path.join(SCRIPT_DIR, "..", "data")
PREV_CACHE  = os.path.join(SCRIPT_DIR, "..", "results", "feature_analysis", "cache")
OUT_DIR     = os.path.join(SCRIPT_DIR, "..", "results", "scoring_comparison")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
FEAT_CACHE  = os.path.join(OUT_DIR, "feat_cache")
for d in [OUT_DIR, FIG_DIR, FEAT_CACHE]: os.makedirs(d, exist_ok=True)

ANOM_CSV    = os.path.join(DATA_DIR, "anomalies.csv")
WINDOW_SIZE = 224
STEP_SIZE   = 56
IMAGE_SIZE  = (224, 224)
DPI         = 100
BATCH_SIZE  = 16
TARGET_LAYER = 10          # MAE L10
N_PATCHES   = 196          # CLIP/MAE: 14×14
PATCH_PX    = 16
GRID_COLS   = 14

INET_MEAN = [0.485, 0.456, 0.406]
INET_STD  = [0.229, 0.224, 0.225]
import torchvision.transforms as T
_inet_norm = T.Normalize(mean=INET_MEAN, std=INET_STD)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    warnings.warn("[WARN] Running on CPU — Method 6 will be very slow.")
print(f"[INFO] Device: {DEVICE}")

# Dataset signal lists ────────────────────────────────────────────────────────
SMAP_CHANNELS = ["D-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7",
                 "F-1","F-2","F-3","P-1","T-1"]          # P-2 removed
MSL_CHANNELS  = ["P-11","P-14","C-2","M-2","M-3","M-7"]  # match prev experiment

# ── Model loading ─────────────────────────────────────────────────────────────
def load_mae_encoder():
    print("[INFO] Loading MAE ViT-B/16 encoder ...")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model.config.mask_ratio = 0.0
    model = model.to(DEVICE).eval()
    return model

def load_mae_full():
    """Full MAE (encoder+decoder) for Method 6."""
    print("[INFO] Loading full MAE for reconstruction ...")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    model = model.to(DEVICE).eval()
    return model

# ── Image generation ─────────────────────────────────────────────────────────
def series_to_windows(values):
    proc = preprocess_time_series(values)
    T = len(proc)
    tp = np.arange(T, dtype=float)
    imgs, starts = [], []
    for start in range(0, T - WINDOW_SIZE + 1, STEP_SIZE):
        img = draw_image(
            series_id=f"sc_{start}",
            save_path=os.path.join(FEAT_CACHE, "_tmp"),
            time_series=proc[start:start+WINDOW_SIZE],
            time_points=tp[start:start+WINDOW_SIZE],
            override=True, save_image=False,
            image_size=IMAGE_SIZE, dpi=DPI,
            plot_params=('-',1,'*',0.1,'black',(0.,1.)),
        )
        if img is not None:
            imgs.append(img)
            starts.append(start)
    return np.stack(imgs) if imgs else np.zeros((0,3,*IMAGE_SIZE)), starts

# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_mae_l10(windows, mae_model):
    """Returns (N_windows, 196, 768) MAE L10 features."""
    win_t = torch.from_numpy(windows).float()
    all_feats = []
    for bs in range(0, len(win_t), BATCH_SIZE):
        batch = win_t[bs:bs+BATCH_SIZE].to(DEVICE)
        norm  = _inet_norm(batch)
        noise = torch.zeros(len(batch), N_PATCHES, device=DEVICE)
        with torch.no_grad():
            out = mae_model(pixel_values=norm, noise=noise,
                            output_hidden_states=True, return_dict=True)
        h = out.hidden_states[TARGET_LAYER]   # (B, 197, 768)
        all_feats.append(h[:, 1:, :].cpu().numpy())   # strip CLS
    return np.concatenate(all_feats, axis=0)

def feat_cache_path(signal, dataset):
    safe = signal.replace("/","_")
    return os.path.join(FEAT_CACHE, f"MAE_L{TARGET_LAYER}_{dataset}_{safe}.npy")

# ── Ground truth labels ───────────────────────────────────────────────────────
def load_anom_intervals(signal, anom_df):
    row = anom_df[anom_df["signal"] == signal]
    if row.empty: return []
    raw = row.iloc[0]["events"]
    if pd.isna(raw) or str(raw).strip() in ("","[]"): return []
    try:    return [(float(a),float(b)) for a,b in ast.literal_eval(str(raw))]
    except: return []

def make_patch_labels(w_start, timestamps, intervals):
    labels = np.zeros(N_PATCHES, dtype=np.int32)
    for i in range(N_PATCHES):
        col = i % GRID_COLS
        ts  = w_start + col * PATCH_PX
        te  = w_start + (col+1) * PATCH_PX
        idx_s = min(ts, len(timestamps)-1)
        idx_e = min(te, len(timestamps))
        if idx_s >= idx_e: continue
        t0, t1 = timestamps[idx_s], timestamps[idx_e-1]
        for (a,b) in intervals:
            if t0 <= b and t1 >= a:
                labels[i] = 1; break
    return labels

# ── Scoring methods ────────────────────────────────────────────────────────────
def score_cosine(F):
    """F: (N,D). Returns (N,) scores."""
    ref = np.median(F, axis=0)
    ref_n = ref / (np.linalg.norm(ref)+1e-8)
    norms = np.linalg.norm(F, axis=1, keepdims=True)+1e-8
    return (1.0 - (F/norms) @ ref_n).clip(0,2)

def score_knn(F, k):
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(F, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return np.mean(np.take_along_axis(D, idx, axis=1), axis=1)

def score_mahalanobis(F, d):
    pca = PCA(n_components=min(d, F.shape[0]-1, F.shape[1]))
    Fp  = pca.fit_transform(F)           # (N, d)
    mu  = Fp.mean(axis=0)
    cov = np.cov(Fp.T) + 1e-4*np.eye(Fp.shape[1])
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    diff = Fp - mu
    scores = np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))
    if np.any(np.isnan(scores)):
        warnings.warn("Mahalanobis NaN — falling back to diagonal cov")
        cov_inv = np.diag(1.0/(np.diag(cov)+1e-4))
        scores  = np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))
    return scores

def score_lof(F, k):
    clf = LocalOutlierFactor(n_neighbors=min(k, len(F)-1), novelty=False)
    clf.fit_predict(F)
    return -clf.negative_outlier_factor_   # higher = more anomalous

def score_iforest(F):
    iso = IsolationForest(contamination='auto', random_state=SEED)
    iso.fit(F)
    return -iso.score_samples(F)            # higher = more anomalous

# ── Method 6: Reconstruction Feature Discrepancy ──────────────────────────────
def score_recon_discrepancy(windows_subset, starts_subset, mae_full, mae_enc, K=2):
    """
    K random masking passes per window (mask_ratio=0.75).
    For masked patches: discrepancy = ||f_orig - f_recon||_2
    Returns (N_windows, 196) score array.
    """
    N = len(windows_subset)
    all_scores = np.zeros((N, N_PATCHES), dtype=np.float32)
    counts     = np.zeros((N, N_PATCHES), dtype=np.float32)

    win_t = torch.from_numpy(windows_subset).float()

    for wi in range(N):
        img    = win_t[wi:wi+1].to(DEVICE)
        norm   = _inet_norm(img)

        # original unmasked features
        noise0 = torch.zeros(1, N_PATCHES, device=DEVICE)
        with torch.no_grad():
            out0   = mae_enc(pixel_values=norm, noise=noise0,
                             output_hidden_states=True, return_dict=True)
        f_orig = out0.hidden_states[TARGET_LAYER][0, 1:].cpu().numpy()  # (196,768)

        # K masked passes
        for _ in range(K):
            noise_k = torch.rand(1, N_PATCHES, device=DEVICE)
            with torch.no_grad():
                out_full = mae_full(pixel_values=norm, noise=noise_k,
                                    return_dict=True)
            # reconstruct full image from decoder output
            # out_full.logits: (1, N_patches, patch_px^2 * 3)
            logits    = out_full.logits          # (1, 196, 768)
            mask_ids  = out_full.mask.squeeze(0).bool().cpu()  # (196,) True=masked

            # build reconstructed image tensor
            # patch pixel layout: (patch_h*patch_w*3)
            pp  = PATCH_PX
            rec_img = img.clone().cpu()          # (1,3,224,224)
            logits_np = logits[0].detach().cpu().numpy()  # (196, pp*pp*3)

            for pi in range(N_PATCHES):
                if not mask_ids[pi]: continue
                row, col = pi // GRID_COLS, pi % GRID_COLS
                patch_pix = logits_np[pi].reshape(pp, pp, 3)  # (H,W,C)
                # undo imagenet normalization for reconstruction
                patch_t = torch.from_numpy(patch_pix).permute(2,0,1).float()
                for c,(m,s) in enumerate(zip(INET_MEAN, INET_STD)):
                    patch_t[c] = patch_t[c] * s + m
                patch_t = patch_t.clamp(0,1)
                r0,r1 = row*pp, (row+1)*pp
                c0,c1 = col*pp, (col+1)*pp
                rec_img[0, :, r0:r1, c0:c1] = patch_t

            # re-encode reconstructed image
            rec_norm  = _inet_norm(rec_img.to(DEVICE))
            noise_re  = torch.zeros(1, N_PATCHES, device=DEVICE)
            with torch.no_grad():
                out_re = mae_enc(pixel_values=rec_norm, noise=noise_re,
                                 output_hidden_states=True, return_dict=True)
            f_recon = out_re.hidden_states[TARGET_LAYER][0, 1:].cpu().numpy()

            # accumulate discrepancy for masked patches only
            for pi in range(N_PATCHES):
                if mask_ids[pi]:
                    disc = np.linalg.norm(f_orig[pi] - f_recon[pi])
                    all_scores[wi, pi] += disc
                    counts[wi, pi]     += 1

    safe_counts = np.where(counts == 0, 1, counts)
    return all_scores / safe_counts

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(scores, labels):
    n_a = labels.sum(); n_n = (labels==0).sum()
    if n_a == 0 or n_n == 0:
        return dict(auroc=float('nan'), ap=float('nan'),
                    cohens_d=float('nan'), f1_max=float('nan'))
    auroc = roc_auc_score(labels, scores)
    ap    = average_precision_score(labels, scores)
    sa, sn = scores[labels==1], scores[labels==0]
    ps    = np.sqrt((sa.std()**2 + sn.std()**2)/2 + 1e-12)
    cd    = (sa.mean() - sn.mean()) / ps
    # F1_max
    best_f1 = 0.0
    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        tp=((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum()
        fn=((pred==0)&(labels==1)).sum()
        pr=tp/(tp+fp+1e-12); re=tp/(tp+fn+1e-12)
        f1=2*pr*re/(pr+re+1e-12)
        if f1 > best_f1: best_f1 = f1
    return dict(auroc=auroc, ap=ap, cohens_d=cd, f1_max=best_f1)

def aggregate_patches_to_series(patch_scores_list, window_starts, T, agg="max"):
    """patch_scores_list: list of (196,) arrays. Returns (T,) series scores."""
    series = np.zeros(T); counts = np.zeros(T)
    for wi, (ps, ws) in enumerate(zip(patch_scores_list, window_starts)):
        for pi in range(N_PATCHES):
            col = pi % GRID_COLS
            t0 = ws + col * PATCH_PX
            t1 = ws + (col+1) * PATCH_PX
            for t in range(t0, min(t1, T)):
                if agg == "max":
                    series[t] = max(series[t], ps[pi])
                else:
                    series[t] += ps[pi]; counts[t] += 1
    if agg == "mean":
        mask = counts > 0; series[mask] /= counts[mask]
    return series

# ── Dataset builder ───────────────────────────────────────────────────────────
def build_entries(anom_df):
    entries = []
    for ch in SMAP_CHANNELS:
        p = os.path.join(DATA_DIR, "SMAP", f"{ch}.csv")
        if not os.path.exists(p): print(f"[WARN] SMAP {ch} not found"); continue
        df = pd.read_csv(p)
        entries.append(dict(dataset="SMAP", signal=ch,
                            timestamps=df["timestamp"].values.astype(float),
                            values=df["value"].values.astype(float),
                            intervals=load_anom_intervals(ch, anom_df)))
    for ch in MSL_CHANNELS:
        p = os.path.join(DATA_DIR, "MSL", f"{ch}.csv")
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        entries.append(dict(dataset="MSL", signal=ch,
                            timestamps=df["timestamp"].values.astype(float),
                            values=df["value"].values.astype(float),
                            intervals=load_anom_intervals(ch, anom_df)))
    nab_dir = os.path.join(DATA_DIR, "realAWSCloudwatch")
    for fname in sorted(os.listdir(nab_dir)):
        if not fname.endswith(".csv"): continue
        sig = fname.replace(".csv","")
        ivs = load_anom_intervals(sig, anom_df)
        if not ivs: continue
        df  = pd.read_csv(os.path.join(nab_dir, fname))
        entries.append(dict(dataset="NAB", signal=sig,
                            timestamps=df["timestamp"].values.astype(float),
                            values=df["value"].values.astype(float),
                            intervals=ivs))
    return entries

# ── Main ──────────────────────────────────────────────────────────────────────
def run(datasets=None):
    if datasets is None:
        datasets = {"SMAP", "MSL", "NAB"}
    anom_df = pd.read_csv(ANOM_CSV)
    entries = [e for e in build_entries(anom_df) if e["dataset"] in datasets]
    print(f"[INFO] {len(entries)} signals to process")

    mae_enc = load_mae_encoder()
    mae_full = None   # loaded lazily for Method 6

    rows = []
    t_global = time.time()

    METHODS_12_TO_5 = [
        ("cosine",          {}),
        ("knn_k3",          {"k":3}),
        ("knn_k5",          {"k":5}),
        ("knn_k10",         {"k":10}),
        ("maha_d32",        {"d":32}),
        ("maha_d64",        {"d":64}),
        ("maha_d128",       {"d":128}),
        ("lof_k5",          {"k":5}),
        ("lof_k10",         {"k":10}),
        ("lof_k20",         {"k":20}),
        ("iforest",         {}),
    ]

    for ei, entry in enumerate(entries):
        ds  = entry["dataset"]
        sig = entry["signal"]
        ts  = entry["timestamps"]
        vals= entry["values"]
        ivs = entry["intervals"]
        T   = len(vals)

        print(f"\n[{ei+1}/{len(entries)}] {ds}/{sig}  T={T}")

        # ── Windows ──────────────────────────────────────────────────────────
        windows, w_starts = series_to_windows(vals)
        N_win = len(windows)
        if N_win == 0:
            print("  [WARN] no windows, skip"); continue
        print(f"  Windows: {N_win}")

        # ── Ground truth labels per window ───────────────────────────────────
        all_labels = []
        for ws in w_starts:
            all_labels.append(make_patch_labels(ws, ts, ivs))

        anom_frac = np.concatenate(all_labels).mean()
        print(f"  Anomaly patch fraction: {anom_frac:.4f}")
        if not (0.001 < anom_frac < 0.50):
            print(f"  [WARN] unusual anomaly fraction — check labels")

        # ── MAE L10 features (extract or load from cache) ────────────────────
        cp = feat_cache_path(sig, ds)
        if os.path.exists(cp):
            feats = np.load(cp)   # (N_win, 196, 768)
            print(f"  Features loaded from cache")
        else:
            print(f"  Extracting MAE L{TARGET_LAYER} features ...")
            t0 = time.time()
            feats = extract_mae_l10(windows, mae_enc)
            np.save(cp, feats)
            print(f"  Extracted in {time.time()-t0:.1f}s, saved to cache")

        assert feats.shape == (N_win, N_PATCHES, 768), \
            f"[FAIL] feature shape {feats.shape} != ({N_win},{N_PATCHES},768)"

        # ── Sanity: cosine direction check ────────────────────────────────────
        cos_scores_all = np.array([score_cosine(feats[i]) for i in range(N_win)])
        cos_flat  = cos_scores_all.flatten()
        lab_flat  = np.concatenate(all_labels)
        if lab_flat.sum() > 0:
            mean_a = cos_flat[lab_flat==1].mean()
            mean_n = cos_flat[lab_flat==0].mean()
            direction = "NORMAL (anom>norm)" if mean_a > mean_n else "INVERTED (anom<norm)"
            print(f"  Cosine direction: {direction}  "
                  f"(mean_anom={mean_a:.4f}, mean_norm={mean_n:.4f})")

        # ── Methods 1-5 ───────────────────────────────────────────────────────
        for method_name, params in METHODS_12_TO_5:
            t0 = time.time()
            win_scores = []

            for wi in range(N_win):
                F = feats[wi]   # (196, 768)

                if method_name == "cosine":
                    sc = score_cosine(F)
                elif method_name.startswith("knn"):
                    sc = score_knn(F, params["k"])
                elif method_name.startswith("maha"):
                    sc = score_mahalanobis(F, params["d"])
                elif method_name.startswith("lof"):
                    sc = score_lof(F, params["k"])
                elif method_name == "iforest":
                    sc = score_iforest(F)
                else:
                    sc = np.zeros(N_PATCHES)

                win_scores.append(sc)

            runtime = time.time() - t0

            for agg in ["max", "mean"]:
                patch_flat  = np.concatenate(win_scores)
                label_flat  = np.concatenate(all_labels)

                # patch-level metrics
                m_patch = compute_metrics(patch_flat, label_flat)

                # series-level aggregation
                series_sc = aggregate_patches_to_series(win_scores, w_starts, T, agg=agg)
                series_lb = np.zeros(T, dtype=np.int32)
                for (a,b) in ivs:
                    mask = (ts >= a) & (ts <= b)
                    series_lb[mask] = 1
                m_series = compute_metrics(series_sc, series_lb)

                # also try inverted scores for SMAP
                inv_patch  = patch_flat.max() - patch_flat
                inv_series = series_sc.max() - series_sc
                m_inv_p = compute_metrics(inv_patch,  label_flat)
                m_inv_s = compute_metrics(inv_series, series_lb)

                best_auroc_p = max(m_patch["auroc"],  m_inv_p["auroc"])
                best_auroc_s = max(m_series["auroc"], m_inv_s["auroc"])
                direction = "normal" if m_patch["auroc"] >= m_inv_p["auroc"] else "inverted"

                rows.append(dict(
                    dataset=ds, signal=sig, method=method_name,
                    aggregation=agg, runtime_s=round(runtime,2),
                    # patch level
                    auroc_patch=m_patch["auroc"], ap_patch=m_patch["ap"],
                    cohens_d=m_patch["cohens_d"], f1_max_patch=m_patch["f1_max"],
                    # series level
                    auroc_series=m_series["auroc"], f1_max_series=m_series["f1_max"],
                    # best of normal/inverted
                    best_auroc_patch=best_auroc_p,
                    best_auroc_series=best_auroc_s,
                    direction=direction,
                ))

            print(f"  {method_name:20s}  "
                  f"AUROC={m_patch['auroc']:.3f}  "
                  f"Cohen_d={m_patch['cohens_d']:+.3f}  "
                  f"F1={m_series['f1_max']:.3f}  "
                  f"({runtime:.1f}s)")

        elapsed = time.time() - t_global
        eta     = elapsed/(ei+1) * (len(entries)-ei-1)
        print(f"  [Progress] {elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m")

    # ── Method 6: Reconstruction Discrepancy (NAB only) ───────────────────────
    print("\n" + "="*60)
    print("[Phase 2] Method 6: Reconstruction Feature Discrepancy (NAB only)")
    mae_full = load_mae_full()
    MAX_WIN_M6 = 10

    for entry in entries:
        if entry["dataset"] != "NAB": continue
        ds  = entry["dataset"]
        sig = entry["signal"]
        ts  = entry["timestamps"]
        ivs = entry["intervals"]
        T   = len(ts)

        cp = feat_cache_path(sig, ds)
        if not os.path.exists(cp): continue
        feats = np.load(cp)
        windows, w_starts = series_to_windows(entry["values"])
        if len(windows) == 0: continue

        # sample up to MAX_WIN_M6 windows
        rng   = np.random.default_rng(SEED)
        n_sel = min(MAX_WIN_M6, len(windows))
        sel   = rng.choice(len(windows), n_sel, replace=False)
        sel   = np.sort(sel)
        win_sub   = windows[sel]
        start_sub = [w_starts[i] for i in sel]

        all_labels_sub = [make_patch_labels(w_starts[i], ts, ivs) for i in sel]
        label_flat_sub = np.concatenate(all_labels_sub)

        if label_flat_sub.sum() == 0:
            print(f"  {sig}: no anomaly patches in subset, skip M6")
            continue

        print(f"  {sig}: running M6 on {n_sel} windows ...")
        t0 = time.time()
        sc_mat = score_recon_discrepancy(win_sub, start_sub, mae_full, mae_enc, K=2)
        runtime = time.time() - t0

        sc_flat = sc_mat.flatten()
        m = compute_metrics(sc_flat, label_flat_sub)
        inv_sc  = sc_flat.max() - sc_flat
        m_inv   = compute_metrics(inv_sc, label_flat_sub)
        best_auroc = max(m["auroc"], m_inv["auroc"])
        direction  = "normal" if m["auroc"] >= m_inv["auroc"] else "inverted"

        for agg in ["max","mean"]:
            series_sc = aggregate_patches_to_series(
                [sc_mat[i] for i in range(n_sel)], start_sub, T, agg=agg)
            series_lb = np.zeros(T, dtype=np.int32)
            for (a,b) in ivs:
                mask=(ts>=a)&(ts<=b); series_lb[mask]=1
            m_s = compute_metrics(series_sc, series_lb)

            rows.append(dict(
                dataset=ds, signal=sig, method="recon_discrepancy",
                aggregation=agg, runtime_s=round(runtime,2),
                auroc_patch=m["auroc"], ap_patch=m["ap"],
                cohens_d=m["cohens_d"], f1_max_patch=m["f1_max"],
                auroc_series=m_s["auroc"], f1_max_series=m_s["f1_max"],
                best_auroc_patch=best_auroc, best_auroc_series=best_auroc,
                direction=direction,
            ))

        print(f"  recon_discrepancy  AUROC={m['auroc']:.3f}  "
              f"Cohen_d={m['cohens_d']:+.3f}  ({runtime:.1f}s)")

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "full_results.csv"), index=False)
    print(f"\n[INFO] Saved full_results.csv ({len(df)} rows)")

    # summary by method (max agg, avg across signals)
    for agg in ["max","mean"]:
        sub = df[df["aggregation"]==agg]
        smry = sub.groupby(["dataset","method"]).agg(
            auroc_patch=("auroc_patch","mean"),
            auroc_series=("auroc_series","mean"),
            best_auroc_patch=("best_auroc_patch","mean"),
            cohens_d=("cohens_d","mean"),
            f1_max_series=("f1_max_series","mean"),
            runtime_s=("runtime_s","mean"),
        ).reset_index()
        smry.to_csv(os.path.join(OUT_DIR, f"summary_{agg}_agg.csv"), index=False)
    print("[INFO] Saved summary CSVs")

    # ── Print comparison tables ───────────────────────────────────────────────
    print_comparison_tables(df)
    generate_figures(df)

    total = time.time() - t_global
    print(f"\n[DONE] Total: {total/60:.1f}m  Results in: {OUT_DIR}")

# ── Comparison tables ─────────────────────────────────────────────────────────
def print_comparison_tables(df):
    max_df = df[df["aggregation"]=="max"]
    for ds in ["NAB","MSL","SMAP"]:
        sub = max_df[max_df["dataset"]==ds]
        if sub.empty: continue
        # best hyperparams per method (highest auroc_patch)
        best = sub.loc[sub.groupby("method")["auroc_patch"].idxmax()]
        cosine_auroc = best[best["method"]=="cosine"]["auroc_patch"].values
        baseline = cosine_auroc[0] if len(cosine_auroc) else float('nan')

        print(f"\n{'='*70}")
        print(f"=== SCORING METHOD COMPARISON — {ds} (max aggregation) ===")
        print(f"{'Method':<22} {'AUROC_patch':>11} {'F1_max':>7} "
              f"{'Cohen_d':>8} {'Dir':>8} {'Runtime':>8}")
        print("-"*70)
        for _, r in best.sort_values("auroc_patch", ascending=False).iterrows():
            delta = r["auroc_patch"] - baseline
            print(f"{r['method']:<22} {r['auroc_patch']:>11.3f} "
                  f"{r['f1_max_series']:>7.3f} {r['cohens_d']:>+8.3f} "
                  f"{r['direction']:>8} {r['runtime_s']:>7.1f}s "
                  f"({'Δ'+f'{delta:+.3f}' if not np.isnan(delta) else ''})")

    # overall best
    max_df2 = df[df["aggregation"]=="max"]
    best_overall = max_df2.loc[max_df2.groupby("dataset")["auroc_patch"].idxmax()]
    print("\n=== OVERALL BEST PER DATASET ===")
    for _, r in best_overall.iterrows():
        print(f"{r['dataset']}: {r['method']}  AUROC={r['auroc_patch']:.3f}  "
              f"Cohen_d={r['cohens_d']:+.3f}  direction={r['direction']}")

    # mean vs max comparison
    print("\n=== AGGREGATION: mean vs max (avg AUROC_patch across all signals) ===")
    for ds in ["NAB","MSL","SMAP"]:
        for method in ["cosine","knn_k5","maha_d64","lof_k10","iforest"]:
            sub = df[(df["dataset"]==ds)&(df["method"]==method)]
            if sub.empty: continue
            mx = sub[sub["aggregation"]=="max"]["auroc_patch"].mean()
            mn = sub[sub["aggregation"]=="mean"]["auroc_patch"].mean()
            print(f"  {ds}/{method}: max={mx:.3f}  mean={mn:.3f}  "
                  f"winner={'max' if mx>=mn else 'mean'}")

# ── Figures ───────────────────────────────────────────────────────────────────
def generate_figures(df):
    max_df = df[df["aggregation"]=="max"]

    # 1. AUROC heatmap per dataset
    for ds in ["NAB","MSL","SMAP"]:
        sub = max_df[max_df["dataset"]==ds]
        if sub.empty: continue
        pivot = sub.groupby("method")["auroc_patch"].mean().reset_index()
        pivot = pivot.sort_values("auroc_patch", ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(3, len(pivot)*0.4)))
        bars = ax.barh(pivot["method"], pivot["auroc_patch"],
                       color=plt.cm.RdYlGn(pivot["auroc_patch"]/1.0))
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='random')
        ax.set_xlabel("AUROC (patch, max agg)")
        ax.set_title(f"Scoring Method Comparison — {ds}")
        ax.set_xlim(0.3, 1.0)
        for bar, v in zip(bars, pivot["auroc_patch"]):
            ax.text(max(v+0.005, 0.31), bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}", va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"auroc_comparison_{ds}.png"), dpi=150)
        plt.close()
        print(f"[FIG] auroc_comparison_{ds}.png")

    # 2. Cohen's d bar chart (shows direction)
    for ds in ["NAB","MSL","SMAP"]:
        sub = max_df[max_df["dataset"]==ds]
        if sub.empty: continue
        pivot = sub.groupby("method")["cohens_d"].mean().reset_index()
        pivot = pivot.sort_values("cohens_d", ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(3, len(pivot)*0.4)))
        colors = ["green" if v >= 0 else "red" for v in pivot["cohens_d"]]
        ax.barh(pivot["method"], pivot["cohens_d"], color=colors)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel("Cohen's d  (positive = correct direction)")
        ax.set_title(f"Score Direction — {ds}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"cohens_d_{ds}.png"), dpi=150)
        plt.close()
        print(f"[FIG] cohens_d_{ds}.png")

    # 3. F1_max comparison
    for ds in ["NAB","MSL","SMAP"]:
        sub = max_df[max_df["dataset"]==ds]
        if sub.empty: continue
        pivot = sub.groupby("method")["f1_max_series"].mean().reset_index()
        pivot = pivot.sort_values("f1_max_series", ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(3, len(pivot)*0.4)))
        ax.barh(pivot["method"], pivot["f1_max_series"])
        ax.set_xlabel("F1_max (series level)")
        ax.set_title(f"F1_max Comparison — {ds}")
        ax.set_xlim(0, 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"f1_comparison_{ds}.png"), dpi=150)
        plt.close()
        print(f"[FIG] f1_comparison_{ds}.png")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+",
                    choices=["SMAP", "MSL", "NAB"], default=["SMAP", "MSL", "NAB"],
                    help="Datasets to run (default: all). Example: --dataset NAB")
    args = ap.parse_args()
    run(datasets=set(args.dataset))
