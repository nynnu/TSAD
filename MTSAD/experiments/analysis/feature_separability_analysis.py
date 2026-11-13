"""
CLIP vs MAE Layer-wise Feature Separability for TSAD
=====================================================
Diagnostic analysis: do MAE intermediate layers produce MORE separable
normal vs anomaly patch features than CLIP final layer on TS line plots?

Fixes applied vs original spec:
- P-2 removed from SMAP list (file doesn't exist; P-2 is an MSL channel)
- NAB: 17 files found (not 16); skip signals with empty anomaly list
- MAE masking disabled via model.config.mask_ratio = 0.0 (not noise=None)
- Per-encoder normalization: CLIP uses CLIP stats, MAE/DINO use ImageNet stats
"""

import os
import sys
import time
import json
import random
import warnings
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── add project src to path ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

# ── imports from existing codebase ──────────────────────────────────────────
from preprocessing.preprocess import draw_image, preprocess_time_series

# ── third-party ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.manifold import TSNE
    from scipy.stats import gaussian_kde
    from transformers import (
        CLIPModel, CLIPProcessor,
        ViTMAEModel,
        AutoModel, AutoImageProcessor,
    )
    import torchvision.transforms as T
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install transformers torch torchvision matplotlib seaborn scikit-learn tqdm Pillow")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results", "feature_analysis")
CACHE_DIR  = os.path.join(RESULTS_DIR, "cache")
ANOM_CSV   = os.path.join(DATA_DIR, "anomalies.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

WINDOW_SIZE = 224
STEP_SIZE   = 56          # stride = window_size / 4
IMAGE_SIZE  = (224, 224)
DPI         = 100
BATCH_SIZE  = 32
LAYERS      = [1, 4, 7, 10, 12]

# SMAP channels (P-2 removed — file doesn't exist in data/SMAP/)
SMAP_CHANNELS = ["D-1", "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7",
                 "F-1", "F-2", "F-3", "P-1", "T-1"]
MSL_PRIMARY   = "P-11"

# Per-encoder normalisation constants
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
INET_MEAN = [0.485,      0.456,      0.406]
INET_STD  = [0.229,      0.224,      0.225]

_clip_norm = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
_inet_norm = T.Normalize(mean=INET_MEAN, std=INET_STD)

def normalize_for_encoder(tensor: torch.Tensor, encoder: str) -> torch.Tensor:
    """Apply the correct normalisation for each encoder."""
    if encoder == "CLIP":
        return _clip_norm(tensor)
    return _inet_norm(tensor)   # MAE and DINOv2

# ════════════════════════════════════════════════════════════════════════════
# DEVICE
# ════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    warnings.warn("[WARN] GPU not available — running on CPU. Expect slow runtime.")
print(f"[INFO] Using device: {DEVICE}")

# ════════════════════════════════════════════════════════════════════════════
# ENCODER LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_clip():
    print("[INFO] Loading CLIP ViT-B/16 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model = model.to(DEVICE).eval()
    return model

def load_mae():
    print("[INFO] Loading MAE ViT-B/16 ...")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    # mask_ratio=0 keeps all patches, but internally patches are still reordered
    # by a random noise vector — so we must also fix the noise to zeros to get
    # deterministic patch ordering: argsort(zeros) = [0, 1, ..., 195] always.
    model.config.mask_ratio = 0.0
    model = model.to(DEVICE).eval()

    # Sanity check: deterministic outputs when noise is fixed
    dummy      = torch.zeros(1, 3, 224, 224).to(DEVICE)
    fixed_noise = torch.zeros(1, 196).to(DEVICE)   # forces canonical patch order
    with torch.no_grad():
        o1 = model(dummy, noise=fixed_noise, output_hidden_states=True).last_hidden_state
        o2 = model(dummy, noise=fixed_noise, output_hidden_states=True).last_hidden_state
    assert torch.allclose(o1, o2, atol=1e-5), \
        "[FAIL] MAE masking not disabled — outputs differ across two identical runs"
    print("[OK] MAE masking disabled (deterministic output verified)")
    return model

def load_dino():
    # facebook/dinov2-vitb14 on HuggingFace Hub now requires authentication.
    # Use timm (already in the project) which has the same weights and works
    # offline after first download.
    print("[INFO] Loading DINOv2 ViT-B/14 via timm ...")
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required: pip install timm")
    # img_size=224: patch_size=14 → 16×16=256 patches (default is 518px)
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224)
    model = model.to(DEVICE).eval()

    # Attach forward hooks on each transformer block so we can read out
    # intermediate hidden states at the layers we care about (1,4,7,10,12).
    # timm stores blocks in model.blocks (list of 12 Block modules).
    model._dino_hook_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output shape: (B, N_tokens, D)  — CLS is index 0
            model._dino_hook_outputs[layer_idx] = output.detach()
        return hook

    model._dino_hooks = []
    for l in LAYERS:
        h = model.blocks[l - 1].register_forward_hook(make_hook(l))
        model._dino_hooks.append(h)

    return model

# ════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════
def extract_hidden_states(images: torch.Tensor, encoder_name: str, models: dict) -> dict:
    """
    Returns dict: {layer_idx: np.ndarray shape (B, N_patches, 768)}
    layers for CLIP/MAE: 196 patches (patch_size=16, 14×14 grid)
    layers for DINO   : 256 patches (patch_size=14, 16×16 grid)
    """
    images = images.to(DEVICE)
    results = {}

    with torch.no_grad():
        if encoder_name == "CLIP":
            norm_imgs = normalize_for_encoder(images, "CLIP")
            out = models["CLIP"].vision_model(
                pixel_values=norm_imgs,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states: tuple of (B, 197, 768), index 0=embed, 1..12=layers
            for l in LAYERS:
                h = out.hidden_states[l]   # (B, 197, 768)
                results[l] = h[:, 1:, :].cpu().numpy()  # strip CLS → (B, 196, 768)

        elif encoder_name == "MAE":
            norm_imgs = normalize_for_encoder(images, "MAE")
            # Pass fixed noise so patch ordering is deterministic (all-zero noise
            # → argsort gives [0,1,...,195] every run, no random reordering)
            fixed_noise = torch.zeros(images.shape[0], 196, device=DEVICE)
            out = models["MAE"](
                pixel_values=norm_imgs,
                noise=fixed_noise,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states: embed + 12 transformer layers → indices 0..12
            for l in LAYERS:
                h = out.hidden_states[l]   # (B, 197, 768)
                results[l] = h[:, 1:, :].cpu().numpy()  # strip CLS → (B, 196, 768)

        elif encoder_name == "DINOv2":
            norm_imgs = normalize_for_encoder(images, "DINOv2")
            dino = models["DINOv2"]
            dino._dino_hook_outputs.clear()
            # timm forward: returns (B, D) class token by default;
            # hooks capture (B, N_tokens, D) after each block.
            dino(norm_imgs)
            for l in LAYERS:
                h = dino._dino_hook_outputs[l]  # (B, 257, 768): CLS + 256 patches
                results[l] = h[:, 1:, :].cpu().numpy()  # strip CLS → (B, 256, 768)

    return results

def verify_feature_shapes(feats: dict, encoder_name: str):
    expected = 196 if encoder_name in ("CLIP", "MAE") else 256
    for l, f in feats.items():
        assert f.shape[1:] == (expected, 768), \
            f"[FAIL] {encoder_name} layer {l}: expected ({expected}, 768), got {f.shape[1:]}"

# ════════════════════════════════════════════════════════════════════════════
# IMAGE GENERATION (wraps existing draw_image)
# ════════════════════════════════════════════════════════════════════════════
def series_to_windows(values: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D time series to windowed images using the SAME pipeline as
    the existing ViT4TS code (draw_image + preprocess_time_series).

    Returns np.ndarray of shape (N_windows, 3, H, W), values in [0, 1].
    """
    values_proc = preprocess_time_series(values)
    T = len(values_proc)
    time_points = np.arange(T, dtype=float)
    y_scale = (0.0, 1.0)   # same as ViT4TS with standardised series
    plot_params = ('-', 1, '*', 0.1, 'black', y_scale)

    windows = []
    for start in range(0, T - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        win_vals  = values_proc[start:end]
        win_times = time_points[start:end]
        img = draw_image(
            series_id=f"tmp_{start}",
            save_path=os.path.join(CACHE_DIR, "_tmp"),
            time_series=win_vals,
            time_points=win_times,
            override=True,
            save_image=False,
            image_size=IMAGE_SIZE,
            dpi=DPI,
            plot_params=plot_params,
        )
        if img is not None:
            windows.append(img)

    # Sanity check: visible line in middle rows
    if len(windows) > 0:
        sample = windows[0]           # (3, H, W), range [0,1]
        mid_rows = sample[:, 80:144, :]
        assert mid_rows.min() < 0.95, \
            "[FAIL] Image sanity check: middle rows appear all-white (no visible line)"

    return np.stack(windows) if windows else np.zeros((0, 3, *IMAGE_SIZE))

# ════════════════════════════════════════════════════════════════════════════
# PATCH SCORE COMPUTATION
# ════════════════════════════════════════════════════════════════════════════
def compute_patch_scores(patch_feats: np.ndarray) -> np.ndarray:
    """
    patch_feats: (N_patches, D)
    v_ref = median over patches
    score(i) = 1 - cosine_similarity(patch_feats[i], v_ref)  ∈ [0, 2]
    """
    v_ref = np.median(patch_feats, axis=0)          # (D,)
    v_ref_n = v_ref / (np.linalg.norm(v_ref) + 1e-8)
    norms = np.linalg.norm(patch_feats, axis=1, keepdims=True) + 1e-8
    normed = patch_feats / norms                    # (N, D)
    cos_sim = normed @ v_ref_n                      # (N,)
    scores = 1.0 - cos_sim
    assert scores.min() >= -0.01 and scores.max() <= 2.01, \
        f"[FAIL] Score range violated: [{scores.min():.4f}, {scores.max():.4f}]"
    return scores.clip(0.0, 2.0)

# ════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH PATCH LABELS
# ════════════════════════════════════════════════════════════════════════════
def load_anomaly_intervals(signal_name: str, anom_df: pd.DataFrame):
    """Return list of (start_ts, end_ts) from anomalies.csv for signal."""
    row = anom_df[anom_df["signal"] == signal_name]
    if row.empty:
        return []
    raw = row.iloc[0]["events"]
    if pd.isna(raw) or str(raw).strip() in ("", "[]"):
        return []
    try:
        intervals = ast.literal_eval(str(raw))
        return [(float(a), float(b)) for a, b in intervals]
    except Exception:
        return []

def make_patch_labels(
    window_start_idx: int,
    timestamps: np.ndarray,
    anomaly_intervals,
    n_patches: int,
    patch_size_pixels: int,
    grid_cols: int,
) -> np.ndarray:
    """
    Returns binary label array of shape (n_patches,).
    A patch is anomalous if ANY timestep it covers overlaps a ground-truth interval.

    patch i → column col = i % grid_cols
    timestep range within window: [col * patch_size_pixels, (col+1) * patch_size_pixels)
    (since 224px image maps 1px → 1 timestep for Lw=224)
    """
    labels = np.zeros(n_patches, dtype=np.int32)
    for i in range(n_patches):
        col = i % grid_cols
        ts_start = window_start_idx + col * patch_size_pixels
        ts_end   = window_start_idx + (col + 1) * patch_size_pixels
        # get actual timestamps for this range
        idx_s = min(ts_start, len(timestamps) - 1)
        idx_e = min(ts_end,   len(timestamps))
        if idx_s >= idx_e:
            continue
        t_start = timestamps[idx_s]
        t_end   = timestamps[idx_e - 1]
        for (a_s, a_e) in anomaly_intervals:
            if t_start <= a_e and t_end >= a_s:
                labels[i] = 1
                break
    return labels

# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════
def cohens_d(scores_a: np.ndarray, scores_n: np.ndarray) -> float:
    if len(scores_a) == 0 or len(scores_n) == 0:
        return float("nan")
    pooled_std = np.sqrt((np.std(scores_a)**2 + np.std(scores_n)**2) / 2 + 1e-12)
    return (np.mean(scores_a) - np.mean(scores_n)) / pooled_std

def overlap_coefficient(scores_a: np.ndarray, scores_n: np.ndarray) -> float:
    if len(scores_a) < 5 or len(scores_n) < 5:
        return float("nan")
    x = np.linspace(
        min(scores_a.min(), scores_n.min()),
        max(scores_a.max(), scores_n.max()),
        500,
    )
    try:
        kde_a = gaussian_kde(scores_a)(x)
        kde_n = gaussian_kde(scores_n)(x)
        overlap = np.trapz(np.minimum(kde_a, kde_n), x)
    except Exception:
        overlap = float("nan")
    return overlap

def f1_max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Sweep threshold → max F1."""
    thresholds = np.unique(y_score)
    best = 0.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        best = max(best, f1)
    return best

# ════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ════════════════════════════════════════════════════════════════════════════
def load_series(csv_path: str):
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values.astype(float)
    values     = df["value"].values.astype(float)
    return timestamps, values

def build_dataset_list(anom_df: pd.DataFrame):
    """
    Returns list of dicts:
      { "dataset", "signal", "csv_path", "timestamps", "values", "anomaly_intervals" }
    """
    entries = []

    # ── SMAP ────────────────────────────────────────────────────────────────
    for ch in SMAP_CHANNELS:
        csv_path = os.path.join(DATA_DIR, "SMAP", f"{ch}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] SMAP {ch}: file not found, skipping")
            continue
        ts, vals = load_series(csv_path)
        intervals = load_anomaly_intervals(ch, anom_df)
        entries.append(dict(dataset="SMAP", signal=ch, csv_path=csv_path,
                            timestamps=ts, values=vals,
                            anomaly_intervals=intervals))

    # ── MSL ─────────────────────────────────────────────────────────────────
    msl_channels = [MSL_PRIMARY]
    # add 5 more random MSL channels
    all_msl = sorted([f.replace(".csv", "") for f in os.listdir(os.path.join(DATA_DIR, "MSL"))
                      if f.endswith(".csv") and f.replace(".csv","") != MSL_PRIMARY])
    rng = np.random.default_rng(SEED)
    extra_msl = rng.choice(all_msl, size=min(5, len(all_msl)), replace=False).tolist()
    msl_channels += extra_msl

    for ch in msl_channels:
        csv_path = os.path.join(DATA_DIR, "MSL", f"{ch}.csv")
        if not os.path.exists(csv_path):
            continue
        ts, vals = load_series(csv_path)
        intervals = load_anomaly_intervals(ch, anom_df)
        entries.append(dict(dataset="MSL", signal=ch, csv_path=csv_path,
                            timestamps=ts, values=vals,
                            anomaly_intervals=intervals))

    # ── NAB-realAWSCloudwatch ────────────────────────────────────────────────
    nab_dir = os.path.join(DATA_DIR, "realAWSCloudwatch")
    for fname in sorted(os.listdir(nab_dir)):
        if not fname.endswith(".csv"):
            continue
        signal = fname.replace(".csv", "")
        intervals = load_anomaly_intervals(signal, anom_df)
        if len(intervals) == 0:
            print(f"[INFO] NAB {signal}: no anomaly intervals, skipping")
            continue
        csv_path = os.path.join(nab_dir, fname)
        ts, vals = load_series(csv_path)
        entries.append(dict(dataset="NAB", signal=signal, csv_path=csv_path,
                            timestamps=ts, values=vals,
                            anomaly_intervals=intervals))

    return entries

# ════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP (per signal × per encoder × per layer)
# ════════════════════════════════════════════════════════════════════════════
# We store:
#   all_patch_scores[encoder][layer][dataset] = {"scores": np.array, "labels": np.array}
#   tsne_feats[encoder][layer][dataset]       = {"feats": ..., "labels": ...}  (subset)

ENCODERS = ["CLIP", "MAE", "DINOv2"]
ENCODER_PATCHES = {"CLIP": 196, "MAE": 196, "DINOv2": 256}
ENCODER_PATCH_PX = {"CLIP": 16, "MAE": 16, "DINOv2": 14}
ENCODER_GRID_COLS = {"CLIP": 14, "MAE": 14, "DINOv2": 16}

def cache_path(encoder, layer_or_tag, dataset, signal):
    safe_sig = signal.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"{encoder}_{layer_or_tag}_{dataset}_{safe_sig}.npz")

def run_analysis():
    anom_df = pd.read_csv(ANOM_CSV)
    entries = build_dataset_list(anom_df)
    print(f"\n[INFO] Total signals to process: {len(entries)}")

    # Load models
    models = {
        "CLIP":   load_clip(),
        "MAE":    load_mae(),
        "DINOv2": load_dino(),
    }

    # Accumulators
    # patch_data[encoder][layer][dataset] → {"scores": list, "labels": list}
    patch_data = {enc: {l: {} for l in LAYERS} for enc in ENCODERS}
    # tsne raw feats: [encoder][layer][dataset] → list of (feat_vec, label)
    tsne_raw   = {enc: {l: {} for l in LAYERS} for enc in ENCODERS}

    results_rows = []
    t_start = time.time()

    for entry_i, entry in enumerate(entries):
        dataset  = entry["dataset"]
        signal   = entry["signal"]
        values   = entry["values"]
        timestamps = entry["timestamps"]
        anom_ivs   = entry["anomaly_intervals"]

        print(f"\n[{entry_i+1}/{len(entries)}] {dataset}/{signal}  "
              f"(T={len(values)}, {len(anom_ivs)} anomaly interval(s))")

        # ── Generate windows ─────────────────────────────────────────────
        windows = series_to_windows(values)
        if len(windows) == 0:
            print(f"  [WARN] No windows generated, skipping")
            continue
        N_win = len(windows)
        window_starts = list(range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE))
        print(f"  Windows: {N_win}")

        # ── Per encoder ──────────────────────────────────────────────────
        for enc in ENCODERS:
            n_patches  = ENCODER_PATCHES[enc]
            patch_px   = ENCODER_PATCH_PX[enc]
            grid_cols  = ENCODER_GRID_COLS[enc]

            # Cache key: scores + labels only (features are ~120MB/signal, scores ~0.6MB)
            score_cache = cache_path(enc, "scores", dataset, signal)
            label_cache = cache_path(enc, "labels", dataset, signal)
            # t-SNE sample cache: small subset of raw features
            tsne_cache  = cache_path(enc, "tsne",   dataset, signal)

            # Try loading from cache; delete and recompute if file is corrupted
            cache_ok = False
            if os.path.exists(score_cache) and os.path.exists(label_cache):
                try:
                    sc = np.load(score_cache)
                    lc = np.load(label_cache)
                    # quick validity check
                    assert f"l{LAYERS[0]}" in sc and f"l{LAYERS[0]}" in lc
                    cache_ok = True
                except Exception as e:
                    print(f"  [{enc}] Cache corrupted ({e}), recomputing ...")
                    for f_ in [score_cache, label_cache, tsne_cache]:
                        if os.path.exists(f_):
                            try: os.remove(f_)
                            except: pass

            if cache_ok:
                print(f"  [{enc}] Loading scores from cache ...")
                for l in LAYERS:
                    key = f"l{l}"
                    all_scores = sc[key]
                    all_labels = lc[key]
                    if dataset not in patch_data[enc][l]:
                        patch_data[enc][l][dataset] = {"scores": [], "labels": []}
                    patch_data[enc][l][dataset]["scores"].append(all_scores)
                    patch_data[enc][l][dataset]["labels"].append(all_labels)
                # restore t-SNE samples if cached
                if os.path.exists(tsne_cache):
                    try:
                        td = np.load(tsne_cache)
                        for l in LAYERS:
                            key_f, key_l = f"feats_l{l}", f"labs_l{l}"
                            if key_f in td:
                                if dataset not in tsne_raw[enc][l]:
                                    tsne_raw[enc][l][dataset] = []
                                tsne_raw[enc][l][dataset].append(
                                    list(zip(td[key_f], td[key_l]))
                                )
                    except Exception:
                        pass  # t-SNE cache corrupted — skip, t-SNE will just have less data
                continue

            # ── Batch forward pass: compute features → scores immediately ──
            win_tensor = torch.from_numpy(windows).float()  # (N, 3, H, W)

            # Accumulators per layer
            layer_scores  = {l: [] for l in LAYERS}
            layer_labels  = {l: [] for l in LAYERS}
            # t-SNE reservoir: keep up to TSNE_RESERVE (feat, label) pairs per layer
            TSNE_RESERVE  = 30    # 30 patches per signal → ~1050 total, sample 600 at plot time
            tsne_buf      = {l: {"feats": [], "labs": []} for l in LAYERS}

            for batch_start in range(0, N_win, BATCH_SIZE):
                batch      = win_tensor[batch_start:batch_start + BATCH_SIZE]
                b_size     = len(batch)
                hs         = extract_hidden_states(batch, enc, models)

                if batch_start == 0 and entry_i == 0:
                    verify_feature_shapes(hs, enc)
                    print(f"  [{enc}] Shape check passed")

                # For each window in batch, compute labels once (same for all layers)
                batch_labels = []
                for bi in range(b_size):
                    win_i  = batch_start + bi
                    assert win_i < len(window_starts), \
                        f"[BUG] win_i={win_i} >= len(window_starts)={len(window_starts)}"
                    w_start = window_starts[win_i]
                    labs   = make_patch_labels(
                        w_start, timestamps, anom_ivs,
                        n_patches, patch_px, grid_cols,
                    )
                    batch_labels.append(labs)

                for l in LAYERS:
                    feats_batch = hs[l]  # (B, N_patches, D)
                    for bi in range(b_size):
                        pf     = feats_batch[bi]          # (N_patches, D)
                        sc     = compute_patch_scores(pf) # (N_patches,)
                        labs   = batch_labels[bi]
                        layer_scores[l].append(sc)
                        layer_labels[l].append(labs)

                        # t-SNE: stratified sampling — prefer anomaly patches
                        # so red dots actually appear even when anomalies are rare
                        anom_idx = np.where(labs == 1)[0]
                        norm_idx = np.where(labs == 0)[0]
                        # collect anomaly patches first (up to half the budget)
                        half = TSNE_RESERVE // 2
                        if len(anom_idx) > 0 and len(tsne_buf[l]["feats"]) < TSNE_RESERVE:
                            pick = anom_idx[np.random.randint(len(anom_idx))]
                            tsne_buf[l]["feats"].append(pf[pick])
                            tsne_buf[l]["labs"].append(int(labs[pick]))
                        # collect normal patches to fill the rest
                        if len(norm_idx) > 0 and len(tsne_buf[l]["feats"]) < TSNE_RESERVE:
                            pick = norm_idx[np.random.randint(len(norm_idx))]
                            tsne_buf[l]["feats"].append(pf[pick])
                            tsne_buf[l]["labs"].append(int(labs[pick]))

            # Concatenate and cache scores (tiny: ~0.6 MB per signal)
            scores_to_save = {}
            labels_to_save = {}
            tsne_to_save   = {}
            for l in LAYERS:
                all_scores = np.concatenate(layer_scores[l])
                all_labels = np.concatenate(layer_labels[l])
                scores_to_save[f"l{l}"] = all_scores
                labels_to_save[f"l{l}"] = all_labels

                if dataset not in patch_data[enc][l]:
                    patch_data[enc][l][dataset] = {"scores": [], "labels": []}
                patch_data[enc][l][dataset]["scores"].append(all_scores)
                patch_data[enc][l][dataset]["labels"].append(all_labels)

                # accumulate t-SNE samples (individual patch vectors, not windows)
                if tsne_buf[l]["feats"]:
                    tsne_feats = np.stack(tsne_buf[l]["feats"])  # (K, D)
                    tsne_labs  = np.array(tsne_buf[l]["labs"])   # (K,)
                    tsne_to_save[f"feats_l{l}"] = tsne_feats
                    tsne_to_save[f"labs_l{l}"]  = tsne_labs
                    if dataset not in tsne_raw[enc][l]:
                        tsne_raw[enc][l][dataset] = []
                    tsne_raw[enc][l][dataset].append(
                        list(zip(tsne_feats, tsne_labs))
                    )

            np.savez_compressed(score_cache, **scores_to_save)
            np.savez_compressed(label_cache, **labels_to_save)
            if tsne_to_save:
                np.savez_compressed(tsne_cache, **tsne_to_save)

        # progress timer
        elapsed = time.time() - t_start
        avg_per = elapsed / (entry_i + 1)
        eta = avg_per * (len(entries) - entry_i - 1)
        print(f"  Elapsed: {elapsed/60:.1f}m  ETA: {eta/60:.1f}m")

    # ── Consolidate and compute metrics ──────────────────────────────────────
    print("\n[INFO] Computing metrics ...")
    datasets_list = sorted({e["dataset"] for e in entries})

    rows = []
    for enc in ENCODERS:
        for l in LAYERS:
            for ds in datasets_list:
                if ds not in patch_data[enc][l]:
                    continue
                scores = np.concatenate(patch_data[enc][l][ds]["scores"])
                labels = np.concatenate(patch_data[enc][l][ds]["labels"])

                n_anom   = labels.sum()
                n_normal = (labels == 0).sum()
                n_total  = len(labels)

                anom_frac = n_anom / n_total if n_total > 0 else 0
                print(f"  {ds}/{enc}/L{l}: anom_frac={anom_frac:.4f}  "
                      f"(n_anom={n_anom}, n_normal={n_normal})")
                assert 0.005 < anom_frac < 0.30, \
                    f"[FAIL] {ds}/{enc}: anomaly patch fraction {anom_frac:.4f} out of [0.5%, 30%]"

                scores_a = scores[labels == 1]
                scores_n = scores[labels == 0]

                auroc = roc_auc_score(labels, scores) if n_anom > 0 and n_normal > 0 else float("nan")
                ap    = average_precision_score(labels, scores) if n_anom > 0 else float("nan")
                cd    = cohens_d(scores_a, scores_n)
                oc    = overlap_coefficient(scores_a, scores_n)
                f1m   = f1_max(labels, scores) if n_anom > 0 else float("nan")

                rows.append({
                    "dataset": ds, "encoder": enc, "layer": l,
                    "AUROC_patch": auroc, "AP_patch": ap,
                    "cohens_d": cd, "overlap_coef": oc, "F1_max_series": f1m,
                    "n_normal_patches": int(n_normal),
                    "n_anomaly_patches": int(n_anom),
                })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(f"[INFO] Saved results.csv")

    # ── Summary: best layer per encoder per dataset ──────────────────────────
    summary_rows = []
    print("\n=== SUMMARY: Best Layer per Encoder ===")
    for ds in datasets_list:
        print(f"\nDataset: {ds}")
        sub = results_df[results_df["dataset"] == ds]
        winner_auroc = -1
        winner_str = ""
        for enc in ENCODERS:
            enc_sub = sub[sub["encoder"] == enc]
            if enc_sub.empty:
                continue
            best_row = enc_sub.loc[enc_sub["AUROC_patch"].idxmax()]
            bl = int(best_row["layer"])
            ba = best_row["AUROC_patch"]
            cd = best_row["cohens_d"]
            enc_short = enc if enc != "DINOv2" else "DINO"
            print(f"{enc_short:6s} → best layer {bl:2d}, AUROC={ba:.3f}, Cohen's d={cd:.2f}")
            summary_rows.append({
                "dataset": ds, "encoder": enc, "best_layer": bl,
                "AUROC_patch": ba, "cohens_d": cd,
            })
            if ba > winner_auroc:
                winner_auroc = ba
                winner_str = f"{enc_short} layer {bl}"
        print(f"→ Winner: {winner_str}")
    print()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print(f"[INFO] Saved summary.csv")

    # ── Final analysis ────────────────────────────────────────────────────────
    print("\n=== FINAL ANALYSIS ===")
    for ds in datasets_list:
        sub = results_df[results_df["dataset"] == ds]
        clip_best = sub[sub["encoder"] == "CLIP"]["AUROC_patch"].max()
        mae_rows  = sub[sub["encoder"] == "MAE"]
        if not mae_rows.empty:
            mae_best     = mae_rows["AUROC_patch"].max()
            mae_best_row = mae_rows.loc[mae_rows["AUROC_patch"].idxmax()]
            mae_best_l   = int(mae_best_row["layer"])
            print(f"{ds}: CLIP_best={clip_best:.3f}  MAE_best={mae_best:.3f} (layer {mae_best_l})")
            if mae_best <= clip_best:
                print(f"  [NEGATIVE RESULT] MAE does NOT beat CLIP on {ds}!")
                # Print full distribution info
                scores_all = np.concatenate(patch_data["MAE"][mae_best_l][ds]["scores"])
                labels_all = np.concatenate(patch_data["MAE"][mae_best_l][ds]["labels"])
                sa = scores_all[labels_all == 1]
                sn = scores_all[labels_all == 0]
                print(f"  MAE L{mae_best_l}: mean_anom={sa.mean():.4f}, mean_norm={sn.mean():.4f}, "
                      f"std_anom={sa.std():.4f}, std_norm={sn.std():.4f}")
                scores_c = np.concatenate(patch_data["CLIP"][12][ds]["scores"])
                labels_c = np.concatenate(patch_data["CLIP"][12][ds]["labels"])
                sa_c = scores_c[labels_c == 1]
                sn_c = scores_c[labels_c == 0]
                print(f"  CLIP L12: mean_anom={sa_c.mean():.4f}, mean_norm={sn_c.mean():.4f}, "
                      f"std_anom={sa_c.std():.4f}, std_norm={sn_c.std():.4f}")

    return results_df, patch_data, tsne_raw, summary_df, datasets_list

# ════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════
def plot_heatmaps(results_df, datasets_list):
    for ds in datasets_list:
        sub = results_df[results_df["dataset"] == ds]
        pivot = sub.pivot_table(index="encoder", columns="layer", values="AUROC_patch")
        pivot = pivot.reindex(index=ENCODERS, columns=LAYERS)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu",
                    vmin=0.4, vmax=1.0, ax=ax,
                    cbar_kws={"label": "AUROC"})
        ax.set_title(f"Patch Anomaly Separability — {ds}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Encoder")
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"heatmap_auroc_{ds}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[INFO] Saved {out}")

def plot_distributions(results_df, patch_data, datasets_list):
    for ds in datasets_list:
        sub = results_df[results_df["dataset"] == ds]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, enc in zip(axes, ENCODERS):
            enc_sub = sub[sub["encoder"] == enc]
            if enc_sub.empty:
                ax.set_visible(False)
                continue
            best_l = int(enc_sub.loc[enc_sub["AUROC_patch"].idxmax(), "layer"])
            if ds not in patch_data[enc][best_l]:
                continue
            scores = np.concatenate(patch_data[enc][best_l][ds]["scores"])
            labels = np.concatenate(patch_data[enc][best_l][ds]["labels"])
            sa = scores[labels == 1]
            sn = scores[labels == 0]
            cd = cohens_d(sa, sn)
            # KDE
            x = np.linspace(scores.min(), scores.max(), 500)
            try:
                if len(sn) > 5:
                    ax.fill_between(x, gaussian_kde(sn)(x), alpha=0.4, color="blue", label="Normal")
                if len(sa) > 5:
                    ax.fill_between(x, gaussian_kde(sa)(x), alpha=0.4, color="red", label="Anomaly")
            except Exception:
                pass
            ax.set_title(f"{enc} L{best_l}\nCohen's d={cd:.2f}")
            ax.set_xlabel("Anomaly score")
            ax.legend(fontsize=8)
        fig.suptitle(f"Score Distributions — {ds}")
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"distribution_{ds}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[INFO] Saved {out}")

def plot_tsne(patch_data, tsne_raw, datasets_list):
    """2×2 grid: CLIP L7 | CLIP L12 / MAE L7 | MAE L12"""
    N_SAMPLE = 300

    for ds in datasets_list:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        configs = [("CLIP", 7), ("CLIP", 12), ("MAE", 7), ("MAE", 12)]

        for ax, (enc, l) in zip(axes.flat, configs):
            if ds not in patch_data[enc][l]:
                ax.set_visible(False)
                continue
            scores = np.concatenate(patch_data[enc][l][ds]["scores"])
            labels = np.concatenate(patch_data[enc][l][ds]["labels"])

            # Gather raw features for t-SNE from tsne_raw
            # Each entry is now a list of (patch_vec (D,), label scalar) pairs
            all_feats_list  = []
            all_labels_list = []
            for sig_entries in tsne_raw[enc][l].get(ds, []):
                for (feat_vec, lab) in sig_entries:
                    all_feats_list.append(feat_vec)
                    all_labels_list.append(lab)

            if len(all_feats_list) == 0:
                ax.set_visible(False)
                continue

            all_feats  = np.stack(all_feats_list)   # (N, D)
            all_labels = np.array(all_labels_list)  # (N,)

            idx_n = np.where(all_labels == 0)[0]
            idx_a = np.where(all_labels == 1)[0]
            rng = np.random.default_rng(SEED)
            sel_n = rng.choice(idx_n, size=min(N_SAMPLE, len(idx_n)), replace=False)
            sel_a = rng.choice(idx_a, size=min(N_SAMPLE, len(idx_a)), replace=False)
            sel_idx = np.concatenate([sel_n, sel_a])
            sel_feats  = all_feats[sel_idx]
            sel_labels = all_labels[sel_idx]

            tsne = TSNE(perplexity=30, n_iter=1000, random_state=SEED)
            emb  = tsne.fit_transform(sel_feats)

            ax.scatter(emb[sel_labels == 0, 0], emb[sel_labels == 0, 1],
                       c="blue", s=20, alpha=0.5, label=f"Normal (n={int((sel_labels==0).sum())})")
            ax.scatter(emb[sel_labels == 1, 0], emb[sel_labels == 1, 1],
                       c="red",  s=60, alpha=0.9, label=f"Anomaly (n={int((sel_labels==1).sum())})",
                       edgecolors="darkred", linewidths=0.5, zorder=5)
            ax.set_title(f"{enc} Layer {l}")
            ax.legend(fontsize=8, markerscale=1)

        fig.suptitle(f"t-SNE Patch Features — {ds}", fontsize=14)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"tsne_{ds}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[INFO] Saved {out}")

def plot_layer_profiles(results_df, datasets_list):
    ENC_COLORS = {"CLIP": "blue", "MAE": "orange", "DINOv2": "green"}
    for ds in datasets_list:
        sub = results_df[results_df["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(8, 5))
        for enc in ENCODERS:
            enc_sub = sub[sub["encoder"] == enc].sort_values("layer")
            if enc_sub.empty:
                continue
            ax.plot(enc_sub["layer"], enc_sub["AUROC_patch"],
                    color=ENC_COLORS[enc], marker="o", label=enc)
            best_row = enc_sub.loc[enc_sub["AUROC_patch"].idxmax()]
            ax.scatter([best_row["layer"]], [best_row["AUROC_patch"]],
                       color=ENC_COLORS[enc], s=200, marker="*", zorder=5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC (patch)")
        ax.set_title(f"Layer Profile — {ds}")
        ax.set_xticks(LAYERS)
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"layer_profile_{ds}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[INFO] Saved {out}")

# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()

    results_df, patch_data, tsne_raw, summary_df, datasets_list = run_analysis()

    print("\n[INFO] Generating visualisations ...")
    plot_heatmaps(results_df, datasets_list)
    plot_distributions(results_df, patch_data, datasets_list)
    plot_tsne(patch_data, tsne_raw, datasets_list)
    plot_layer_profiles(results_df, datasets_list)

    total = time.time() - t0
    print(f"\n[DONE] Total runtime: {total/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
