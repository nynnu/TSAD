"""
Feature-Level Reconstruction Discrepancy Scoring
=================================================
Tests a new anomaly scoring signal on top of the Experiment 1 pipeline.

Baseline:   cosine distance between patch features and global median reference
New method: 1 - cosine_sim(encode(original_patch), encode(MAE_reconstructed_patch))

Key difference from pixel-level recon (already tested, failed):
  pixel-level: ||reconstructed - original||^2  →  measures "how normal it looks"
  feature-level: cos_disc(f_orig, f_recon)     →  measures "how much MAE expectation
                                                    deviates from actual"

Pipeline (same as Experiment 1):
  timm MAE vit_base_patch16_224.mae (L12, forward_features, 3-scale)
  + HuggingFace ViTMAEForPreTraining (facebook/vit-mae-base, for reconstruction)
  Interval F1, alpha=0.01, NAB + SMAP + MSL

Usage (Colab):
  1. from google.colab import drive; drive.mount('/content/drive')
  2. !python "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/compare_recon_discrepancy.py"
"""

import os, sys, subprocess, ast, tempfile, time, io
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get(
    'VLM4TS_ROOT',
    '/content/drive/Othercomputers/내 노트북/VLM4TS')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'preprocessing'))

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

# ── Dependencies ──────────────────────────────────────────────
subprocess.run(
    ['pip', 'install', 'timm', 'open-clip-torch', 'transformers', '--quiet'],
    check=True)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining

from preprocessing.preprocess        import preprocess_time_series, apply_ewma, draw_windowed_images
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils        import orion_to_internal, intervals_from_indices
from models.mae_vision               import MAE_AD
from models.model_utils              import (
    build_memory, compute_patch_dissimilarity,
    harmonic_aggregation, stitch_anomaly_maps,
    align_anomaly_vector, compute_detection_intervals,
)
from models.vit4ts_mae  import ViT4TS_MAE
from evaluation.evaluate import evaluate_intervals

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, 'data')
NAB_DIR     = os.path.join(DATA_DIR, 'realAWSCloudwatch')
SMAP_DIR    = os.path.join(DATA_DIR, 'SMAP')
MSL_DIR     = os.path.join(DATA_DIR, 'MSL')
ANOM_CSV    = os.path.join(DATA_DIR, 'anomalies.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_recon_discrepancy')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────
PATCH_SIZE  = 16
IMAGE_SIZE  = 224
N_PATCHES   = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196
GRID_COLS   = IMAGE_SIZE // PATCH_SIZE          # 14
PATCH_PX    = PATCH_SIZE
BATCH_SIZE  = 8    # smaller due to K×B expansion
K_PASSES    = 5    # number of random masking passes per batch
INET_MEAN   = [0.485, 0.456, 0.406]
INET_STD    = [0.229, 0.224, 0.225]
_inet_norm  = T.Normalize(mean=INET_MEAN, std=INET_STD)

BASE_PARAMS = dict(
    window_size=224, window_step_ratio=4.0,
    agg_percent=0.25, patch_size=16,
    model_name="vit_base_patch16_224.mae",
    image_size=(224, 224), dpi=100,
    standardize=True, smoothing_alpha=1.0,
    alpha=0.01, verbose=False,
)

# ── Datasets ──────────────────────────────────────────────────
NAB_CHANNELS  = None   # filled after GT loaded
SMAP_CHANNELS = ["D-1", "E-1", "E-2", "E-3", "F-1", "F-3", "T-1"]
MSL_CHANNELS  = ["P-11"]

# ============================================================
# Step 0: Codebase summary (printed, not computed)
# ============================================================
print("\n" + "=" * 60)
print("[Step 0] Codebase summary")
print("=" * 60)
print("  Encoder  : timm vit_base_patch16_224.mae via MAE_AD")
print("    → forward_features() returns [B, 197, 768] (L12, CLS+196 patches)")
print("    → multi-scale: large(3×3), mid(2×2), patch(1×1) tokens")
print("  Scoring  : build_memory (global median) + compute_patch_dissimilarity (cosine)")
print("  F1 eval  : evaluate_intervals, alpha=0.01 (interval-level)")
print("  Cache    : none in baseline; recon features NOT cached (large)")
print("")
print("  [CASE A] timm has NO decoder → load ViTMAEForPreTraining separately")
print("    Encoder for recon : timm (same as pipeline)")
print("    Reconstruction    : HuggingFace ViTMAEForPreTraining (facebook/vit-mae-base)")
print("    Shared weights    : same MAE checkpoint → consistent features")
print("=" * 60)

# ============================================================
# Helpers: patchify / unpatchify / denorm
# ============================================================
def patchify(images: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """[B, 3, H, W] → [B, N, P²×3]"""
    B, C, H, W = images.shape
    h = w = H // patch_size
    x = images.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)              # [B, h, w, C, P, P]
    x = x.reshape(B, h * w, C * patch_size ** 2)  # [B, N, P²×3]
    return x


def unpatchify(patches: torch.Tensor, patch_size: int = PATCH_SIZE,
               img_size: int = IMAGE_SIZE) -> torch.Tensor:
    """[B, N, P²×3] → [B, 3, H, W]"""
    B = patches.shape[0]
    h = w = img_size // patch_size
    C = 3
    x = patches.reshape(B, h, w, C, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)              # [B, C, h, P, w, P]
    x = x.reshape(B, C, h * patch_size, w * patch_size)
    return x


def denorm_logits(logits: torch.Tensor,
                  orig_patches: torch.Tensor) -> torch.Tensor:
    """
    Undo per-patch normalization that facebook/vit-mae-base applies as its loss target.
    logits      : [B, N, D]   — decoder predictions (per-patch-normalized space)
    orig_patches: [B, N, D]   — original patchified image (ImageNet-normalized)
    Returns     : [B, N, D]   — back in ImageNet-normalized space
    """
    mean = orig_patches.mean(dim=-1, keepdim=True)           # [B, N, 1]
    std  = (orig_patches.var(dim=-1, keepdim=True) + 1e-6).sqrt()
    return logits * std + mean

# ============================================================
# Core scorer: feature-level reconstruction discrepancy
# ============================================================
@torch.no_grad()
def compute_recon_discrepancy_scores(
    raw_images:  torch.Tensor,        # [B, 3, H, W] float32 in [0, 1]
    recon_model: ViTMAEForPreTraining, # used for both reconstruction AND encoding
    device:      torch.device,
    k_passes:    int = K_PASSES,
) -> torch.Tensor:
    """
    For each image in the batch, compute per-patch reconstruction discrepancy.
    Returns [B, N_patches] — higher score = more anomalous.

    Both f_orig and f_recon are extracted via recon_model.vit (HF encoder),
    ensuring consistent normalization and feature space throughout.

    Algorithm (k_passes × random 75% masking):
      1. Apply ImageNet normalization
      2. f_orig = recon_model.vit(norm_images).last_hidden_state[:, 1:, :]
      For each pass:
        3. Run recon_model(norm_images, random_noise)  → logits + mask
        4. Denorm logits → replace only masked patches → recon_image (inet-norm)
        5. f_recon = recon_model.vit(recon_image).last_hidden_state[:, 1:, :]
        6. score[b, i] = 1 - cos_sim(f_orig[b,i], f_recon[b,i])   for masked i
      Average across passes for each patch.
    """
    B = raw_images.shape[0]
    images    = raw_images.to(device)
    norm_imgs = torch.stack([_inet_norm(img) for img in images])   # [B, 3, H, W]

    hf_enc      = recon_model.vit
    saved_ratio = recon_model.config.mask_ratio   # 0.75

    # Step 1: original features — disable masking to get all 196 patches
    # With mask_ratio=0.75 the encoder only outputs 49 kept patches; we need 196.
    recon_model.config.mask_ratio = 0.0
    enc_out = hf_enc(norm_imgs, output_hidden_states=False)
    f_orig  = enc_out.last_hidden_state[:, 1:, :]   # [B, 196, 768] strip CLS
    recon_model.config.mask_ratio = saved_ratio      # restore 0.75

    orig_patches = patchify(norm_imgs)               # [B, 196, P²×3=768]

    score_sum  = torch.zeros(B, N_PATCHES, device=device)
    mask_count = torch.zeros(B, N_PATCHES, device=device)

    for _ in range(k_passes):
        # Step 2: reconstruction with 75% masking (as trained)
        # HF MAE: argsort(noise) ascending → smallest noise = KEPT, largest = MASKED
        noise  = torch.rand(B, N_PATCHES, device=device)
        output = recon_model(norm_imgs, noise=noise)
        # output.logits : [B, 196, P²×3] in per-patch-normalized space
        # output.mask   : [B, 196], 1=masked 0=visible

        logits = output.logits   # [B, 196, 768]
        mask   = output.mask     # [B, 196]

        # Step 3: denorm → replace only masked patches in inet-norm image
        recon_patches = denorm_logits(logits, orig_patches)
        replaced      = orig_patches.clone()
        replaced[mask.bool()] = recon_patches[mask.bool()]
        recon_imgs    = unpatchify(replaced).clamp(-4.0, 4.0)

        # Step 4: re-encode reconstructed image — also disable masking
        recon_model.config.mask_ratio = 0.0
        enc_recon = hf_enc(recon_imgs, output_hidden_states=False)
        f_recon   = enc_recon.last_hidden_state[:, 1:, :]   # [B, 196, 768]
        recon_model.config.mask_ratio = saved_ratio          # restore 0.75

        # Step 5: cosine discrepancy — only count masked patches
        disc = 1.0 - F.cosine_similarity(f_orig, f_recon, dim=-1)  # [B, 196]

        score_sum  += disc  * mask.float()
        mask_count += mask.float()

    avg = torch.where(
        mask_count > 0,
        score_sum / mask_count.clamp(min=1.0),
        torch.zeros_like(score_sum),
    )
    return avg   # [B, 196]

# ============================================================
# Step 6: Sanity checks (run before full experiment)
# ============================================================
def run_sanity_checks(timm_model, recon_model):
    print("\n" + "=" * 60)
    print("[Step 6] Sanity checks")
    print("=" * 60)

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Check 1: determinism ────────────────────────────────
    dummy = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    norm  = _inet_norm(dummy[0]).unsqueeze(0)
    fixed_noise = torch.zeros(1, N_PATCHES, device=DEVICE)

    with torch.no_grad():
        o1 = recon_model(norm, noise=fixed_noise).logits
        o2 = recon_model(norm, noise=fixed_noise).logits

    assert torch.allclose(o1, o2, atol=1e-5), "FAIL: non-deterministic reconstruction!"
    print("  [1] Determinism          PASS ✓")

    # ── Check 2: reconstruction visual quality ──────────────
    # Synthetic sine wave → render as image → reconstruct → save
    t      = np.linspace(0, 4 * np.pi, IMAGE_SIZE)
    signal = np.sin(t).astype(np.float32)

    # Quick render
    fig_size = IMAGE_SIZE / 100
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
    g_min, g_max = float(signal.min()) - 0.1, float(signal.max()) + 0.1
    ax.plot(signal, color='black', linewidth=1.0)
    ax.set_xlim(0, IMAGE_SIZE - 1); ax.set_ylim(g_min, g_max)
    ax.axis('off'); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0); plt.margins(0, 0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0); plt.close(fig); buf.seek(0)
    from PIL import Image as PILImage
    img_arr = np.array(PILImage.open(buf).convert('RGB').resize(
        (IMAGE_SIZE, IMAGE_SIZE), PILImage.LANCZOS), dtype=np.float32) / 255.0
    buf.close()
    raw_t   = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    norm_t  = _inet_norm(raw_t[0]).unsqueeze(0)

    # Fixed mask: mask first 50% of patches
    fixed_noise = torch.cat([
        torch.ones(1, N_PATCHES // 2, device=DEVICE),      # large = masked
        torch.zeros(1, N_PATCHES - N_PATCHES // 2, device=DEVICE),  # small = kept
    ], dim=1)

    with torch.no_grad():
        out = recon_model(norm_t, noise=fixed_noise)
    orig_p   = patchify(norm_t)
    recon_p  = denorm_logits(out.logits, orig_p)
    replaced = orig_p.clone(); replaced[out.mask.bool()] = recon_p[out.mask.bool()]
    recon_img = unpatchify(replaced).clamp(-4, 4)

    # Denormalize for saving (roughly back to [0,1])
    inet_mean_t = torch.tensor(INET_MEAN, device=DEVICE).view(1, 3, 1, 1)
    inet_std_t  = torch.tensor(INET_STD,  device=DEVICE).view(1, 3, 1, 1)
    orig_vis  = (norm_t  * inet_std_t + inet_mean_t).clamp(0, 1).cpu().squeeze()
    recon_vis = (recon_img * inet_std_t + inet_mean_t).clamp(0, 1).cpu().squeeze()

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(orig_vis.permute(1, 2, 0)); ax1.set_title('Original'); ax1.axis('off')
    ax2.imshow(recon_vis.permute(1, 2, 0)); ax2.set_title('Reconstructed (50% masked)'); ax2.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sanity_reconstruction.png'), dpi=150)
    plt.close()
    print(f"  [2] Reconstruction quality  saved → {RESULTS_DIR}/sanity_reconstruction.png")

    # ── Check 3 & 4: score direction + over-generalization ──
    t2      = np.arange(IMAGE_SIZE, dtype=np.float32)
    normal  = np.sin(2 * np.pi * t2 / 50).astype(np.float32)
    anomaly = normal.copy(); anomaly[100:112] += 4.0   # spike

    def quick_render(signal):
        fig_size = IMAGE_SIZE / 100
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
        ax.plot(signal, color='black', linewidth=1.0)
        ax.set_xlim(0, IMAGE_SIZE - 1)
        ymin, ymax = float(signal.min()) - 0.2, float(signal.max()) + 0.2
        ax.set_ylim(ymin, ymax); ax.axis('off')
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0); plt.margins(0, 0)
        buf2 = io.BytesIO()
        fig.savefig(buf2, format='png', pad_inches=0); plt.close(fig); buf2.seek(0)
        arr = np.array(PILImage.open(buf2).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), PILImage.LANCZOS), dtype=np.float32) / 255.0
        buf2.close()
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    raw_norm  = quick_render(normal)
    raw_anom  = quick_render(anomaly)

    scores_n = compute_recon_discrepancy_scores(raw_norm, recon_model, DEVICE, k_passes=3)
    scores_a = compute_recon_discrepancy_scores(raw_anom, recon_model, DEVICE, k_passes=3)

    # Spike is at t=100..112 → column pixels 100..112 → patch col = 100//16 = 6..7
    spike_cols  = list(range(6, 8))   # patch columns covering t=100-112
    spike_patch_ids = [r * GRID_COLS + c for r in range(GRID_COLS) for c in spike_cols]
    other_patch_ids = [i for i in range(N_PATCHES) if i not in spike_patch_ids]

    s_spike   = scores_a[0, spike_patch_ids].mean().item()
    s_normal  = scores_n[0, other_patch_ids].mean().item()
    s_anom_bg = scores_a[0, other_patch_ids].mean().item()

    print(f"  [3] Score direction check:")
    print(f"      Anomaly window — spike patches   score = {s_spike:.4f}")
    print(f"      Anomaly window — other patches   score = {s_anom_bg:.4f}")
    print(f"      Normal  window — all patches     score = {s_normal:.4f}")
    if s_spike > s_anom_bg:
        print("      PASS ✓  spike patches score higher than background")
    else:
        print("      WARN ✗  spike patches NOT scoring higher (method may fail on point anomalies)")

    # ── Check 4: over-generalization check ──────────────────
    print(f"  [4] Over-generalization check:")
    # For the anomaly window: what does MAE reconstruct the spike patch as?
    norm_anom = _inet_norm(raw_anom[0]).unsqueeze(0)
    fixed_noise4 = torch.ones(1, N_PATCHES, device=DEVICE) * 0.001
    fixed_noise4[0, spike_patch_ids] = 0.999   # spike patches get large noise = masked
    with torch.no_grad():
        out4 = recon_model(norm_anom, noise=fixed_noise4)
    orig_p4  = patchify(norm_anom)
    recon_p4 = denorm_logits(out4.logits, orig_p4)

    orig_spike_pix   = orig_p4[0, spike_patch_ids[0]].cpu()
    recon_spike_pix  = recon_p4[0, spike_patch_ids[0]].cpu()
    orig_range  = f"[{orig_spike_pix.min().item():.3f}, {orig_spike_pix.max().item():.3f}]"
    recon_range = f"[{recon_spike_pix.min().item():.3f}, {recon_spike_pix.max().item():.3f}]"
    print(f"      Original spike patch pixel range  : {orig_range}")
    print(f"      Reconstructed spike patch range   : {recon_range}")

    orig_max  = orig_spike_pix.max().item()
    recon_max = recon_spike_pix.max().item()
    if recon_max < orig_max - 0.05:
        print(f"      PASS ✓  MAE reconstructs spike as lower-amplitude (over-generalization present)")
        print(f"              → feature discrepancy signal should be detectable")
    else:
        print(f"      WARN ✗  Reconstructed spike ≈ original spike (no over-generalization)")
        print(f"              → feature discrepancy may not work for point anomalies")

    print()


# ============================================================
# Load models
# ============================================================
print("\n[INFO] Loading models ...")
_timm_enc   = MAE_AD(model_name="vit_base_patch16_224.mae", device=DEVICE)
_recon_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE).eval()
print(f"  timm MAE_AD  : loaded (cosine baseline, forward_features → L12)")
print(f"  HF recon     : ViTMAEForPreTraining facebook/vit-mae-base")
print(f"    mask_ratio={_recon_model.config.mask_ratio}  (75% masking as trained)")
print(f"    recon_model.vit used for BOTH f_orig and f_recon (consistent feature space)")

# Run sanity checks before experiment
run_sanity_checks(_timm_enc, _recon_model)

# ============================================================
# Aggregate patch scores → series level (same as Exp 1)
# ============================================================
def aggregate_to_series(patch_scores_list, window_starts, T, agg='max'):
    """patch_scores_list: list of (196,) arrays. Returns (T,) series scores."""
    series = np.zeros(T)
    for ps, ws in zip(patch_scores_list, window_starts):
        for pi in range(N_PATCHES):
            col = pi % GRID_COLS
            t0  = ws + col * PATCH_PX
            t1  = ws + (col + 1) * PATCH_PX
            for t in range(t0, min(t1, T)):
                if agg == 'max':
                    series[t] = max(series[t], float(ps[pi]))
                else:
                    series[t] += float(ps[pi])
    return series

# ============================================================
# ViT4TS_MAE_ReconDisc
# ============================================================
class ViT4TS_MAE_ReconDisc(ViT4TS_MAE):
    """
    Same as ViT4TS_MAE but replaces patch scoring with feature-level
    reconstruction discrepancy. Inherits all windowing, threshold, and
    interval logic unchanged.
    """

    def __init__(self, recon_model: ViTMAEForPreTraining, k_passes: int = K_PASSES, **kwargs):
        super().__init__(**kwargs)
        self._recon_model = recon_model
        self._k_passes    = k_passes

    def _run_inference_recon(self, results_dir: str, base_series_id: str,
                              T_full: int, window_starts: list):
        dataset    = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_series_id,
            sample_size=None, no_anomaly=True, plot_type="line")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Total windows → choose full vs approximate mode
        n_windows = len(dataset)
        if n_windows > 100:
            k = max(3, self._k_passes - 1)
            print(f"    [recon] Approximate mode (n_windows={n_windows} > 100), K={k}")
        else:
            k = self._k_passes
            print(f"    [recon] Full mode (n_windows={n_windows}), K={k}")

        all_scores = []   # list of (196,) numpy arrays, in window order
        all_wids   = []

        for items in dataloader:
            raw_images = items["img"]            # [B, 3, H, W] float32 in [0,1]
            window_ids = items["window_id"]

            scores_b = compute_recon_discrepancy_scores(
                raw_images, self._recon_model, self.device, k_passes=k)
            # scores_b: [B, 196] tensor

            for b in range(scores_b.shape[0]):
                all_scores.append(scores_b[b].cpu().numpy())
                all_wids.append(int(window_ids[b]))

        # Sort by window id (dataloader might shuffle)
        order     = np.argsort(all_wids)
        sorted_sc = [all_scores[i] for i in order]
        sorted_ws = [window_starts[all_wids[i]] for i in order]

        series = aggregate_to_series(sorted_sc, sorted_ws, T_full)
        return series

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full      = len(values)
        values_proc = preprocess_time_series(values) if self.standardize \
                      else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        with tempfile.TemporaryDirectory() as temp_dir:
            step_size     = int(self.window_size / self.window_step_ratio)
            time_points   = np.arange(len(values_proc))
            n_windows     = int((T_full - self.window_size) / step_size) + 1
            window_starts = list(range(0, T_full - self.window_size + 1, step_size))

            plot_params = ("-", 1, "*", 0.1, "black",
                           (0, 1) if self.standardize else None)
            success = draw_windowed_images(
                base_series_id="series",
                save_path=temp_dir,
                time_series=values_proc,
                time_points=time_points,
                override=True,
                window_size=self.window_size,
                step_size=step_size,
                image_size=self.image_size,
                dpi=self.dpi,
                plot_params=plot_params,
            )
            if not success:
                return np.zeros(T_full), timestamps

            series_scores = self._run_inference_recon(
                temp_dir, "series", T_full, window_starts)
            if series_scores is None:
                return np.zeros(T_full), timestamps

        aligned = align_anomaly_vector(
            series_scores, T_full, self.window_size, step_size, n_windows)
        return aligned, timestamps

# ============================================================
# Ground truth & evaluation helpers
# ============================================================
def load_gt(anom_csv: str) -> dict:
    gt = {}
    with open(anom_csv, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt

def run_detector(detector, csv_path: str, gt_intervals: list) -> dict:
    data     = pd.read_csv(csv_path)
    result   = detector.detect(data)
    detected = result[["start", "end"]].values.tolist() if len(result) > 0 else []
    m = evaluate_intervals(gt_intervals, detected)
    return {"f1": m["F1"], "precision": m["precision"], "recall": m["recall"]}

# ============================================================
# Step 4: Experiment
# ============================================================
gt = load_gt(ANOM_CSV)

nab_files = sorted(f for f in os.listdir(NAB_DIR) if f.endswith('.csv'))
NAB_CHANNELS = [f.replace('.csv', '') for f in nab_files if gt.get(f.replace('.csv', ''))]

DATASET_CONFIGS = {
    "NAB":  {"dir": NAB_DIR,  "channels": NAB_CHANNELS},
    "SMAP": {"dir": SMAP_DIR, "channels": SMAP_CHANNELS},
    "MSL":  {"dir": MSL_DIR,  "channels": MSL_CHANNELS},
}

print("\n" + "=" * 60)
print("[Step 4] Running experiment")
print("  Scoring A: cosine (Experiment 1 baseline)")
print("  Scoring B: feature-level recon discrepancy (new)")
print("=" * 60)

rows = []

det_cosine = ViT4TS_MAE(**BASE_PARAMS)
det_recon  = ViT4TS_MAE_ReconDisc(recon_model=_recon_model, k_passes=K_PASSES, **BASE_PARAMS)

for ds_name, cfg in DATASET_CONFIGS.items():
    print(f"\n{'─'*60}")
    print(f"Dataset: {ds_name}  ({len(cfg['channels'])} signals)")
    print(f"{'─'*60}")

    for sig in cfg['channels']:
        csv_path  = os.path.join(cfg['dir'], f"{sig}.csv")
        intervals = gt.get(sig, [])
        if not os.path.exists(csv_path):
            print(f"  SKIP {sig}: file not found"); continue
        if not intervals:
            print(f"  SKIP {sig}: no GT"); continue

        print(f"\n  [{sig}]")
        try:
            t0 = time.time()
            m_cos  = run_detector(det_cosine, csv_path, intervals)
            t_cos  = time.time() - t0

            t0 = time.time()
            m_rec  = run_detector(det_recon, csv_path, intervals)
            t_rec  = time.time() - t0
        except Exception as e:
            print(f"    ERROR: {e}"); continue

        delta = m_rec['f1'] - m_cos['f1']
        winner = "RECON" if delta > 0 else ("COSINE" if delta < 0 else "TIE")
        print(f"    cosine  F1={m_cos['f1']:.4f}  ({t_cos:.1f}s)")
        print(f"    recon   F1={m_rec['f1']:.4f}  ({t_rec:.1f}s)  Δ={delta:+.4f}  [{winner}]")

        rows.append(dict(
            dataset=ds_name, signal=sig,
            cosine_f1=m_cos['f1'], cosine_prec=m_cos['precision'], cosine_rec=m_cos['recall'],
            recon_f1 =m_rec['f1'],  recon_prec =m_rec['precision'],  recon_rec =m_rec['recall'],
            delta_f1=delta, winner=winner,
            t_cosine=round(t_cos, 1), t_recon=round(t_rec, 1),
        ))

# ============================================================
# Step 5: Results table
# ============================================================
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, 'comparison.csv'), index=False)

SEP = 80

def _fmt(v):
    try:
        f = float(v)
        return "  NaN" if (f != f) else f"{f:.4f}"
    except (TypeError, ValueError):
        return "  NaN"

for ds_name in ["NAB", "SMAP", "MSL"]:
    sub = results_df[results_df['dataset'] == ds_name]
    if sub.empty: continue

    print(f"\n{'='*SEP}")
    print(f"=== Feature-level Recon Discrepancy vs Cosine — {ds_name} ===")
    print(f"{'='*SEP}")
    print(f"{'Signal':<40} {'Cosine F1':>10} {'Recon F1':>9} {'Delta':>7} {'Winner':>8}")
    print(f"{'-'*SEP}")
    for _, r in sub.iterrows():
        print(f"{r['signal']:<40} {_fmt(r['cosine_f1']):>10} {_fmt(r['recon_f1']):>9}"
              f"  {r['delta_f1']:>+6.4f}  {r['winner']:>8}")
    print(f"{'-'*SEP}")
    avg_c = sub['cosine_f1'].mean()
    avg_r = sub['recon_f1'].mean()
    print(f"{'AVG F1':<40} {avg_c:>10.4f} {avg_r:>9.4f}  {avg_r-avg_c:>+6.4f}")
    wins_c = (sub['winner'] == 'COSINE').sum()
    wins_r = (sub['winner'] == 'RECON').sum()
    print(f"  WINS — cosine: {wins_c}/{len(sub)}  recon: {wins_r}/{len(sub)}")

# ============================================================
# Step 7: Key findings
# ============================================================
print("\n" + "=" * SEP)
print("KEY FINDINGS")
print("=" * SEP)

# 1. Over-generalization?
print("1. Over-generalization check: see sanity output above.")
print("   (If PASS: recon discrepancy should work for point/spike anomalies)")
print("   (If WARN: recon discrepancy may only work for distribution-shift anomalies)")

# 2. Level-shift signals
level_shift_sigs = ['ec2_cpu_utilization_ac20cd', 'rds_cpu_utilization_cc0c53']
print("\n2. Level-shift signals (NAB: ac20cd, cc0c53):")
for sig in level_shift_sigs:
    row = results_df[results_df['signal'] == sig]
    if not row.empty:
        r = row.iloc[0]
        print(f"   {sig:<45} cosine={r['cosine_f1']:.4f}  recon={r['recon_f1']:.4f}  Δ={r['delta_f1']:+.4f}")

# 3. Spike/point signals in SMAP
print("\n3. SMAP point/amplitude anomaly signals (F-1, F-3, T-1):")
for sig in ['F-1', 'F-3', 'T-1']:
    row = results_df[(results_df['dataset'] == 'SMAP') & (results_df['signal'] == sig)]
    if not row.empty:
        r = row.iloc[0]
        print(f"   {sig:<10} cosine={r['cosine_f1']:.4f}  recon={r['recon_f1']:.4f}  Δ={r['delta_f1']:+.4f}")

# 4. Overall: complementary or redundant?
print("\n4. Overall Δ distribution:")
print(f"   Recon > cosine (Δ>0.05) : {(results_df['delta_f1'] > 0.05).sum()} signals")
print(f"   Recon < cosine (Δ<-0.05): {(results_df['delta_f1'] < -0.05).sum()} signals")
print(f"   Roughly tied (|Δ|≤0.05) : {(results_df['delta_f1'].abs() <= 0.05).sum()} signals")

# 5. Runtime
avg_t_cos = results_df['t_cosine'].mean()
avg_t_rec = results_df['t_recon'].mean()
print(f"\n5. Runtime per signal — cosine: {avg_t_cos:.1f}s   recon: {avg_t_rec:.1f}s"
      f"  (×{avg_t_rec/max(avg_t_cos,0.1):.1f} overhead)")

print(f"\nAll results → {RESULTS_DIR}/comparison.csv")
print(f"Sanity vis  → {RESULTS_DIR}/sanity_reconstruction.png")
