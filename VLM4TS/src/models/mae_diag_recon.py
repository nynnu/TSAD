"""MAE Diagonal Reconstruction Scorer for GAF images.

Core idea
---------
In a GAF image (224x224, 14x14 ViT patch grid):

  GASF(i, j) = cos(phi_i + phi_j)

  Main diagonal  GASF(i,i) = 2*x_i^2 - 1  ← encodes signal VALUES
  Off-diagonal   GASF(i,j) for i≠j          ← encodes temporal CORRELATIONS

Diagonal band patches (|row - col| <= band) encode the time series values.
Off-diagonal patches encode the correlation structure.

Scoring principle
-----------------
  Mask: diagonal band patches  (signal-encoding region)
  Visible: off-diagonal patches (correlation structure)
  Question: "Can MAE predict the signal values from the correlation structure?"

  Normal window:   signal consistent with correlations → low recon error
  Anomalous window: signal violates correlation structure → high recon error

This is fundamentally different from:
  - LTR: external comparison to temporal neighbors
  - Random recon: averages over all patches, dilutes anomaly signal

Implementation
--------------
  noise[:, diag_indices] = 1.0   → diagonal patches always masked
  noise[:, other_indices] = U[0,1) → fill remaining mask budget randomly
  mask_ratio = 0.75               → 147 masked, 49 kept
  score = recon_error on diagonal patches ONLY
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from transformers import ViTMAEForPreTraining
except ImportError:
    raise ImportError("transformers>=4.26 required: pip install transformers")


class MAE_DiagRecon:
    """GAF-specific anomaly scorer using diagonal-masked MAE reconstruction.

    Parameters
    ----------
    device      : torch.device
    diag_band   : int   half-width of diagonal band to mask (default 1 → 40 patches)
    n_iter      : int   number of forward passes to average (default 3)
    batch_size  : int   images per GPU batch (default 16)
    seed        : int   RNG seed for non-diagonal random noise
    """

    GRID = 14        # 224 / 16 = 14 patches per side
    N_PATCHES = 196  # 14 * 14

    def __init__(
        self,
        device: torch.device,
        diag_band: int = 1,
        n_iter: int = 3,
        batch_size: int = 16,
        seed: int = 42,
    ):
        self.device     = device
        self.diag_band  = diag_band
        self.n_iter     = n_iter
        self.batch_size = batch_size
        self.seed       = seed

        # Compute diagonal band patch indices (row, col) in 14×14 grid
        diag_idx = []
        for r in range(self.GRID):
            for c in range(self.GRID):
                if abs(r - c) <= diag_band:
                    diag_idx.append(r * self.GRID + c)
        self.diag_indices     = torch.tensor(diag_idx, dtype=torch.long)
        self.n_diag           = len(diag_idx)

        print(f"MAE_DiagRecon: diag_band={diag_band}  "
              f"n_diag_patches={self.n_diag}/{self.N_PATCHES}  "
              f"n_iter={n_iter}")

        print("  Loading facebook/vit-mae-base ...")
        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.model.config.mask_ratio = 0.75  # standard — ensures diagonal fits
        self.model = self.model.to(device)
        self.model.eval()

    def _make_noise(self, B: int, it: int) -> torch.Tensor:
        """Noise tensor that guarantees diagonal patches are masked.

        Diagonal patches → noise = 1.0      (highest  → always masked)
        Other patches    → noise ~ U[0, 0.9) (random   → partially masked)
        """
        rng   = torch.Generator()
        rng.manual_seed(self.seed + it)
        noise = torch.rand(B, self.N_PATCHES, generator=rng) * 0.9   # [0, 0.9)
        noise[:, self.diag_indices] = 1.0
        return noise

    @torch.no_grad()
    def score_windows(self, imgs: np.ndarray) -> np.ndarray:
        """Score a batch of GAF window images.

        Parameters
        ----------
        imgs : np.ndarray  [N, 3, 224, 224] float32, values in [0, 1]

        Returns
        -------
        scores : np.ndarray [N]  reconstruction error on diagonal patches
        """
        N = len(imgs)
        all_scores = np.zeros((self.n_iter, N), dtype=np.float32)

        for it in range(self.n_iter):
            iter_scores = []
            for start in range(0, N, self.batch_size):
                batch_np = imgs[start : start + self.batch_size]
                B = len(batch_np)
                images = torch.from_numpy(batch_np).to(self.device)

                noise = self._make_noise(B, it).to(self.device)
                out   = self.model(pixel_values=images, noise=noise)

                pred   = out.logits  # [B, 196, patch_size^2 * 3]
                target = self.model.patchify(images)  # [B, 196, patch_size^2 * 3]

                if self.model.config.norm_pix_loss:
                    mu  = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mu) / (var + 1e-6).sqrt()

                # Per-patch MSE, then select diagonal patches only
                patch_mse = (pred - target).pow(2).mean(dim=-1)  # [B, 196]
                diag_mse  = patch_mse[:, self.diag_indices.to(self.device)]  # [B, n_diag]
                score     = diag_mse.mean(dim=-1)  # [B]
                iter_scores.append(score.cpu().numpy())

            all_scores[it] = np.concatenate(iter_scores)

        return all_scores.mean(axis=0)  # [N], averaged over iterations


def window_scores_to_timeseries(
    scores: np.ndarray,
    T: int,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Stitch per-window scalar scores into a T-length score vector.

    Uses mean aggregation for overlapping windows (smoother than max).
    """
    out   = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)
    for i, sc in enumerate(scores):
        s = i * step_size
        e = min(s + window_size, T)
        out[s:e]   += sc
        count[s:e] += 1.0
    count = np.maximum(count, 1.0)
    return out / count
