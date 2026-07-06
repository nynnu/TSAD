"""MAE pixel-space reconstruction error for anomaly detection.

Uses HuggingFace facebook/vit-mae-base (full encoder + lightweight decoder).
Per-window reconstruction error averaged over n_iter random masks (MAEDAY style).

Key difference from mae_vision.py (MAE_AD):
  MAE_AD    : encoder only, feature-space comparison
  MAE_Recon : encoder + decoder, pixel-space reconstruction error
"""

import numpy as np
import torch

try:
    from transformers import ViTMAEForPreTraining
except ImportError:
    raise ImportError("transformers>=4.26 required: pip install transformers")


class MAE_Recon:
    """Per-window MAE reconstruction error scorer.

    Parameters
    ----------
    device     : torch.device
    n_iter     : int    Random mask iterations to average (default 5)
    mask_ratio : float  Fraction of patches masked per iteration (default 0.5)
    seed       : int    Base seed; iteration i uses (seed + i)
    """

    def __init__(self, device, n_iter: int = 5, mask_ratio: float = 0.5, seed: int = 42):
        self.device     = device
        self.n_iter     = n_iter
        self.mask_ratio = mask_ratio
        self.seed       = seed

        print("Loading MAE_Recon (facebook/vit-mae-base) ...")
        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.model.config.mask_ratio = mask_ratio
        self.model = self.model.to(device)
        self.model.eval()
        print(f"  device={device}  n_iter={n_iter}  mask_ratio={mask_ratio}  seed={seed}")

    @torch.no_grad()
    def score_batch(
        self, images: torch.Tensor, return_std: bool = False
    ) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
        """Compute per-image reconstruction error for one batch.

        Parameters
        ----------
        images     : torch.Tensor  [B, 3, 224, 224]
        return_std : bool  If True, also return per-image std across n_iter masks.

        Returns
        -------
        mean : np.ndarray [B]  mean MSE across n_iter masks
        std  : np.ndarray [B]  std  MSE across n_iter masks  (only if return_std=True)

        Why std matters
        ---------------
        Normal windows reconstruct consistently regardless of which patches are
        masked → low std.  Anomalous windows are unstable: masking the anomaly
        region yields high error; masking elsewhere yields low error → high std.
        std therefore captures a complementary aspect of anomalousness.
        """
        images = images.to(self.device)
        B      = images.shape[0]

        # Patchify + normalise target once — independent of mask
        target = self.model.patchify(images)          # [B, 196, p^2*3]
        if self.model.config.norm_pix_loss:
            mean   = target.mean(dim=-1, keepdim=True)
            var    = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        iter_scores: list[torch.Tensor] = []

        for it in range(self.n_iter):
            # Different random mask per iteration, reproducible across runs
            torch.manual_seed(self.seed + it)

            out  = self.model(pixel_values=images)         # uses config.mask_ratio
            pred = out.logits                              # [B, 196, p^2*3]
            mask = out.mask                                # [B, 196]  1=masked

            patch_loss = (pred - target).pow(2).mean(dim=-1)          # [B, 196]
            n_masked   = mask.sum(dim=1).clamp(min=1)                 # [B]
            per_img    = (patch_loss * mask).sum(dim=1) / n_masked    # [B]
            iter_scores.append(per_img)

        stacked    = torch.stack(iter_scores, dim=0)                  # [n_iter, B]
        mean_score = stacked.mean(dim=0).cpu().numpy()                # [B]

        if return_std:
            # unbiased (ddof=1) requires n_iter >= 2; n_iter=1 would give NaN
            std_score = stacked.std(dim=0, unbiased=(self.n_iter > 1)).cpu().numpy()
            return mean_score, std_score

        return mean_score   # [B]  — backward compatible
