"""PatchCore anomaly scorer using MAE patch-level features.

PatchCore (Roth et al., CVPR 2022) adapted for time series:
  1. Build memory bank from reference (normal) windows
  2. For each test window, score each of the 196 MAE patches by its
     nearest-neighbour distance to the memory bank
  3. Window score = mean of top-1% patch scores (soft max)

Why PatchCore over LTR
----------------------
LTR pools patch features → one vector per window → throws away all
spatial structure within the window.

PatchCore keeps all 196 patch features → memory bank captures the
full distribution of normal patch appearances → test patches that
deviate from ANY normal patch pattern are flagged.

For spectrogram images where every patch is information-dense, this
per-patch scoring is far more discriminative.

Reference selection
-------------------
n_ref_ratio = 0.2 → first 20% of windows (assumed pre-anomaly normal).
Minimum 10 windows.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import ViTMAEModel
except ImportError:
    raise ImportError("transformers>=4.26 required")


class _ImgDataset(Dataset):
    def __init__(self, imgs: np.ndarray):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx]).float()


class MAE_PatchCore:
    """PatchCore scorer using MAE ViT-B/16 patch features.

    Parameters
    ----------
    device        : torch.device
    n_ref_ratio   : float   fraction of windows used as normal reference
    top_p         : float   fraction of highest-scoring patches to average
                            (1.0 = mean, ~0.0 ≈ max)
    batch_size    : int
    """

    N_PATCHES  = 196   # 14 × 14
    FEAT_DIM   = 768   # ViT-B hidden dim

    def __init__(
        self,
        device: torch.device,
        n_ref_ratio: float = 0.2,
        top_p: float = 0.1,
        batch_size: int = 16,
    ):
        self.device      = device
        self.n_ref_ratio = n_ref_ratio
        self.top_p       = top_p
        self.batch_size  = batch_size

        print("  Loading facebook/vit-mae-base for PatchCore ...")
        self._model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self._model.config.mask_ratio = 0.0   # no masking → all 196 patches visible
        self._model = self._model.to(device)
        self._model.eval()
        print("  MAE ready.  (mask_ratio=0 → 196 patches per image)")

    @torch.no_grad()
    def _extract(self, imgs: np.ndarray) -> np.ndarray:
        """Extract patch features.

        Returns
        -------
        np.ndarray  [N, 196, 768]
        """
        loader  = DataLoader(_ImgDataset(imgs), batch_size=self.batch_size,
                             shuffle=False)
        feats   = []
        for batch in loader:
            batch = batch.to(self.device)
            out   = self._model(pixel_values=batch)
            # last_hidden_state: [B, 1+196, 768]  (cls + patch tokens)
            patch_tokens = out.last_hidden_state[:, 1:, :]  # [B, 196, 768]
            feats.append(patch_tokens.cpu().float().numpy())
        return np.concatenate(feats, axis=0)   # [N, 196, 768]

    def fit_and_score(self, imgs: np.ndarray) -> np.ndarray:
        """Build memory bank then score all windows.

        Parameters
        ----------
        imgs : np.ndarray  [N, 3, 224, 224] float32

        Returns
        -------
        scores : np.ndarray [N]  higher = more anomalous
        """
        N     = len(imgs)
        n_ref = max(10, int(N * self.n_ref_ratio))
        n_ref = min(n_ref, N)

        print(f"    PatchCore: N={N}  n_ref={n_ref}  top_p={self.top_p}")

        # Extract all features at once
        all_feats = self._extract(imgs)   # [N, 196, 768]

        # Memory bank: flatten reference patches → [n_ref*196, 768]
        ref_feats = all_feats[:n_ref].reshape(-1, self.FEAT_DIM)  # [M, 768]

        # Convert to torch tensors for batched distance computation
        ref_t  = torch.from_numpy(ref_feats).to(self.device)  # [M, 768]

        scores = np.empty(N, dtype=np.float32)
        k_top  = max(1, int(self.N_PATCHES * self.top_p))

        for i in range(N):
            q = torch.from_numpy(all_feats[i]).to(self.device)  # [196, 768]

            # L2 distances: [196, M]
            dists = torch.cdist(q, ref_t, p=2)         # [196, M]
            nn_dist = dists.min(dim=1).values           # [196]  min over memory bank

            # Score = mean of top-k patch distances
            topk  = nn_dist.topk(k_top).values
            scores[i] = topk.mean().item()

        return scores


def scores_to_timeseries(
    scores: np.ndarray,
    T: int,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Stitch per-window scores to a T-length vector (mean aggregation)."""
    out   = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)
    for i, sc in enumerate(scores):
        s = i * step_size
        e = min(s + window_size, T)
        out[s:e]   += sc
        count[s:e] += 1.0
    return out / np.maximum(count, 1.0)
