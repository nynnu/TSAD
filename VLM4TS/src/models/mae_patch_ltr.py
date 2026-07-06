"""Patch-level LTR scoring with MAE features.

Core idea
---------
Standard LTR pools 196 patch features → 1 vector per window.
Patch-LTR keeps all 196 features and compares each patch position
independently to the same position in temporal neighbor windows.

For window i, patch position p:
  ref(i, p) = mean of feature(j, p) for j in k-nearest temporal neighbors
  score(i, p) = L2( feature(i, p) − ref(i, p) )
  window_score(i) = mean of top-P% patch scores

Why this is better than window-level LTR
-----------------------------------------
Line + window-LTR: each "feature vector" averages over 85% background
patches → anomaly signal diluted.

Spectrogram + patch-LTR: every patch contains spectral information.
Comparison at the same patch position across windows asks:
  "Does this frequency band deviate from my temporal neighbors?"
That is a precise, spatially-resolved anomaly question.

Why this is better than PatchCore (fixed memory bank)
------------------------------------------------------
PatchCore assumes first N% of windows are normal → fragile.
Patch-LTR uses temporal neighbors → same robustness as standard LTR,
no assumption about which period is anomaly-free.
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
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx): return torch.from_numpy(self.imgs[idx]).float()


class MAE_PatchLTR:
    """Patch-level Local Temporal Reference scorer.

    Parameters
    ----------
    device    : torch.device
    k         : int    number of temporal neighbors
    exclusion : int    exclude windows within ±exclusion of target
    top_p     : float  fraction of highest-scoring patches to average
    batch_size: int
    """

    N_PATCHES = 196
    FEAT_DIM  = 768

    def __init__(
        self,
        device: torch.device,
        k: int = 5,
        exclusion: int = 5,
        top_p: float = 0.10,
        batch_size: int = 16,
    ):
        self.device    = device
        self.k         = k
        self.exclusion = exclusion
        self.top_p     = top_p
        self.batch_size = batch_size

        print("  Loading facebook/vit-mae-base ...")
        self._model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self._model.config.mask_ratio = 0.0   # all 196 patches visible
        self._model = self._model.to(device)
        self._model.eval()
        print("  MAE ready.  (mask_ratio=0 → 196 patches per image)")

    @torch.no_grad()
    def extract(self, imgs: np.ndarray) -> np.ndarray:
        """Extract patch features for all windows.

        Returns
        -------
        np.ndarray  [N, 196, 768]
        """
        loader = DataLoader(_ImgDataset(imgs), batch_size=self.batch_size,
                            shuffle=False)
        feats = []
        for batch in loader:
            out = self._model(pixel_values=batch.to(self.device))
            # last_hidden_state: [B, 1+196, 768]
            patches = out.last_hidden_state[:, 1:, :]   # [B, 196, 768]
            feats.append(patches.cpu().float().numpy())
        return np.concatenate(feats, axis=0)   # [N, 196, 768]

    def score(self, feats: np.ndarray, k: int | None = None) -> np.ndarray:
        """Compute patch-level LTR scores.

        Parameters
        ----------
        feats : np.ndarray  [N, 196, 768]  pre-extracted patch features
        k     : int  override self.k if provided

        Returns
        -------
        scores : np.ndarray  [N]
        """
        k     = k if k is not None else self.k
        N     = len(feats)
        k_top = max(1, int(self.N_PATCHES * self.top_p))
        scores = np.zeros(N, dtype=np.float32)

        for i in range(N):
            neighbors = _temporal_neighbors(i, N, k, self.exclusion)
            if len(neighbors) == 0:
                continue

            ref_mean  = feats[neighbors].mean(axis=0)   # [196, 768]
            diff      = feats[i] - ref_mean              # [196, 768]
            pscore    = np.linalg.norm(diff, axis=1)     # [196]

            # mean of top-k_top patch scores
            scores[i] = np.partition(pscore, -k_top)[-k_top:].mean()

        return scores

    def fit_and_score(
        self,
        imgs: np.ndarray,
        k_list: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Extract features once, score at multiple k values.

        Parameters
        ----------
        imgs   : np.ndarray  [N, 3, 224, 224]
        k_list : list of k values to score (default [self.k])

        Returns
        -------
        dict  {k: scores [N]}
        """
        if k_list is None:
            k_list = [self.k]

        print(f"    Extracting patch features for {len(imgs)} windows ...")
        feats = self.extract(imgs)   # [N, 196, 768]
        print(f"    Features: {feats.shape}  → scoring at k={k_list}")

        return {k: self.score(feats, k) for k in k_list}


def _temporal_neighbors(i: int, N: int, k: int, exclusion: int) -> list[int]:
    """k nearest temporal neighbors of window i, excluding ±exclusion."""
    candidates = [j for j in range(N) if abs(j - i) > exclusion]
    if not candidates:
        # relax exclusion if not enough candidates
        candidates = [j for j in range(N) if j != i]
    candidates.sort(key=lambda j: abs(j - i))
    return candidates[:k]


def scores_to_timeseries(
    scores: np.ndarray,
    T: int,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Stitch per-window scores into a T-length vector (mean aggregation)."""
    out   = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)
    for i, sc in enumerate(scores):
        s = i * step_size
        e = min(s + window_size, T)
        out[s:e]   += sc
        count[s:e] += 1.0
    return out / np.maximum(count, 1.0)
