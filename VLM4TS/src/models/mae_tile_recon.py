"""MAE Reconstruction Anomaly Detector for 2D Tile images.

Scoring principle
-----------------
The tile image stacks 14 windows as rows. The target window is always
at row 7 (patches 98-111 in the 14x14 patch grid).

We force those 14 patches to be masked and ask MAE to reconstruct them
from the 13 surrounding context rows (temporal neighbors).

Normal window  -> consistent with neighbors -> low reconstruction error
Anomalous window -> deviates from neighbors -> high reconstruction error

This is fundamentally different from LTR:
  LTR      : compare window embedding to k neighbors
  TileRecon: MAE *predicts* the target window from neighbors directly

Why this should work where GAF diagonal masking failed
------------------------------------------------------
GAF diagonal masking: MAE has no knowledge of GAF math structure.
  The reconstruction asks "predict cos(phi_i+phi_j) from off-diagonal"
  which is meaningless to a model trained on natural images.

2D Tile: MAE sees stacked waveforms -> texture-like natural image.
  The reconstruction asks "predict what this row looks like given the
  rows above and below" -> same as "predict masked patches from context"
  which is exactly MAE's training objective.

Masking strategy
----------------
  Target patches (98-111) -> noise = 1.0  (always masked)
  Other 182 patches       -> noise ~ U[0, 0.9)
  Total masked ~= 147/196 = 75%  (matches training distribution)
  Score = MSE on the 14 target patches only
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import ViTMAEForPreTraining
except ImportError:
    raise ImportError("transformers>=4.26 required")

_GRID           = 14
_N_PATCHES      = 196
_TARGET_ROW     = 7     # row index of target window (N_BEFORE=7)
_TARGET_PATCHES = list(range(_TARGET_ROW * _GRID, _TARGET_ROW * _GRID + _GRID))
# = [98, 99, 100, ..., 111]


class _ImgDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
    def __len__(self):  return len(self.imgs)
    def __getitem__(self, i): return torch.from_numpy(self.imgs[i]).float()


class MAE_TileRecon:
    """MAE reconstruction scorer for 2D tile images.

    Parameters
    ----------
    device     : torch.device
    n_iter     : int   forward passes with different random non-target masks
    batch_size : int
    seed       : int
    """

    def __init__(
        self,
        device: torch.device,
        n_iter: int = 3,
        batch_size: int = 16,
        seed: int = 42,
    ):
        self.device     = device
        self.n_iter     = n_iter
        self.batch_size = batch_size
        self.seed       = seed
        self._target    = torch.tensor(_TARGET_PATCHES, dtype=torch.long)

        print("  Loading facebook/vit-mae-base for TileRecon ...")
        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.model.config.mask_ratio = 0.75
        self.model = self.model.to(device)
        self.model.eval()
        print(f"  MAE ready.  target_row={_TARGET_ROW}  "
              f"target_patches={_TARGET_PATCHES[0]}..{_TARGET_PATCHES[-1]}")

    def _make_noise(self, B: int, it: int) -> torch.Tensor:
        rng   = torch.Generator()
        rng.manual_seed(self.seed + it)
        noise = torch.rand(B, _N_PATCHES, generator=rng) * 0.9
        noise[:, self._target] = 1.0   # target row always masked
        return noise

    @torch.no_grad()
    def score_windows(self, imgs: np.ndarray) -> np.ndarray:
        """Score tile images by target-row reconstruction error.

        Parameters
        ----------
        imgs : np.ndarray  [N, 3, 224, 224]  float32

        Returns
        -------
        np.ndarray  [N]  higher = more anomalous
        """
        N          = len(imgs)
        all_scores = np.zeros((self.n_iter, N), dtype=np.float32)

        for it in range(self.n_iter):
            iter_scores = []
            for start in range(0, N, self.batch_size):
                batch  = imgs[start : start + self.batch_size]
                B      = len(batch)
                images = torch.from_numpy(batch).to(self.device)
                noise  = self._make_noise(B, it).to(self.device)

                out    = self.model(pixel_values=images, noise=noise)
                pred   = out.logits                               # [B, 196, patch_dim]
                tgt    = self.model.patchify(images)              # [B, 196, patch_dim]

                if self.model.config.norm_pix_loss:
                    mu  = tgt.mean(dim=-1, keepdim=True)
                    var = tgt.var(dim=-1, keepdim=True)
                    tgt = (tgt - mu) / (var + 1e-6).sqrt()

                patch_mse  = (pred - tgt).pow(2).mean(dim=-1)       # [B, 196]
                target_mse = patch_mse[:, self._target.to(self.device)]  # [B, 14]
                iter_scores.append(target_mse.mean(dim=-1).cpu().numpy())

            all_scores[it] = np.concatenate(iter_scores)

        return all_scores.mean(axis=0)


def scores_to_timeseries(
    scores: np.ndarray,
    T: int,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    out   = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)
    for i, sc in enumerate(scores):
        s = i * step_size
        e = min(s + window_size, T)
        out[s:e]   += sc
        count[s:e] += 1.0
    return out / np.maximum(count, 1.0)
