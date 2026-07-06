"""
Spectral Magnitude Aligner (SMA)
=================================
Aligns the frequency magnitude spectrum of rendered time-series images toward
the 1/f^alpha distribution of natural images, while preserving the spatial
phase (which encodes the actual signal structure).

Theory
------
Natural images follow P(f) ∝ 1/f^alpha (alpha ≈ 2) in 2D radial power spectrum.
Rendered line-plot images have a flat or high-frequency-heavy spectrum.
MAE was pretrained on natural images, so this spectral gap degrades feature
quality for out-of-distribution inputs.

SMA corrects only the amplitude — phase is kept intact — so the structural
information (where the line is, shape of the anomaly) is fully preserved.

Algorithm (per channel)
-----------------------
1. F = fft2(channel)           — complex spectrum
2. A = |F|, P = angle(F)       — amplitude and phase
3. Build target_A following 1/f^alpha profile (DC excluded)
4. Scale target_A to match source RMS energy
5. A_new = (1 - beta)*A + beta*target_A    — beta controls blend strength
6. Reconstruct: ifft2(A_new * exp(i*P)), clip to original range

Parameters
----------
beta  : float [0, 1] — blend toward natural-image spectrum (0 = no-op, 1 = full)
alpha : float        — spectral slope; 2.0 matches natural images
"""

from __future__ import annotations

import numpy as np
import torch


def _build_target_amplitude(H: int, W: int, alpha: float = 2.0) -> np.ndarray:
    """Radially symmetric 1/f^(alpha/2) target amplitude profile, shape [H, W]."""
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    f  = np.sqrt(fy**2 + fx**2)
    f[0, 0] = 1.0                         # avoid divide-by-zero at DC
    target = 1.0 / (f ** (alpha / 2.0))
    target[0, 0] = 0.0                    # will be replaced by source DC
    return target.astype(np.float32)


def sma_channel(channel: np.ndarray, target_amp: np.ndarray,
                beta: float = 0.5) -> np.ndarray:
    """Apply SMA to a single 2D channel [H, W], float values."""
    F     = np.fft.fft2(channel)
    A     = np.abs(F)
    phase = np.angle(F)

    # Scale target to match source RMS (excluding DC)
    src_rms = np.sqrt(np.mean(A[1:, 1:]**2)) + 1e-8
    tgt_rms = np.sqrt(np.mean(target_amp[1:, 1:]**2)) + 1e-8
    scaled_target = target_amp * (src_rms / tgt_rms)
    scaled_target[0, 0] = A[0, 0]        # preserve mean brightness (DC)

    A_new = (1.0 - beta) * A + beta * scaled_target

    F_new   = A_new * np.exp(1j * phase)
    ch_new  = np.real(np.fft.ifft2(F_new)).astype(np.float32)

    ch_min, ch_max = channel.min(), channel.max()
    return np.clip(ch_new, ch_min, ch_max)


def sma_image(img: np.ndarray, beta: float = 0.5,
              alpha: float = 2.0) -> np.ndarray:
    """
    Apply SMA to a single image.

    Parameters
    ----------
    img   : np.ndarray, shape [C, H, W] or [H, W], float32
    beta  : blend strength (0 = no-op, 1 = full alignment)
    alpha : spectral slope (2.0 = natural images)

    Returns
    -------
    np.ndarray, same shape and dtype as img
    """
    if img.ndim == 2:
        H, W = img.shape
        target = _build_target_amplitude(H, W, alpha)
        return sma_channel(img, target, beta)

    C, H, W  = img.shape
    target   = _build_target_amplitude(H, W, alpha)
    out      = np.empty_like(img)
    for c in range(C):
        out[c] = sma_channel(img[c], target, beta)
    return out


def sma_batch(imgs: np.ndarray, beta: float = 0.5,
              alpha: float = 2.0) -> np.ndarray:
    """
    Apply SMA to a batch of images.

    Parameters
    ----------
    imgs : np.ndarray, shape [N, C, H, W]
    beta : blend strength
    alpha: spectral slope

    Returns
    -------
    np.ndarray, shape [N, C, H, W]
    """
    N, C, H, W = imgs.shape
    target     = _build_target_amplitude(H, W, alpha)
    out        = np.empty_like(imgs)
    for i in range(N):
        for c in range(C):
            out[i, c] = sma_channel(imgs[i, c], target, beta)
    return out


class SMADataset(torch.utils.data.Dataset):
    """
    Wraps CLIPTimeSeriesDataset and applies SMA on-the-fly per image.

    Usage
    -----
    from preprocessing.sma_transform import SMADataset
    from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset

    base_ds = CLIPTimeSeriesDataset(...)
    sma_ds  = SMADataset(base_ds, beta=0.5)
    """

    def __init__(self, base_dataset, beta: float = 0.5, alpha: float = 2.0):
        self.base    = base_dataset
        self.beta    = beta
        self.alpha   = alpha
        # Pre-build target amplitude once (all images same size)
        sample = base_dataset[0]
        _, H, W  = sample["img"].shape
        self._target = _build_target_amplitude(H, W, alpha)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        img_np = sample["img"].numpy()          # [C, H, W]
        C      = img_np.shape[0]
        out    = np.empty_like(img_np)
        for c in range(C):
            out[c] = sma_channel(img_np[c], self._target, self.beta)
        sample["img"] = torch.from_numpy(out).float()
        return sample
