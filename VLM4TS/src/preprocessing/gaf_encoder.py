"""Gramian Angular Field (GAF) encoder for time series windows.

Converts each 1-D window into a 2-D image where every pixel encodes
temporal correlation — unlike line plots where ~94% of pixels are
background.

GASF(i, j) = cos(phi_i + phi_j)   where phi = arccos(normalized_value)

Key property: the full 224x224 image is information-dense. An anomaly
at time step t ripples through the entire t-th row AND t-th column,
making it spatially extensive and much more visible to ViT patches.
"""

from __future__ import annotations

import numpy as np


def window_to_gaf(window: np.ndarray, img_size: int = 224) -> np.ndarray:
    """Convert a 1-D time series window to a GASF image.

    Parameters
    ----------
    window : np.ndarray, shape (W,)
    img_size : int  output image side length (default 224)

    Returns
    -------
    np.ndarray, shape (3, img_size, img_size), dtype float32, values in [0, 1]
    """
    W = len(window)

    # Normalize to [-1, 1] per window
    lo, hi = window.min(), window.max()
    if hi - lo < 1e-8:
        x = np.zeros(W, dtype=np.float64)
    else:
        x = 2.0 * (window - lo) / (hi - lo) - 1.0
    x = np.clip(x, -1.0, 1.0)

    # Compute GASF: cos(phi_i + phi_j)
    phi = np.arccos(x)                        # (W,)
    gasf = np.cos(phi[:, None] + phi[None, :])  # (W, W), in [-1, 1]

    # Resize to img_size if W != img_size (bilinear via PIL)
    if W != img_size:
        from PIL import Image as _PIL
        # gasf in [-1, 1] → [0, 255] for PIL
        img8 = ((gasf + 1.0) / 2.0 * 255).astype(np.uint8)
        pil = _PIL.fromarray(img8, mode='L')
        pil = pil.resize((img_size, img_size), _PIL.BILINEAR)
        gasf_resized = np.array(pil, dtype=np.float64) / 255.0 * 2.0 - 1.0
    else:
        gasf_resized = gasf

    # Map [-1, 1] → [0, 1], apply viridis colormap → RGB
    norm = (gasf_resized + 1.0) / 2.0          # (H, W), in [0, 1]
    rgb = _viridis_colormap(norm)               # (H, W, 3), in [0, 1]
    return rgb.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)


def render_gaf_windows(
    values: np.ndarray,
    window_size: int = 224,
    step_size: int = 56,
    img_size: int = 224,
) -> np.ndarray:
    """Render GAF images for all sliding windows of a time series.

    Parameters
    ----------
    values     : np.ndarray, shape (T,)
    window_size: int  number of time steps per window
    step_size  : int  stride between windows
    img_size   : int  output image side length

    Returns
    -------
    np.ndarray, shape (N, 3, img_size, img_size), dtype float32
    """
    T = len(values)
    starts = list(range(0, T - window_size + 1, step_size))
    imgs = np.empty((len(starts), 3, img_size, img_size), dtype=np.float32)
    for i, s in enumerate(starts):
        imgs[i] = window_to_gaf(values[s : s + window_size], img_size)
    return imgs


# ── Viridis colormap (hard-coded, no matplotlib dep at import time) ────────────

def _viridis_colormap(x: np.ndarray) -> np.ndarray:
    """Apply viridis colormap to values in [0, 1].

    Uses a piecewise-linear approximation of the 256-entry viridis LUT.
    Returns float32 RGB in [0, 1], same spatial shape as input.
    """
    import matplotlib.cm as _cm
    cmap = _cm.get_cmap("viridis")
    rgba = cmap(x.astype(np.float32))   # (..., 4)
    return rgba[..., :3].astype(np.float32)
