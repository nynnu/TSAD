"""Recurrence Plot (RP) encoder for time series windows.

Within-window soft RP:
  RP(i, j) = exp(-( (x_i - x_j) / sigma )^2 / 2)

  Gaussian kernel → continuous values in (0, 1]
  Main diagonal   = 1.0 (always self-recurrent)
  Off-diagonal    = decreases with value difference

Why RP > GAF for zero-shot MAE
-------------------------------
GAF: smooth gradient matrix → looks nothing like natural images
     → MAE features poorly calibrated → domain gap dominates

RP:  binary-like textures with diagonal line structures → closer to
     natural image textures (stripes, edges, checkerboard patterns)
     → MAE features more discriminative

Anomaly types visible in within-window RP
------------------------------------------
  Point anomaly  : one row + one column become sparse  (cross shape)
  Transitional   : block structure appears at transition boundary
  Periodic change: diagonal line spacing changes

Limitation
----------
  Sustained mean shift (full-window): self-similarity unchanged
  → Need CRP (cross-recurrence) to detect this type
"""

from __future__ import annotations
import numpy as np


def window_to_rp(
    window: np.ndarray,
    sigma_scale: float = 1.0,
    img_size: int = 224,
) -> np.ndarray:
    """Convert a 1-D time series window to a soft Recurrence Plot image.

    Parameters
    ----------
    window      : np.ndarray, shape (W,)
    sigma_scale : float  controls recurrence density (default 1.0)
                  smaller → sparser RP; larger → denser RP
    img_size    : int    output image side length (default 224)

    Returns
    -------
    np.ndarray, shape (3, img_size, img_size), dtype float32, values in [0, 1]
    """
    W = len(window)

    # Z-score normalize per window so sigma is interpretable
    mu  = window.mean()
    std = window.std() + 1e-8
    x   = (window - mu) / std          # shape (W,), mean=0 std=1

    # Pairwise squared differences → Gaussian kernel
    diff  = x[:, None] - x[None, :]   # (W, W)
    sigma = sigma_scale                # after z-score, std=1 so sigma=scale
    rp    = np.exp(-diff ** 2 / (2.0 * sigma ** 2))  # (W, W), in (0, 1]

    # Resize if W != img_size
    if W != img_size:
        from PIL import Image as _PIL
        img8 = (rp * 255).astype(np.uint8)
        pil  = _PIL.fromarray(img8, mode='L')
        resample = getattr(_PIL, 'Resampling', _PIL).BILINEAR
        pil  = pil.resize((img_size, img_size), resample)
        rp   = np.array(pil, dtype=np.float32) / 255.0
    else:
        rp = rp.astype(np.float32)

    rgb = _hot_colormap(rp)
    return rgb.transpose(2, 0, 1)     # (3, H, W)


def render_rp_windows(
    values: np.ndarray,
    window_size: int = 224,
    step_size: int = 56,
    sigma_scale: float = 1.0,
    img_size: int = 224,
) -> np.ndarray:
    """Render RP images for all sliding windows of a time series.

    Parameters
    ----------
    values      : np.ndarray, shape (T,)
    window_size : int
    step_size   : int
    sigma_scale : float  passed to window_to_rp
    img_size    : int

    Returns
    -------
    np.ndarray, shape (N, 3, img_size, img_size), dtype float32
    """
    T      = len(values)
    starts = list(range(0, T - window_size + 1, step_size))
    imgs   = np.empty((len(starts), 3, img_size, img_size), dtype=np.float32)
    for i, s in enumerate(starts):
        imgs[i] = window_to_rp(
            values[s : s + window_size], sigma_scale, img_size
        )
    return imgs


def compute_crp(
    window_a: np.ndarray,
    window_b: np.ndarray,
    sigma_scale: float = 1.0,
    img_size: int = 224,
) -> np.ndarray:
    """Cross Recurrence Plot between two windows.

    CRP(i, j) = exp(-( (x_a_i - x_b_j) / sigma )^2 / 2)

    Encodes how similar window_a is to window_b at each lag combination.
    Normal window vs normal reference  → dense (high values everywhere)
    Anomalous window vs normal ref     → sparse (low values = anomaly)

    Parameters
    ----------
    window_a, window_b : np.ndarray, shape (W,)
    sigma_scale        : float
    img_size           : int

    Returns
    -------
    np.ndarray, shape (3, img_size, img_size), dtype float32
    """
    # Z-score each window independently
    def _zscore(w):
        return (w - w.mean()) / (w.std() + 1e-8)

    xa = _zscore(window_a)
    xb = _zscore(window_b)

    diff  = xa[:, None] - xb[None, :]   # (W, W)
    sigma = sigma_scale
    crp   = np.exp(-diff ** 2 / (2.0 * sigma ** 2)).astype(np.float32)

    if crp.shape[0] != img_size:
        from PIL import Image as _PIL
        img8 = (crp * 255).astype(np.uint8)
        pil  = _PIL.fromarray(img8, mode='L')
        resample = getattr(_PIL, 'Resampling', _PIL).BILINEAR
        pil  = pil.resize((img_size, img_size), resample)
        crp  = np.array(pil, dtype=np.float32) / 255.0

    rgb = _hot_colormap(crp)
    return rgb.transpose(2, 0, 1)       # (3, H, W)


# ── 'hot' colormap (pure numpy — no matplotlib) ───────────────────────────────

def _hot_colormap(x: np.ndarray) -> np.ndarray:
    """'hot' colormap in pure numpy. Values in [0,1] → RGB in [0,1].

    Segments:
      [0,   1/3] : black → red   (R: 0→1, G=0, B=0)
      [1/3, 2/3] : red → yellow  (R=1, G: 0→1, B=0)
      [2/3, 1  ] : yellow → white(R=1, G=1, B: 0→1)
    """
    v = x.astype(np.float32)
    r = np.clip(v * 3.0,             0., 1.)
    g = np.clip(v * 3.0 - 1.0,       0., 1.)
    b = np.clip(v * 3.0 - 2.0,       0., 1.)
    return np.stack([r, g, b], axis=-1).astype(np.float32)
