"""Spectrogram encoder for time series windows.

STFT log-power spectrogram → 224×224 image.

Why spectrogram for MAE
-----------------------
Line plot: ~85% background pixels → 85% of MAE patches carry no signal.
Spectrogram: every pixel encodes power at (time, frequency) → 100% patch
density → PatchCore memory bank has no wasted entries.

Temporal direction is preserved (time on x-axis).
Looks like natural-image textures (fabric, heat maps) → MAE features
are well-calibrated (no domain gap like GAF).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import stft as _stft


def window_to_spectrogram(
    window: np.ndarray,
    n_fft: int = 64,
    hop_length: int = 4,
    img_size: int = 224,
) -> np.ndarray:
    """Convert a 1-D window to a log-power STFT spectrogram image.

    Parameters
    ----------
    window     : np.ndarray, shape (W,)
    n_fft      : int  FFT size (controls freq resolution)
    hop_length : int  hop between successive frames
    img_size   : int  output image side length

    Returns
    -------
    np.ndarray, shape (3, img_size, img_size), dtype float32, values in [0, 1]
    """
    W = len(window)

    # Z-score normalize per window
    mu  = window.mean()
    std = window.std() + 1e-8
    x   = (window - mu) / std

    # STFT → complex spectrogram
    _, _, Zxx = _stft(x, nperseg=n_fft, noverlap=n_fft - hop_length)
    # Zxx shape: (n_freqs, n_times)  n_freqs = n_fft//2 + 1

    # Log power spectrum
    power     = np.abs(Zxx) ** 2
    log_power = np.log1p(power).astype(np.float32)  # log(1 + p), avoids log(0)

    # Normalize to [0, 1]
    lo, hi = log_power.min(), log_power.max()
    spec   = np.zeros_like(log_power) if hi - lo < 1e-8 else (log_power - lo) / (hi - lo)

    # Flip freq axis: low freq at bottom (natural orientation)
    spec = spec[::-1, :].copy()   # (n_freqs, n_times), contiguous

    # Resize to img_size × img_size
    from PIL import Image as _PIL
    img8     = (spec * 255).astype(np.uint8)
    pil      = _PIL.fromarray(img8, mode="L")
    resample = getattr(_PIL, "Resampling", _PIL).BILINEAR
    pil      = pil.resize((img_size, img_size), resample)
    spec_r   = np.array(pil, dtype=np.float32) / 255.0   # (H, W)

    # Replicate to 3 channels — MAE expects RGB
    rgb = np.stack([spec_r, spec_r, spec_r], axis=0)     # (3, H, W)
    return rgb


def render_spectrogram_windows(
    values: np.ndarray,
    window_size: int = 224,
    step_size: int = 56,
    n_fft: int = 64,
    hop_length: int = 4,
    img_size: int = 224,
) -> np.ndarray:
    """Render spectrogram images for all sliding windows.

    Returns
    -------
    np.ndarray, shape (N, 3, img_size, img_size), dtype float32
    """
    T      = len(values)
    starts = list(range(0, T - window_size + 1, step_size))
    imgs   = np.empty((len(starts), 3, img_size, img_size), dtype=np.float32)
    for i, s in enumerate(starts):
        imgs[i] = window_to_spectrogram(
            values[s : s + window_size], n_fft, hop_length, img_size
        )
    return imgs
