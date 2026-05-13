"""FFT-based adaptive window size estimation for time series anomaly detection.

Algorithm
---------
1. Remove mean from the signal (suppress DC component).
2. Compute magnitude spectrum via rfft.
3. Find the dominant non-DC frequency.
4. Require SNR >= 3 over the median noise floor; fall back to default otherwise.
5. Convert dominant frequency → period (= 1 / freq).
6. From the candidate window sizes [56, 112, 224], pick the one whose nearest
   integer multiple of the estimated period is closest to that candidate.
"""

import numpy as np


WINDOW_CANDIDATES = (56, 112, 224)


def estimate_window_size(
    values: np.ndarray,
    candidates: tuple = WINDOW_CANDIDATES,
    default: int = 224,
    snr_threshold: float = 3.0,
) -> int:
    """Return the best window size from `candidates` for this time series.

    Parameters
    ----------
    values : np.ndarray
        1-D time series (raw or preprocessed).
    candidates : tuple of int
        Allowed window sizes, in ascending order.
    default : int
        Fallback when no clear period is detected.
    snr_threshold : float
        Peak / median(spectrum) ratio below which period is considered unreliable.

    Returns
    -------
    int  — selected window size from `candidates`.
    """
    n = len(values)
    if n < min(candidates):
        return default

    # 1. Remove mean to zero out the DC bin cleanly
    x = values - values.mean()

    # 2. Magnitude spectrum (one-sided)
    fft_mag = np.abs(np.fft.rfft(x))
    fft_mag[0] = 0.0   # zero DC explicitly

    if len(fft_mag) < 2:
        return default

    # 3. Dominant non-DC peak
    peak_idx = int(np.argmax(fft_mag[1:])) + 1   # offset back to full array

    # 4. SNR check against median noise floor
    noise = np.median(fft_mag[1:])
    if noise < 1e-10 or fft_mag[peak_idx] < snr_threshold * noise:
        return default   # no clear periodicity → fallback

    # 5. Period estimate
    freqs = np.fft.rfftfreq(n)
    dominant_freq = freqs[peak_idx]
    if dominant_freq < 1e-6:
        return default

    period = 1.0 / dominant_freq

    # 6. For each candidate, find the integer multiple of `period` closest to
    #    that candidate, then pick the candidate with the smallest gap.
    best_candidate = default
    best_dist = float("inf")

    for cand in candidates:
        k = max(1, round(cand / period))          # nearest multiple index
        approx = k * period                        # closest multiple of period
        dist = abs(float(cand) - approx)
        if dist < best_dist:
            best_dist = dist
            best_candidate = cand

    return best_candidate
