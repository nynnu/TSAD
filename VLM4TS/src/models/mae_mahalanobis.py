"""MAE + Mahalanobis distance scorer with optional STL de-seasonalization.

Scoring
-------
LTR compares each window to k temporal neighbors → fails for long anomalies
(all neighbors anomalous → reference IS the anomaly → score ≈ 0).

Mahalanobis uses the GLOBAL distribution of all windows as reference:
  F  : [N, D]  mean-pooled patch features
  μ  = mean(F)
  Σ  = cov(F)  + reg·I
  score_i = sqrt((f_i − μ)ᵀ Σ⁻¹ (f_i − μ))

Long anomalies are still far from the global centroid even when all local
neighbors are also anomalous — no locality assumption required.

STL pre-processing (optional)
------------------------------
Decompose signal into trend + seasonal + residual, render the residual.
Removes cyclic variation so MAE features see only genuine deviations.

Risk: level-shift anomalies can be absorbed into the trend component.
Mitigation: fuse STL+Mahal with LTR k=5 in the experiment script.
"""
from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from preprocessing.preprocess import (
    preprocess_time_series,
    draw_windowed_images,
    apply_ewma,
)
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils import orion_to_internal
from models.vit4ts_mae import ViT4TS_MAE
from models.model_utils import align_anomaly_vector
from models.model_utils_local_v2 import build_ordered_embeddings


# ── STL helpers ───────────────────────────────────────────────────────────────

def detect_period(values: np.ndarray, min_period: int = 4) -> int:
    """Dominant period via FFT peak. Falls back to min_period if unreliable."""
    n = len(values)
    if n < 2 * min_period:
        return min_period
    fft_mag = np.abs(np.fft.rfft(values - values.mean()))
    freqs   = np.fft.rfftfreq(n)
    mask    = (freqs > 0) & (freqs < 1.0 / min_period)
    if not mask.any():
        return min_period
    dom_freq = freqs[mask][np.argmax(fft_mag[mask])]
    period   = int(round(1.0 / dom_freq))
    # Need at least 2 full periods to fit STL reliably
    return period if 2 * period <= n else max(min_period, n // 8)


def stl_residual(values: np.ndarray, period: int | None = None) -> np.ndarray:
    """STL residual component. Falls back to linear detrend on failure."""
    from statsmodels.tsa.seasonal import STL
    if period is None:
        period = detect_period(values)
    period = max(4, period)
    try:
        return STL(values, period=period, robust=True).fit().resid
    except Exception:
        from scipy.signal import detrend
        return detrend(values)


# ── Mahalanobis scorer ────────────────────────────────────────────────────────

def mahalanobis_scores(
    feats: np.ndarray,
    n_components: int = 50,
    reg: float = 1e-4,
) -> np.ndarray:
    """Mahalanobis distance from global centroid in PCA-reduced space.

    Parameters
    ----------
    feats        : [N, D]  window-level features (e.g. mean-pooled patches)
    n_components : PCA target dims (D >> N requires reduction)
    reg          : regularization added to cov diagonal

    Returns
    -------
    scores : [N]  float32
    """
    N, D   = feats.shape
    n_comp = min(n_components, N - 1, D)
    if n_comp < 1:
        return np.zeros(N, dtype=np.float32)

    pca = PCA(n_components=n_comp, whiten=False)
    F   = pca.fit_transform(feats)              # [N, n_comp]

    mu      = F.mean(axis=0)
    cov     = np.cov(F, rowvar=False) if n_comp > 1 else np.array([[F.var()]])
    cov    += reg * np.eye(n_comp)
    inv_cov = np.linalg.inv(cov)

    diff   = F - mu                             # [N, n_comp]
    sq     = np.einsum("nd,de,ne->n", diff, inv_cov, diff)
    return np.sqrt(np.maximum(sq, 0.0)).astype(np.float32)


# ── Window scores → time-series vector ───────────────────────────────────────

def _stitch_window_scores(
    scores: np.ndarray,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Mean-aggregate per-window scalar scores into a T-length vector."""
    N = len(scores)
    T = step_size * (N - 1) + window_size
    out = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    for i, s in enumerate(scores):
        start = i * step_size
        out[start : start + window_size] += s
        cnt[start : start + window_size] += 1.0
    return (out / np.maximum(cnt, 1.0)).astype(np.float32)


# ── Main class ────────────────────────────────────────────────────────────────

class MAE_Mahalanobis(ViT4TS_MAE):
    """MAE + Mahalanobis anomaly scorer with optional STL residual rendering.

    Parameters
    ----------
    use_stl        : render STL residual instead of raw signal (default True)
    stl_period     : period for STL; None = auto-detect via FFT
    pca_components : PCA dims before Mahalanobis (default 50)
    cov_reg        : covariance regularization (default 1e-4)
    All other params forwarded to ViT4TS_MAE.
    """

    def __init__(
        self,
        *args,
        use_stl: bool = True,
        stl_period: int | None = None,
        pca_components: int = 50,
        cov_reg: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_stl        = use_stl
        self.stl_period     = stl_period
        self.pca_components = pca_components
        self.cov_reg        = cov_reg

    # Override predict_scores to inject STL before rendering
    def predict_scores(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values, timestamps = orion_to_internal(data)
        T_full   = len(values)
        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        if self.use_stl:
            period = self.stl_period or detect_period(values_proc)
            if self.verbose:
                print(f"  STL decompose  period={period} ...")
            resid = stl_residual(values_proc, period)
            lo, hi = resid.min(), resid.max()
            values_render = (resid - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(resid)
        else:
            values_render = values_proc

        with tempfile.TemporaryDirectory() as tmp:
            ok = draw_windowed_images(
                base_series_id="series",
                save_path=tmp,
                time_series=values_render,
                time_points=np.arange(len(values_render)),
                window_size=self.window_size,
                step_size=step_size,
                override=True,
                save_image=False,
                image_size=self.image_size,
                dpi=self.dpi,
                plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
            )
            if not ok:
                warnings.warn("draw_windowed_images failed.")
                return np.zeros(T_full, dtype=np.float32), timestamps

            scores_win = self._run_mahalanobis(tmp, "series")

        if scores_win is None or len(scores_win) == 0:
            return np.zeros(T_full, dtype=np.float32), timestamps

        raw_vec = _stitch_window_scores(scores_win, self.window_size, step_size)
        aligned = align_anomaly_vector(raw_vec, T_full, self.window_size, step_size, n_windows)
        return aligned, timestamps

    def _run_mahalanobis(self, results_dir: str, base_series_id: str):
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir,
            base_series_id=base_series_id,
            sample_size=None,
            no_anomaly=True,
            plot_type="line",
        )
        if len(dataset) == 0:
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        _, _, patch_embeds, _, _, _ = build_ordered_embeddings(
            self.model, loader, self.patch_size, self.device
        )
        # patch_embeds: [N, N_patches, D]  → mean-pool patches → [N, D]
        feats = patch_embeds.mean(dim=1).numpy()

        return mahalanobis_scores(feats, self.pca_components, self.cov_reg)
