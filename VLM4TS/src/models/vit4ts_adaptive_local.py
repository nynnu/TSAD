"""ViT4TS with Adaptive Window + Local Temporal Reference (LTR).

New file — no existing files modified.

Key differences from ViT4TS_Local
-----------------------------------
1. window_size is determined per-series via determine_window_size():
   - high_freq_ratio > 0.4  →  56  (short-anomaly mode)
   - SNR >= 3.0 (clear period) →  nearest {56, 112, 224} to dominant period
   - SNR < 3.0  (no period)   →  224 (default)

2. Ordering fix: window_size is determined on the preprocessed-but-NOT-smoothed
   signal so that high-frequency components are not yet removed.
   EWMA is applied AFTER window_size selection.

3. LTR k is adapted to the number of windows L:
   k = max(3, min(10, L // 10))
   Falls back to global if L < 15.
"""

import os
import sys
import tempfile
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from preprocessing.preprocess import preprocess_time_series, draw_windowed_images, apply_ewma
from preprocessing.adaptive_window import estimate_window_size
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils import orion_to_internal, intervals_from_indices
from models.model_utils import (
    harmonic_aggregation,
    stitch_anomaly_maps,
    align_anomaly_vector,
    compute_detection_intervals,
)
from models.model_utils_local import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)

WINDOW_CANDIDATES = (56, 112, 224)


# ---------------------------------------------------------------------------
# Window-size determination (raw signal, before EWMA)
# ---------------------------------------------------------------------------

def determine_window_size(
    values: np.ndarray,
    candidates: tuple = WINDOW_CANDIDATES,
    default: int = 224,
    snr_threshold: float = 3.0,
    high_freq_threshold: float = 0.4,
) -> int:
    """Select window size based on signal characteristics.

    Priority
    --------
    1. high_freq_ratio > high_freq_threshold  →  smallest candidate (short-anomaly mode)
    2. clear dominant period (SNR >= snr_threshold)  →  nearest candidate to period
    3. no clear period  →  default (224)

    Parameters
    ----------
    values : np.ndarray  Raw (preprocessed, pre-EWMA) time series.
    candidates : tuple   Candidate window sizes in ascending order.
    default : int        Fallback window size.
    snr_threshold : float
    high_freq_threshold : float  high_freq_ratio cutoff for short-anomaly mode.
    """
    n = len(values)
    if n < min(candidates):
        return default

    x = values - values.mean()
    fft_mag = np.abs(np.fft.rfft(x))
    fft_mag[0] = 0.0  # zero DC

    if len(fft_mag) < 2:
        return default

    total_power = np.sum(fft_mag ** 2) + 1e-12
    cutoff_idx  = max(1, int(len(fft_mag) * 0.75))   # top-25% frequency band
    high_power  = np.sum(fft_mag[cutoff_idx:] ** 2)
    high_freq_ratio = high_power / total_power

    # Rule 1: short-anomaly mode
    if high_freq_ratio > high_freq_threshold:
        return min(candidates)

    # Rule 2 & 3: delegate to existing FFT-period logic
    return estimate_window_size(values, candidates=candidates,
                                default=default, snr_threshold=snr_threshold)


def adaptive_k(L: int, min_k: int = 3, max_k: int = 10, fallback_threshold: int = 15) -> int:
    """Compute LTR k proportional to number of windows L."""
    if L < fallback_threshold:
        return 0   # global fallback
    return max(min_k, min(max_k, L // 10))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ViT4TS_AdaptiveLocal:
    """Stage-1 detector with Adaptive Window + Local Temporal Reference.

    Parameters
    ----------
    backbone : nn.Module
        Vision encoder with encode_image() interface.
    patch_size : int
    min_ref : int
        Minimum neighbours for LTR; falls back to global if below.
    smoothing_alpha : float
        EWMA alpha applied AFTER window-size determination.
        Default 1.0 = no smoothing (recommended when using adaptive window).
    """

    def __init__(
        self,
        backbone,
        patch_size: int = 16,
        min_ref: int = 5,
        window_step_ratio: float = 4.0,
        agg_percent: float = 0.25,
        device: str = "auto",
        batch_size: int = 20,
        image_size: tuple = (224, 224),
        dpi: int = 100,
        standardize: bool = True,
        alpha: float = 0.01,
        smoothing_alpha: float = 1.0,
        window_candidates: tuple = WINDOW_CANDIDATES,
        snr_threshold: float = 3.0,
        high_freq_threshold: float = 0.4,
        verbose: bool = True,
    ):
        self.backbone           = backbone
        self.patch_size         = patch_size
        self.min_ref            = min_ref
        self.window_step_ratio  = window_step_ratio
        self.agg_percent        = agg_percent
        self.batch_size         = batch_size
        self.image_size         = image_size
        self.dpi                = dpi
        self.standardize        = standardize
        self.alpha              = alpha
        self.smoothing_alpha    = smoothing_alpha
        self.window_candidates  = window_candidates
        self.snr_threshold      = snr_threshold
        self.high_freq_threshold = high_freq_threshold
        self.verbose            = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        if self.verbose:
            print(f"ViT4TS_AdaptiveLocal initialized | device={self.device} | "
                  f"candidates={window_candidates}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)
        return self.get_intervals(scores, timestamps, self.alpha)

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        # Step 1: preprocess (standardize) — NO EWMA yet
        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)

        # Step 2: determine window size on the raw preprocessed signal
        window_size = determine_window_size(
            values_proc,
            candidates=self.window_candidates,
            snr_threshold=self.snr_threshold,
            high_freq_threshold=self.high_freq_threshold,
        )
        step_size = max(1, int(window_size / self.window_step_ratio))
        n_windows = max(1, int((T_full - window_size) / step_size) + 1)
        local_k   = adaptive_k(n_windows)

        if self.verbose:
            print(f"  {T_full} pts | window={window_size} step={step_size} "
                  f"L={n_windows} k={local_k if local_k > 0 else 'global'}")

        # Step 3: apply EWMA AFTER window-size decision
        values_smoothed = apply_ewma(values_proc, self.smoothing_alpha)

        with tempfile.TemporaryDirectory() as tmp:
            time_pts    = np.arange(len(values_smoothed))
            plot_params = ("-", 1, "*", 0.1, "black", (0, 1) if self.standardize else None)

            success = draw_windowed_images(
                base_series_id="series", save_path=tmp,
                time_series=values_smoothed, time_points=time_pts,
                window_size=window_size, step_size=step_size,
                override=True, save_image=False,
                image_size=self.image_size, dpi=self.dpi,
                plot_params=plot_params,
            )
            if not success:
                warnings.warn("No windowed images generated.")
                return np.zeros(T_full), timestamps

            anomaly_scores = self._run_inference_local(tmp, "series", local_k)

            if anomaly_scores is None or len(anomaly_scores) == 0:
                return np.zeros(T_full), timestamps

        aligned = align_anomaly_vector(anomaly_scores, T_full, window_size, step_size, n_windows)
        return aligned, timestamps

    def get_intervals(self, scores, timestamps, alpha=None):
        if alpha is None:
            alpha = self.alpha
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
        return intervals_from_indices(idx, timestamps, scores)

    # ------------------------------------------------------------------
    # LTR 2-pass inference
    # ------------------------------------------------------------------

    def _run_inference_local(self, results_dir: str, base_id: str, local_k: int) -> Optional[np.ndarray]:
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            warnings.warn("Empty dataset.")
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.verbose:
            print("  [AdaptiveLTR] Pass 1: encoding all windows...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = w = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        mode = f"local k={local_k}" if local_k > 0 else "global"
        if self.verbose:
            print(f"  [AdaptiveLTR] Pass 2: anomaly maps ({mode}, L={L})...")

        anomaly_maps = []

        with torch.no_grad():
            for i in range(L):
                if local_k > 0:
                    l_ref, _ = get_local_reference(large_embeds, i, local_k, self.min_ref)
                    m_ref, _ = get_local_reference(mid_embeds,   i, local_k, self.min_ref)
                    p_ref, _ = get_local_reference(patch_embeds, i, local_k, self.min_ref)
                else:
                    idx_all = list(range(0, i)) + list(range(i + 1, L))
                    l_ref = torch.median(large_embeds[idx_all], dim=0).values
                    m_ref = torch.median(mid_embeds[idx_all],   dim=0).values
                    p_ref = torch.median(patch_embeds[idx_all], dim=0).values

                l_tok = large_embeds[i].unsqueeze(0).to(self.device)
                m_tok = mid_embeds[i].unsqueeze(0).to(self.device)
                p_tok = patch_embeds[i].unsqueeze(0).to(self.device)
                l_ref = l_ref.to(self.device)
                m_ref = m_ref.to(self.device)
                p_ref = p_ref.to(self.device)

                m_l = compute_dissimilarity_with_ref(l_tok, l_ref)
                m_m = compute_dissimilarity_with_ref(m_tok, m_ref)
                m_p = compute_dissimilarity_with_ref(p_tok, p_ref)

                m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(self.device)
                m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(self.device)
                m_p = m_p.reshape((1, ph, pw)).to(self.device)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.)
                score = F.interpolate(score.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)
