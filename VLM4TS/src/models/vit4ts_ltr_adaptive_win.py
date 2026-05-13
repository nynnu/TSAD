"""ViT4TS with fixed LTR k + Adaptive Window size.

New file — no existing files modified.

Difference from ViT4TS_Local
------------------------------
- window_size is determined per-series via determine_window_size() (FFT + high_freq)
- LTR k is FIXED (default k=5), not adaptive
- EWMA applied AFTER window-size determination (ordering fix)

This isolates exactly one variable vs ViT4TS_Local(k=5, fixed window):
  ViT4TS_Local(k=5)        : LTR k=5 + Fixed 224
  ViT4TS_LTR_AdaptiveWin   : LTR k=5 + Adaptive window   ← one variable added
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


def determine_window_size(
    values: np.ndarray,
    candidates: tuple = WINDOW_CANDIDATES,
    default: int = 224,
    snr_threshold: float = 3.0,
    high_freq_threshold: float = 0.4,
) -> int:
    n = len(values)
    if n < min(candidates):
        return default

    x = values - values.mean()
    fft_mag = np.abs(np.fft.rfft(x))
    fft_mag[0] = 0.0

    if len(fft_mag) < 2:
        return default

    total_power    = np.sum(fft_mag ** 2) + 1e-12
    cutoff_idx     = max(1, int(len(fft_mag) * 0.75))
    high_freq_ratio = np.sum(fft_mag[cutoff_idx:] ** 2) / total_power

    if high_freq_ratio > high_freq_threshold:
        return min(candidates)

    return estimate_window_size(values, candidates=candidates,
                                default=default, snr_threshold=snr_threshold)


class ViT4TS_LTR_AdaptiveWin:
    """LTR with fixed k + per-series adaptive window size.

    Parameters
    ----------
    backbone  : vision encoder with encode_image() interface
    patch_size: int
    local_k   : int   Fixed LTR k (default 5). Set 0 for global.
    min_ref   : int   Minimum neighbours before global fallback.
    """

    def __init__(
        self,
        backbone,
        patch_size: int = 16,
        local_k: int = 5,
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
        self.backbone            = backbone
        self.patch_size          = patch_size
        self.local_k             = local_k
        self.min_ref             = min_ref
        self.window_step_ratio   = window_step_ratio
        self.agg_percent         = agg_percent
        self.batch_size          = batch_size
        self.image_size          = image_size
        self.dpi                 = dpi
        self.standardize         = standardize
        self.alpha               = alpha
        self.smoothing_alpha     = smoothing_alpha
        self.window_candidates   = window_candidates
        self.snr_threshold       = snr_threshold
        self.high_freq_threshold = high_freq_threshold
        self.verbose             = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        ref_mode = f"LTR k={local_k}" if local_k > 0 else "global"
        if self.verbose:
            print(f"ViT4TS_LTR_AdaptiveWin | device={self.device} | "
                  f"ref={ref_mode} | candidates={window_candidates}")

    # ------------------------------------------------------------------
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)
        return self.get_intervals(scores, timestamps, self.alpha)

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        # Step 1: standardize — NO EWMA yet
        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)

        # Step 2: determine window size on raw preprocessed signal
        window_size = determine_window_size(
            values_proc,
            candidates=self.window_candidates,
            snr_threshold=self.snr_threshold,
            high_freq_threshold=self.high_freq_threshold,
        )
        step_size = max(1, int(window_size / self.window_step_ratio))
        n_windows = max(1, int((T_full - window_size) / step_size) + 1)

        if self.verbose:
            print(f"  {T_full} pts | win={window_size} step={step_size} "
                  f"L={n_windows} k={self.local_k if self.local_k > 0 else 'global'}")

        # Step 3: EWMA after window-size decision
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

            anomaly_scores = self._run_inference_ltr(tmp, "series")

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
    def _run_inference_ltr(self, results_dir: str, base_id: str) -> Optional[np.ndarray]:
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            warnings.warn("Empty dataset.")
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.verbose:
            print("  [LTR-AW] Pass 1: encoding all windows...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = w = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        mode = f"LTR k={self.local_k}" if self.local_k > 0 else "global"
        if self.verbose:
            print(f"  [LTR-AW] Pass 2: {mode} (L={L})...")

        anomaly_maps = []

        with torch.no_grad():
            for i in range(L):
                if self.local_k > 0:
                    l_ref, _ = get_local_reference(large_embeds, i, self.local_k, self.min_ref)
                    m_ref, _ = get_local_reference(mid_embeds,   i, self.local_k, self.min_ref)
                    p_ref, _ = get_local_reference(patch_embeds, i, self.local_k, self.min_ref)
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
