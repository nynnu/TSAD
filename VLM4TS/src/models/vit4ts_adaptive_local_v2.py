"""ViT4TS with Adaptive Window + Stationarity-based Reference Strategy.

v2 changes from vit4ts_adaptive_local.py
-----------------------------------------
- Stationarity (CV) measured per-series to choose reference strategy:
    cv > 0.3  → LTR k=5   (non-stationary: local temporal reference)
    cv ≤ 0.3  → early_ref  (stable: first 20% windows as reference)

- early_ref rationale:
    For stable signals (T-1, T-2, etc.), the anomaly is a distribution shift
    in the latter part. Global median mixes before+after states → both get
    low dissimilarity. Using only the first 20% as reference preserves the
    "what normal looked like at the start" baseline.

No existing files modified.
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
# Stationarity measurement
# ---------------------------------------------------------------------------

def compute_cv(series: np.ndarray, n_segments: int = 10) -> float:
    """Coefficient of variation of per-segment variance.

    High CV  → non-stationary (variance changes across time)
    Low CV   → stable (variance is consistent across time)
    """
    segs = np.array_split(series, n_segments)
    variances = np.array([np.var(s) for s in segs])
    return float(np.std(variances) / (np.mean(variances) + 1e-8))


def determine_ref_strategy(series: np.ndarray, cv_threshold: float = 0.3) -> str:
    """Choose reference strategy based on signal stationarity.

    Returns
    -------
    'ltr_k5'    : non-stationary → Local Temporal Reference k=5
    'early_ref' : stable         → first 20% windows as reference
    """
    cv = compute_cv(series)
    if cv > cv_threshold:
        return "ltr_k5"
    else:
        return "early_ref"


# ---------------------------------------------------------------------------
# Window-size determination (identical to v1)
# ---------------------------------------------------------------------------

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

    total_power = np.sum(fft_mag ** 2) + 1e-12
    cutoff_idx  = max(1, int(len(fft_mag) * 0.75))
    high_power  = np.sum(fft_mag[cutoff_idx:] ** 2)
    high_freq_ratio = high_power / total_power

    if high_freq_ratio > high_freq_threshold:
        return min(candidates)

    return estimate_window_size(values, candidates=candidates,
                                default=default, snr_threshold=snr_threshold)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ViT4TS_AdaptiveLocalV2:
    """Adaptive Window + Stationarity-based Reference Strategy.

    Reference selection per series:
      non-stationary (cv > cv_threshold) → LTR k=5
      stable         (cv ≤ cv_threshold) → early_ref (first early_frac of windows)
    """

    def __init__(
        self,
        backbone,
        patch_size: int = 16,
        min_ref: int = 5,
        ltr_k: int = 5,
        early_frac: float = 0.2,
        cv_threshold: float = 0.3,
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
        self.min_ref             = min_ref
        self.ltr_k               = ltr_k
        self.early_frac          = early_frac
        self.cv_threshold        = cv_threshold
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

        if self.verbose:
            print(f"ViT4TS_AdaptiveLocalV2 | device={self.device} | "
                  f"ltr_k={ltr_k} early_frac={early_frac} cv_thr={cv_threshold}")

    # ------------------------------------------------------------------
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)
        return self.get_intervals(scores, timestamps, self.alpha)

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        # Step 1: standardize (NO EWMA yet)
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

        # Step 3: determine reference strategy based on stationarity
        cv       = compute_cv(values_proc)
        strategy = "ltr_k5" if cv > self.cv_threshold else "early_ref"

        if self.verbose:
            print(f"  {T_full} pts | win={window_size} step={step_size} "
                  f"L={n_windows} | cv={cv:.3f} → {strategy}")

        # Step 4: EWMA after window/strategy decision
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

            anomaly_scores = self._run_inference(tmp, "series", strategy, n_windows)

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
    # 2-pass inference with strategy selection
    # ------------------------------------------------------------------

    def _run_inference(self, results_dir: str, base_id: str,
                       strategy: str, n_windows: int) -> Optional[np.ndarray]:
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            warnings.warn("Empty dataset.")
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.verbose:
            print(f"  [V2] Pass 1: encoding ({strategy})...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = w = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        # Pre-compute early_ref once (used for all windows if strategy=early_ref)
        if strategy == "early_ref":
            n_early = max(self.min_ref, int(L * self.early_frac))
            n_early = min(n_early, L - 1)
            if self.verbose:
                print(f"  [V2] Pass 2: early_ref (first {n_early}/{L} windows)...")
            l_early = torch.median(large_embeds[:n_early], dim=0).values
            m_early = torch.median(mid_embeds[:n_early],   dim=0).values
            p_early = torch.median(patch_embeds[:n_early], dim=0).values
        else:
            if self.verbose:
                print(f"  [V2] Pass 2: LTR k={self.ltr_k} (L={L})...")

        anomaly_maps = []

        with torch.no_grad():
            for i in range(L):
                if strategy == "early_ref":
                    l_ref = l_early.to(self.device)
                    m_ref = m_early.to(self.device)
                    p_ref = p_early.to(self.device)
                elif strategy == "ltr_k5":
                    l_ref, _ = get_local_reference(large_embeds, i, self.ltr_k, self.min_ref)
                    m_ref, _ = get_local_reference(mid_embeds,   i, self.ltr_k, self.min_ref)
                    p_ref, _ = get_local_reference(patch_embeds, i, self.ltr_k, self.min_ref)
                    l_ref = l_ref.to(self.device)
                    m_ref = m_ref.to(self.device)
                    p_ref = p_ref.to(self.device)
                else:  # global fallback
                    idx_all = list(range(0, i)) + list(range(i + 1, L))
                    l_ref = torch.median(large_embeds[idx_all], dim=0).values.to(self.device)
                    m_ref = torch.median(mid_embeds[idx_all],   dim=0).values.to(self.device)
                    p_ref = torch.median(patch_embeds[idx_all], dim=0).values.to(self.device)

                l_tok = large_embeds[i].unsqueeze(0).to(self.device)
                m_tok = mid_embeds[i].unsqueeze(0).to(self.device)
                p_tok = patch_embeds[i].unsqueeze(0).to(self.device)

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
