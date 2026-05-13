"""ViT4TS Dual-Anchor: LTR (local) + GAP (global) combined anomaly detection.

New file — no existing files modified.

Motivation
----------
LTR k=5 detects local pattern changes well but misses drift anomalies
because the local reference shifts along with the signal.
GAP (Global Average Pooling) uses a fixed global median reference,
so it detects gradual distribution shifts that LTR misses.

Architecture
------------
Pass 1 : encode all windows → patch_embeds [L, N, D]
         extract gap_vectors [L, D] for FREE (just .mean(dim=1))

GAP path : cosine dissimilarity vs global median → gap_scores [L]
           direct window→time mapping (max overlap) → gap_score_1d

LTR path : per-window local comparison → anomaly_maps [L, H, W]
           stitch_anomaly_maps → ltr_score_1d

Combine  : α * norm(ltr) + (1-α) * norm(gap)
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
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils import orion_to_internal, intervals_from_indices
from models.model_utils import (
    harmonic_aggregation,
    stitch_anomaly_maps,
    align_anomaly_vector,
    compute_detection_intervals,
)
from models.model_utils_local_v2 import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _gap_scores_to_1d(gap_scores: np.ndarray, L: int,
                      step_size: int, window_size: int) -> np.ndarray:
    """Map window-level GAP scalar scores → 1D time series.

    Uses max-overlap aggregation: if two windows cover the same
    time step, keep the higher score (conservative for drift detection).

    Parameters
    ----------
    gap_scores : [L]  per-window scalar anomaly score (already normalised)
    L          : number of windows
    step_size  : stride between windows
    window_size: window width in time steps

    Returns
    -------
    np.ndarray  shape [step_size*(L-1) + window_size]
    """
    T_final = step_size * (L - 1) + window_size
    out = np.zeros(T_final, dtype=float)
    for i, score in enumerate(gap_scores):
        start = i * step_size
        end   = min(start + window_size, T_final)
        out[start:end] = np.maximum(out[start:end], score)
    return out


class ViT4TS_DualAnchor:
    """Dual-Anchor anomaly detector: LTR + GAP.

    Parameters
    ----------
    backbone   : vision encoder with encode_image() interface
    patch_size : int
    local_k    : int   LTR half-window (default 5)
    alpha      : float weight for LTR score; (1-alpha) for GAP score
    """

    def __init__(
        self,
        backbone,
        patch_size: int = 16,
        local_k: int = 5,
        min_ref: int = 5,
        alpha: float = 0.7,
        window_size: int = 224,
        window_step_ratio: float = 4.0,
        agg_percent: float = 0.25,
        device: str = "auto",
        batch_size: int = 32,
        image_size: tuple = (224, 224),
        dpi: int = 100,
        standardize: bool = True,
        alpha_detect: float = 0.01,
        smoothing_alpha: float = 1.0,
        verbose: bool = True,
    ):
        self.backbone         = backbone
        self.patch_size       = patch_size
        self.local_k          = local_k
        self.min_ref          = min_ref
        self.alpha            = alpha          # LTR weight
        self.window_size      = window_size
        self.window_step_ratio = window_step_ratio
        self.agg_percent      = agg_percent
        self.batch_size       = batch_size
        self.image_size       = image_size
        self.dpi              = dpi
        self.standardize      = standardize
        self.alpha_detect     = alpha_detect   # threshold for anomaly detection
        self.smoothing_alpha  = smoothing_alpha
        self.verbose          = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        if self.verbose:
            print(f"ViT4TS_DualAnchor | device={self.device} | "
                  f"LTR k={local_k} | alpha(LTR)={alpha:.1f} | alpha(GAP)={1-alpha:.1f}")

    # ------------------------------------------------------------------
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)
        return self.get_intervals(scores, timestamps, self.alpha_detect)

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        if self.verbose:
            print(f"  {T_full} pts | win={self.window_size} L={n_windows} "
                  f"α={self.alpha:.1f}(LTR)+{1-self.alpha:.1f}(GAP)")

        with tempfile.TemporaryDirectory() as tmp:
            time_pts    = np.arange(len(values_proc))
            plot_params = ("-", 1, "*", 0.1, "black", (0, 1) if self.standardize else None)

            success = draw_windowed_images(
                base_series_id="series", save_path=tmp,
                time_series=values_proc, time_points=time_pts,
                window_size=self.window_size, step_size=step_size,
                override=True, save_image=False,
                image_size=self.image_size, dpi=self.dpi,
                plot_params=plot_params,
            )
            if not success:
                warnings.warn("No windowed images generated.")
                return np.zeros(T_full), timestamps

            anomaly_scores = self._run_inference_dual(tmp, "series", step_size, n_windows)
            if anomaly_scores is None or len(anomaly_scores) == 0:
                return np.zeros(T_full), timestamps

        aligned = align_anomaly_vector(anomaly_scores, T_full,
                                       self.window_size, step_size, n_windows)
        return aligned, timestamps

    def get_intervals(self, scores, timestamps, alpha=None):
        if alpha is None:
            alpha = self.alpha_detect
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
        return intervals_from_indices(idx, timestamps, scores)

    # ------------------------------------------------------------------
    # Dual-Anchor inference
    # ------------------------------------------------------------------

    def _run_inference_dual(self, results_dir: str, base_id: str,
                            step_size: int, n_windows: int) -> Optional[np.ndarray]:
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            warnings.warn("Empty dataset.")
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # ---- Pass 1: encode all windows ----
        if self.verbose:
            print("  [DA] Pass 1: encoding...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = w = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        # ---- Point 1: GAP vectors — FREE from patch_embeds ----
        # patch_embeds: [L, N, D] → mean over patches → [L, D]
        gap_vectors = patch_embeds.mean(dim=1)  # [L, D]

        # ---- Point 2: GAP anomaly scores ----
        median_gap = torch.median(gap_vectors, dim=0).values  # [D]
        gap_scores_raw = []
        with torch.no_grad():
            for i in range(L):
                sim = F.cosine_similarity(
                    gap_vectors[i].unsqueeze(0),
                    median_gap.unsqueeze(0),
                )
                gap_scores_raw.append((1.0 - sim.item()))
        gap_scores_raw = np.array(gap_scores_raw)           # [L]
        gap_scores_norm = _normalize_01(gap_scores_raw)     # [L] in [0,1]

        # ---- Point 3: GAP [L] → 1D time series (direct, max-overlap) ----
        gap_score_1d = _gap_scores_to_1d(
            gap_scores_norm, L, step_size, self.window_size
        )

        # ---- Pass 2: LTR patch-level comparison ----
        if self.verbose:
            print(f"  [DA] Pass 2: LTR k={self.local_k} + GAP (L={L})...")
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

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                         nan=0., posinf=0., neginf=0.)
                score = F.interpolate(score.unsqueeze(1), size=(h, w),
                                      mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()          # [L, H, W]
        ltr_score_1d = stitch_anomaly_maps(maps_arr, self.window_step_ratio,
                                           self.agg_percent)          # [T_final]

        # ---- Point 4: normalize + combine ----
        ltr_norm = _normalize_01(ltr_score_1d)

        # align lengths (gap uses window_size, ltr uses stitch output)
        T_final = len(ltr_norm)
        if len(gap_score_1d) > T_final:
            gap_score_1d = gap_score_1d[:T_final]
        elif len(gap_score_1d) < T_final:
            gap_score_1d = np.pad(gap_score_1d, (0, T_final - len(gap_score_1d)))

        final_score = self.alpha * ltr_norm + (1 - self.alpha) * gap_score_1d
        return final_score
