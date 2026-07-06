"""ViT4TS: LTR × MAE Reconstruction Error (RESTAD-style product fusion).

score_LTR   = Local Temporal Reference k=5  (feature-space, temporal)
score_recon = MAE pixel-space reconstruction error (spatial, within-image)

final_score = normalize(score_LTR) × normalize(score_recon)

Why product instead of weighted sum:
  - Sum: one noisy signal contaminates the other → FP on non-stationary signals
  - Product: both detectors must agree → FP suppressed
  - No β hyperparameter needed

No existing files modified.
"""

import os
import sys
import tempfile
import warnings
from typing import Optional, Tuple

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


class ViT4TS_LTR_Recon:
    """LTR k=5 × MAE Reconstruction Error — product fusion.

    Parameters
    ----------
    backbone  : MAE_AD   feature extractor for LTR
    recon     : MAE_Recon  pixel-space reconstruction scorer
    local_k   : int    LTR half-window (default 5)
    """

    def __init__(
        self,
        backbone,
        recon,
        local_k:    int   = 5,
        min_ref:    int   = 5,
        patch_size: int   = 16,
        window_size: int  = 224,
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
        self.backbone          = backbone
        self.recon             = recon
        self.local_k           = local_k
        self.min_ref           = min_ref
        self.patch_size        = patch_size
        self.window_size       = window_size
        self.window_step_ratio = window_step_ratio
        self.agg_percent       = agg_percent
        self.batch_size        = batch_size
        self.image_size        = image_size
        self.dpi               = dpi
        self.standardize       = standardize
        self.alpha_detect      = alpha_detect
        self.smoothing_alpha   = smoothing_alpha
        self.verbose           = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        if self.verbose:
            print(f"ViT4TS_LTR_Recon | device={self.device} | "
                  f"LTR k={local_k} × Recon n_iter={recon.n_iter} | fusion=product")

    # ------------------------------------------------------------------
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=self.alpha_detect)
        return intervals_from_indices(idx, timestamps, scores)

    def predict_scores(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        if self.verbose:
            print(f"  {T_full} pts | win={self.window_size} L={n_windows} | LTR×Recon")

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

            score_ltr, score_recon = self._compute_both(
                tmp, "series", step_size, n_windows, T_full
            )

        if score_ltr is None:
            return np.zeros(T_full), timestamps

        ltr_norm   = _normalize_01(score_ltr)
        recon_norm = _normalize_01(score_recon)

        # Align lengths
        T_out = min(len(ltr_norm), len(recon_norm), T_full)
        ltr_norm   = ltr_norm[:T_out]
        recon_norm = recon_norm[:T_out]

        # Product fusion — both must agree
        combined = ltr_norm * recon_norm

        if len(combined) < T_full:
            combined = np.pad(combined, (0, T_full - len(combined)))

        return combined, timestamps

    # ------------------------------------------------------------------
    def _compute_both(
        self, results_dir: str, base_id: str,
        step_size: int, n_windows: int, T_full: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            warnings.warn("Empty dataset.")
            return None, None

        # ---- Pass 1: LTR only ----
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.verbose:
            print("  [LTR] Pass 1: encoding...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device,
        )

        L  = large_embeds.shape[0]
        h  = w = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        if self.verbose:
            print(f"  [LTR] Pass 2: k={self.local_k} (L={L})...")
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

                m_l = compute_dissimilarity_with_ref(
                    large_embeds[i].unsqueeze(0).to(self.device), l_ref.to(self.device))
                m_m = compute_dissimilarity_with_ref(
                    mid_embeds[i].unsqueeze(0).to(self.device),   m_ref.to(self.device))
                m_p = compute_dissimilarity_with_ref(
                    patch_embeds[i].unsqueeze(0).to(self.device), p_ref.to(self.device))

                m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(self.device)
                m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(self.device)
                m_p = m_p.reshape((1, ph, pw)).to(self.device)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                          nan=0., posinf=0., neginf=0.)
                score = F.interpolate(score.unsqueeze(1), size=(h, w),
                                      mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        ltr_1d   = stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)
        score_ltr = align_anomaly_vector(ltr_1d, T_full, self.window_size, step_size, n_windows)

        # ---- Pass 2: MAE reconstruction ----
        if self.verbose:
            print(f"  [Recon] scoring (n_iter={self.recon.n_iter})...")
        loader2    = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        recon_raw  = []
        recon_wids = []
        for batch in loader2:
            imgs = batch["img"]
            wids = batch["window_id"]
            errs = self.recon.score_batch(imgs)
            recon_raw.extend(errs.tolist())
            recon_wids.extend(wids.tolist())

        order        = sorted(range(len(recon_wids)), key=lambda i: recon_wids[i])
        recon_scores = np.array([recon_raw[i] for i in order])

        recon_1d    = np.zeros(step_size * (L - 1) + self.window_size)
        for i, sc in enumerate(recon_scores):
            s = i * step_size
            e = min(s + self.window_size, len(recon_1d))
            recon_1d[s:e] = np.maximum(recon_1d[s:e], sc)
        score_recon = align_anomaly_vector(recon_1d, T_full, self.window_size, step_size, n_windows)

        return score_ltr, score_recon
