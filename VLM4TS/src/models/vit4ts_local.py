"""ViT4TS with Local Temporal Reference (LTR).

Drop-in replacement for ViT4TS / ViT4TS_DINO / ViT4TS_MAE.
Backbone is injected at construction time — no existing files are modified.

Instead of comparing each window's patches to a single global-median reference
built from ALL windows, this class compares to the median of the k temporally
nearest windows (excluding self), reducing false positives on non-stationary series.

Computational cost vs global reference
---------------------------------------
Global:  O(L²·N·D)    (each of L windows compared to median over L)
Local k: O(L·2k·N·D)  (each window compared to median over 2k neighbors)
When k < L/2, local is CHEAPER.  Typical NAB-AWS (L≈69, k=10): ~29% of global.
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
from models.model_utils_local import (
    build_ordered_embeddings,
    get_local_reference,
    compute_dissimilarity_with_ref,
)


class ViT4TS_Local:
    """Stage-1 anomaly detector using Local Temporal Reference.

    Parameters
    ----------
    backbone : nn.Module
        Any vision encoder with encode_image() interface
        (CLIP_AD, DINO_AD, MAE_AD, ConvNeXtV2_AD).
    patch_size : int
        Patch size matching the backbone.
    local_k : int
        Half-window for local reference. Set to 0 for global (original behaviour).
    min_ref : int
        Minimum neighbours required; falls back to global if below this.
    smoothing_alpha : float
        EWMA smoothing applied before rendering (1.0 = no smoothing).
    """

    def __init__(
        self,
        backbone,
        patch_size: int = 16,
        local_k: int = 10,
        min_ref: int = 5,
        window_size: int = 224,
        window_step_ratio: float = 4.0,
        agg_percent: float = 0.25,
        device: str = "auto",
        batch_size: int = 20,
        image_size: tuple = (224, 224),
        dpi: int = 100,
        standardize: bool = True,
        alpha: float = 0.01,
        smoothing_alpha: float = 1.0,
        verbose: bool = True,
    ):
        self.backbone         = backbone
        self.patch_size       = patch_size
        self.local_k          = local_k
        self.min_ref          = min_ref
        self.window_size      = window_size
        self.window_step_ratio = window_step_ratio
        self.agg_percent      = agg_percent
        self.batch_size       = batch_size
        self.image_size       = image_size
        self.dpi              = dpi
        self.standardize      = standardize
        self.alpha            = alpha
        self.smoothing_alpha  = smoothing_alpha
        self.verbose          = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        mode = f"local k={local_k}" if local_k > 0 else "global (fallback)"
        if self.verbose:
            print(f"ViT4TS_Local initialized | device={self.device} | ref={mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)
        return self.get_intervals(scores, timestamps, self.alpha)

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        if self.verbose:
            print(f"  {T_full} points...")

        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        with tempfile.TemporaryDirectory() as tmp:
            step_size  = int(self.window_size / self.window_step_ratio)
            time_pts   = np.arange(len(values_proc))
            n_windows  = int((T_full - self.window_size) / step_size) + 1
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

            anomaly_scores = self._run_inference_local(tmp, "series")

            if anomaly_scores is None or len(anomaly_scores) == 0:
                return np.zeros(T_full), timestamps

        aligned = align_anomaly_vector(anomaly_scores, T_full, self.window_size, step_size, n_windows)
        return aligned, timestamps

    def get_intervals(self, scores, timestamps, alpha=None):
        if alpha is None:
            alpha = self.alpha
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
        return intervals_from_indices(idx, timestamps, scores)

    # ------------------------------------------------------------------
    # Local-reference inference (2-pass)
    # ------------------------------------------------------------------

    def _run_inference_local(self, results_dir: str, base_id: str) -> Optional[np.ndarray]:
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            warnings.warn("Empty dataset.")
            return None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # --- Pass 1: collect all embeddings ---
        if self.verbose:
            print("  [LTR] Pass 1: encoding all windows...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, window_ids) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device
        )

        L  = large_embeds.shape[0]
        h  = w = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        if self.verbose:
            mode = f"local k={self.local_k}" if self.local_k > 0 else "global"
            print(f"  [LTR] Pass 2: computing anomaly maps ({mode}, L={L})...")

        # --- Pass 2: window-by-window local comparison ---
        anomaly_maps = []

        with torch.no_grad():
            for i in range(L):
                if self.local_k > 0:
                    l_ref, _ = get_local_reference(large_embeds, i, self.local_k, self.min_ref)
                    m_ref, _ = get_local_reference(mid_embeds,   i, self.local_k, self.min_ref)
                    p_ref, _ = get_local_reference(patch_embeds, i, self.local_k, self.min_ref)
                else:
                    # global: median of all windows except self
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

                m_l = compute_dissimilarity_with_ref(l_tok, l_ref)   # [1, N_L]
                m_m = compute_dissimilarity_with_ref(m_tok, m_ref)   # [1, N_M]
                m_p = compute_dissimilarity_with_ref(p_tok, p_ref)   # [1, N]

                m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(self.device)
                m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(self.device)
                m_p = m_p.reshape((1, ph, pw)).to(self.device)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0, nan=0., posinf=0., neginf=0.)
                score = F.interpolate(score.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())  # [H, W]

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()   # [L, H, W]
        return stitch_anomaly_maps(maps_arr, self.window_step_ratio, self.agg_percent)
