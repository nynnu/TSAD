"""ViT4TS_ConvNeXt: Stage-1 anomaly detector using ConvNeXt V2."""

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
from models.convnext_vision import ConvNeXtV2_AD
from models.model_utils import (
    build_memory,
    compute_patch_dissimilarity,
    harmonic_aggregation,
    stitch_anomaly_maps,
    align_anomaly_vector,
    compute_detection_intervals,
)


class ViT4TS_ConvNeXt:
    """Zero-shot Stage-1 anomaly detector using ConvNeXt V2 Base.

    Stage-2 feature map (14×14 = 196 spatial tokens, virtual patch_size=16)
    is used as patch tokens — identical downstream pipeline to ViT4TS.
    """

    def __init__(
        self,
        window_size: int = 224,
        window_step_ratio: float = 4.0,
        agg_percent: float = 0.25,
        patch_size: int = 16,          # virtual: 224 // 14 = 16
        model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        device: str = "auto",
        batch_size: int = 20,
        image_size: tuple = (224, 224),
        dpi: int = 100,
        standardize: bool = True,
        alpha: float = 0.01,
        verbose: bool = True,
        smoothing_alpha: float = 1.0,
    ):
        self.window_size = window_size
        self.window_step_ratio = window_step_ratio
        self.agg_percent = agg_percent
        self.patch_size = patch_size
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.dpi = dpi
        self.standardize = standardize
        self.alpha = alpha
        self.verbose = verbose
        self.smoothing_alpha = smoothing_alpha

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.verbose:
            print(f"ViT4TS_ConvNeXt initialized with device: {self.device}")

        self.model = ConvNeXtV2_AD(
            model_name=self.model_name, device=self.device, image_size=image_size
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        aligned_scores, timestamps = self.predict_scores(data)
        return self.get_intervals(aligned_scores, timestamps, self.alpha)

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        if self.verbose:
            print(f"Processing time series with {T_full} points...")

        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        with tempfile.TemporaryDirectory() as temp_dir:
            base_series_id = "series"
            step_size = int(self.window_size / self.window_step_ratio)
            time_points = np.arange(len(values_proc))
            n_windows = int((T_full - self.window_size) / step_size) + 1

            plot_params = ("-", 1, "*", 0.1, "black", (0, 1) if self.standardize else None)

            if self.verbose:
                print(f"Generating windowed visualizations (window_size={self.window_size}, step_size={step_size})...")

            success = draw_windowed_images(
                base_series_id=base_series_id,
                save_path=temp_dir,
                time_series=values_proc,
                time_points=time_points,
                window_size=self.window_size,
                step_size=step_size,
                override=True,
                save_image=False,
                image_size=self.image_size,
                dpi=self.dpi,
                plot_params=plot_params,
            )

            if not success:
                warnings.warn("Failed to generate windowed images.")
                return np.zeros(T_full), timestamps

            if self.verbose:
                print("Running ConvNeXt V2 inference...")

            anomaly_scores = self._run_inference(temp_dir, base_series_id)

            if anomaly_scores is None or len(anomaly_scores) == 0:
                warnings.warn("No anomaly scores generated.")
                return np.zeros(T_full), timestamps

        aligned_scores = align_anomaly_vector(
            anomaly_scores, T_full, self.window_size, step_size, n_windows
        )
        return aligned_scores, timestamps

    def get_intervals(self, scores: np.ndarray, timestamps: np.ndarray, alpha: float = None) -> pd.DataFrame:
        if alpha is None:
            alpha = self.alpha

        if self.verbose:
            print(f"Converting anomaly scores to intervals (alpha={alpha})...")

        interval_indices, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
        intervals = intervals_from_indices(
            interval_indices=interval_indices,
            timestamps=timestamps,
            scores=scores,
        )

        if self.verbose:
            print(f"Detected {len(intervals)} anomaly intervals")

        return intervals

    # ------------------------------------------------------------------
    # Internal inference
    # ------------------------------------------------------------------

    def _run_inference(self, results_dir: str, base_series_id: str) -> Optional[np.ndarray]:
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir,
            base_series_id=base_series_id,
            sample_size=None,
            no_anomaly=True,
            plot_type="line",
        )

        if len(dataset) == 0:
            warnings.warn("No windowed images found.")
            return None

        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        large_memory_normal, mid_memory_normal, patch_memory_normal = build_memory(
            self.model, test_dataloader, self.patch_size, self.device
        )

        results = {"window_id": [], "anomaly_maps": []}

        with torch.no_grad():
            for items in test_dataloader:
                images = items["img"].to(self.device)
                cls_names = items["cls_name"]
                window_ids = items["window_id"]

                results["window_id"].extend(window_ids.tolist())

                b, _, h, w = images.shape

                (
                    large_scale_tokens,
                    mid_scale_tokens,
                    patch_tokens,
                    _,
                    large_scale,
                    mid_scale,
                ) = self.model.encode_image(images, self.patch_size)

                m_l = compute_patch_dissimilarity(large_memory_normal, large_scale_tokens, cls_names)
                m_m = compute_patch_dissimilarity(mid_memory_normal, mid_scale_tokens, cls_names)
                m_p = compute_patch_dissimilarity(patch_memory_normal, patch_tokens, cls_names)

                m_l = harmonic_aggregation(
                    (b, h // self.patch_size, w // self.patch_size), m_l, large_scale
                ).to(self.device)
                m_m = harmonic_aggregation(
                    (b, h // self.patch_size, w // self.patch_size), m_m, mid_scale
                ).to(self.device)
                m_p = m_p.reshape((b, h // self.patch_size, w // self.patch_size)).to(self.device)

                score = torch.nan_to_num(
                    (m_l + m_m + m_p) / 3.0, nan=0.0, posinf=0.0, neginf=0.0
                )

                score = F.interpolate(score.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
                results["anomaly_maps"].append(score.detach().cpu())

        results["anomaly_maps"] = torch.cat(results["anomaly_maps"], dim=0).numpy()

        window_ids_arr = np.array(results["window_id"])
        sorted_idx = np.argsort(window_ids_arr)
        sorted_maps = results["anomaly_maps"][sorted_idx]

        return stitch_anomaly_maps(sorted_maps, self.window_step_ratio, self.agg_percent)
