"""ViT4TS Sub-Adjacent LTR (Sub-Adjacent Transformer, IJCAI 2024).

기존 LTR k=5: window_i를 i±1..5 이웃과 비교
              → spike 주변 윈도우들도 이웃에 포함 → 참조가 오염될 수 있음

SubLTR [k_min, k_max]: window_i를 i±[k_min..k_max] 이웃과 비교
                        → 가까운 이웃 skip → 깨끗한 정상 참조 확보
                        → spike 탐지에 유리

단일 forward pass로 LTR + SubLTR 동시 계산:
  Pass 1: 임베딩 빌드 (batch-aware, GPU 효율)
  Pass 2: 같은 루프에서 LTR ref + SubLTR ref 동시 계산

반환:
  "ltr"     : LTR k=5 (기존 방식)
  "sub_ltr" : SubLTR [k_min, k_max]
  "max"     : max(normalize(ltr), normalize(sub_ltr))

No existing files modified.
"""

import os
import sys
import tempfile
import warnings
from typing import Dict, Optional, Tuple

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


def _get_sub_ref(embeds: torch.Tensor, i: int,
                 k_min: int, k_max: int, sub_min_ref: int) -> torch.Tensor:
    """Sub-adjacent reference: windows at distance [k_min, k_max], skipping immediate.

    Fallback chain:
      1. Sub-adjacent [k_min, k_max]  → if len >= sub_min_ref
      2. Global (all non-i windows)   → if sub-adjacent not enough
    """
    L = embeds.shape[0]

    left  = list(range(max(0, i - k_max), max(0, i - k_min + 1)))
    right = list(range(min(L, i + k_min), min(L, i + k_max + 1)))
    idx   = left + right

    if len(idx) < sub_min_ref:
        idx = list(range(0, i)) + list(range(i + 1, L))

    if len(idx) == 0:
        return embeds.mean(dim=0)

    return torch.median(embeds[idx], dim=0).values


class ViT4TS_SubLTR:
    """LTR k=5 + Sub-adjacent LTR, single forward pass.

    Parameters
    ----------
    backbone    : MAE_AD (or any backbone with encode_image)
    local_k     : int   LTR immediate half-window (default 5)
    sub_k_min   : int   Sub-adjacent skip distance, inclusive (default 10)
    sub_k_max   : int   Sub-adjacent far distance, inclusive  (default 20)
    sub_min_ref : int   Min sub-adjacent windows before global fallback (default 3)
    """

    def __init__(
        self,
        backbone,
        patch_size:  int   = 16,
        local_k:     int   = 5,
        sub_k_min:   int   = 10,
        sub_k_max:   int   = 20,
        sub_min_ref: int   = 3,
        min_ref:     int   = 5,
        window_size: int   = 224,
        window_step_ratio: float = 4.0,
        agg_percent: float = 0.25,
        device: str = "auto",
        batch_size:  int   = 32,
        image_size:  tuple = (224, 224),
        dpi:         int   = 100,
        standardize: bool  = True,
        alpha_detect: float = 0.01,
        smoothing_alpha: float = 1.0,
        verbose:     bool  = True,
    ):
        self.backbone          = backbone
        self.patch_size        = patch_size
        self.local_k           = local_k
        self.sub_k_min         = sub_k_min
        self.sub_k_max         = sub_k_max
        self.sub_min_ref       = sub_min_ref
        self.min_ref           = min_ref
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
            print(f"ViT4TS_SubLTR | device={self.device} | "
                  f"LTR k={local_k} | SubLTR [{sub_k_min},{sub_k_max}] | fusion=max")

    # ------------------------------------------------------------------
    # Public API (vit4ts_local.py 인터페이스와 동일)
    # ------------------------------------------------------------------

    def detect(self, data: pd.DataFrame, mode: str = "max") -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)[mode]
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=self.alpha_detect)
        return intervals_from_indices(idx, timestamps, scores)

    def predict_scores(self, data: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns
        -------
        {
          "ltr"     : (scores_1d, timestamps),
          "sub_ltr" : (scores_1d, timestamps),
          "max"     : (max(norm_ltr, norm_subltr), timestamps),
        }
        """
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        values_proc = (preprocess_time_series(values)
                       if self.standardize else values.astype(float))
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        if self.verbose:
            print(f"  {T_full} pts | win={self.window_size} step={step_size} L={n_windows}")

        with tempfile.TemporaryDirectory() as tmp:
            plot_params = ("-", 1, "*", 0.1, "black",
                           (0, 1) if self.standardize else None)

            success = draw_windowed_images(
                base_series_id="series", save_path=tmp,
                time_series=values_proc,
                time_points=np.arange(len(values_proc)),
                window_size=self.window_size, step_size=step_size,
                override=True, save_image=False,
                image_size=self.image_size, dpi=self.dpi,
                plot_params=plot_params,
            )
            if not success:
                warnings.warn("No windowed images generated.")
                zero = np.zeros(T_full)
                return {"ltr": (zero, timestamps),
                        "sub_ltr": (zero, timestamps),
                        "max": (zero, timestamps)}

            score_ltr, score_sub = self._run_inference(
                tmp, "series", step_size, n_windows, T_full
            )

        if score_ltr is None:
            zero = np.zeros(T_full)
            return {"ltr": (zero, timestamps),
                    "sub_ltr": (zero, timestamps),
                    "max": (zero, timestamps)}

        ltr_norm = _normalize_01(score_ltr)
        sub_norm = _normalize_01(score_sub)

        # align to T_full
        T_out    = min(len(ltr_norm), len(sub_norm), T_full)
        ltr_norm = np.pad(ltr_norm[:T_out], (0, T_full - T_out))
        sub_norm = np.pad(sub_norm[:T_out], (0, T_full - T_out))

        return {
            "ltr":     (ltr_norm,                    timestamps),
            "sub_ltr": (sub_norm,                    timestamps),
            "max":     (np.maximum(ltr_norm, sub_norm), timestamps),
        }

    # ------------------------------------------------------------------
    # Core inference — single forward pass, dual reference
    # ------------------------------------------------------------------

    def _run_inference(
        self, results_dir: str, base_id: str,
        step_size: int, n_windows: int, T_full: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build embeddings once; compute LTR and SubLTR maps in one loop."""

        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            warnings.warn("Empty dataset.")
            return None, None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # ---- Pass 1: encode all windows (batch-aware) ----
        if self.verbose:
            print("  [Pass 1] encoding embeddings...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device,
        )

        L  = large_embeds.shape[0]
        h  = w = self.image_size[0]
        ph = h // self.patch_size
        pw = w // self.patch_size

        if self.verbose:
            print(f"  [Pass 2] LTR k={self.local_k} + "
                  f"SubLTR [{self.sub_k_min},{self.sub_k_max}] (L={L})...")

        ltr_maps = []
        sub_maps = []

        with torch.no_grad():
            for i in range(L):
                # ---- LTR reference ----
                l_ref_ltr, _ = get_local_reference(large_embeds, i, self.local_k, self.min_ref)
                m_ref_ltr, _ = get_local_reference(mid_embeds,   i, self.local_k, self.min_ref)
                p_ref_ltr, _ = get_local_reference(patch_embeds, i, self.local_k, self.min_ref)

                # ---- SubLTR reference ----
                l_ref_sub = _get_sub_ref(large_embeds, i, self.sub_k_min, self.sub_k_max, self.sub_min_ref)
                m_ref_sub = _get_sub_ref(mid_embeds,   i, self.sub_k_min, self.sub_k_max, self.sub_min_ref)
                p_ref_sub = _get_sub_ref(patch_embeds, i, self.sub_k_min, self.sub_k_max, self.sub_min_ref)

                # tokens for window i
                l_tok = large_embeds[i].unsqueeze(0).to(self.device)
                m_tok = mid_embeds[i].unsqueeze(0).to(self.device)
                p_tok = patch_embeds[i].unsqueeze(0).to(self.device)

                # ---- compute both anomaly maps ----
                for refs, maps in [
                    ((l_ref_ltr, m_ref_ltr, p_ref_ltr), ltr_maps),
                    ((l_ref_sub, m_ref_sub, p_ref_sub), sub_maps),
                ]:
                    l_ref, m_ref, p_ref = [r.to(self.device) for r in refs]

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
                    maps.append(score.squeeze(0).detach().cpu())

        # ---- stitch to 1D ----
        ltr_arr   = torch.stack(ltr_maps, dim=0).numpy()
        sub_arr   = torch.stack(sub_maps, dim=0).numpy()

        ltr_1d = stitch_anomaly_maps(ltr_arr, self.window_step_ratio, self.agg_percent)
        sub_1d = stitch_anomaly_maps(sub_arr, self.window_step_ratio, self.agg_percent)

        score_ltr = align_anomaly_vector(ltr_1d, T_full, self.window_size, step_size, n_windows)
        score_sub = align_anomaly_vector(sub_1d, T_full, self.window_size, step_size, n_windows)

        if self.verbose:
            print(f"  LTR    min={score_ltr.min():.4f} max={score_ltr.max():.4f}")
            print(f"  SubLTR min={score_sub.min():.4f} max={score_sub.max():.4f}")

        return score_ltr, score_sub
