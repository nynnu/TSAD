"""ViT4TS: LTR + Sub-adjacent LTR + Intra-window Spatial, max fusion.

세 가지 신호를 단일 forward pass로 계산:

  score_LTR      : window_i vs 시간 이웃 i±k (k=5)
                   → level shift, drift 감지

  score_SubLTR   : window_i vs 떨어진 이웃 i±[k_min, k_max] (기본 10~20)
                   skip immediate neighbors → spike 주변 오염 방지
                   → spike, 짧은 burst anomaly 감지 (Sub-Adjacent Transformer, IJCAI 2024)

  score_Spatial  : window 내부에서 patch(r,c) vs 공간 이웃 patch
                   → spike, point anomaly 감지 (intra-image)

Fusion:
  max(LTR, Spatial)          → 기본 max fusion
  max(SubLTR, Spatial)       → sub-adjacent + spatial
  max(LTR, SubLTR, Spatial)  → 세 신호 모두 OR

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


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _window_scores_to_1d(scores: np.ndarray, L: int,
                          step_size: int, window_size: int) -> np.ndarray:
    T = step_size * (L - 1) + window_size
    out = np.zeros(T, dtype=float)
    for i, sc in enumerate(scores):
        s = i * step_size
        e = min(s + window_size, T)
        out[s:e] = np.maximum(out[s:e], sc)
    return out


def _get_sub_adjacent_ref(embeds: torch.Tensor, i: int,
                           k_min: int, k_max: int) -> torch.Tensor:
    """Reference from windows at distance [k_min, k_max], skipping immediate neighbors.

    If not enough sub-adjacent windows, falls back to global (all non-i windows).
    """
    L = embeds.shape[0]
    idx = (list(range(max(0, i - k_max), max(0, i - k_min + 1))) +
           list(range(min(L, i + k_min), min(L, i + k_max + 1))))

    if len(idx) < 2:
        idx = list(range(0, i)) + list(range(i + 1, L))
    if len(idx) == 0:
        return embeds.mean(dim=0)

    return torch.median(embeds[idx], dim=0).values


def _compute_spatial_scores(patch_embeds: torch.Tensor,
                             top_k_ratio: float = 0.1) -> np.ndarray:
    """Intra-window spatial patch comparison. [L, 196, D] → [L] scores."""
    L, N, D = patch_embeds.shape
    ph = pw = 14

    p = F.normalize(patch_embeds.reshape(L, ph, pw, D), dim=-1)

    h_diff = 1.0 - (p[:, :, :-1, :] * p[:, :, 1:, :]).sum(dim=-1)   # [L, 14, 13]
    v_diff = 1.0 - (p[:, :-1, :, :] * p[:, 1:, :, :]).sum(dim=-1)   # [L, 13, 14]

    all_diff = torch.cat([h_diff.reshape(L, -1),
                          v_diff.reshape(L, -1)], dim=1)              # [L, 364]

    k = max(1, int(all_diff.shape[1] * top_k_ratio))
    return all_diff.topk(k, dim=1).values.mean(dim=1).cpu().numpy()  # [L]


def _ltr_anomaly_maps(large_embeds, mid_embeds, patch_embeds,
                      large_mask, mid_mask,
                      local_k: int, min_ref: int,
                      ph: int, pw: int, h: int, w: int,
                      device: torch.device,
                      ref_fn) -> list:
    """Compute LTR anomaly maps given a reference function ref_fn(embeds, i)."""
    L = large_embeds.shape[0]
    maps = []
    with torch.no_grad():
        for i in range(L):
            l_ref = ref_fn(large_embeds, i)
            m_ref = ref_fn(mid_embeds,   i)
            p_ref = ref_fn(patch_embeds, i)

            m_l = compute_dissimilarity_with_ref(
                large_embeds[i].unsqueeze(0).to(device), l_ref.to(device))
            m_m = compute_dissimilarity_with_ref(
                mid_embeds[i].unsqueeze(0).to(device),   m_ref.to(device))
            m_p = compute_dissimilarity_with_ref(
                patch_embeds[i].unsqueeze(0).to(device), p_ref.to(device))

            m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(device)
            m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(device)
            m_p = m_p.reshape((1, ph, pw)).to(device)

            score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                      nan=0., posinf=0., neginf=0.)
            score = F.interpolate(score.unsqueeze(1), size=(h, w),
                                  mode="bilinear").squeeze(1)
            maps.append(score.squeeze(0).detach().cpu())
    return maps


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ViT4TS_MAE_Spatial:
    """LTR + Sub-adjacent LTR + Intra-window Spatial, max fusion.

    Single forward pass produces all three signals.

    Parameters
    ----------
    backbone      : MAE_AD
    local_k       : int    LTR immediate half-window (default 5)
    sub_k_min     : int    Sub-adjacent skip distance (default 10)
    sub_k_max     : int    Sub-adjacent far distance  (default 20)
    top_k_ratio   : float  Fraction of patch-pair diffs for spatial score
    """

    def __init__(
        self,
        backbone,
        local_k:     int   = 5,
        sub_k_min:   int   = 10,
        sub_k_max:   int   = 20,
        top_k_ratio: float = 0.1,
        min_ref:     int   = 5,
        patch_size:  int   = 16,
        window_size: int   = 224,
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
        self.local_k           = local_k
        self.sub_k_min         = sub_k_min
        self.sub_k_max         = sub_k_max
        self.top_k_ratio       = top_k_ratio
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
            print(f"ViT4TS_MAE_Spatial | device={self.device} | "
                  f"LTR k={local_k} | SubLTR [{sub_k_min},{sub_k_max}] | "
                  f"Spatial top_k={top_k_ratio:.0%} | fusion=max")

    # ------------------------------------------------------------------
    def detect(self, data: pd.DataFrame, mode: str = "max_all") -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)[mode]
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=self.alpha_detect)
        return intervals_from_indices(idx, timestamps, scores)

    def predict_scores(self, data: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns dict with keys:
          "ltr"           : LTR k=5 only
          "sub_ltr"       : Sub-adjacent LTR only
          "spatial"       : Intra-window spatial only
          "max_ltr_sp"    : max(LTR, Spatial)
          "max_subltr_sp" : max(SubLTR, Spatial)
          "max_all"       : max(LTR, SubLTR, Spatial)
        """
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        values_proc = preprocess_time_series(values) if self.standardize else values.astype(float)
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        step_size = int(self.window_size / self.window_step_ratio)
        n_windows = int((T_full - self.window_size) / step_size) + 1

        if self.verbose:
            print(f"  {T_full} pts | win={self.window_size} L={n_windows}")

        with tempfile.TemporaryDirectory() as tmp:
            success = draw_windowed_images(
                base_series_id="series", save_path=tmp,
                time_series=values_proc,
                time_points=np.arange(len(values_proc)),
                window_size=self.window_size, step_size=step_size,
                override=True, save_image=False,
                image_size=self.image_size, dpi=self.dpi,
                plot_params=("-", 1, "*", 0.1, "black",
                             (0, 1) if self.standardize else None),
            )
            if not success:
                zero = np.zeros(T_full)
                empty = {k: (zero, timestamps) for k in
                         ["ltr","sub_ltr","spatial","max_ltr_sp","max_subltr_sp","max_all"]}
                return empty

            s_ltr, s_sub, s_sp = self._compute_all(
                tmp, "series", step_size, n_windows, T_full
            )

        if s_ltr is None:
            zero = np.zeros(T_full)
            return {k: (zero, timestamps) for k in
                    ["ltr","sub_ltr","spatial","max_ltr_sp","max_subltr_sp","max_all"]}

        T_out  = min(len(s_ltr), len(s_sub), len(s_sp), T_full)
        ltr_n  = np.pad(_normalize_01(s_ltr[:T_out]),  (0, T_full - T_out))
        sub_n  = np.pad(_normalize_01(s_sub[:T_out]),  (0, T_full - T_out))
        sp_n   = np.pad(_normalize_01(s_sp[:T_out]),   (0, T_full - T_out))

        return {
            "ltr":           (ltr_n,                           timestamps),
            "sub_ltr":       (sub_n,                           timestamps),
            "spatial":       (sp_n,                            timestamps),
            "max_ltr_sp":    (np.maximum(ltr_n, sp_n),         timestamps),
            "max_subltr_sp": (np.maximum(sub_n, sp_n),         timestamps),
            "max_all":       (np.maximum(np.maximum(ltr_n, sub_n), sp_n), timestamps),
        }

    # ------------------------------------------------------------------
    def _compute_all(
        self, results_dir: str, base_id: str,
        step_size: int, n_windows: int, T_full: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Single forward pass → (score_ltr, score_sub_ltr, score_spatial)."""

        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir, base_series_id=base_id,
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return None, None, None

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.verbose:
            print("  [Pass 1] encoding embeddings...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            self.backbone, loader, self.patch_size, self.device,
        )

        L   = large_embeds.shape[0]
        h   = w = self.image_size[0]
        ph  = h // self.patch_size
        pw  = w // self.patch_size

        # ---- Spatial (from patch_embeds, no extra forward) ----
        if self.verbose:
            print("  [Spatial] intra-window patch comparison...")
        sp_raw     = _compute_spatial_scores(patch_embeds, self.top_k_ratio)
        sp_1d      = _window_scores_to_1d(sp_raw, L, step_size, self.window_size)
        score_sp   = align_anomaly_vector(sp_1d, T_full, self.window_size, step_size, n_windows)

        # ---- LTR k=5 (immediate neighbors) ----
        if self.verbose:
            print(f"  [LTR] k={self.local_k}...")

        def ref_ltr(embeds, i):
            ref, _ = get_local_reference(embeds, i, self.local_k, self.min_ref)
            return ref

        ltr_maps  = _ltr_anomaly_maps(
            large_embeds, mid_embeds, patch_embeds,
            large_mask, mid_mask,
            self.local_k, self.min_ref,
            ph, pw, h, w, self.device, ref_ltr,
        )
        ltr_arr   = torch.stack(ltr_maps, dim=0).numpy()
        ltr_1d    = stitch_anomaly_maps(ltr_arr, self.window_step_ratio, self.agg_percent)
        score_ltr = align_anomaly_vector(ltr_1d, T_full, self.window_size, step_size, n_windows)

        # ---- Sub-adjacent LTR (skip immediate neighbors) ----
        if self.verbose:
            print(f"  [SubLTR] skip i±{self.sub_k_min-1}, ref i±[{self.sub_k_min},{self.sub_k_max}]...")

        def ref_sub(embeds, i):
            return _get_sub_adjacent_ref(embeds, i, self.sub_k_min, self.sub_k_max)

        sub_maps  = _ltr_anomaly_maps(
            large_embeds, mid_embeds, patch_embeds,
            large_mask, mid_mask,
            self.local_k, self.min_ref,
            ph, pw, h, w, self.device, ref_sub,
        )
        sub_arr   = torch.stack(sub_maps, dim=0).numpy()
        sub_1d    = stitch_anomaly_maps(sub_arr, self.window_step_ratio, self.agg_percent)
        score_sub = align_anomaly_vector(sub_1d, T_full, self.window_size, step_size, n_windows)

        if self.verbose:
            print(f"  LTR    min={score_ltr.min():.4f} max={score_ltr.max():.4f}")
            print(f"  SubLTR min={score_sub.min():.4f} max={score_sub.max():.4f}")
            print(f"  Spatial min={score_sp.min():.4f} max={score_sp.max():.4f}")

        return score_ltr, score_sub, score_sp
