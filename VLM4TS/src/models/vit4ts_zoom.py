"""ViT4TS Multi-Resolution Zoom.

Two-stage anomaly detection:
  Stage 1 (Coarse): window=224, stride=56, MAE + LTR k=5 + GAP
                    → coarse_score[T_full], candidate intervals
  Stage 2 (Fine):   window=56,  stride=14, MAE + LTR k=5 + GAP
                    on candidate intervals ± padding only
                    → fine_score[T_full] (0 outside candidates)

Score fusion:
  final[t] = α_coarse * coarse[t] + α_fine * fine[t]   (inside candidate)
  final[t] = coarse[t]                                   (outside candidate)

New file — no existing files modified.
"""

import os
import sys
import tempfile
import warnings
from typing import List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Candidate interval helpers
# ---------------------------------------------------------------------------

def _extract_candidates(score: np.ndarray, tau: float) -> List[Tuple[int, int]]:
    """Find contiguous runs where score >= (1-tau) quantile.

    Parameters
    ----------
    score : [T_full]
    tau   : top fraction to flag, e.g. 0.30 → top 30%

    Returns
    -------
    List of (t_start, t_end) pairs (end is exclusive)
    """
    threshold = np.quantile(score, 1.0 - tau)
    above = score >= threshold
    candidates, in_run = [], False
    start = 0
    for t, v in enumerate(above):
        if v and not in_run:
            start, in_run = t, True
        elif not v and in_run:
            candidates.append((start, t))
            in_run = False
    if in_run:
        candidates.append((start, len(above)))
    return candidates


def _merge_intervals(candidates: List[Tuple[int, int]],
                     gap_thresh: int) -> List[Tuple[int, int]]:
    """Merge candidate intervals whose gap <= gap_thresh."""
    if not candidates:
        return []
    merged = [list(candidates[0])]
    for s, e in candidates[1:]:
        if s - merged[-1][1] <= gap_thresh:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# Core LTR + GAP inference (shared by coarse and fine stages)
# ---------------------------------------------------------------------------

def _run_ltr_gap(backbone, values_segment: np.ndarray, patch_size: int,
                 device: torch.device, window_size: int, step_size: int,
                 image_size: tuple, dpi: int, agg_percent: float,
                 alpha_ltr: float, min_ref: int, batch_size: int,
                 verbose: bool, tag: str) -> Optional[np.ndarray]:
    """Run LTR k=5 + GAP on a segment, return 1D score aligned to len(segment).

    Returns None if window generation fails or too few windows.
    """
    T_seg = len(values_segment)
    if T_seg < window_size:
        return None

    n_windows  = (T_seg - window_size) // step_size + 1
    local_k    = 5 if n_windows >= 15 else 0   # global fallback if too few
    plot_params = ("-", 1, "*", 0.1, "black", (0, 1))

    with tempfile.TemporaryDirectory() as tmp:
        success = draw_windowed_images(
            base_series_id="seg", save_path=tmp,
            time_series=values_segment,
            time_points=np.arange(T_seg),
            window_size=window_size, step_size=step_size,
            override=True, save_image=False,
            image_size=image_size, dpi=dpi,
            plot_params=plot_params,
        )
        if not success:
            return None

        dataset = CLIPTimeSeriesDataset(
            results_dir=tmp, base_series_id="seg",
            sample_size=None, no_anomaly=True, plot_type="line",
        )
        if len(dataset) == 0:
            return None

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if verbose:
            print(f"    [{tag}] Pass 1: encoding (win={window_size} L={n_windows} k={local_k})...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            backbone, loader, patch_size, device
        )

        L  = large_embeds.shape[0]
        h  = w = image_size[0]
        ph = h // patch_size
        pw = w // patch_size

        # GAP global score
        gap_vecs   = patch_embeds.mean(dim=1)                    # [L, D]
        median_gap = torch.median(gap_vecs, dim=0).values        # [D]
        gap_scores = []
        with torch.no_grad():
            for i in range(L):
                sim = F.cosine_similarity(
                    gap_vecs[i].unsqueeze(0),
                    median_gap.unsqueeze(0), dim=1,
                )
                gap_scores.append(1.0 - sim.item())
        gap_scores_norm = _normalize_01(np.array(gap_scores))    # [L]

        # GAP [L] → 1D (max-overlap)
        T_final = step_size * (L - 1) + window_size
        gap_1d  = np.zeros(T_final)
        for i, sc in enumerate(gap_scores_norm):
            s = i * step_size
            e = min(s + window_size, T_final)
            gap_1d[s:e] = np.maximum(gap_1d[s:e], sc)

        # LTR Pass 2
        if verbose:
            print(f"    [{tag}] Pass 2: LTR k={local_k}...")
        anomaly_maps = []
        with torch.no_grad():
            for i in range(L):
                if local_k > 0:
                    l_ref, _ = get_local_reference(large_embeds, i, local_k, min_ref)
                    m_ref, _ = get_local_reference(mid_embeds,   i, local_k, min_ref)
                    p_ref, _ = get_local_reference(patch_embeds, i, local_k, min_ref)
                else:
                    idx_all  = list(range(0, i)) + list(range(i + 1, L))
                    l_ref = torch.median(large_embeds[idx_all], dim=0).values
                    m_ref = torch.median(mid_embeds[idx_all],   dim=0).values
                    p_ref = torch.median(patch_embeds[idx_all], dim=0).values

                l_tok = large_embeds[i].unsqueeze(0).to(device)
                m_tok = mid_embeds[i].unsqueeze(0).to(device)
                p_tok = patch_embeds[i].unsqueeze(0).to(device)
                l_ref = l_ref.to(device)
                m_ref = m_ref.to(device)
                p_ref = p_ref.to(device)

                m_l = compute_dissimilarity_with_ref(l_tok, l_ref)
                m_m = compute_dissimilarity_with_ref(m_tok, m_ref)
                m_p = compute_dissimilarity_with_ref(p_tok, p_ref)

                m_l = harmonic_aggregation((1, ph, pw), m_l, large_mask).to(device)
                m_m = harmonic_aggregation((1, ph, pw), m_m, mid_mask).to(device)
                m_p = m_p.reshape((1, ph, pw)).to(device)

                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                         nan=0., posinf=0., neginf=0.)
                score = F.interpolate(score.unsqueeze(1), size=(h, w),
                                      mode="bilinear").squeeze(1)
                anomaly_maps.append(score.squeeze(0).detach().cpu())

        maps_arr  = torch.stack(anomaly_maps, dim=0).numpy()
        ltr_1d    = stitch_anomaly_maps(maps_arr, window_size / step_size, agg_percent)

    # Normalize + combine
    ltr_norm = _normalize_01(ltr_1d)

    # Align lengths
    T_out = len(ltr_norm)
    if len(gap_1d) > T_out:
        gap_1d = gap_1d[:T_out]
    elif len(gap_1d) < T_out:
        gap_1d = np.pad(gap_1d, (0, T_out - len(gap_1d)))

    combined = alpha_ltr * ltr_norm + (1 - alpha_ltr) * gap_1d

    # Align to T_seg
    return align_anomaly_vector(combined, T_seg, window_size, step_size, n_windows)


# ---------------------------------------------------------------------------
# ViT4TS_Zoom
# ---------------------------------------------------------------------------

class ViT4TS_Zoom:
    """Multi-Resolution Zoom anomaly detector.

    Stage 1 (Coarse): window=224, stride=56, LTR k=5 + GAP
    Stage 2 (Fine)  : window=56,  stride=14, LTR k=5 + GAP
                      on candidate intervals only

    Parameters
    ----------
    backbone      : MAE (or any) vision encoder
    patch_size    : int
    alpha_ltr     : float  LTR weight inside each stage (GAP = 1-alpha_ltr)
    alpha_coarse  : float  coarse weight in fusion (fine = 1-alpha_coarse)
    tau_coarse    : float  top fraction flagged as candidate (default 0.30)
    """

    def __init__(
        self,
        backbone,
        patch_size: int = 16,
        min_ref: int = 5,
        alpha_ltr: float = 0.7,       # LTR vs GAP weight (both stages)
        alpha_coarse: float = 0.3,    # coarse weight in fusion
        window_coarse: int = 224,
        window_fine: int = 56,
        tau_coarse: float = 0.30,
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
        self.backbone       = backbone
        self.patch_size     = patch_size
        self.min_ref        = min_ref
        self.alpha_ltr      = alpha_ltr
        self.alpha_coarse   = alpha_coarse
        self.alpha_fine     = 1.0 - alpha_coarse
        self.window_coarse  = window_coarse
        self.window_fine    = window_fine
        self.stride_coarse  = window_coarse // 4
        self.stride_fine    = window_fine // 4
        self.padding        = window_fine // 2
        self.tau_coarse     = tau_coarse
        self.gap_thresh     = window_coarse // 2
        self.agg_percent    = agg_percent
        self.batch_size     = batch_size
        self.image_size     = image_size
        self.dpi            = dpi
        self.standardize    = standardize
        self.alpha_detect   = alpha_detect
        self.smoothing_alpha = smoothing_alpha
        self.verbose        = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        if self.verbose:
            print(f"ViT4TS_Zoom | device={self.device} | "
                  f"coarse={window_coarse} fine={window_fine} "
                  f"τ={tau_coarse} α_ltr={alpha_ltr} "
                  f"α_coarse={alpha_coarse} α_fine={self.alpha_fine}")

    # ------------------------------------------------------------------
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        scores, timestamps = self.predict_scores(data)
        return self.get_intervals(scores, timestamps, self.alpha_detect)

    def predict_scores(self, data: pd.DataFrame) -> tuple:
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        values_proc = (preprocess_time_series(values)
                       if self.standardize else values.astype(float))
        values_proc = apply_ewma(values_proc, self.smoothing_alpha)

        if self.verbose:
            print(f"  {T_full} pts | coarse={self.window_coarse} fine={self.window_fine}")

        # ---- Stage 1: Coarse screening ----
        coarse_score = self._coarse_screening(values_proc, T_full)
        if coarse_score is None:
            return np.zeros(T_full), timestamps

        # ---- Extract + merge candidate intervals ----
        raw_cands = _extract_candidates(coarse_score, self.tau_coarse)
        candidates = _merge_intervals(raw_cands, self.gap_thresh)

        if self.verbose:
            print(f"  Candidates: {len(candidates)} intervals "
                  f"(τ={self.tau_coarse}, gap_thresh={self.gap_thresh})")
            for s, e in candidates:
                print(f"    [{s}, {e})  len={e-s}")

        # ---- Stage 2: Fine zoom on candidates ----
        fine_score = np.zeros(T_full)
        fine_mask  = np.zeros(T_full, dtype=bool)

        for idx, (t_start, t_end) in enumerate(candidates):
            seg_start = max(0, t_start - self.padding)
            seg_end   = min(T_full, t_end + self.padding)
            segment   = values_proc[seg_start:seg_end]

            if self.verbose:
                print(f"  Fine zoom [{idx+1}/{len(candidates)}]: "
                      f"t=[{seg_start},{seg_end}) len={len(segment)}")

            seg_fine = self._fine_zoom(segment)
            if seg_fine is not None:
                L_seg = min(len(seg_fine), seg_end - seg_start)
                fine_score[seg_start:seg_start + L_seg] = np.maximum(
                    fine_score[seg_start:seg_start + L_seg],
                    seg_fine[:L_seg],
                )
                fine_mask[seg_start:seg_start + L_seg] = True

        # ---- Score fusion ----
        final_score = coarse_score.copy()
        if fine_mask.any():
            fine_norm   = _normalize_01(fine_score)
            coarse_norm = _normalize_01(coarse_score)
            final_score[fine_mask] = (
                self.alpha_coarse * coarse_norm[fine_mask] +
                self.alpha_fine   * fine_norm[fine_mask]
            )
            # outside candidates: keep original coarse (already in final_score)
            final_score[~fine_mask] = coarse_norm[~fine_mask]

        if self.verbose:
            print(f"  Coarse score  min={coarse_score.min():.4f} max={coarse_score.max():.4f}")
            if fine_mask.any():
                print(f"  Fine score    min={fine_score[fine_mask].min():.4f} "
                      f"max={fine_score[fine_mask].max():.4f}")

        return final_score, timestamps

    def get_intervals(self, scores, timestamps, alpha=None):
        if alpha is None:
            alpha = self.alpha_detect
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
        return intervals_from_indices(idx, timestamps, scores)

    # ------------------------------------------------------------------
    def _coarse_screening(self, values_proc: np.ndarray,
                          T_full: int) -> Optional[np.ndarray]:
        score = _run_ltr_gap(
            backbone=self.backbone,
            values_segment=values_proc,
            patch_size=self.patch_size,
            device=self.device,
            window_size=self.window_coarse,
            step_size=self.stride_coarse,
            image_size=self.image_size,
            dpi=self.dpi,
            agg_percent=self.agg_percent,
            alpha_ltr=self.alpha_ltr,
            min_ref=self.min_ref,
            batch_size=self.batch_size,
            verbose=self.verbose,
            tag="COARSE",
        )
        if score is None:
            return None
        # align to T_full
        n_windows = (T_full - self.window_coarse) // self.stride_coarse + 1
        return align_anomaly_vector(
            score, T_full, self.window_coarse, self.stride_coarse, n_windows
        )

    def _fine_zoom(self, segment: np.ndarray) -> Optional[np.ndarray]:
        return _run_ltr_gap(
            backbone=self.backbone,
            values_segment=segment,
            patch_size=self.patch_size,
            device=self.device,
            window_size=self.window_fine,
            step_size=self.stride_fine,
            image_size=self.image_size,
            dpi=self.dpi,
            agg_percent=self.agg_percent,
            alpha_ltr=self.alpha_ltr,
            min_ref=self.min_ref,
            batch_size=self.batch_size,
            verbose=self.verbose,
            tag="FINE",
        )
