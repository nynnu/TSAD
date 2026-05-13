"""ViT4TS Multi-Resolution Zoom v3.

v3 changes from vit4ts_zoom_v2.py (2 points):

1. Coarse: α_ltr=0.9 (LTR 90% + GAP 10%)
   - 기존 α_ltr=0.7 → 0.9
   - LTR 비중 높여서 false candidate 줄이기
   - GAP는 10%만 유지 (drift 탐지 보조)

2. Fine: LTR k=5 ONLY (GAP 제거)
   - 기존: Fine도 Dual-Anchor (LTR+GAP)
   - v3:   Fine은 LTR만
   - 이유: Fine segment는 L이 작아 GAP median 불안정
           Fine의 역할은 drift 탐지가 아닌 localization
           LTR k=5가 "candidate 내 어느 시점이 가장 이상한가"에 집중

나머지 (max fusion, mean+k*std threshold) 는 v2와 동일.
No existing files modified.
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


def _extract_candidates_adaptive(score: np.ndarray,
                                  k_sigma: float = 1.5) -> List[Tuple[int, int]]:
    threshold = score.mean() + k_sigma * score.std()
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
# Core inference — use_gap flag로 coarse/fine 분기
# ---------------------------------------------------------------------------

def _run_ltr_gap(backbone, values_segment: np.ndarray, patch_size: int,
                 device: torch.device, window_size: int, step_size: int,
                 image_size: tuple, dpi: int, agg_percent: float,
                 alpha_ltr: float, min_ref: int, batch_size: int,
                 verbose: bool, tag: str,
                 use_gap: bool = True) -> Optional[np.ndarray]:
    """LTR (+ optional GAP) inference on a segment.

    use_gap=True  → Dual-Anchor: α_ltr * LTR + (1-α_ltr) * GAP  [coarse]
    use_gap=False → LTR only                                       [fine]
    """
    T_seg = len(values_segment)
    if T_seg < window_size:
        return None

    n_windows = (T_seg - window_size) // step_size + 1
    if n_windows < 2:
        return None

    local_k     = 5 if n_windows >= 15 else 0
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
            gap_tag = "LTR+GAP" if use_gap else "LTR only"
            print(f"    [{tag}] Pass 1: encoding "
                  f"(win={window_size} L={n_windows} k={local_k} {gap_tag})...")
        (large_embeds, mid_embeds, patch_embeds,
         large_mask, mid_mask, _) = build_ordered_embeddings(
            backbone, loader, patch_size, device
        )

        L  = large_embeds.shape[0]
        h  = w = image_size[0]
        ph = h // patch_size
        pw = w // patch_size

        # ---- GAP (coarse only) ----
        if use_gap:
            gap_vecs   = patch_embeds.mean(dim=1)
            median_gap = torch.median(gap_vecs, dim=0).values
            gap_scores = []
            with torch.no_grad():
                for i in range(L):
                    sim = F.cosine_similarity(
                        gap_vecs[i].unsqueeze(0),
                        median_gap.unsqueeze(0), dim=1,
                    )
                    gap_scores.append(1.0 - sim.item())
            gap_norm = _normalize_01(np.array(gap_scores))

            T_final = step_size * (L - 1) + window_size
            gap_1d  = np.zeros(T_final)
            for i, sc in enumerate(gap_norm):
                s = i * step_size
                e = min(s + window_size, T_final)
                gap_1d[s:e] = np.maximum(gap_1d[s:e], sc)

        # ---- LTR Pass 2 ----
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

        maps_arr = torch.stack(anomaly_maps, dim=0).numpy()
        ltr_1d   = stitch_anomaly_maps(maps_arr, window_size / step_size, agg_percent)

    ltr_norm = _normalize_01(ltr_1d)

    # ---- Combine ----
    if use_gap:
        T_out = len(ltr_norm)
        if len(gap_1d) > T_out:
            gap_1d = gap_1d[:T_out]
        elif len(gap_1d) < T_out:
            gap_1d = np.pad(gap_1d, (0, T_out - len(gap_1d)))
        combined = alpha_ltr * ltr_norm + (1 - alpha_ltr) * gap_1d
    else:
        combined = ltr_norm   # LTR only

    return align_anomaly_vector(combined, T_seg, window_size, step_size, n_windows)


# ---------------------------------------------------------------------------
# ViT4TS_Zoom_V3
# ---------------------------------------------------------------------------

class ViT4TS_Zoom_V3:
    """Multi-Resolution Zoom v3.

    Coarse: Dual-Anchor  α_ltr=0.9 (LTR 90% + GAP 10%)
    Fine:   LTR only     (GAP 제거, localization에 집중)
    Threshold: mean + k_sigma * std
    Fusion: max(coarse, fine)
    """

    def __init__(
        self,
        backbone,
        patch_size: int = 16,
        min_ref: int = 5,
        alpha_ltr_coarse: float = 0.9,   # coarse LTR weight
        k_sigma: float = 1.5,
        window_coarse: int = 224,
        window_fine: int = 56,
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
        self.backbone           = backbone
        self.patch_size         = patch_size
        self.min_ref            = min_ref
        self.alpha_ltr_coarse   = alpha_ltr_coarse
        self.k_sigma            = k_sigma
        self.window_coarse      = window_coarse
        self.window_fine        = window_fine
        self.stride_coarse      = window_coarse // 4
        self.stride_fine        = window_fine // 4
        self.padding            = window_fine // 2
        self.gap_thresh         = window_coarse // 2
        self.agg_percent        = agg_percent
        self.batch_size         = batch_size
        self.image_size         = image_size
        self.dpi                = dpi
        self.standardize        = standardize
        self.alpha_detect       = alpha_detect
        self.smoothing_alpha    = smoothing_alpha
        self.verbose            = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        if self.verbose:
            print(f"ViT4TS_Zoom_V3 | device={self.device} | "
                  f"coarse={window_coarse}(α_ltr={alpha_ltr_coarse} GAP={1-alpha_ltr_coarse}) "
                  f"fine={window_fine}(LTR only) "
                  f"k_sigma={k_sigma} fusion=max")

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

        # ---- Stage 1: Coarse (Dual-Anchor, α_ltr=0.9) ----
        coarse_score = self._coarse_screening(values_proc, T_full)
        if coarse_score is None:
            return np.zeros(T_full), timestamps

        # ---- Candidate 추출 (adaptive threshold) ----
        thr        = coarse_score.mean() + self.k_sigma * coarse_score.std()
        raw_cands  = _extract_candidates_adaptive(coarse_score, self.k_sigma)
        candidates = _merge_intervals(raw_cands, self.gap_thresh)

        if self.verbose:
            print(f"  Threshold: {thr:.4f}  Candidates: {len(candidates)}")
            for s, e in candidates:
                print(f"    [{s}, {e})  len={e-s}")

        # ---- Stage 2: Fine (LTR only) ----
        fine_score = np.zeros(T_full)
        fine_mask  = np.zeros(T_full, dtype=bool)

        for idx, (t_start, t_end) in enumerate(candidates):
            seg_start = max(0, t_start - self.padding)
            seg_end   = min(T_full, t_end + self.padding)
            segment   = values_proc[seg_start:seg_end]

            if self.verbose:
                print(f"  Fine [{idx+1}/{len(candidates)}]: "
                      f"t=[{seg_start},{seg_end}) L_approx="
                      f"{(len(segment)-self.window_fine)//self.stride_fine+1}")

            seg_fine = self._fine_zoom(segment)
            if seg_fine is not None:
                L_seg = min(len(seg_fine), seg_end - seg_start)
                fine_score[seg_start:seg_start + L_seg] = np.maximum(
                    fine_score[seg_start:seg_start + L_seg],
                    seg_fine[:L_seg],
                )
                fine_mask[seg_start:seg_start + L_seg] = True

        # ---- Max fusion ----
        final_score = coarse_score.copy()
        if fine_mask.any():
            coarse_norm = _normalize_01(coarse_score)
            fine_norm   = _normalize_01(fine_score)
            final_score[fine_mask]  = np.maximum(coarse_norm[fine_mask],
                                                  fine_norm[fine_mask])
            final_score[~fine_mask] = coarse_norm[~fine_mask]

        if self.verbose:
            print(f"  Coarse min={coarse_score.min():.4f} max={coarse_score.max():.4f}")
            if fine_mask.any():
                print(f"  Fine   min={fine_score[fine_mask].min():.4f} "
                      f"max={fine_score[fine_mask].max():.4f}")

        return final_score, timestamps

    def get_intervals(self, scores, timestamps, alpha=None):
        if alpha is None:
            alpha = self.alpha_detect
        idx, _, _ = compute_detection_intervals(score_vector=scores, alpha=alpha)
        return intervals_from_indices(idx, timestamps, scores)

    def _coarse_screening(self, values_proc, T_full):
        score = _run_ltr_gap(
            backbone=self.backbone, values_segment=values_proc,
            patch_size=self.patch_size, device=self.device,
            window_size=self.window_coarse, step_size=self.stride_coarse,
            image_size=self.image_size, dpi=self.dpi,
            agg_percent=self.agg_percent,
            alpha_ltr=self.alpha_ltr_coarse,  # 0.9
            min_ref=self.min_ref, batch_size=self.batch_size,
            verbose=self.verbose, tag="COARSE",
            use_gap=True,   # Dual-Anchor
        )
        if score is None:
            return None
        n_windows = (T_full - self.window_coarse) // self.stride_coarse + 1
        return align_anomaly_vector(
            score, T_full, self.window_coarse, self.stride_coarse, n_windows
        )

    def _fine_zoom(self, segment):
        return _run_ltr_gap(
            backbone=self.backbone, values_segment=segment,
            patch_size=self.patch_size, device=self.device,
            window_size=self.window_fine, step_size=self.stride_fine,
            image_size=self.image_size, dpi=self.dpi,
            agg_percent=self.agg_percent,
            alpha_ltr=1.0,   # fine: LTR only (alpha_ltr=1.0 → GAP term = 0)
            min_ref=self.min_ref, batch_size=self.batch_size,
            verbose=self.verbose, tag="FINE",
            use_gap=False,   # LTR only
        )
