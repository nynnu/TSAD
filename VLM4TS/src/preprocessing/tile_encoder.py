"""2D Tile encoder — VisionTS-style stacked window image.

Each image = 14 windows stacked as rows (7 before + target + 6 after).
Each row = filled area chart of one window (16 px tall × 224 px wide).

Why this works for MAE
-----------------------
- Stacked waveforms create texture-like images (oscilloscope / EEG displays)
  → MAE features well-calibrated (closer to natural images than GAF/RP)
- Every pixel carries signal information (no 85% background)
- Target window is always in the same position (row 7) → patch indices fixed
- MAE can exploit vertical (temporal) context to predict the target row

Scoring
-------
Mask the 14 patches of the target row. MAE reconstructs from the 13
surrounding context rows. High reconstruction error = the target window
is unusual given its temporal neighbors.
"""
from __future__ import annotations
import numpy as np


# Fixed geometry for 224×224 MAE input
N_ROWS      = 14    # 224 // 16 = 14 rows of patches
ROW_H       = 16    # pixels per row  = one MAE patch height
N_BEFORE    = 7     # context rows before target
N_AFTER     = 6     # context rows after target
TARGET_ROW  = N_BEFORE   # row index of the target window (0-indexed)


def _render_row(v_norm: np.ndarray, row_h: int, width: int) -> np.ndarray:
    """Filled area chart: fill from bottom to v_norm[x] * row_h.

    Parameters
    ----------
    v_norm : np.ndarray  [width]  values in [0, 1]
    Returns
    -------
    np.ndarray  [row_h, width]  float32
    """
    if len(v_norm) != width:
        v_norm = np.interp(
            np.linspace(0, 1, width),
            np.linspace(0, 1, len(v_norm)),
            v_norm,
        )
    # fill_start[x] = first pixel row (from top) that should be filled
    fill_start = np.clip(((1.0 - v_norm) * row_h).astype(int), 0, row_h)
    row_idx    = np.arange(row_h)[:, None]          # [row_h, 1]
    return (row_idx >= fill_start[None, :]).astype(np.float32)  # [row_h, width]


def build_tile_image(
    values: np.ndarray,
    center_win_idx: int,
    window_size: int,
    step_size: int,
    img_size: int = 224,
) -> np.ndarray:
    """Build a 14-row tile image centered on center_win_idx.

    Parameters
    ----------
    values         : np.ndarray  full signal
    center_win_idx : int         index of the target window
    window_size    : int
    step_size      : int
    img_size       : int         224

    Returns
    -------
    np.ndarray  [3, img_size, img_size]  float32, values in [0, 1]
    """
    T      = len(values)
    rows   = []
    offsets = list(range(-N_BEFORE, N_AFTER + 1))   # -7 … +6 (14 total)

    for off in offsets:
        w_idx = center_win_idx + off
        s     = w_idx * step_size
        e     = s + window_size

        if s < 0 or e > T:
            # Out-of-bounds → black row (zero-pad)
            row = np.zeros((ROW_H, img_size), dtype=np.float32)
        else:
            win    = values[s:e].astype(np.float32)
            lo, hi = win.min(), win.max()
            v_norm = np.full(len(win), 0.5) if hi - lo < 1e-8 \
                     else (win - lo) / (hi - lo)
            row    = _render_row(v_norm, ROW_H, img_size)

        rows.append(row)

    img = np.vstack(rows).astype(np.float32)            # [224, 224]
    return np.stack([img, img, img], axis=0)            # [3, 224, 224]


def render_tile_windows(
    values: np.ndarray,
    window_size: int = 224,
    step_size: int = 56,
    img_size: int = 224,
) -> np.ndarray:
    """Render tile images for every window in the signal.

    Returns
    -------
    np.ndarray  [N, 3, img_size, img_size]  float32
    """
    T      = len(values)
    starts = list(range(0, T - window_size + 1, step_size))
    N      = len(starts)
    imgs   = np.empty((N, 3, img_size, img_size), dtype=np.float32)
    for i in range(N):
        imgs[i] = build_tile_image(values, i, window_size, step_size, img_size)
    return imgs
