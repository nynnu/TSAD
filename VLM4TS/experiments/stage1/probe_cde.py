"""
CDE Visual Probe — Minimum Viable Test
=======================================
Delay-Embedding + KDE density maps for SMAP F-1 and F-3.

Before committing to a full experiment, we answer two questions:
  Q1. Do normal windows produce visually distinct density maps from
      anomalous windows in phase space?
  Q2. Is the embedding discriminative enough to survive window-scale sparsity?

τ is computed automatically as the first zero-crossing of the signal ACF
(Takens' theorem: optimal τ = decorrelation time).

Output
------
  results_cde_probe/cde_probe_F-1.png
  results_cde_probe/cde_probe_F-3.png
  results_cde_probe/cde_probe_stats.txt  — quantitative discrimination metrics

Usage
-----
  python experiments/probe_cde.py
"""

from __future__ import annotations

import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
OUT_DIR      = os.path.join(PROJECT_ROOT, "results_cde_probe")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 224
STEP        = 56         # window_step_ratio = 4.0
IMAGE_SIZE  = 224

# ── ACF-based τ ───────────────────────────────────────────────────────────────

def compute_tau(x: np.ndarray, max_lag_frac: float = 0.25) -> int:
    """First zero-crossing of normalized ACF — principled τ for Takens embedding."""
    x_c  = x - x.mean()
    var  = float(np.dot(x_c, x_c))
    if var < 1e-10:
        return 1
    max_lag = max(2, int(len(x) * max_lag_frac))
    for tau in range(1, max_lag):
        acf = float(np.dot(x_c[:-tau], x_c[tau:])) / var
        if acf <= 0:
            return tau
    return max(1, max_lag // 4)


# ── CDE image ─────────────────────────────────────────────────────────────────

def cde_image(window: np.ndarray, tau: int,
              x_range: tuple[float, float],
              image_size: int = IMAGE_SIZE,
              sigma_px: float | None = None) -> np.ndarray:
    """
    Delay-embedding density map for one sliding window.

    Parameters
    ----------
    window   : 1D signal segment, shape [W]
    tau      : delay (from full-signal ACF zero-crossing)
    x_range  : (global_min, global_max) — fixed axes for cross-window comparability
    sigma_px : Gaussian blur radius in pixels (None → Scott's rule)

    Returns
    -------
    density  : float32 array [image_size, image_size], normalised to [0, 1]
    """
    tau = min(tau, len(window) - 2)
    u   = window[:-tau]       # x_t
    v   = window[tau:]        # x_{t+tau}
    n   = len(u)

    x_min, x_max = x_range
    span = x_max - x_min + 1e-8

    # Map to pixel coordinates using global range (preserves inter-window scale)
    u_px = np.clip(((u - x_min) / span * (image_size - 1)).astype(int), 0, image_size - 1)
    v_px = np.clip(((v - x_min) / span * (image_size - 1)).astype(int), 0, image_size - 1)

    # Accumulate into 2D histogram (v = row / y-axis, u = col / x-axis)
    H = np.zeros((image_size, image_size), dtype=np.float32)
    np.add.at(H, (v_px, u_px), 1.0)

    # Bandwidth: Scott's rule adapted to pixel space
    if sigma_px is None:
        std_px   = max(1.0, float(np.std(u_px)))
        sigma_px = max(3.0, (n ** (-1.0 / 6.0)) * std_px)

    density = gaussian_filter(H, sigma=sigma_px)
    if density.max() > 1e-10:
        density /= density.max()
    return density


# ── Window utilities ──────────────────────────────────────────────────────────

def find_windows(timestamps: np.ndarray, values: np.ndarray,
                 anomaly_ivs: list,
                 n_normal: int = 4, n_anom: int = 4):
    """
    Return indices of purely-normal and most-anomalous windows.

    Spacing: normal windows are evenly spread across the pre-anomaly region
    so they represent different phases of the signal, not just one patch.
    """
    T        = len(timestamps)
    n_wins   = (T - WINDOW_SIZE) // STEP + 1

    is_anom  = np.zeros(T, dtype=bool)
    for s, e in anomaly_ivs:
        is_anom[(timestamps >= s) & (timestamps <= e)] = True

    anom_idx   = int(np.where(is_anom)[0][0])   # first anomalous timestep
    safe_limit = max(0, anom_idx - WINDOW_SIZE)  # windows that can't touch anomaly

    # Normal windows: sample evenly from safe region
    safe_wins = [i for i in range(n_wins) if i * STEP + WINDOW_SIZE <= safe_limit]
    if not safe_wins:
        safe_wins = [i for i in range(n_wins)
                     if is_anom[i * STEP: i * STEP + WINDOW_SIZE].sum() == 0]
    step_n   = max(1, len(safe_wins) // n_normal)
    normal_wins = safe_wins[::step_n][:n_normal]

    # Anomalous windows: sorted by overlap fraction (descending)
    anom_wins = []
    for i in range(n_wins):
        overlap = int(is_anom[i * STEP: i * STEP + WINDOW_SIZE].sum())
        if overlap > 0:
            anom_wins.append((overlap, i))
    anom_wins.sort(reverse=True)
    anom_wins = [i for _, i in anom_wins[:n_anom]]

    return normal_wins, anom_wins


# ── Discrimination metric ─────────────────────────────────────────────────────

def discrimination_score(normal_imgs: list[np.ndarray],
                          anom_imgs:   list[np.ndarray]) -> dict:
    """
    Mean L2 distance between flat density vectors:
      within_normal : avg pairwise distance among normal images
      cross         : avg distance normal↔anomalous
    Ratio cross/within > 1 means anomalies are more distinct than normals.
    """
    def l2(a, b):
        return float(np.linalg.norm(a.ravel() - b.ravel()))

    within, cross = [], []
    for i in range(len(normal_imgs)):
        for j in range(i + 1, len(normal_imgs)):
            within.append(l2(normal_imgs[i], normal_imgs[j]))
    for ni in normal_imgs:
        for ai in anom_imgs:
            cross.append(l2(ni, ai))

    return {
        "within_normal_L2": float(np.mean(within)) if within else 0.0,
        "cross_L2":         float(np.mean(cross))  if cross  else 0.0,
        "ratio":            float(np.mean(cross) / np.mean(within))
                            if within and cross and np.mean(within) > 0 else 0.0,
    }


# ── Per-signal probe ──────────────────────────────────────────────────────────

def probe_signal(sig: str, tau_override: int | None = None) -> dict:
    df         = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "SMAP", f"{sig}.csv"))
    timestamps = df.iloc[:, 0].values
    values     = df["value"].values if "value" in df.columns else df.iloc[:, 1].values

    gt = {}
    with open(os.path.join(PROJECT_ROOT, "data", "anomalies.csv"), encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    anomaly_ivs = gt[sig]

    # τ from ACF, capped so each window has ≥ WINDOW_SIZE*7/8 phase-space points
    tau_acf = tau_override if tau_override is not None else compute_tau(values)
    tau     = min(tau_acf, WINDOW_SIZE // 8)
    print(f"\n  SMAP {sig}  T={len(values)}  τ_acf={tau_acf}  τ_used={tau}")

    x_range     = (float(values.min()), float(values.max()))
    normal_wins, anom_wins = find_windows(timestamps, values, anomaly_ivs,
                                          n_normal=4, n_anom=4)

    print(f"  Normal window indices : {normal_wins}")
    print(f"  Anomaly window indices: {anom_wins}")

    # Render CDE images
    def render(win_idx):
        start = win_idx * STEP
        seg   = values[start: start + WINDOW_SIZE]
        img   = cde_image(seg, tau, x_range)
        return img, seg

    normal_data = [render(i) for i in normal_wins]
    anom_data   = [render(i) for i in anom_wins]
    normal_imgs = [d[0] for d in normal_data]
    anom_imgs   = [d[0] for d in anom_data]
    normal_segs = [d[1] for d in normal_data]
    anom_segs   = [d[1] for d in anom_data]

    # ── Figure ────────────────────────────────────────────────────────────────
    n_col    = max(len(normal_wins), len(anom_wins))
    fig_w    = max(12, n_col * 3.5)
    fig      = plt.figure(figsize=(fig_w, 10), facecolor="#1a1a2e")
    fig.suptitle(f"CDE Visual Probe  —  SMAP {sig}   (τ={tau})",
                 fontsize=14, color="white", y=0.98)

    # 4 rows: normal line | normal CDE | anomalous line | anomalous CDE
    gs = gridspec.GridSpec(4, n_col, figure=fig, hspace=0.05, wspace=0.08,
                           top=0.93, bottom=0.04, left=0.04, right=0.97)

    ROW_LABELS = ["Normal\n(line)", "Normal\n(CDE)", "Anomalous\n(line)", "Anomalous\n(CDE)"]
    ROW_COLORS = ["#4fc3f7", "#4fc3f7", "#ef5350", "#ef5350"]

    for col, (seg, img) in enumerate(zip(normal_segs, normal_imgs)):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(seg, color="#4fc3f7", linewidth=0.8)
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        if col == 0:
            ax.set_ylabel("Normal\n(line)", color="#4fc3f7", fontsize=8)
        ax.set_title(f"win {normal_wins[col]}", color="#aaa", fontsize=7)
        ax.set_ylim(*x_range)

        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(img, cmap="plasma", origin="lower", aspect="auto",
                   vmin=0, vmax=1)
        ax2.set_facecolor("#12122a")
        ax2.set_xticks([]); ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel("Normal\n(CDE)", color="#4fc3f7", fontsize=8)

    for col, (seg, img) in enumerate(zip(anom_segs, anom_imgs)):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(seg, color="#ef5350", linewidth=0.8)
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="white", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        if col == 0:
            ax.set_ylabel("Anomalous\n(line)", color="#ef5350", fontsize=8)
        ax.set_title(f"win {anom_wins[col]}", color="#aaa", fontsize=7)
        ax.set_ylim(*x_range)

        ax2 = fig.add_subplot(gs[3, col])
        ax2.imshow(img, cmap="plasma", origin="lower", aspect="auto",
                   vmin=0, vmax=1)
        ax2.set_facecolor("#12122a")
        ax2.set_xticks([]); ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel("Anomalous\n(CDE)", color="#ef5350", fontsize=8)

    out_path = os.path.join(OUT_DIR, f"cde_probe_{sig}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.close("all")
    print(f"  Saved → {out_path}")

    # ── Discrimination metrics ────────────────────────────────────────────────
    metrics = discrimination_score(normal_imgs, anom_imgs)
    print(f"  within-normal L2 : {metrics['within_normal_L2']:.4f}")
    print(f"  cross  (N vs A)  : {metrics['cross_L2']:.4f}")
    print(f"  ratio cross/within: {metrics['ratio']:.3f}  "
          f"({'DISCRIMINATIVE ✓' if metrics['ratio'] > 1.5 else 'WEAK' if metrics['ratio'] > 1.0 else 'NOT DISCRIMINATIVE ✗'})")

    return {"signal": sig, "tau": tau, **metrics}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CDE Visual Probe")
    print("Signals : SMAP F-1 (subtle), SMAP F-3 (stuck sensor)")
    print("=" * 60)

    results = []
    for sig in ["F-3", "F-1"]:
        r = probe_signal(sig)
        results.append(r)

    # ── Summary text ─────────────────────────────────────────────────────────
    stats_path = os.path.join(OUT_DIR, "cde_probe_stats.txt")
    with open(stats_path, "w") as f:
        f.write("CDE Probe — Discrimination Metrics\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"SMAP {r['signal']}  (τ={r['tau']})\n")
            f.write(f"  within-normal L2  : {r['within_normal_L2']:.4f}\n")
            f.write(f"  cross L2 (N vs A) : {r['cross_L2']:.4f}\n")
            f.write(f"  ratio             : {r['ratio']:.3f}\n")
            verdict = ("DISCRIMINATIVE — CDE worth pursuing"
                       if r["ratio"] > 1.5 else
                       "MARGINAL — CDE may help, needs more investigation"
                       if r["ratio"] > 1.0 else
                       "NOT DISCRIMINATIVE — CDE unlikely to help this signal")
            f.write(f"  verdict           : {verdict}\n\n")
        f.write("Decision criterion:\n")
        f.write("  ratio > 1.5 → anomalous windows are visually distinct → run full experiment\n")
        f.write("  ratio < 1.0 → CDE encoding is not discriminative → do not proceed\n")

    print(f"\nStats saved → {stats_path}")
    print("\nCheck the PNG files for visual inspection.")
    print("If anomalous CDE images look visually distinct from normal ones, proceed.")
