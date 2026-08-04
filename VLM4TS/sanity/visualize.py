from io import BytesIO
from pathlib import Path

import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DPI, FIGSIZE


def render_overlay(a: np.ndarray, b: np.ndarray, save_path: Path) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(a))
    ax.plot(x, a, color="#1f77b4", linewidth=1.2, label="Channel A")
    ax.plot(x, b, color="#ff7f0e", linewidth=1.2, label="Channel B")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    fig.savefig(save_path, format="png", dpi=DPI)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_boundary_overlay(
    a: np.ndarray,
    b: np.ndarray,
    left_candidates: dict,
    right_candidates: dict,
    corr: np.ndarray,
    threshold: float,
    save_path: Path,
) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * 1.6), dpi=DPI,
        sharex=True, gridspec_kw={"height_ratios": [2, 1]},
    )
    x = np.arange(len(a))
    ax.plot(x, a, color="#1f77b4", linewidth=1.2, label="Channel A")
    ax.plot(x, b, color="#ff7f0e", linewidth=1.2, label="Channel B")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_ylabel("Value")

    y_lo, y_hi = ax.get_ylim()
    label_y = y_hi - 0.08 * (y_hi - y_lo)
    for name, pos in left_candidates.items():
        ax.axvline(pos, color="#9467bd", linestyle="--", linewidth=1.0)
        ax.text(pos, label_y, name, color="#9467bd", fontsize=8, ha="center", va="top")
    for name, pos in right_candidates.items():
        ax.axvline(pos, color="#2ca02c", linestyle="--", linewidth=1.0)
        ax.text(pos, label_y, name, color="#2ca02c", fontsize=8, ha="center", va="top")
    ax.set_ylim(y_lo, y_hi)

    ax2.plot(x, corr, color="#555555", linewidth=1.0, label="Rolling correlation")
    ax2.axhline(threshold, color="#d62728", linestyle=":", linewidth=1.0, label=f"threshold={threshold}")
    for pos in left_candidates.values():
        ax2.axvline(pos, color="#9467bd", linestyle="--", linewidth=0.8)
    for pos in right_candidates.values():
        ax2.axvline(pos, color="#2ca02c", linestyle="--", linewidth=0.8)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("corr(A, B)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, format="png", dpi=DPI)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_boundary_overlay_no_corr(
    a: np.ndarray,
    b: np.ndarray,
    left_candidates: dict,
    right_candidates: dict,
    save_path: Path,
) -> str:
    """Same candidate markers as render_boundary_overlay, but without the rolling
    correlation subplot -- isolates whether the model anchors on the correlation
    curve rather than the raw channel shapes when both are shown."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(a))
    ax.plot(x, a, color="#1f77b4", linewidth=1.2, label="Channel A")
    ax.plot(x, b, color="#ff7f0e", linewidth=1.2, label="Channel B")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")

    y_lo, y_hi = ax.get_ylim()
    label_y = y_hi - 0.08 * (y_hi - y_lo)
    for name, pos in left_candidates.items():
        ax.axvline(pos, color="#9467bd", linestyle="--", linewidth=1.0)
        ax.text(pos, label_y, name, color="#9467bd", fontsize=8, ha="center", va="top")
    for name, pos in right_candidates.items():
        ax.axvline(pos, color="#2ca02c", linestyle="--", linewidth=1.0)
        ax.text(pos, label_y, name, color="#2ca02c", fontsize=8, ha="center", va="top")
    ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    fig.savefig(save_path, format="png", dpi=DPI)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_multichannel_overlay(channels: dict, save_path: Path) -> str:
    """Overlay N named channels with a fixed, distinguishable qualitative palette
    (tab10 for <=10 channels, tab20 beyond that) and a legend. Solid lines only
    (line-style variation is a possible follow-up axis, not used here)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(channels.keys())
    n = len(names)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")

    width = FIGSIZE[0] + 0.15 * max(0, n - 2)
    fig, ax = plt.subplots(figsize=(width, FIGSIZE[1]), dpi=DPI)
    x = np.arange(len(next(iter(channels.values()))))
    for i, name in enumerate(names):
        ax.plot(x, channels[name], color=cmap(i % cmap.N), linewidth=1.1, label=name)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=8, ncol=min(4, max(1, n // 4)))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    fig.savefig(save_path, format="png", dpi=DPI)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_subplot_multichannel(channels: dict, save_path: Path) -> str:
    """Stack N named channels as separate subplots sharing a time axis, one
    color per channel (tab10/tab20, matching render_multichannel_overlay).
    Channel identity is trivial to read (own row + label) but cross-channel
    lag/co-reaction is harder to eyeball than in the overlay version."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(channels.keys())
    n = len(names)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")

    fig, axes = plt.subplots(n, 1, figsize=(FIGSIZE[0], 1.1 * n), dpi=DPI, sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(next(iter(channels.values()))))
    for i, name in enumerate(names):
        axes[i].plot(x, channels[name], color=cmap(i % cmap.N), linewidth=1.0)
        axes[i].set_ylabel(f"Ch {name}", rotation=0, ha="right", va="center", fontsize=9)
        axes[i].grid(True, alpha=0.25, linewidth=0.5)
    axes[-1].set_xlabel("Time step")
    fig.tight_layout()

    fig.savefig(save_path, format="png", dpi=DPI)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_candidate_overlay(a: np.ndarray, b: np.ndarray, candidate_start: int, candidate_end: int, save_path: Path) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(a))
    ax.axvspan(candidate_start, candidate_end, color="#d62728", alpha=0.18, label="Candidate interval")
    ax.axvline(candidate_start, color="#d62728", linestyle="--", linewidth=1.0)
    ax.axvline(candidate_end, color="#d62728", linestyle="--", linewidth=1.0)
    ax.plot(x, a, color="#1f77b4", linewidth=1.2, label="Channel A")
    ax.plot(x, b, color="#ff7f0e", linewidth=1.2, label="Channel B")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    fig.savefig(save_path, format="png", dpi=DPI)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
