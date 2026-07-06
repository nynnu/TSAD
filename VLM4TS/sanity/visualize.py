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
