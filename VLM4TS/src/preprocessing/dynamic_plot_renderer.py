import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
from typing import List, Tuple, Optional


class DynamicPlotRenderer:
    """
    Replaces VLM4TS Stage 2 static plot with dynamic multi-scale plot.

    Two-panel layout:
    - Top panel: full series (gray) + red shading over all candidate intervals
    - Bottom panel: high-res zoom of top-1 candidate ± padding

    Drop-in replacement: produces PIL.Image like original renderer.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 6),
        dpi: int = 150,
        zoom_padding: float = 0.15,
        n_ticks_global: int = 20,
        n_ticks_zoom: int = 15,
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.zoom_padding = zoom_padding
        self.n_ticks_global = n_ticks_global
        self.n_ticks_zoom = n_ticks_zoom

    def render(
        self,
        series: np.ndarray,
        candidates: List[Tuple[int, int, float]],  # (start, end, score)
        series_name: str = "",
    ) -> Image.Image:
        """
        Render dynamic multi-scale plot.
        Returns PIL Image ready to send to GPT-4o.
        """
        if len(candidates) == 0:
            return self._render_fallback(series, series_name)

        T = len(series)
        top = self._get_top_candidate(candidates)
        zoom_start, zoom_end = self._compute_zoom_region(top[0], top[1], T)

        fig, (ax_global, ax_zoom) = plt.subplots(
            2, 1, figsize=self.figsize, dpi=self.dpi
        )

        # --- Top panel: global context ---
        ax_global.plot(
            np.arange(T), series,
            color='#555555', linewidth=0.7, alpha=0.7
        )
        # Shade all candidates (light red)
        for start, end, score in candidates:
            ax_global.axvspan(start, end, alpha=0.12, color='red')
        # Highlight top-1 candidate (darker red)
        ax_global.axvspan(top[0], top[1], alpha=0.30, color='red')

        ax_global.set_xlim(0, T - 1)
        ax_global.set_xticks(
            np.linspace(0, T - 1, self.n_ticks_global, dtype=int)
        )
        ax_global.tick_params(axis='x', labelsize=7)
        ax_global.grid(True, alpha=0.25, linewidth=0.5)
        ax_global.set_title(
            f'Global Context — {series_name}  '
            f'(red = ViT4TS candidates, {len(candidates)} total)',
            fontsize=9
        )
        ax_global.set_ylabel('Value', fontsize=8)

        # --- Bottom panel: zoom-in ---
        zoom_series = series[zoom_start:zoom_end]
        zoom_x = np.arange(zoom_start, zoom_end)

        ax_zoom.plot(zoom_x, zoom_series, color='#1a6faf', linewidth=1.0)
        # Shade actual candidate region within zoom
        ax_zoom.axvspan(top[0], top[1], alpha=0.30, color='red')

        ax_zoom.set_xlim(zoom_start, zoom_end - 1)
        ax_zoom.set_xticks(
            np.linspace(zoom_start, zoom_end - 1, self.n_ticks_zoom, dtype=int)
        )
        ax_zoom.tick_params(axis='x', labelsize=7)
        ax_zoom.grid(True, alpha=0.25, linewidth=0.5)
        ax_zoom.set_title(
            f'Zoom: Candidate [{top[0]}–{top[1]}]  '
            f'score={top[2]:.3f}  '
            f'(view: [{zoom_start}–{zoom_end}])',
            fontsize=9
        )
        ax_zoom.set_ylabel('Value', fontsize=8)
        ax_zoom.set_xlabel('Time Index (original)', fontsize=8)

        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _render_fallback(self, series: np.ndarray, name: str) -> Image.Image:
        """No candidates: render plain full-series plot."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(series, color='#555555', linewidth=0.7)
        ax.grid(True, alpha=0.25)
        ax.set_title(f'{name} (no candidates from ViT4TS)', fontsize=9)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _get_top_candidate(
        self, candidates: List[Tuple[int, int, float]]
    ) -> Tuple[int, int, float]:
        return max(candidates, key=lambda x: x[2])

    def _compute_zoom_region(
        self, start: int, end: int, series_len: int
    ) -> Tuple[int, int]:
        span = end - start
        pad = max(int(span * self.zoom_padding), 20)
        zoom_start = max(0, start - pad)
        zoom_end = min(series_len, end + pad)
        return zoom_start, zoom_end
