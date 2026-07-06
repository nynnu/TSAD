"""
RGBTimeSeriesEncoder
====================
Encodes a 1-D time series window into a 224×224 3-channel image.

Supported strategies
--------------------
  trend_vol  (default)
      R = original series
      G = deviation from moving average  (trend shift)
      B = rolling standard deviation     (volatility)

  grayscale
      R = G = B = original series  (identical to existing pipeline)

  vetime
      R = original series
      G = double moving average (MA of MA, smooth trend)
      B = residual (X − double MA)

  freq_split
      R = original series
      G = low-pass  (single MA)
      B = high-pass (X − low-pass)

Output: np.ndarray [3, H, W] float32 in [0, 1]
        white background, black line — same convention as draw_image()
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

STRATEGIES = ("trend_vol", "grayscale", "vetime", "freq_split")


class RGBTimeSeriesEncoder:
    def __init__(
        self,
        strategy:    str   = "trend_vol",
        image_size:  int   = 224,
        ma_window:   int   = 11,
        std_window:  int   = 11,
        line_width:  float = 1.0,
        dpi:         int   = 100,
    ):
        assert strategy in STRATEGIES, \
            f"strategy must be one of {STRATEGIES}, got '{strategy}'"
        assert ma_window  % 2 == 1, "ma_window must be odd"
        assert std_window % 2 == 1, "std_window must be odd"
        self.strategy   = strategy
        self.image_size = image_size
        self.ma_window  = ma_window
        self.std_window = std_window
        self.line_width = line_width
        self.dpi        = dpi

    # ------------------------------------------------------------------
    def _render_channel(self, values: np.ndarray,
                        y_min: float, y_max: float) -> np.ndarray:
        """1-D array → [H, W] float32 in [0, 1], white bg / black line."""
        fig_size = self.image_size / self.dpi
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size), dpi=self.dpi)
        ax.plot(values, color="black", linewidth=self.line_width)
        ax.set_xlim(0, max(len(values) - 1, 1))
        if abs(y_max - y_min) < 1e-8:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        ax.set_ylim(y_min, y_max)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        buf.close()
        return arr

    def _ma(self, x: np.ndarray, window: int) -> np.ndarray:
        pad    = window // 2
        padded = np.pad(x, pad, mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(x)]

    def _rolling_std(self, x: np.ndarray) -> np.ndarray:
        pad = self.std_window // 2
        return np.array([
            x[max(0, i - pad): min(len(x), i + pad + 1)].std()
            for i in range(len(x))
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def encode(
        self,
        window:            np.ndarray,
        series_global_min: float = None,
        series_global_max: float = None,
    ) -> np.ndarray:
        """
        Encode a 1-D window into a 3-channel image.

        series_global_min / series_global_max should be computed once
        over the FULL series before the sliding-window loop.

        Returns
        -------
        np.ndarray [3, image_size, image_size] float32 in [0, 1]
        """
        w = np.asarray(window, dtype=np.float32)
        g_min = float(w.min()) if series_global_min is None else series_global_min
        g_max = float(w.max()) if series_global_max is None else series_global_max

        if self.strategy == "grayscale":
            ch = self._render_channel(w, g_min, g_max)
            return np.stack([ch, ch, ch])

        elif self.strategy == "trend_vol":
            ch_r = self._render_channel(w, g_min, g_max)
            ma   = self._ma(w, self.ma_window)
            dev  = w - ma
            d_abs = max(float(np.abs(dev).max()), 1e-8)
            ch_g  = self._render_channel(dev, -d_abs, d_abs)
            rstd  = self._rolling_std(w)
            ch_b  = self._render_channel(rstd, 0.0, max(float(rstd.max()), 1e-8))
            return np.stack([ch_r, ch_g, ch_b])

        elif self.strategy == "vetime":
            ch_r  = self._render_channel(w, g_min, g_max)
            ma2   = self._ma(self._ma(w, self.ma_window), self.ma_window)
            ch_g  = self._render_channel(ma2, g_min, g_max)
            res   = w - ma2
            r_abs = max(float(np.abs(res).max()), 1e-8)
            ch_b  = self._render_channel(res, -r_abs, r_abs)
            return np.stack([ch_r, ch_g, ch_b])

        elif self.strategy == "freq_split":
            ch_r   = self._render_channel(w, g_min, g_max)
            lp     = self._ma(w, self.ma_window)
            ch_g   = self._render_channel(lp, g_min, g_max)
            hp     = w - lp
            hp_abs = max(float(np.abs(hp).max()), 1e-8)
            ch_b   = self._render_channel(hp, -hp_abs, hp_abs)
            return np.stack([ch_r, ch_g, ch_b])

    # ------------------------------------------------------------------
    def encode_to_pil(
        self,
        window:            np.ndarray,
        series_global_min: float = None,
        series_global_max: float = None,
    ) -> Image.Image:
        """Returns PIL Image (RGB mode) for visualisation."""
        arr  = self.encode(window, series_global_min, series_global_max)
        u8   = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(u8.transpose(1, 2, 0), mode="RGB")
