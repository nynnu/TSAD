import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from PIL import Image

from preprocessing.dynamic_plot_renderer import DynamicPlotRenderer


@pytest.fixture
def renderer():
    return DynamicPlotRenderer()


@pytest.fixture
def series_500():
    rng = np.random.default_rng(0)
    return rng.standard_normal(500)


# ---------------------------------------------------------------------------
# Core render tests
# ---------------------------------------------------------------------------

def test_render_returns_pil_image(renderer, series_500):
    img = renderer.render(series_500, [(200, 250, 0.8)], "test")
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_no_candidates_returns_fallback(renderer, series_500):
    # Must not crash and must return a valid PIL image
    img = renderer.render(series_500, [], "fallback")
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_render_multiple_candidates(renderer, series_500):
    candidates = [(50, 80, 0.5), (200, 250, 0.9), (350, 400, 0.3)]
    img = renderer.render(series_500, candidates, "multi")
    assert isinstance(img, Image.Image)


# ---------------------------------------------------------------------------
# Zoom region tests
# ---------------------------------------------------------------------------

def test_zoom_region_clamping(renderer):
    # Candidate near start — zoom_start must not go below 0
    zoom_start, zoom_end = renderer._compute_zoom_region(5, 15, 1000)
    assert zoom_start >= 0
    assert zoom_end <= 1000

    # Candidate near end — zoom_end must not exceed series length
    zoom_start, zoom_end = renderer._compute_zoom_region(990, 998, 1000)
    assert zoom_start >= 0
    assert zoom_end <= 1000


def test_zoom_region_padding(renderer):
    # candidate [400, 500] in series length 1000
    # span = 100, pad = max(int(100 * 0.15), 20) = max(15, 20) = 20
    zoom_start, zoom_end = renderer._compute_zoom_region(400, 500, 1000)
    assert zoom_start == 380    # 400 - 20
    assert zoom_end == 520      # 500 + 20


def test_zoom_region_minimum_padding(renderer):
    # Very short candidate (span < 134) → minimum pad of 20 kicks in
    zoom_start, zoom_end = renderer._compute_zoom_region(100, 105, 500)
    span = 5
    expected_pad = max(int(span * renderer.zoom_padding), 20)
    assert zoom_start == max(0, 100 - expected_pad)
    assert zoom_end == min(500, 105 + expected_pad)


def test_zoom_region_large_padding(renderer):
    # Large candidate (span=200) → 15% pad = 30 > 20
    zoom_start, zoom_end = renderer._compute_zoom_region(300, 500, 1000)
    span = 200
    expected_pad = max(int(span * renderer.zoom_padding), 20)  # max(30, 20) = 30
    assert zoom_start == 300 - expected_pad
    assert zoom_end == 500 + expected_pad


# ---------------------------------------------------------------------------
# Top-candidate selection
# ---------------------------------------------------------------------------

def test_top_candidate_selection(renderer):
    candidates = [(0, 50, 0.3), (100, 150, 0.9), (200, 250, 0.5)]
    top = renderer._get_top_candidate(candidates)
    assert top == (100, 150, 0.9)


def test_top_candidate_single(renderer):
    candidates = [(10, 20, 0.4)]
    top = renderer._get_top_candidate(candidates)
    assert top == (10, 20, 0.4)


def test_top_candidate_tie_is_deterministic(renderer):
    # When two candidates share the max score, result must be stable across calls
    candidates = [(0, 10, 0.7), (20, 30, 0.7)]
    result1 = renderer._get_top_candidate(candidates)
    result2 = renderer._get_top_candidate(candidates)
    assert result1 == result2


# ---------------------------------------------------------------------------
# Image dimension sanity
# ---------------------------------------------------------------------------

def test_render_resolution_matches_figsize(renderer, series_500):
    # With figsize=(14,6) and dpi=150, width ≈ 14*150=2100 px (bbox_inches='tight'
    # may trim a small margin, so allow ±100 px tolerance)
    img = renderer.render(series_500, [(100, 200, 0.8)], "res_test")
    expected_width = renderer.figsize[0] * renderer.dpi
    assert abs(img.size[0] - expected_width) < 100

    fallback = renderer._render_fallback(series_500, "fallback")
    assert abs(fallback.size[0] - expected_width) < 100
