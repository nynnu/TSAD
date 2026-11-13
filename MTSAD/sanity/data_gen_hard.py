import math
from typing import Callable

import numpy as np

from config import BASE_FREQ, SIGMA_BASE, T
from data_gen import _clean_signal, _meta, middle_third

# Round-1 "hard case" generators: anomaly/negative-control types that C0-C6 never
# covered, because every C0-C6 pair shares the exact same clean waveform (so
# "maintained" always looked visually identical, and every "broken" case was a
# sharp, localized change). These test whether the model reasons about the
# abstract relationship or uses visual-similarity-of-shape as a proxy for it.
#
# H1a-H1d: pairs that look different from t=0 (different frequency / scale /
#          sign / phase) but whose relationship never changes -- ground truth is
#          always "maintained". These are hard negative controls for the
#          "different shape == broken" confound.
# H2:      noise variance spikes locally in the middle third while the mean
#          relationship is untouched -- ground truth "maintained" under the
#          strict "relationship, not absolute value" definition.
# H3:      a slow trend divergence starting at the middle third and persisting
#          to the end -- ground truth "broken", gradual/global rather than
#          sharp/local.
# H4:      periodic structure in B is replaced by a random walk in the middle
#          third (loses periodicity without going flat) -- ground truth "broken".


def gen_H1a(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    """Different frequency, fixed ratio, from t=0. Always maintained."""
    rng = np.random.default_rng(seed)
    a = _clean_signal(t, freq=BASE_FREQ) + rng.normal(0.0, sigma, t)
    b = _clean_signal(t, freq=BASE_FREQ * 5 / 3) + rng.normal(0.0, sigma, t)
    return a, b, _meta("maintained", None, None)


def gen_H1b(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    """Different scale/offset, fixed linear transform, from t=0. Always maintained."""
    rng = np.random.default_rng(seed)
    clean = _clean_signal(t)
    a = clean + rng.normal(0.0, sigma, t)
    b = 3.0 * clean + 2.0 + rng.normal(0.0, sigma, t)
    return a, b, _meta("maintained", None, None)


def gen_H1c(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    """Always anti-phase (B = -A) from t=0, never flips. Always maintained."""
    rng = np.random.default_rng(seed)
    clean = _clean_signal(t)
    a = clean + rng.normal(0.0, sigma, t)
    b = -clean + rng.normal(0.0, sigma, t)
    return a, b, _meta("maintained", None, None)


def gen_H1d(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    """Fixed quarter-period phase offset from t=0, never changes. Always maintained."""
    rng = np.random.default_rng(seed)
    a = _clean_signal(t) + rng.normal(0.0, sigma, t)
    b = _clean_signal(t, phase=math.pi / 2) + rng.normal(0.0, sigma, t)
    return a, b, _meta("maintained", None, None)


def gen_H2(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    """B's noise variance spikes in the middle third; mean relationship intact.
    Strictly maintained under the "relationship, not absolute value" definition."""
    rng = np.random.default_rng(seed)
    clean = _clean_signal(t)
    a = clean + rng.normal(0.0, sigma, t)
    b = clean.copy()
    s, e = middle_third(t)
    noisy_sigma = sigma * rng.uniform(4.0, 6.0)
    b[:s] += rng.normal(0.0, sigma, s)
    b[s:e] += rng.normal(0.0, noisy_sigma, e - s)
    b[e:] += rng.normal(0.0, sigma, t - e)
    return a, b, _meta("maintained", None, None)


def gen_H3(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    """B's trend slope diverges from A's starting at the middle third and never
    recovers -- gradual/global divergence rather than a sharp local change."""
    rng = np.random.default_rng(seed)
    x = np.arange(t)
    oscillation = _clean_signal(t)
    a = oscillation + rng.normal(0.0, sigma, t)
    s, _ = middle_third(t)
    extra_trend = np.zeros(t)
    slope = rng.uniform(0.008, 0.014)
    extra_trend[s:] = slope * (x[s:] - s)
    b = oscillation + extra_trend + rng.normal(0.0, sigma, t)
    return a, b, _meta("broken", s, t)


def gen_H4(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    """B's periodic oscillation is replaced by a random walk in the middle
    third (loses periodicity without flattening), then resumes."""
    rng = np.random.default_rng(seed)
    clean = _clean_signal(t)
    a = clean + rng.normal(0.0, sigma, t)
    b = clean.copy()
    s, e = middle_third(t)
    walk = np.cumsum(rng.normal(0.0, sigma * 2.0, e - s))
    b[s:e] = clean[s] + walk
    b = b + rng.normal(0.0, sigma, t)
    return a, b, _meta("broken", s, e)


GENERATORS: dict[str, Callable] = {
    "H1a": gen_H1a,
    "H1b": gen_H1b,
    "H1c": gen_H1c,
    "H1d": gen_H1d,
    "H2": gen_H2,
    "H3": gen_H3,
    "H4": gen_H4,
}

CASES = list(GENERATORS)
_CASE_INDEX = {name: i for i, name in enumerate(CASES)}


def generate_instance(case_type: str, index: int, t: int = T):
    seed = 10_000 * _CASE_INDEX[case_type] + index
    a, b, meta = GENERATORS[case_type](seed=seed, t=t)
    case_id = f"{case_type}_{index:03d}"
    return case_id, case_type, a, b, meta


def generate_all(n_per_case: int, t: int = T, cases: list[str] | None = None):
    rows = []
    for case_type in cases or CASES:
        for i in range(n_per_case):
            rows.append(generate_instance(case_type, i, t))
    return rows
