import math
from typing import Callable

import numpy as np

from config import BASE_FREQ, SIGMA_BASE, T


def middle_third(t: int) -> tuple[int, int]:
    return t // 3, 2 * t // 3


def _clean_signal(t: int, freq: float = BASE_FREQ, phase: np.ndarray | float = 0.0) -> np.ndarray:
    x = np.arange(t)
    return np.sin(2 * math.pi * freq * x / t + phase)


def _base_pair(seed: int, t: int = T, sigma: float = SIGMA_BASE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean = _clean_signal(t)
    a = clean + rng.normal(0.0, sigma, t)
    b = clean + rng.normal(0.0, sigma, t)
    return a, b, clean


def _meta(label: str, start: int | None, end: int | None) -> dict:
    return {"label": label, "break_start": start, "break_end": end}


def gen_C0(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    a, b, _ = _base_pair(seed, t, sigma)
    return a, b, _meta("maintained", None, None)


def gen_C1(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    a, b, _ = _base_pair(seed, t, sigma)
    s, e = middle_third(t)
    b[s:e] *= 2.0
    return a, b, _meta("broken", s, e)


def gen_C2(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    a, b, _ = _base_pair(seed, t, sigma)
    rng = np.random.default_rng(seed + 10_000)
    s, e = middle_third(t)
    b[s:e] = float(np.mean(b[s:e])) + rng.normal(0.0, sigma * 0.1, e - s)
    return a, b, _meta("broken", s, e)


def gen_C3(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    a, b, _ = _base_pair(seed, t, sigma)
    s, e = middle_third(t)
    b[s:e] = -a[s:e]
    return a, b, _meta("broken", s, e)


def gen_C4(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    rng = np.random.default_rng(seed)
    x = np.arange(t)
    clean_a = _clean_signal(t)
    s, e = middle_third(t)
    target = math.pi / 2
    phase = np.zeros(t)
    phase[s:e] = np.linspace(0.0, target, e - s, endpoint=False)
    phase[e:] = target
    clean_b = np.sin(2 * math.pi * BASE_FREQ * x / t + phase)
    a = clean_a + rng.normal(0.0, sigma, t)
    b = clean_b + rng.normal(0.0, sigma, t)
    return a, b, _meta("broken", s, t)


def gen_C5(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    rng = np.random.default_rng(seed)
    clean = _clean_signal(t)
    s, e = middle_third(t)
    drift = rng.uniform(0.05, 0.10)
    x_mid = np.arange(e - s)
    b_clean = clean.copy()
    phase0 = 2 * math.pi * BASE_FREQ * s / t
    b_clean[s:e] = np.sin(phase0 + 2 * math.pi * BASE_FREQ * (1 + drift) * x_mid / t)
    a = clean + rng.normal(0.0, sigma, t)
    b = b_clean + rng.normal(0.0, sigma, t)
    return a, b, _meta("broken", s, e)


def gen_C6(seed: int, t: int = T, sigma: float = SIGMA_BASE):
    rng = np.random.default_rng(seed)
    clean = _clean_signal(t)
    noisy_sigma = sigma * rng.uniform(4.0, 6.0)
    a = clean + rng.normal(0.0, sigma, t)
    b = clean + rng.normal(0.0, noisy_sigma, t)
    return a, b, _meta("maintained", None, None)


GENERATORS: dict[str, Callable] = {
    "C0": gen_C0,
    "C1": gen_C1,
    "C2": gen_C2,
    "C3": gen_C3,
    "C4": gen_C4,
    "C5": gen_C5,
    "C6": gen_C6,
}


def generate_instance(case_type: str, index: int, t: int = T):
    seed = 10_000 * int(case_type[1:]) + index
    a, b, meta = GENERATORS[case_type](seed=seed, t=t)
    case_id = f"{case_type}_{index:03d}"
    return case_id, case_type, a, b, meta


def generate_all(n_per_case: int, t: int = T, cases: list[str] | None = None):
    rows = []
    for case_type in cases or list(GENERATORS):
        for i in range(n_per_case):
            rows.append(generate_instance(case_type, i, t))
    return rows
