import math

import numpy as np

from config import BASE_FREQ, SIGMA_BASE, T
from data_gen import _clean_signal

# Sanity-9: causal root-cause / propagation structure across 6 channels, with no
# fixed grouping (channels play a different role every sample). Every non-root
# channel independently rolls one of three roles:
#   root      - the one channel where the anomaly originates (C1-C5 break type,
#               abrupt onset, full intensity)
#   cascade   - reacts strongly after root_onset + lag (abrupt, full intensity),
#               using the SAME break type as root (homogeneous) or a DIFFERENT
#               one (heterogeneous), 50/50
#   mild      - reacts weakly and gradually after root_onset + lag (ramped-in,
#               partial intensity), same homogeneous/heterogeneous draw
#   unrelated - stays on the shared clean pattern with only its own noise
#
# All perturbations are expressed as a blend between the untouched "clean"
# trajectory and a "fully transformed" trajectory (the same formulas as
# data_gen.py's gen_C1-C5, but parameterized by onset instead of a fixed middle
# third and persisting to the end of the series, since Sanity-9 only ever asks
# for onset_time, not an end boundary). The blend envelope controls how quickly
# (ramp_len) and how strongly (intensity) the transformation phases in, so
# "cascade" vs "mild" is just two different envelope settings on the same
# underlying break-type formulas.

CHANNEL_NAMES = [str(i) for i in range(1, 7)]
ROOT_TYPES = ["C1", "C2", "C3", "C4", "C5"]
ROOT_ONSET = 100

CASCADE_INTENSITY = 1.0
CASCADE_RAMP = 1
MILD_INTENSITY = 0.35
MILD_RAMP = 40


def _fully_transformed(clean: np.ndarray, x: np.ndarray, t: int, onset: int, freq: float, break_type: str, rng: np.random.Generator) -> np.ndarray:
    out = clean.copy()
    if onset >= t:
        return out
    if break_type == "C1":  # amplitude jump (2x)
        out[onset:] = clean[onset:] * 2.0
    elif break_type == "C2":  # flatline at the value seen at onset
        out[onset:] = clean[onset]
    elif break_type == "C3":  # full phase inversion
        out[onset:] = np.sin(2 * math.pi * freq * x[onset:] / t + math.pi)
    elif break_type == "C4":  # persistent phase drift to 90 degrees
        target = math.pi / 2
        ramp_len = min(50, t - onset)
        phase = np.zeros(t)
        phase[onset:onset + ramp_len] = np.linspace(0.0, target, ramp_len, endpoint=False)
        phase[onset + ramp_len:] = target
        out[onset:] = np.sin(2 * math.pi * freq * x[onset:] / t + phase[onset:])
    elif break_type == "C5":  # persistent frequency drift
        drift = rng.uniform(0.05, 0.10)
        phase0 = 2 * math.pi * freq * onset / t
        x_rel = np.arange(t - onset)
        out[onset:] = np.sin(phase0 + 2 * math.pi * freq * (1 + drift) * x_rel / t)
    else:
        raise ValueError(f"unknown break_type {break_type}")
    return out


def _envelope(t: int, onset: int, ramp_len: int, intensity: float) -> np.ndarray:
    env = np.zeros(t)
    if onset >= t:
        return env
    ramp_len = max(1, ramp_len)
    end_ramp = min(t, onset + ramp_len)
    if end_ramp > onset:
        env[onset:end_ramp] = intensity * (np.arange(end_ramp - onset) / ramp_len)
    env[end_ramp:] = intensity
    return env


def _apply_break(clean: np.ndarray, x: np.ndarray, t: int, onset: int, freq: float, break_type: str, intensity: float, ramp_len: int, rng: np.random.Generator) -> np.ndarray:
    transformed = _fully_transformed(clean, x, t, onset, freq, break_type, rng)
    env = _envelope(t, onset, ramp_len, intensity)
    return (1.0 - env) * clean + env * transformed


def _pick_type(rng: np.random.Generator, exclude: str | None = None) -> str:
    pool = [c for c in ROOT_TYPES if c != exclude] if exclude else ROOT_TYPES
    return pool[int(rng.integers(0, len(pool)))]


def generate_sample(
    n_affected: int,
    lag: int,
    seed: int,
    t: int = T,
    freq: float = BASE_FREQ,
    sigma: float = SIGMA_BASE,
    force_normal: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate one 6-channel sample. `n_affected` is how many of the 5 non-root
    channels react to the root (cascade or mild, chosen 50/50); the rest stay
    unrelated. `force_normal=True` ignores n_affected/lag and produces a fully
    normal sample (no root at all) as the negative control."""
    rng = np.random.default_rng(seed)
    x = np.arange(t)
    clean = _clean_signal(t, freq)

    if force_normal:
        roles = {name: {"role": "unrelated", "break_type": None, "onset": None, "homogeneous": None} for name in CHANNEL_NAMES}
        root, root_type, onset = None, None, None
    else:
        root_idx = int(rng.integers(0, 6))
        root = CHANNEL_NAMES[root_idx]
        root_type = _pick_type(rng)
        onset = ROOT_ONSET
        others = [n for n in CHANNEL_NAMES if n != root]
        n_affected = min(n_affected, len(others))
        affected_positions = set(rng.choice(len(others), size=n_affected, replace=False).tolist()) if n_affected else set()

        roles = {root: {"role": "root", "break_type": root_type, "onset": onset, "homogeneous": None}}
        for i, name in enumerate(others):
            if i in affected_positions:
                role = "cascade" if rng.random() < 0.5 else "mild"
                homogeneous = bool(rng.random() < 0.5)
                ctype = root_type if homogeneous else _pick_type(rng, exclude=root_type)
                eff_onset = min(onset + lag, t - 5)
                roles[name] = {"role": role, "break_type": ctype, "onset": eff_onset, "homogeneous": homogeneous}
            else:
                roles[name] = {"role": "unrelated", "break_type": None, "onset": None, "homogeneous": None}

    channels: dict[str, np.ndarray] = {}
    for name in CHANNEL_NAMES:
        info = roles[name]
        if info["role"] == "unrelated":
            ideal = clean
        elif info["role"] in ("root", "cascade"):
            ideal = _apply_break(clean, x, t, info["onset"], freq, info["break_type"], CASCADE_INTENSITY, CASCADE_RAMP, rng)
        else:  # mild
            ideal = _apply_break(clean, x, t, info["onset"], freq, info["break_type"], MILD_INTENSITY, MILD_RAMP, rng)
        channels[name] = ideal + rng.normal(0.0, sigma, t)

    affected = [n for n in CHANNEL_NAMES if roles[n]["role"] in ("cascade", "mild")]
    unaffected = [n for n in CHANNEL_NAMES if roles[n]["role"] == "unrelated"]

    meta = {
        "root_cause_channel": root if root is not None else "none",
        "affected_channels": affected,
        "unaffected_channels": unaffected,
        "onset_time": onset,
        "root_type": root_type,
        "roles": roles,
    }
    return channels, meta
