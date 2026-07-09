import numpy as np

from config import T
from data_gen import GENERATORS

# Break types are restricted to the three patterns Sanity-1/3/4 showed GPT-4o
# reliably detects at the 2-channel scale (amplitude jump, flatline, phase
# flip). Using only well-established break types isolates the variable under
# test here (channel count / visual crowding) from the confound of break-type
# difficulty that Sanity-1 already characterized separately.
BREAK_TYPES = ["C1", "C2", "C3"]


def _pair_names(index: int) -> tuple[str, str]:
    return f"P{index + 1}a", f"P{index + 1}b"


def generate_scene(n_channels: int, n_broken_pairs: int, seed: int, t: int = T) -> dict:
    """Compose n_channels/2 independent synchronized pairs into one multi-channel
    scene. Each pair is generated exactly like a Sanity-1 case (shared clean
    signal + independent noise, optionally perturbed in the middle third).
    `n_broken_pairs` of the pairs are perturbed (ground truth "broken"); the
    rest stay maintained (ground truth "maintained").
    """
    if n_channels % 2 != 0:
        raise ValueError("n_channels must be even (channels are shown as synchronized pairs)")
    n_pairs = n_channels // 2
    if not 0 <= n_broken_pairs <= n_pairs:
        raise ValueError(f"n_broken_pairs must be in [0, {n_pairs}]")

    rng = np.random.default_rng(seed)
    broken_idx = set(rng.choice(n_pairs, size=n_broken_pairs, replace=False).tolist()) if n_broken_pairs else set()

    channels: dict[str, np.ndarray] = {}
    pair_names: list[str] = []
    ground_truth: dict[str, str] = {}
    break_types: dict[str, str | None] = {}

    for i in range(n_pairs):
        pair_name = f"P{i + 1}"
        pair_seed = seed * 1000 + i
        name_a, name_b = _pair_names(i)

        if i in broken_idx:
            break_type = BREAK_TYPES[int(rng.integers(0, len(BREAK_TYPES)))]
            a, b, meta = GENERATORS[break_type](seed=pair_seed, t=t)
        else:
            break_type = None
            a, b, meta = GENERATORS["C0"](seed=pair_seed, t=t)

        channels[name_a] = a
        channels[name_b] = b
        pair_names.append(pair_name)
        ground_truth[pair_name] = meta["label"]
        break_types[pair_name] = break_type

    return {
        "channels": channels,
        "pair_names": pair_names,
        "ground_truth": ground_truth,
        "break_types": break_types,
    }
