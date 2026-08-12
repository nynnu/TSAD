import numpy as np

from config import T

WINDOW = 24
THRESHOLD = 0.5
MARGIN = 15
MIN_GAP = 6

LEFT_LABELS = ["L0", "L1", "L2", "L3"]
RIGHT_LABELS = ["R0", "R1", "R2", "R3"]


def rolling_correlation(a: np.ndarray, b: np.ndarray, window: int = WINDOW) -> np.ndarray:
    t = len(a)
    half = window // 2
    corr = np.full(t, np.nan)
    for i in range(half, t - half):
        wa = a[i - half:i + half]
        wb = b[i - half:i + half]
        if wa.std() < 1e-9 or wb.std() < 1e-9:
            corr[i] = 0.0
        else:
            corr[i] = float(np.corrcoef(wa, wb)[0, 1])
    return corr


def _first_below(corr: np.ndarray, idxs: list[int], threshold: float) -> int | None:
    for i in idxs:
        if not np.isnan(corr[i]) and corr[i] < threshold:
            return i
    return None


def _first_above(corr: np.ndarray, idxs: list[int], threshold: float) -> int | None:
    for i in idxs:
        if not np.isnan(corr[i]) and corr[i] > threshold:
            return i
    return None


def _argmin(corr: np.ndarray, idxs: list[int]) -> int:
    valid = [i for i in idxs if not np.isnan(corr[i])]
    if not valid:
        return idxs[len(idxs) // 2]
    return min(valid, key=lambda i: corr[i])


def _argmax(corr: np.ndarray, idxs: list[int]) -> int:
    valid = [i for i in idxs if not np.isnan(corr[i])]
    if not valid:
        return idxs[len(idxs) // 2]
    return max(valid, key=lambda i: corr[i])


def _steepest_fall(corr: np.ndarray, idxs: list[int]) -> int:
    best_i, best_delta = idxs[0], np.inf
    for i, j in zip(idxs[:-1], idxs[1:]):
        if np.isnan(corr[i]) or np.isnan(corr[j]):
            continue
        delta = corr[j] - corr[i]
        if delta < best_delta:
            best_delta, best_i = delta, i
    return best_i


def _steepest_rise(corr: np.ndarray, idxs: list[int]) -> int:
    best_i, best_delta = idxs[0], -np.inf
    for i, j in zip(idxs[:-1], idxs[1:]):
        if np.isnan(corr[i]) or np.isnan(corr[j]):
            continue
        delta = corr[j] - corr[i]
        if delta > best_delta:
            best_delta, best_i = delta, i
    return best_i


def _enforce_min_gap(raw: dict[str, int], order: list[str], lo: int, hi: int, min_gap: int) -> dict[str, int]:
    """Assign each candidate its desired position, or the nearest free slot
    (searched outward in both directions) that is >= min_gap from every
    already-placed candidate. A fixed-direction nudge can bounce forever when
    a value is squeezed between two neighbors closer than 2*min_gap apart, so
    this searches outward instead of only forward/backward.
    """
    fixed: list[int] = []
    out: dict[str, int] = {}
    for key in order:
        want = max(lo, min(hi, int(raw[key])))
        if not fixed or all(abs(want - f) >= min_gap for f in fixed):
            out[key] = want
            fixed.append(want)
            continue
        found = None
        for step in range(1, hi - lo + 1):
            for cand in (want - step, want + step):
                if lo <= cand <= hi and all(abs(cand - f) >= min_gap for f in fixed):
                    found = cand
                    break
            if found is not None:
                break
        out[key] = found if found is not None else want
        fixed.append(out[key])
    return out


def generate_boundary_candidates(
    a: np.ndarray,
    b: np.ndarray,
    break_start: int,
    break_end: int,
    seed: int,
    t: int = T,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
    margin: int = MARGIN,
    min_gap: int = MIN_GAP,
) -> dict:
    """Derive 4 start-side (L0-L3) and 4 end-side (R0-R3) boundary candidates
    from a rolling cross-correlation curve, for constrained-selection prompting.

    L0/R0: ground-truth break point plus a seeded jitter (not the exact point).
    L1/R1: first threshold crossing (drop below / recover above) scanning forward in time.
    L2/R2: local minimum of the correlation curve within the search half.
    L3/R3: steepest fall / rise (max |derivative|) within the search half.
    """
    corr = rolling_correlation(a, b, window)
    half = window // 2
    valid_lo, valid_hi = half, t - half - 1

    search_lo = max(valid_lo, break_start - margin)
    search_hi = min(valid_hi, break_end + margin)
    mid = (break_start + break_end) // 2
    mid = max(search_lo + 1, min(search_hi - 1, mid))

    left_idxs = list(range(search_lo, mid))
    right_idxs = list(range(mid, search_hi + 1))
    if len(left_idxs) < 2:
        left_idxs = list(range(search_lo, search_lo + 2))
    if len(right_idxs) < 2:
        right_idxs = list(range(search_hi - 1, search_hi + 1))

    rng = np.random.default_rng(seed)
    offset_start = int(rng.choice([-1, 1])) * int(rng.integers(4, 13))
    offset_end = int(rng.choice([-1, 1])) * int(rng.integers(4, 13))

    l1 = _first_below(corr, left_idxs, threshold)
    if l1 is None:
        l1 = _argmin(corr, left_idxs)
    r1 = _first_above(corr, right_idxs, threshold)
    if r1 is None:
        r1 = _argmax(corr, right_idxs)

    left_raw = {
        "L0": break_start + offset_start,
        "L1": l1,
        "L2": _argmin(corr, left_idxs),
        "L3": _steepest_fall(corr, left_idxs),
    }
    right_raw = {
        "R0": break_end + offset_end,
        "R1": r1,
        # When the trough is a flat plateau (correlation pinned at its floor for a
        # stretch), argmin ties are broken by np.min's "first occurrence" rule.
        # Scanning right_idxs in reverse breaks ties toward the *latest* index, so
        # R2 lands near the recovery edge of the plateau instead of snapping to mid.
        "R2": _argmin(corr, list(reversed(right_idxs))),
        "R3": _steepest_rise(corr, right_idxs),
    }

    left = _enforce_min_gap(left_raw, LEFT_LABELS, search_lo, mid, min_gap)
    right = _enforce_min_gap(right_raw, RIGHT_LABELS, mid, search_hi, min_gap)

    gold_left = min(left, key=lambda k: abs(left[k] - break_start))
    gold_right = min(right, key=lambda k: abs(right[k] - break_end))

    return {
        "left": left,
        "right": right,
        "gold_left": gold_left,
        "gold_right": gold_right,
        "corr": corr,
        "threshold": threshold,
        "search_lo": search_lo,
        "search_hi": search_hi,
        "mid": mid,
    }
