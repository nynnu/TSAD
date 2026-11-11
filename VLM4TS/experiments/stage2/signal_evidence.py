"""Signal-derived temporal evidence, separate from anomaly validity."""
import numpy as np


def signal_span(values, start, end, context=28, recovery=60):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not 0 <= start < end <= len(values):
        raise ValueError("Invalid signal or candidate bounds")
    if start < 8 or not np.isfinite(values[max(0,start-context):min(len(values),end+recovery)]).all():
        return dict(status="insufficient_context", interval=[start,end])
    baseline = float(np.median(values[max(0,start-context):start]))
    deviation = np.abs(values-baseline)
    peak = start + int(np.argmax(deviation[start:end]))
    amplitude = float(deviation[peak])
    noise = float(np.median(np.abs(values[max(0,start-context):start]-baseline)))
    # Reject uninformative local peaks; this is a display-evidence gate.
    if amplitude <= max(1e-12, 3 * noise):
        return dict(status="no_distinct_local_peak", interval=[start,end])
    threshold = amplitude * .5
    left = peak
    lower = max(0,start-context)
    while left > lower and deviation[left-1] >= threshold:
        left -= 1
    right = peak+1
    upper = min(len(values),end+recovery)
    while right < upper and deviation[right] >= threshold:
        right += 1
    recovered = right < upper and np.all(deviation[right:min(right+3,upper)] < threshold)
    bounded = left > lower and recovered
    active = deviation[lower:upper] >= threshold
    transitions = np.diff(np.r_[False, active, False].astype(int))
    components = [[int(a+lower), int(b+lower)] for a,b in
                  zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1))
                  if max(start,a+lower) < min(end,b+lower)]
    all_bounded = bounded and all(a>lower and b<upper for a,b in components)
    return dict(intervals=components if all_bounded else [[start,end]],
                status="bounded_peak" if bounded else "unresolved_boundary",
                interval=[left,right] if bounded else [start,end],
                half_height_span=[left,right],peak=peak,baseline=baseline,
                amplitude=amplitude,recovered=bool(recovered),
                meaning="half-height shape span; not GT duration or anomaly decision")


def excursion_end(values, start, end, context=60, k=3.0, quiet=30, horizon=1000, floor=0.01, band="mad"):
    """First tick after which a channel stays back at its pre-candidate level, or None.

    Baseline and noise come only from the `context` ticks before the candidate (local,
    so a train/test level mismatch cannot trigger it). The excursion ends where the signal
    stays inside the band for `quiet` consecutive ticks. band="mad": median +- k*MAD noise.
    band="range": the pre-candidate min..max widened by half its width, so a channel whose
    ordinary behaviour is a recurring spike train is not "away" all the time. None means: no
    distinct excursion inside the candidate, or no return within `horizon` (a level shift
    that never comes back is not extended). Extent evidence only, not an anomaly decision.
    """
    values = np.asarray(values, dtype=float)
    if start < context:
        return None
    before = values[start-context:start]
    after = values[start:start+horizon+quiet]
    if band == "range":
        margin = max(0.5*np.ptp(before), floor)
        away = (after < before.min()-margin) | (after > before.max()+margin)
    else:
        baseline = np.median(before)
        away = np.abs(after-baseline) > k*max(1.4826*np.median(np.abs(before-baseline)), floor)
    if not away[:end-start].any():
        return None
    first = int(np.argmax(away))
    calm = np.convolve(~away, np.ones(quiet, dtype=int), "valid") == quiet  # calm[i]: ticks i..i+quiet-1 all back
    later = np.flatnonzero(calm[first:])
    return start+first+int(later[0]) if len(later) else None


def demo():
    x = np.zeros(600); x[100:300] = 1.0                      # bounded excursion returns at 300
    assert excursion_end(x, 100, 114) == 300
    x = np.zeros(600); x[100:103] = 1.0                      # spike: ends where it ends
    assert excursion_end(x, 100, 114) == 103
    x = np.zeros(3000); x[100:] = 1.0                        # level shift never returns
    assert excursion_end(x, 100, 114) is None
    x = np.zeros(600); x[100:300:2] = 1.0                    # oscillating burst is one excursion
    assert excursion_end(x, 100, 114) == 299
    assert excursion_end(np.zeros(600), 100, 114) is None    # nothing happens in the candidate
    x = np.zeros(900); x[::20] = 1.0; x[100:300] += 3.0      # normal spike train + bounded excursion
    assert excursion_end(x, 100, 114) is None                # MAD band: spikes never look calm
    assert excursion_end(x, 100, 114, band="range") == 300


if __name__ == "__main__":
    demo(); print("ok")
