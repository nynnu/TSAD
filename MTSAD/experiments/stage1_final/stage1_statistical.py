"""Train-calibrated statistical Stage 1.

This is intentionally separate from ``stage1.py``.  The older script is kept
unchanged so that its published fixed-percent results remain reproducible.

The detector has two outputs:
  * event p-values: when should Stage 2 be called?
  * channel evidence rankings: which channels deserve inspection in that event?

All calibration distributions are constructed from the training split only.
No test-set percentile, median, or MAD is used to decide a threshold.

Components
----------
level        absolute departure from the train median
jump         departure of x[t]-x[t-1] from normal train changes
persistence  trailing mean of level departure (no future look-ahead)
seasonal     trailing residual from a train-estimated periodic profile
dino_spike   DINO spike-direction score
dino_height  DINO height score

Each component is converted to an empirical upper-tail p-value against its
own training distribution.  Components are combined per channel and then
across channels, with a second training-only empirical calibration at each
level.  ``--alpha`` is therefore a false-alarm *level*, not "top x% of test".
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from stage1 import (P, ROLL_WIN, WIN, ZOOM_HALF, dino_scores, drop_flat,
                    load_dino, period, segs)

HERE = Path(__file__).resolve().parent
COMPONENTS = ("level", "jump", "persistence", "seasonal", "dino_spike", "dino_height")


def merge_close(intervals, gap):
    """Merge nearby score runs without depending on a legacy Stage1 helper."""
    out = []
    for a, b in intervals:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def causal_mean(x, window):
    """Trailing mean for a 1-D or 2-D (channel, time) array.

    ``uniform_filter1d`` used in the legacy score is symmetric and therefore
    includes future observations.  A deployed detector only receives history.
    """
    x = np.asarray(x, dtype=float)
    cs = np.cumsum(x, axis=-1)
    out = np.empty_like(x, dtype=float)
    n = x.shape[-1]
    for t in range(n):
        left = max(0, t - window + 1)
        total = cs[..., t] - (cs[..., left - 1] if left else 0.0)
        out[..., t] = total / (t - left + 1)
    return out


def robust_scale(x):
    med = np.nanmedian(x, axis=-1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=-1, keepdims=True)
    return med, np.maximum(mad, 1e-6)


def level_component(train, test):
    med, scale = robust_scale(train.T)
    return np.abs((train.T - med) / scale), np.abs((test.T - med) / scale)


def jump_component(train, test):
    tr_jump = np.abs(np.diff(train, axis=0, prepend=train[:1])).T
    # The first test point must be compared with the last observation available
    # at deployment time, rather than silently resetting the difference to zero.
    te_jump = np.abs(np.diff(np.vstack([train[-1:], test]), axis=0)).T
    med, scale = robust_scale(tr_jump)
    return np.abs((tr_jump - med) / scale), np.abs((te_jump - med) / scale)


def seasonal_component(train, test):
    """Causal periodic-residual magnitude using a profile fitted on train only."""
    C = train.shape[1]
    tr_out = np.zeros((C, len(train)), dtype=float)
    te_out = np.zeros((C, len(test)), dtype=float)
    for c in range(C):
        p = period(train[:, c])
        if p is None or np.std(train[:, c]) < 1e-6:
            continue
        prof = np.array([train[k::p, c].mean() for k in range(p)])
        tr_resid = np.abs(train[:, c] - prof[np.arange(len(train)) % p])
        te_resid = np.abs(test[:, c] - prof[np.arange(len(test)) % p])
        tr_out[c] = causal_mean(tr_resid, ROLL_WIN)
        te_out[c] = causal_mean(te_resid, ROLL_WIN)
    return tr_out, te_out


def dino_component(train, test, cache, entity, dino_fn):
    """Load or compute DINO evidence for train and test independently."""
    cache.mkdir(parents=True, exist_ok=True)
    ans = []
    for split, values, ref in (("train", train, train), ("test", test, train)):
        path = cache / f"{entity}_dino_{split}.npz"
        if path.exists():
            d = np.load(path); spike, height = d["spike"], d["height"]
        else:
            # DINO rendering requires full 224-tick images.  Keeping only the
            # scored prefix is explicit rather than mixing an unscored tail in.
            usable = (len(values) // WIN) * WIN
            spike, height = dino_scores(dino_fn(), ref, values[:usable])
            np.savez_compressed(path, spike=spike, height=height)
        ans.append((spike.astype(float), height.astype(float)))
    (tr_spike, tr_height), (te_spike, te_height) = ans
    return tr_spike, tr_height, te_spike, te_height


def empirical_surprisal(reference, values, stride=1):
    """Upper-tail surprisal -log10(p) from a train-only empirical CDF.

    Both reference and values are (channels, time).  DINO values are repeated
    inside a 14-tick column, so their reference is sampled once per column.
    """
    C, T = values.shape
    out = np.full((C, T), np.nan, dtype=float)
    ref_score = np.full_like(reference, np.nan, dtype=float)
    for c in range(C):
        r = reference[c, ::stride]
        r = r[np.isfinite(r)]
        if len(r) < 20:
            continue
        r = np.sort(r)
        for target, dest in ((reference[c], ref_score[c]), (values[c], out[c])):
            ok = np.isfinite(target)
            # P(R >= x) with add-one smoothing.  This prevents infinite scores
            # and makes scores comparable across entities with different length.
            ge = len(r) - np.searchsorted(r, target[ok], side="left")
            p = (ge + 1.0) / (len(r) + 1.0)
            dest[ok] = -np.log10(p)
    return ref_score, out


def combine_components(train_components, test_components):
    """Return calibrated channel and event surprisals plus component evidence."""
    train_s, test_s = {}, {}
    for name in COMPONENTS:
        stride = P if name.startswith("dino_") else 1
        train_s[name], test_s[name] = empirical_surprisal(train_components[name], test_components[name], stride)
    tr_stack = np.stack([train_s[n] for n in COMPONENTS])
    te_stack = np.stack([test_s[n] for n in COMPONENTS])
    tr_raw = np.nanmax(tr_stack, axis=0)
    te_raw = np.nanmax(te_stack, axis=0)
    tr_channel, te_channel = empirical_surprisal(tr_raw, te_raw)
    # A time is surprising only if the maximum evidence across channels is
    # surprising compared with maxima observed at normal training times.
    tr_event_raw = np.nanmax(tr_channel, axis=0, keepdims=True)
    te_event_raw = np.nanmax(te_channel, axis=0, keepdims=True)
    tr_event, te_event = empirical_surprisal(tr_event_raw, te_event_raw)
    return train_s, test_s, tr_channel, te_channel, tr_event[0], te_event[0]


def component_at(component_scores, channel, tick):
    pairs = [(name, float(component_scores[name][channel, tick])) for name in COMPONENTS
             if np.isfinite(component_scores[name][channel, tick])]
    if not pairs:
        return "unknown", {}
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[0][0], {name: round(value, 3) for name, value in pairs}


def make_intervals(event_surprisal, channel_surprisal, component_scores, alpha, channel_alpha, merge_gap, rank_top_k):
    event_cut = -np.log10(alpha)
    channel_cut = -np.log10(channel_alpha)
    mask = np.isfinite(event_surprisal) & (event_surprisal >= event_cut)
    intervals = []
    for a, b in merge_close(segs(mask), gap=merge_gap):
        ranked = []
        for c in range(channel_surprisal.shape[0]):
            v = channel_surprisal[c, a:b]
            if not np.isfinite(v).any():
                continue
            local = int(np.nanargmax(v)); tick = a + local
            dominant, parts = component_at(component_scores, c, tick)
            ranked.append(dict(channel=int(c), evidence_score=round(float(v[local]), 3), peak_tick=int(tick),
                               support_fraction=round(float(np.nanmean(v >= channel_cut)), 3),
                               dominant_component=dominant, component_scores=parts))
        # Evidence ranking, not causal proof.  Support is only a deterministic
        # tie-breaker, so downstream papers can inspect exactly what it means.
        ranked.sort(key=lambda r: (-r["evidence_score"], -r["support_fraction"], r["channel"]))
        channels = []
        for r in ranked:
            if r["evidence_score"] >= channel_cut:
                channels.append(dict(channel=r["channel"], source=r["dominant_component"],
                                     spike_score=r["evidence_score"], deseason_score=None,
                                     zoom_center=r["peak_tick"],
                                     zoom=[max(0, r["peak_tick"] - ZOOM_HALF), r["peak_tick"] + ZOOM_HALF],
                                     evidence_score=r["evidence_score"], support_fraction=r["support_fraction"],
                                     dominant_component=r["dominant_component"]))
        intervals.append(dict(start=int(a), end=int(b), channels=channels,
                              root_cause_candidates=ranked[:rank_top_k]))
    return intervals, mask


def run_machine(entity, data, out, cache, alpha, channel_alpha, merge_gap, rank_top_k, dino_fn):
    train = np.loadtxt(data / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(data / "test" / f"{entity}.txt", delimiter=",")
    tr_spike, tr_height, te_spike, te_height = dino_component(train, test, cache, entity, dino_fn)
    # Every component must describe the same test prefix.  DINO currently scores
    # complete 224-tick windows, so the explicit truncation avoids a hidden tail.
    T = min(len(test), te_spike.shape[1]); test = test[:T]
    train_for_dino = train[:tr_spike.shape[1]]
    tr_level, te_level = level_component(train_for_dino, test)
    tr_jump, te_jump = jump_component(train_for_dino, test)
    tr_persist, te_persist = causal_mean(tr_level, ROLL_WIN), causal_mean(te_level, ROLL_WIN)
    tr_season, te_season = seasonal_component(train_for_dino, test)
    tr_components = dict(level=tr_level, jump=tr_jump, persistence=tr_persist, seasonal=tr_season,
                         dino_spike=drop_flat(tr_spike, train_for_dino, train_for_dino),
                         dino_height=drop_flat(tr_height, train_for_dino, train_for_dino))
    te_components = dict(level=te_level, jump=te_jump, persistence=te_persist, seasonal=te_season,
                         dino_spike=drop_flat(te_spike[:, :T], test, train_for_dino),
                         dino_height=drop_flat(te_height[:, :T], test, train_for_dino))
    _, te_component_s, _, channel_s, _, event_s = combine_components(tr_components, te_components)
    intervals, mask = make_intervals(event_s, channel_s, te_component_s, alpha, channel_alpha, merge_gap, rank_top_k)
    out.mkdir(parents=True, exist_ok=True); score_dir = out / "scores"; score_dir.mkdir(exist_ok=True)
    np.savez_compressed(score_dir / f"{entity}.npz", event_score=event_s.astype(np.float32),
                        event_p=np.power(10.0, -event_s).astype(np.float32), channel_score=channel_s.astype(np.float32),
                        channel_p=np.power(10.0, -channel_s).astype(np.float32),
                        **{f"component_{n}": te_component_s[n].astype(np.float32) for n in COMPONENTS})
    result = dict(machine=entity, length=int(T), n_channels=int(test.shape[1]),
                  config=dict(calibration="train empirical upper-tail p-values", alpha=alpha,
                              channel_alpha=channel_alpha, merge_gap=merge_gap,
                              components=list(COMPONENTS), root_cause_note="evidence ranking, not causal proof"),
                  intervals=intervals)
    (out / f"{entity}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "output_statistical")
    ap.add_argument("--cache", type=Path, default=HERE / "cache_statistical")
    ap.add_argument("--machines", nargs="*")
    ap.add_argument("--alpha", type=float, default=0.01, help="train-calibrated event false-alarm level")
    ap.add_argument("--channel-alpha", type=float, default=0.10)
    ap.add_argument("--merge-gap", type=int, default=60)
    ap.add_argument("--rank-top-k", type=int, default=10)
    args = ap.parse_args()
    if not 0 < args.alpha < 1 or not 0 < args.channel_alpha < 1:
        raise ValueError("--alpha and --channel-alpha must be in (0, 1)")
    holder = {}
    dino_fn = lambda: holder.setdefault("model", load_dino())
    machines = args.machines or sorted(p.stem for p in (args.data / "test").glob("*.txt"))
    for entity in machines:
        r = run_machine(entity, args.data, args.out, args.cache, args.alpha, args.channel_alpha,
                        args.merge_gap, args.rank_top_k, dino_fn)
        print(f"{entity}: intervals={len(r['intervals'])}, candidates="
              f"{np.mean([len(i['channels']) for i in r['intervals']] or [0]):.1f} per interval", flush=True)


if __name__ == "__main__":
    main()
