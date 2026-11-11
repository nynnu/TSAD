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
    return med, np.maximum(mad,1e-6)


def sparse_scale_candidate(x):
    """Experimental scale; not enabled in the detector.

    ECDF ranks of pointwise deviations are invariant in exact arithmetic, but
    changing scale before cumulative trailing means perturbs ties numerically.
    """
    med = np.nanmedian(x,axis=-1,keepdims=True)
    mad = 1.4826*np.nanmedian(np.abs(x-med),axis=-1,keepdims=True)
    # Zero MAD is common for sparse/binary channels. Avoid a misleading
    # million-fold raw z-score. Full-pipeline invariance must be checked.
    std=np.nanstd(x,axis=-1,keepdims=True)
    fallback=np.where(std>1e-6,std,1.)
    return med, np.where(mad>1e-6,mad,fallback)


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


def combine_components(train_components, test_components, components=None):
    """Return calibrated channel and event surprisals plus component evidence."""
    names=tuple(COMPONENTS if components is None else components)
    if not names: raise ValueError('At least one component is required')
    train_s, test_s = {}, {}
    for name in names:
        stride = P if name.startswith("dino_") else 1
        train_s[name], test_s[name] = empirical_surprisal(train_components[name], test_components[name], stride)
    tr_stack = np.stack([train_s[n] for n in names])
    te_stack = np.stack([test_s[n] for n in names])
    tr_raw = np.nanmax(tr_stack, axis=0)
    te_raw = np.nanmax(te_stack, axis=0)
    tr_channel, te_channel = empirical_surprisal(tr_raw, te_raw)
    # A time is surprising only if the maximum evidence across channels is
    # surprising compared with maxima observed at normal training times.
    tr_event_raw = np.nanmax(tr_channel, axis=0, keepdims=True)
    te_event_raw = np.nanmax(te_channel, axis=0, keepdims=True)
    tr_event, te_event = empirical_surprisal(tr_event_raw, te_event_raw)
    return train_s, test_s, tr_channel, te_channel, tr_event[0], te_event[0]


def tail_routing_score(train_components, test_components, calibrated_components):
    """Continuous evidence for routing ONLY; this is not a p-value.

    ECDF gives identical probabilities to every value above the train maximum.
    Preserve that probability for detection, but retain the amount of excess
    for channel review. Scale is fitted on train; no test quantile is used.
    """
    routing=[]
    for name in COMPONENTS:
        tr=np.asarray(train_components[name],float)
        te=np.asarray(test_components[name],float)
        score=np.array(calibrated_components[name],copy=True)
        for c in range(len(tr)):
            ref=tr[c,np.isfinite(tr[c])]
            if len(ref)<20: continue
            q90,q99=np.quantile(ref,[.90,.99]); maximum=np.max(ref)
            spread=max(float(q99-q90),float(np.std(ref)),1e-6)
            excess=np.maximum(0.,te[c]-maximum)/spread
            score[c]+=np.log10(1.+excess)
        routing.append(score)
    return np.nanmax(np.stack(routing),axis=0)


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
    routing_s=tail_routing_score(tr_components,te_components,te_component_s)
    intervals, mask = make_intervals(event_s, channel_s, te_component_s, alpha, channel_alpha, merge_gap, rank_top_k)
    out.mkdir(parents=True, exist_ok=True); score_dir = out / "scores"; score_dir.mkdir(exist_ok=True)
    np.savez_compressed(score_dir / f"{entity}.npz", event_score=event_s.astype(np.float32),
                        channel_rank_score=routing_s.astype(np.float32),
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


def link_tiles(event, channel, parts, args, ranking=None):
    """Build fixed-width evidence tiles; all coordinates are test-global."""
    high = merge_close(segs(np.isfinite(event) & (event >= -np.log10(args.alpha))), args.merge_gap)
    broad = merge_close(segs(np.isfinite(event) & (event >= -np.log10(args.review_alpha))), args.merge_gap)
    # Preserve weak evidence before/after a high event inside the same merged
    # review interval. Dropping the entire broad interval discarded it.
    review=[]
    for a,b in broad:
        remaining=[(a,b)]
        for c,d in high:
            next_remaining=[]
            for u,v in remaining:
                if not (u<d and c<v): next_remaining.append((u,v)); continue
                if u<c: next_remaining.append((u,c))
                if d<v: next_remaining.append((d,v))
            remaining=next_remaining
        review.extend(remaining)
    tiles = []
    for band, events in (("high", high), ("review", review)):
        for a,b in events:
            # Cover the complete parent interval. A peak-only cap silently
            # discarded most evidence in long events. The last tile is moved
            # to the event end, allowing overlap but never an uncovered gap.
            if b-a <= args.tile_length:
                peak=int(a+np.nanargmax(event[a:b]))
                lower=max(0,b-args.tile_length)
                upper=min(a,len(event)-args.tile_length)
                starts=[int(np.clip(peak-args.tile_length//2,lower,upper))]
            else:
                starts=list(range(a,b-args.tile_length+1,args.tile_length))
                if starts[-1] != b-args.tile_length: starts.append(b-args.tile_length)
            for rank,start in enumerate(starts,1):
                end=start+args.tile_length
                left=max(a,start); right=min(b,end)
                peak=int(left+np.nanargmax(event[left:right]))
                rows=[]
                rank_start,rank_end=(left,right) if getattr(args,'channel_scope','tile')=='parent' else (start,end)
                for c in range(channel.shape[0]):
                    v=channel[c,rank_start:rank_end]
                    if not np.isfinite(v).any(): continue
                    route=(ranking if ranking is not None else channel)[c,rank_start:rank_end]
                    tick=rank_start+int(np.nanargmax(route)); score=float(channel[c,tick])
                    name, detail=component_at(parts,c,tick)
                    rows.append(dict(channel=c,zoom_center=tick,dominant_component=name,
                                     evidence_score=score,mean_evidence=float(np.nanmean(v)),
                                     routing_score=float(route[tick-rank_start]),
                                     tile_p=min(1.,len(v)*10.**(-score)),component_scores=detail))
                rows.sort(key=lambda r:(-r['routing_score'],-r['mean_evidence'],r['channel']))
                # Evidence routing, not a calibrated hypothesis test. Keep
                # the strongest channels for VLM review without claiming FDR.
                count=min(args.max_channels,len(rows))
                policy=getattr(args,'candidate_policy','topk')
                if policy=='boundary-ties' and count:
                    boundary=rows[count-1]['routing_score']
                    while count<len(rows) and np.isclose(rows[count]['routing_score'],boundary,rtol=0,atol=1e-6):count+=1
                chosen=[dict(r,selection='score_topk' if policy=='topk' else 'boundary_ties') for r in rows[:count]]
                tiles.append(dict(start=int(start),end=int(end),band=band,parent_start=int(a),parent_end=int(b),
                                  tile_peak=peak,tile_rank=rank,tile_count=len(starts),channels=chosen,
                                  candidate_policy=policy,eligible_channels=len(rows),
                                  channel_ranking_start=int(rank_start),channel_ranking_end=int(rank_end),
                                  omitted_channels=[r['channel'] for r in rows[count:]],
                                  boundary_score_tie=bool(len(rows)>args.max_channels and
                                      np.isclose(rows[args.max_channels-1]['evidence_score'],rows[args.max_channels]['evidence_score'])),
                                  root_cause_candidates=rows[:args.rank_top_k]))
    return tiles,[dict(start=int(a),end=int(b),band='high') for a,b in high],[dict(start=int(a),end=int(b),band='review') for a,b in review]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "output_statistical")
    ap.add_argument("--cache", type=Path, default=HERE / "cache_statistical")
    ap.add_argument("--machines", nargs="*")
    ap.add_argument("--alpha", type=float, default=0.01, help="train-calibrated event false-alarm level")
    ap.add_argument("--channel-alpha", type=float, default=0.10)
    ap.add_argument("--review-alpha", type=float, default=0.05)
    ap.add_argument("--channel-fdr", type=float, default=0.05)
    ap.add_argument("--tile-length", type=int, choices=[224], default=224)
    ap.add_argument("--max-tiles-per-event", type=int, default=3)
    ap.add_argument("--max-channels", type=int, default=8)
    ap.add_argument('--candidate-policy',choices=['topk','boundary-ties'],default='topk')
    ap.add_argument('--channel-scope',choices=['tile','parent'],default='tile',help='Candidate ranking span; parent excludes display-only padding')
    ap.add_argument("--scores-only", type=Path, help="Rebuild proposals from saved score NPZ files without DINO recomputation")
    ap.add_argument('--ranking-mode',choices=['empirical','tail-routing'],default='empirical')
    ap.add_argument("--merge-gap", type=int, default=60)
    ap.add_argument("--rank-top-k", type=int, default=10)
    args = ap.parse_args()
    if not args.alpha <= args.review_alpha < 1 or not 0 < args.channel_fdr < 1:
        raise ValueError('Require alpha <= review-alpha < 1 and 0 < channel-fdr < 1')
    if args.max_tiles_per_event < 1 or args.max_channels < 1 or args.merge_gap<0:
        raise ValueError('Require positive tile/channel counts and a nonnegative merge gap')
    if not 0 < args.alpha < 1 or not 0 < args.channel_alpha < 1:
        raise ValueError("--alpha and --channel-alpha must be in (0, 1)")
    holder = {}
    def dino_fn():
        if 'model' not in holder: holder['model']=load_dino()
        return holder['model']
    machines = args.machines or sorted(p.stem for p in (args.data / "test").glob("*.txt"))
    for entity in machines:
        if args.scores_only:
            r=dict(machine=entity, config={'score_source':str(args.scores_only)})
            score_path=args.scores_only / (entity+'.npz')
        else:
            r = run_machine(entity, args.data, args.out, args.cache, args.alpha, args.channel_alpha,
                            args.merge_gap, args.rank_top_k, dino_fn)
            score_path=args.out / 'scores' / (entity+'.npz')
        with np.load(score_path) as scores:
            event=scores['event_score']; channel=scores['channel_score']
            if len(event)<224: raise ValueError('At least 224 scored ticks required')
            parts={n:scores['component_'+n] for n in COMPONENTS}
            if args.ranking_mode=='tail-routing' and 'channel_rank_score' not in scores:
                raise ValueError('tail-routing requires newly computed scores with channel_rank_score')
            ranking=scores['channel_rank_score'] if args.ranking_mode=='tail-routing' else None
            r['intervals'],r['detected_events'],r['review_events']=link_tiles(event,channel,parts,args,ranking)
        r['schema_version']='stage2-evidence-tiles-v2-coverage-topk'
        r['config'].update(high_alpha=args.alpha,review_alpha=args.review_alpha,channel_fdr=args.channel_fdr)
        r['config']['ranking_mode']=args.ranking_mode
        r['config'].update(candidate_policy=args.candidate_policy,max_channels=args.max_channels,
                           merge_gap=args.merge_gap,evidence_tile_length=args.tile_length,channel_scope=args.channel_scope)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/(entity+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"{entity}: intervals={len(r['intervals'])}, candidates="
              f"{np.mean([len(i['channels']) for i in r['intervals']] or [0]):.1f} per interval", flush=True)


if __name__ == "__main__":
    main()
