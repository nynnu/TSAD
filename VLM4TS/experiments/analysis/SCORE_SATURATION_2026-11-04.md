# Stage1 score saturation audit — 2026-11-04

## Scope and reproduction

Three SMD entities, identical raw DINO train/test caches, six current score
components. `audit_score_saturation.py` prohibits DINO recomputation if caches
are absent. Test labels are read only after score construction for reporting.
Current source is the empirical calibration model, not the earlier proposed
forward-calibrated relational model.

Audit outputs: `analysis/results/saturation_20261104/`.
Experimental proposals: `stage1_final/output_statistical/SMD_saturation_routing/`.
Baseline proposals: `stage1_final/output_statistical/SMD_link_v3_audit/`.

## Mechanism confirmed

For n finite training observations the empirical upper-tail probability is
`(count(train >= x)+1)/(n+1)`. Every x above the train maximum receives the
same probability, even when raw magnitudes differ. This is finite-sample
resolution, not floating-point overflow or a GPU issue. Subsequent channel
and event ECDF transformations cannot recover differences lost upstream.

The audit traces all six component transformations, channel recalibration and
event recalibration. Distinct test raw values collapse in all six components.
For channel/event recalibration, above-train-maximum inputs have already
collapsed upstream; the audit finds no distinct excess values at those stages.

| Entity | Persistence values above train max | Seasonal values above train max | Event recalibration inputs above train max | >=8-way channel max ties |
|---|---:|---:|---:|---:|
| machine-1-1 | 5.07% | 5.78% | 57.36% | 9.72% |
| machine-1-2 | 0.30% | 0.50% | 9.60% | 0.17% |
| machine-1-6 | 18.59% | 19.10% | 86.54% | 38.43% |

Component rates count finite channel-time observations. Event rates count
scored ticks. These are not false-alarm probabilities.

## Controlled experimental change

Detection scores and thresholds remain byte-for-byte numerically identical.
Only routing can use the additional `channel_rank_score` via the explicit
`--ranking-mode tail-routing` option. Default remains `empirical`.

For each component, routing adds `log10(1 + positive_excess / train_scale)` to
the component empirical surprisal, where positive_excess is raw value minus
the train maximum and train_scale is max(q99-q90, std, 1e-6). Components are
combined by maximum for routing. This score is an experimental heuristic,
not a p-value or a tail probability estimator. Scale sensitivity, sparse and
constant train channels, and component dominance need separate validation.

The direct component routing path also avoids channel-level ECDF rounding;
therefore this experiment combines two changes and cannot attribute its
effect to excess magnitude alone. A follow-up ablation must separate them.

## Results, identical events/tiles/candidate budget

| Entity | >=8-way ties: empirical → routing | Event-union channel recall | Event-union channel precision |
|---|---:|---:|---:|
| machine-1-1 | 9.72% → 0% | .9092 → .8824 | .5543 → .5505 |
| machine-1-2 | .17% → 0% | .8065 → .8070 | .4677 → .4698 |
| machine-1-6 | 38.43% → 0% | .2101 → .2430 | .2290 → .2415 |

These channel metrics union candidates from all display tiles overlapping
each interpretation event and macro-average across events. They do not
measure VLM accuracy, event F1, or per-tick localization. Detection events,
review events, tile coordinates and the candidate count of eight are checked
unchanged; original event/channel score arrays equal rebuilt arrays exactly.

In machine-1-6, >=8-way ties occur at 85.36% of GT anomaly ticks and 29.65%
of normal ticks. Saturation removes useful amplitude ordering, but excess
alone is not sufficient to distinguish anomalies from normal regime changes.
No claim that these normal changes are periodic follows from this audit.

## Verification and decision

Six regression checks passed, including a constructed case where empirical
probabilities tie above the train maximum while routing preserves ordering.
The original empirical probabilities are checked unchanged.

Do not promote tail-routing as the final model: results are mixed and
machine-1-6 candidate recall remains low. No new VLM API calls were made.
Next isolate direct component aggregation from tail excess, inspect normal
vs anomaly channels with zero-MAD/sparse train distributions, and reconstruct
the intended forward-calibration model in a separately versioned experiment.
Event score saturation is diagnosed but intentionally not repaired using an
unvalidated extrapolated probability.
