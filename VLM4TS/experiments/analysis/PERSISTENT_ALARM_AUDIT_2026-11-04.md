# Persistent normal-alarm mechanism audit

## Scope and causal limits

Same three SMD entities and cached raw DINO evidence as previous audits.
GT labels are used only for reporting/plots. No test-label threshold or window
selection. Local-centering widths61/225 are fixed ablations, not selected
winners. All comparisons use cut2.0; the initial225 ticks are excluded from
both normal/anomaly point-rate denominators to avoid cross-split initialization.
GT event reachability/matching is reported separately on the scored prefix.
Strict matching uses gap60 and maximum-cardinality one-to-one matching.

## Mechanism traced

machine-1-6 has a continuously high score on[3315,23520), without gap merging.
This run contains16520 evaluated label-normal ticks and3685 anomaly ticks.
It cannot be explained by a60-tick smoothing tail or merge gap.

Train median / label-normal test median on dominant channels:

| Channel | Train median | Normal test median | Train q99 |
|---|---:|---:|---:|
| ch25 | .554522 | .221189 | .620672 |
| ch22 | .171457 | .315412 | .287305 |
| ch15 | .006572 | .079255 | .026328 |
| ch14 | .030107 | .143993 | .114658 |
| ch13 | .011992 | .113319 | .028624 |

ch25's normal high-score samples are below the train median99.95% of the time.
This is not merely exaggerated upward spikes: the absolute departure score
also treats a persistent lower operating level as abnormal. Other channels
move upward, so a one-channel anomaly filter cannot explain the full episode.
Persistence averages ongoing departure, not just the initial transition.
Seasonal residuals also inherit the level mismatch; prior removal experiments
show one component can replace another. ECDF saturation then removes ordering.

The data establish a level/distribution mismatch relative to supplied labels.
They do not identify its physical cause or rule out preprocessing/label issues.

## Controlled representation experiment

Subtract a trailing median from both train and test statistical branches,
recompute level/jump/persistence/seasonal, and re-fit all empirical calibrations.
The original DINO branches are held fixed to isolate the statistical-path
change. This is an experimental hybrid, not a fully re-rendered/retrained
visual pipeline. Trailing median includes current and past samples only.
Full-train reference scores remain in-sample; alpha has no demonstrated
deployment false-alarm guarantee.

| Entity | Normal alarm: static -> center61 -> center225 | Anomaly tick hit: static -> center61 -> center225 |
|---|---:|---:|
| machine-1-1 |54.62% ->16.08% ->21.40% |99.22% ->94.36% ->97.03% |
| machine-1-2 |9.50% ->4.72% ->5.13% |37.27% ->35.98% ->32.10% |
| machine-1-6 |85.12% ->18.51% ->23.50% |99.78% ->91.64% ->89.05% |

Longest continuous run in1-6:20205 ->2042/2208 ticks. Strict matches:4 ->10/10
out of30 GTs. But per-event macro tick coverage drops .9444 ->.5810/.5989.
Event reachability stays29/30, masking substantial loss within events.
machine-1-2 loses the entire GT[15540,15605) under both adaptive variants:
high-event reachability9/10 ->8/10. No new GT events are recovered.

Static-only evidence under center225 occurs in63.04% of normal ticks and
10.73% of anomaly ticks in1-6. These cannot be automatically classified normal
or removed from candidate review. Exported evidence_state is diagnostic only.

## Counterfactual controls

Statistical components only, train prefix60% defines normal reference, normal
train suffix is the query. Adding a constant vector of one prefix std to the
same suffix waveform changes alarm fraction from46.39% to78.94% in1-1,
43.99% to100% in1-2,33.65% to100% in1-6. The large unaltered-suffix rates are
already evidence of poor temporal generalization; the injection tests offset
sensitivity, not whether such an offset is semantically normal.

Inject a1200-tick shift of3 prefix std into the suffix. Static statistics flag
100% of the shift after300 ticks. Center225 flags only20.33%,57.67%,2.11%
respectively after300 ticks. A sustained fault with this shape would be
absorbed. This directly rejects universal adaptive centering as a safe fix.

## Checks and artifacts

Three new tests pass: causal no-future behavior, constant-offset equivariance,
and deliberate demonstration that centering erases a sustained shift. Entire
three-entity score arrays reproduce exactly in two runs for all three variants.
Saved per-GT coverage and lost/recovered event lists avoid count-only claims.
Timeline plots were inspected; a separate trailing alarm-occupancy plot avoids
overlapping score traces hiding saturation. Plot smoothing is reporting only.

Scripts: audit_persistent_alarms.py, summarize_persistent_alarms.py,
test_persistent_alarms.py. Results: results/persistent_alarms_checked_20261104,
including summary_enriched.json, NPZ branch scores/evidence states and plots.
No VLM calls or production detector default changes.

## Research decision

Supported mechanism: persistent absolute-state mismatch is a major source of
normal alarms, with redundant smoothed components and ECDF saturation.
The representation intervention reduces alarm occupancy substantially but
also removes genuine anomaly evidence. Keep absolute-state evidence and
local-change evidence separate; neither can determine event validity alone.
Do not erase the former whenever the latter is low. Next validation must
address supported normal-state transitions, adaptation contamination, anomaly
duration/coverage and actual channel selection on held-out entities/ASD.
The present three-entity test results do not certify a final adaptive model.
