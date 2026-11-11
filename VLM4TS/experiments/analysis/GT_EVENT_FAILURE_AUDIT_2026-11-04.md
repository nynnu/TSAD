# Per-GT event failure audit

## Scope

Current SMD_link_v3_audit proposals and original SMD_link_v1 scores, entities
machine-1-1/1-2/1-6: 48 GT time events. This is not the older ASD experiment or
the forward-calibrated model. ASD data are not present in the current mv_data
folder. Every GT event has an individual JSON row, not only aggregate F1.
No test label changes scores, thresholds or proposals.

## Distinguish failure stages

| Entity | GT events | High score crosses inside GT | High parents overlap GT | Strict one-to-one high matches | GT fully covered by display tiles | All GT channels retained |
|---|---:|---:|---:|---:|---:|---:|
| machine-1-1 | 8 | 8 | 8 | 1 | 8 | 5 |
| machine-1-2 | 10 | 9 | 9 | 9 | 10 | 4 |
| machine-1-6 | 30 | 29 | 29 | 4 | 30 | 1 |

46/48 events cross the high threshold, one crosses review only, one crosses
neither. Yet only 14/48 independently match high parent intervals, largely
because multiple GT events are merged into a single broad prediction.
Full display coverage does not mean the score detected the event: the below-
threshold event is visible incidentally in padding around a nearby candidate.
Similarly, unioning channels across all overlapping tiles is optimistic;
only 10/48 events retain every interpretation-labelled channel even under
that metric. These are distinct errors and require distinct remedies.

33 events have length <=14 ticks; both non-high-crossing events belong to
this group. The other 31 short events cross high, so short duration alone is
not a sufficient cause. There are 4 events of length15..60, 4 of61..224, and
7 above224. All-channel omissions occur in27/33 short,4/4 medium,2/4 of61..224,
and5/7 long events.

## Miss case 1: machine-1-2 [15415,15418), channel17

3-tick GT, single isolated raw spike. Within GT: level surprisal max3.330,
jump3.070, GT-channel score2.974, final event score1.715. High cut2.0 is not
crossed; review cut1.301 is crossed. Display and GT-channel routing succeed.
This is a high-band miss, not a complete Stage2 reachability miss.

The observed evidence supports event aggregation/calibration attenuation of
an already strong local spike. Threshold is the immediate decision cause;
it does not establish that a lower threshold is a general solution. Lowering
to this event's maximum would flag18.34% of label-normal scored ticks.

GT-centred full-query train retrieval p is .00189 at length3, .01320 at14,
.02516 at60. The spike occupies only5% of the60-tick query. This is evidence
of scale-sensitive dilution, not a deployable detector: GT chooses the shortest
window length and query centre in this post-hoc comparison. Train nearest
reference has a smaller spike, so do not call this an established normal repeat.

## Miss case 2: machine-1-6 [2884,2888), channel5

4-tick local spike on a broader gradual waveform. Within GT: level .521,
jump1.894, persistence .298, seasonal .081, DINO spike .829, DINO height1.312;
GT-channel score1.310 and final event score .691. Neither high nor review
crosses. However the event is fully displayed and its channel retained because
another candidate's padded tile includes it. This incidental recovery must
not be credited to event scoring.

A real train reference reproduces the broader gradual waveform, but not the
local spike's full height. GT-centred retrieval p is .04784 at length4,
.08223 at14 and .19497 at60. At60 ticks the GT occupies6.7% of the query.
The evidence supports local weak-spike/scale dilution as a contributing
mechanism. It does not prove that the abnormal spike itself repeats normally.
Lowering the event cut to .691 would flag87.81% of normal scored ticks; this
counterfactual is diagnostic, not a threshold-selection recommendation.

## Evaluation correction

evaluate_tiled_events.match previously greedily assigned largest overlaps.
That can undercount valid one-to-one matches. Replaced with maximum-cardinality
bipartite matching. Three regression tests cover a greedy trap, a broad
prediction that cannot count multiple GTs, and half-open/empty intervals.
Current three-entity counts stay1/9/4; pooled strict high event P=.0986,
R=.2917,F1=.1474. This correction does not improve this model's results.

## Outputs and verification

- audit_gt_event_failures.py outputs all48 event records with thresholds,
  score status, first crossing delay, coverage, missing channels and actual
  train reference locations in results/gt_failures_checked_20261104/events.json.
- summaries separately count score misses, merging failures and channel loss.
- plot_gt_score_misses.py renders raw waveform, component/event scores and
  actual train overlay for the two misses. Both plots were opened and inspected.
- Normal-repeat is not assigned as a proven root cause from one nearest match.
- GT-based multiscale examples are explicitly post-hoc, never model inputs.

## Research decisions

Do not lower a global threshold or suppress all recurring shapes. Investigate
short-scale residual/change evidence against multiple normal-pattern scales,
with train-only selection and controlled false alarms. Separately fix event
boundaries and channel eligibility, since they account for many more current
failures than two high-score misses. No new detector variant or VLM result is
claimed by this audit; failure taxonomy is now concrete and per-event.
