# Routing repair experiments

## Scope

Three SMD entities, original empirical scores, fixed high/review thresholds.
Labels report results only. Settings below are ablations, not winners chosen
for deployment using test results. No new VLM calls were made.

## 1. Gap merging is not the sole source of giant events

| Entity | gap60 largest high run | gap0 largest high run | strict matches gap60 -> gap0 |
|---|---:|---:|---:|
| machine-1-1 | 12601 | 11487 | 1 -> 2 |
| machine-1-2 | 610 | 233 | 9 -> 9 |
| machine-1-6 | 20229 | 20205 | 4 -> 4 |

Removing gap merging does not split a continuously high score. The prior
description of long events as mostly a merge-gap problem was incomplete.
gap0 also increases high-event counts from61/70/11 to125/164/16 and display
tile counts from216/220/122 to468/589/164. No global change to gap0 or gap14
is adopted. A context tile limit must not be used to manufacture event matches.
Persistent normal-score inflation remains the main boundary blocker in1-6.

## 2. Channel routing and pagination

Same gap60 and tile locations for all candidate-budget experiments. Channel
recall unions candidates across display tiles overlapping each GT event and
macro-averages across interpretation events. It is optimistic, not VLM recall.

| Entity | top8 recall | top16 | top24 | include cutoff-score ties |
|---|---:|---:|---:|---:|
| machine-1-1 | .9092 | .9879 | .9960 | .9960 |
| machine-1-2 | .8065 | .9757 | 1.0000 | .8451 |
| machine-1-6 | .2101 | .6034 | .9013 | .3324 |

More candidates mechanically increase opportunity and cost. top24 triples
the page budget; all38 channels need five pages and trivially reach full
candidate coverage. This does not establish superior final precision.
Cutoff-tie policy retains an average9.69/8.15/11.57 channels and uses285/238/206
pages versus216/220/122 for top8. It remains an explicit experiment.

Functional implementation: Stage1 supports budgets above8 and optional
boundary-ties policy; Stage2 partitions candidates into disjoint pages of8,
never silently truncates them. Page index/count, parent range and cache keys
are retained. Page normal_repeat is explicitly not a whole-event judgment.
No page-level event aggregation is invented by this patch.

Display padding can also redirect ranking to a different spike. Added optional
channel-scope=parent to restrict channel ranking to the parent/tile intersection
while keeping full display context. Synthetic padding interference test passes.
Real event-union recall changes .9092->.9092, .8065->.8532, .2101->.2278;
this is a limited-effect ablation,
not a solution to continuously inflated scores. Default scope remains tile.

## 3. Short impulse experiment

Local contrast uses absolute residual from median filters with halfwidth1/3/7.
Prefix60% train calibrates per-channel/component ranks; suffix40% calibrates
the maximum across channels/scales. Split/boundary margins of7 ticks excluded.
This is retrospective evidence with up to7-tick look-ahead, not zero-latency
causal detection. Empirical p-values under temporal dependence have no exact
deployment false-alarm guarantee. Finite boundary scores are unavailable.

At cut2.0, unioning impulse with baseline high scores reaches8/8,9/10,29/30
GT events, unchanged. Neither of the two score-missed events is newly reached.
Normal tick alarm fractions increase .5414->.5432, .0970->.1022, .8415->.8420.
This feature is not adopted into Stage1. Synthetic recurrent spikes versus
novel heights are distinguishable, but that functional test does not imply
the real missed events are solved.

## Verification and outputs

Five new routing/pulse checks, seven evidence checks and three matching checks
pass (15 total). Baseline merge/routing/pulse ablations reproduce after adding
the optional parent-scope comparison. For machine-1-6, top24 proposals were
generated and a real tile rendered as3 pages. The second page was opened and
inspected; result rows contain24 distinct channels,8 per page. Heatmap labels
now use sparse axis labels plus candidate markers to avoid overlapping text.

Outputs: analysis/results/routing_repairs_checked_20261104/summary.json and
stage2/results/SMD_paged24_audit. The latter is three render-only pages of one
tile; it is not a completed VLM run or an end-to-end result.

## Current decision

Pagination fixes a functional bottleneck and permits fair candidate-budget
experiments. Score-episode segmentation remains unresolved because high
scores persist continuously in normal regimes. Short-spike feature tested
negative. Keep defaults reproducible and do not claim all three research
problems are solved; next model changes must address score specificity and
event boundaries with held-out normal-state evidence.
