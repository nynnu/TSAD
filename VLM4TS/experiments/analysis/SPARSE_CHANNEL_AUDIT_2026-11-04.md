                                                                              # Sparse-channel and routing-ablation audit

## Protocol

Three SMD entities, cached DINO evidence, unchanged empirical score baseline.
Train-defined groups: constant (no deviation from median), sparse (at most
1% active ticks), zero-MAD (remaining channels with MAD below 1e-6), variable.
Activity tolerance is max(1e-8, train range * 1e-6). These are diagnostic groups,
not learned decision thresholds. Test GT is used only for reporting.
High score means calibrated channel surprisal >=2; this is a reporting cutoff,
not a claim of 1% deployment error control. Each group rate averages channel
rates. Label-normal includes all ticks without a GT anomaly label and may
contain unlabeled changes.

## Normal-tick high-score rates

| Entity | Constant | Sparse | Other zero-MAD | Variable |
|---|---:|---:|---:|---:|
| machine-1-1 | 0.34% (8 ch) | absent | 1.93% (2 ch) | 9.37% (28 ch) |
| machine-1-2 | 0% (5 ch) | 1.47% (6 ch) | 6.55% (5 ch) | 1.44% (22 ch) |
| machine-1-6 | 0.058% (5 ch) | 0.80% (2 ch) | 1.73% (2 ch) | 36.46% (29 ch) |

Sparse/constant channels are not the primary source of machine-1-6 score
inflation. For machine-1-2, some zero-MAD channels do merit inspection.
Do not suppress all sparse channels: their anomaly-tick high-score rate is
15.19% in machine-1-2 versus 1.47% on label-normal ticks.

Examples (zero-based IDs): machine-1-6 ch25 is high-scoring on 84.9% of normal
ticks; 77.7% of its normal values lie outside the train value range. Ch22:
73.1% high score, 47.4% outside range. These support a train/test distribution
mismatch diagnosis but do not establish whether GT is complete or which
operating change occurred. Ch14 is high-scoring on 70.4% of normal ticks even
though none of those raw values is outside the train range; support-frequency,
temporal-pattern and component analysis are needed, not just range checks.

## Factor separation

All modes use exactly the same tiles, candidate budget=8, and channel peak
positions determined by the baseline empirical score. Keeping peak positions
fixed removes the previous experiment's peak-localization confound.

A: channel ECDF baseline. B: direct maximum of component ECDF scores.
C: baseline primary order, tail-routing score used only to resolve ties.
D: direct component plus tail-excess routing score.
All tie rules are deterministic; no mode is selected using test metrics.

| Event-union channel recall | A | B | C | D |
|---|---:|---:|---:|---:|
| machine-1-1 | .9092 | .9092 | .8962 | .8962 |
| machine-1-2 | .8065 | .8070 | .7951 | .7856 |
| machine-1-6 | .2101 | .2089 | .2658 | .2647 |

These metrics union candidates in all tiles overlapping each interpretation
event. They are optimistic routing diagnostics, not event F1 or VLM results.
For machine-1-6, C versus A attributes the improvement to resolving saturation
ties using excess evidence, while B alone does not help. Across entities the
effect is mixed; none is promoted as the final policy.
Full precision and channel-level records: results/sparse_20261104/sparse_ablation.json.

## Scale modification tested and rejected for default use

An experimental MAD-zero fallback used train std (or 1 for a constant channel).
Pointwise ECDF invariance passed a synthetic check. Full-pipeline replay then
found changes in persistence scores because the cumulative-sum trailing mean
and exact ECDF tie comparisons are numerically scale-sensitive. For machine-1-2
the maximum channel-score difference was .9545, and event-score difference
.4590. This change is not neutral and was reverted from robust_scale.
The candidate remains an explicitly unused sparse_scale_candidate function.
It must not be described as a validated detector improvement.

Seven regression checks cover sparse pointwise scaling, tail ordering,
tile/review coverage, periodic controls, focus leakage, context boundaries,
and malformed VLM responses. Full replay after reverting the scale is checked
separately against the saved baseline arrays.

## Next unresolved cause

Machine-1-6 needs per-component analysis of normal operating-state mismatch,
and numerical stability of persistence tie handling needs a separately tested
implementation. Blindly reducing scores on sparse channels would target the
wrong dominant group. Event threshold calibration and final aggregation are
outside this experiment; no new VLM calls were made.
