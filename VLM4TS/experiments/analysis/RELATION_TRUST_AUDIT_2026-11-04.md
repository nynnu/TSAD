# Forward relation-trust audit

## Decision contract

Each channel/60-tick window exports `relation_supported`, `relation_deviation`,
or `uncertain`, plus a reason. These are association-evidence states, never
normal/anomalous event labels. Uncertain candidates are retained for review.
Even relation_supported does not authorize deleting an anomaly candidate.

## Train-only protocol

Expanding history endpoints at approximately 60%,73.3%,86.7% of full normal
train; evaluate the next disjoint block through 100%. Every model, correlation,
scaler, penalty choice and residual calibration uses only history before its
audit block. No fit uses that block's observations. A channel is forward-stable
when prediction MSE beats a historical-mean baseline by >10% in at least two
of three future blocks. Missing models count as non-passes.

The relation family uses the previously fixed fit_relation_models procedure:
first 60% of each history fits the model, remaining 40% calibrates residuals;
inner prefix split selects hyperparameters. Final test models use the same
recipe on full train, not a refit on all training samples. Outer folds validate
this training procedure, not the exact final coefficients. The two-of-three
and >10% rules are predeclared exploratory settings, not statistical guarantees.

Predictor-state support uses only neighboring channels: window mean, std and
mean absolute change. A prefix-normalized nearest-neighbor feature distance
is calibrated on the suffix train. Support p>.05 is required; target values
are excluded from this test. Joint feature support is stronger than a simple
per-channel min/max check but is still limited by this feature representation.

Eligible evidence requires final model availability, forward stability, and
predictor support. Residual p<=.01 => relation_deviation; p>.05 =>
relation_supported. Intermediate residuals and failed gates => uncertain.
Unavailable model, unstable forward skill, unsupported predictor state and
ambiguous residual are distinguished in exported reasons.

Test labels only summarize already computed states. Same three entities and
60-tick windows as earlier audits. Cell rates below are not event metrics.

## Results: denominators include all assessed channel-windows

| Entity | Forward stable channels | Normal uncertain | Anomaly uncertain | Normal relation deviation | Anomaly relation deviation |
|---|---:|---:|---:|---:|---:|
| machine-1-1 | 23/38 | 51.95% | 81.22% | .39% | .15% |
| machine-1-2 | 24/38 | 39.65% | 68.67% | .09% | 3.01% |
| machine-1-6 | 24/38 | 74.60% | 87.42% | 3.58% | 6.90% |

Normal eligible coverage is 51.09%,61.86%,32.97%; anomaly eligible coverage is
19.65%,40.96%,19.48%. Lower deviation rates result substantially from abstention,
not demonstrated improvements in anomaly detection. Among eligible cells,
machine-1-6 has 10.87% normal residual novelty vs 35.44% anomaly novelty.

Relation-supported fractions among actual anomaly cells are 18.63%,28.31%,
5.67%. Treating support as a normal label would therefore discard true anomaly
evidence. Coherent multichannel anomalies can preserve normal relations.

## Previously ambiguous novel-level / familiar-shape subset

machine-1-6: normal uncertainty 97.25% (1674 cells), anomaly uncertainty 85.59%
(222 cells). Only 133 normal and 41 anomaly cells pass eligibility gates;
conditional deviation rates are 27.07% vs 75.61%. These small selected subsets
cannot support a broad claim that normal level changes are solved.
Normal unsupported-predictor reason accounts for 1182/1674 cells; anomaly
unsupported-predictor reason accounts for 179/222. machine-1-1's entire
corresponding subset remains uncertain. Current linear model coverage is
insufficient precisely in the difficult cases.

## Validation

Four new tests pass: forward boundary ordering/unavailable channels,
unseen predictor states remain uncertain, test cannot alter forward skill or
fitted model metadata, predictor-support score excludes the target channel.
Three relation-model and four regime-retrieval regression tests also pass.
Deterministic rerun checked separately; state counts add to each group total.

Artifacts: results/relation_trust_checked_20261104/summary.json, per-entity NPZ
(states/reasons/p-values/predictions/eligibility), trust_coverage.png. No VLM
API calls or changes to Stage1 default detection/Stage2 event aggregation.

## Interpretation and remaining issues

The new functional behavior is explicit abstention, not a validated final
anomaly classifier. Predictor support and target residual have distinct roles;
using target residual to define trust would hide real relation anomalies.
Input outside support may itself be anomaly evidence and must not be erased.

Current remaining limitations: only three entities, no ASD, one window scale,
linear contemporaneous dependencies, no lag/nonlinear regime models, uncertain
handling not yet connected to final event aggregation. Nested train splits and
normal-state changes reduce effective data. Feature-supported is not guaranteed
in-distribution, and temporal dependence limits empirical p-value guarantees.
Next assess conditional performance versus coverage across train-fixed gate
settings and alternative relation families, preserving all uncertain candidates.
