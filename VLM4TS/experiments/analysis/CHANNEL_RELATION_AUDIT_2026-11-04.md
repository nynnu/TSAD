# Channel relation audit

## Protocol

Same disjoint 60-tick test windows and interpretation-label masks as the
normal-regime audit. No test label enters fit, neighbor selection, penalty
selection, availability gating, residual probabilities or extrapolation flags.

First 60% of train is the model prefix; remaining train calibrates residual
window RMS. Within the prefix, first 70% fits standardized correlations and
ridge coefficients; last 30% selects penalty among .01/.1/1 and measures
prediction gain against the prefix-fit mean. Up to five other channels with
absolute correlation >=.3 are selected. Self prediction is prohibited.
Models with validation gain <=.1 or no usable neighbors are unavailable.
Accepted models refit on the full prefix. These train validation choices are
fixed exploratory settings; selecting penalty and reporting skill on the same
inner segment can be optimistic and needs nested forward evaluation later.

Window RMS residuals are calibrated against train suffix RMS. p<=.01 is novel,
p>.05 supported, the middle interval undecided. Unavailable models produce
NaN, never automatic normal decisions. Predictor extrapolation means any
neighbor value in the window exceeds that neighbor's prefix min/max. Passing
this marginal-range check does not establish joint-distribution support.

## Results

All rates are conditional on available models, and count channel-windows.
They are not event recall/F1 or causal evidence.

| Entity | Models available | Normal residual-novel | Anomaly residual-novel | Normal predictor extrapolation | Anomaly predictor extrapolation |
|---|---:|---:|---:|---:|---:|
| machine-1-1 | 24/38 | 5.99% | 46.14% | 21.02% | 63.47% |
| machine-1-2 | 27/38 | .28% | 13.79% | 1.50% | 35.17% |
| machine-1-6 | 27/38 | 18.42% | 55.62% | 34.94% | 57.62% |

Available anomaly-cell coverage is 77.29%,87.35%,92.41% respectively; normal
coverage is 63.16%,71.05%,71.05%. Comparing rates without these denominators
would hide channels for which no train relation model works.

## Focus: novel level but supported shape

machine-1-6 has 1674 normal and 222 labelled anomaly cells in this subset.
Models cover 1474 normal and 220 anomaly cells. Residual novelty is 36.84%
versus 75.45%, respectively. Residual support is 31.21% versus 8.64%.
Thus relation evidence helps differentiate some ambiguous level changes but
does not solve the task. Requiring residual novelty would discard 24.55% of
the available anomaly cells in this subset; this is not an acceptable default
without a stated tradeoff and event-level evaluation.

Predictor extrapolation is 65.20% for available normal cells and 74.55% for
available anomaly cells in the subset. Restricting to predictors within their
train marginal ranges still gives residual novelty of 40.16% versus 80.36%.
The remaining normal mismatch shows extrapolation alone is not the explanation;
normal relation regimes, nonlinearities, lags and label uncertainty remain.

machine-1-1 gives an opposing warning: in the same ambiguous subset, 85.47%
of available normal cells have novel residuals. Only three anomaly cells have
available models there, too few to generalize. machine-1-2 has no labelled
anomaly cells in this subset. No universal relation gate is supported.

## Functional validation and visualization

Three new regression checks passed: coherent multi-channel shifts versus a
broken relation, test values cannot alter fitted models, unavailable constant
channels are not marked normal. Four prior regime checks also pass.
Model metadata includes neighbors, chosen penalties, validation gain and
coefficients. Outputs preserve residuals, predictions, p-values and marginal
extrapolation flags in relations_20261104/*.npz and summary.json.

Actual/predicted plots were rendered for all entities. machine-1-6 was opened
and inspected: both a GT-normal ch25 case and an anomalous ch14 case show
large errors while predictors extrapolate. A large orange/black separation
alone cannot distinguish the two. Plots are post-hoc QA, not VLM inputs.

## Decision

Keep relational evidence as a separate diagnostic; do not overwrite the
detector or certify a normal/anomaly event from residuals alone. Linear
contemporaneous models are a limited association baseline. They do not prove
root cause, cover delayed dependencies, or handle every normal operating
regime. Multiple anomalies can affect predictor channels, causing residuals
in downstream or otherwise normal channels.

The defensible next experiment is train-forward evaluation of model skill and
operating-regime support, with absolute/shape/relation evidence jointly
reported and unavailable/OOD cases retained as uncertain. No VLM API calls,
final aggregation changes or new end-to-end performance claims were made.
