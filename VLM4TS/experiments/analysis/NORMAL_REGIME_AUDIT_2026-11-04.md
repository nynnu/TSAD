# Normal-regime support audit

## Fixed protocol

Non-overlapping 60-tick windows in SMD machine-1-1, machine-1-2, machine-1-6.
First 60% of train, rounded down to 60 ticks, forms the reference bank and
channel scale. Remaining train calibrates nearest-neighbor distances. No
test label enters retrieval, scaling, probability calculation or cutoffs.
Train reference windows do not overlap the calibration suffix.

Absolute retrieval uses full-waveform RMSE with a prefix-fitted shared channel
scale. Shape retrieval removes each window's mean before RMSE, retaining
amplitude, width and temporal changes. Shape is invariant to constant offsets
but not amplitude multiplication. No DTW, phase shift or period matching is
used here; this intentionally simple diagnostic is separate from HFZ retrieval.
Unlike predictive overlay, this offline audit searches the entire query
window, including its anomaly. It must not be presented as pre-focus prediction.

Train suffix distances define upper-tail empirical p-values. A distance is
flagged novel at p<=.01 and supported at p>.05. Middle values remain undecided.
These cutoffs are predeclared diagnostics, not test-optimized thresholds or
guarantees under temporal dependence/multichannel testing. Calibration has
190/158/158 windows, limiting minimum p to about .0052/.0063/.0063.

## Results

Rates below count channel-window observations, not event recall or F1.
Normal means the entire window has no positive time label. Anomaly means an
interpretation-labelled channel in a window overlapping that event; other
channels in mixed windows are excluded from this comparison.

| Entity | Normal absolute-novel | Normal shape-novel | Anomaly absolute-novel | Anomaly shape-novel |
|---|---:|---:|---:|---:|
| machine-1-1 | 4.42% | 2.13% | 39.59% | 40.17% |
| machine-1-2 | .82% | 1.78% | 19.88% | 15.06% |
| machine-1-6 | 22.20% | 1.42% | 60.22% | 29.05% |

For machine-1-6, 14.49% of normal channel-windows have a novel absolute level
but supported centered shape. However 15.17% of labelled anomaly channel-
windows also do. Automatically accepting a familiar shape would lose true
level-shift anomalies. Both absolute and shape are supported in 67.50% of
normal cells but also 23.72% of anomaly cells. Train support is not a normality
certificate. Both are novel in 1.27% of normal cells vs 28.37% of anomaly cells.

At least one novel channel exists in 86.18% of machine-1-6 normal windows for
absolute retrieval and 33.88% for shape retrieval. Small channel-level rates
must not be reported as small event false-alarm rates.

Shape distance AUROC on these cells: .8119/.8537/.8233. Correlated cells and
unequal channel/event lengths preclude treating these pooled values as a
statistical significance claim or cross-dataset generalization result.

## Examples and interpretation

Figures contain actual train traces rather than medians. The level/shape
mismatch figures additionally show the same train trace shifted by its mean
for visualization, explicitly dashed and labelled. This alignment is never
used to erase the absolute-level novelty score. GT selects examples only.
Inspecting the most novel normal example also finds spike-like behavior absent
from the train bank. It may be unseen normal operation, incomplete labels or
an inappropriate reference; this audit does not establish which.

Evidence supports separating absolute-state novelty from shape novelty.
It does not support suppressing a level shift whenever shape matches, nor
calling every unsupported train state anomalous. The detector default and
HFZ reference routine remain unchanged by this audit.

## Verification and remaining limitations

Four tests pass: shape offset invariance with amplitude sensitivity, suffix
cannot change the reference bank, one query cannot change other queries, and
finite monotone empirical probabilities. Outputs in results/regimes_checked_20261104
were regenerated after adding mismatch diagnostics and actual-reference QA.

Single split, fixed 60-tick alignment, limited bank coverage, no phase alignment,
and univariate channel distances limit the result. Future checks should use
train forward folds and cross-channel relations before routing these two
evidence types into Stage2. No VLM calls or final detector performance claims.
