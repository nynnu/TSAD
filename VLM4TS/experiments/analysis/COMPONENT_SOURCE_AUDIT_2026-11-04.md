# machine-1-6 component-source audit

Uses cached raw DINO scores and current empirical model. Test labels only
report normal/anomaly tick metrics and select post-hoc QA plots. All subset
ablations re-fit every empirical calibration on the same train data; simply
removing an already calibrated test score would not be a fair ablation.

## Findings

At component surprisal >=2, label-normal channel-time high-score rates are
level 24.38%, persistence 29.46%, seasonal 29.45%; anomaly rates are 42.54%,
46.51%, 46.82%. The components overlap and must not be added as independent
evidence. Among high component maxima on normal channel-time cells,
persistence ties for the maximum in 78.45% and seasonal in 66.77%; these shares
overlap. An argmax-only table would hide this redundancy.

ch25 normal high-score fractions: level 83.86%, persistence 84.54%, seasonal
85.18%. ch22: 70.59%, 76.56%, 75.77%. ch11: level only 2.26%, but persistence
70.52% and seasonal 70.51%, showing smoothed temporal behavior can dominate
even without frequently extreme individual values.

## Recalibrated removal experiments

Fixed alpha=.01, no test threshold tuning. Rates below are tick rates before
60-tick event merging; they are not strict event precision/recall.

| Components removed | Normal tick alarm rate | Anomaly tick hit rate |
|---|---:|---:|
| none | 84.15% | 99.78% |
| level | 84.06% | 99.78% |
| persistence | 84.16% | 99.78% |
| seasonal | 83.91% | 99.49% |
| level + persistence | 84.07% | 99.78% |
| level + persistence + seasonal | 24.07% | 56.93% |

One-at-a-time removal is insufficient because redundant components retain
the same abnormality signal. Removing all three loses substantial true
anomaly evidence and is not an acceptable default fix.
Normal ticks farther than 60 ticks after any GT anomaly still have an alarm
rate of 83.77%; post-anomaly smoothing alone does not explain the failure.

## Seasonal validity

Full-train phase profiles explain only 4.62%, 6.85%, 5.01%, 3.33% of train
variance for ch25/ch22/ch14/ch11. A detected FFT peak is not evidence that
the phase profile meaningfully predicts future normal data.
Fit period/profile on first 70% of train, evaluate last 30% against the prefix
mean: 13 of 31 tested channels have nonpositive MSE improvement. Ch25 gain
is -15.37%, ch22 -14.30%; ch14 +1.77%, ch11 +3.39% (small improvements).

A train-only experimental gate sets seasonal evidence to zero on the 13
nonpositive-gain channels and recalibrates everything. Normal tick alarms
remain 84.15%, anomaly hits 99.78%; normal channel-time high scores decrease
only from 27.97% to 27.85%. Other components replace the discarded evidence.
The gate is therefore retained only in the audit, not enabled in the detector.
This single split is a diagnostic, not robust forward-validation evidence.

## Checks and artifacts

- Optional component subset API added without changing default behavior.
- All saved baseline score arrays exactly reproduce the earlier baseline.
- Subset ablations checked not to mutate input arrays or the full baseline.
- Seven prior regression checks pass.
- QA selection corrected: a normal anchor tick does not imply a normal whole
  window. The normal plot now uses a verified GT-free 224-tick window.
- Outputs: results/component_sources_final_20261104/component_sources.json,
  normal_components.png and anomaly_components.png. Red shading is GT used
  for reporting only; figures are not VLM inputs.

## Decision

The dominant problem is shared sensitivity to train/test level and temporal
distribution mismatch, compounded by score saturation. Unreliable seasonality
is a real contributing issue but cannot explain the detector failure alone.
No component is disabled by default. Further work must compare train-normal
regime matching and temporal-change evidence while measuring anomaly loss,
using the same candidate protocol across entities. Do not normalize away a
persistent level shift automatically: it may itself be a genuine anomaly.
