# HFZ Stage 2: comparative evaluation record

This folder records the reproducible Stage 2 decision as of the current SMD
evaluation.  The final method uses train-only calibrated Stage 1 candidate
screening and the HFZ evidence page: a whole-channel heatmap, a full 224-tick
trace for each candidate channel, and a 60-tick zoom centred on a DINOv2
column-evidence peak.

## Confirmed final comparison (48 SMD segments)

| Layout | Macro F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| HZ (heatmap + zoom) | 0.4681 | — | — |
| **HFZ (final)** | **0.5486** | **0.6074** | **0.5818** |
| HFZR (HFZ + normal-reference overlay) | 0.5472 | — | — |

HFZ improves over HZ by +0.0805 macro F1 (paired Wilcoxon p=0.0082).  Adding
the reference overlay did not improve the same protocol (HFZR versus HFZ,
p=1.0), so HFZ is retained as the final page.

The Stage 1 screen for the same 48 segments has candidate-channel F1=0.4460,
precision=0.3949, and recall=0.7169.  Stage 2 therefore functions as a
precision-oriented visual adjudicator rather than an all-channel renderer.

## Supporting and rejected alternatives

The accompanying scripts and compact output folders preserve the tested
alternatives.  They are evidence records, not new final claims.

* K32 renders all valid channels; it was inferior to selected-channel input
  (reported macro F1 0.4017), supporting Stage 1 channel screening.
* K33 overlays a retrieved normal reference band.  Its 12-segment smoke result
  was directionally interesting but was not a sufficiently matched final
  evaluation.
* K37 (joint local evidence), K38 (index-aware text), and K39 (interval
  summary) were smoke experiments.  None has a later, matched 48-segment
  result exceeding HFZ.

The original cached feature tensors and large intermediate artifacts are
intentionally excluded.  All files in this bundle are source code, compact
summaries/checkpoints, or rendered examples required to inspect the decision.
