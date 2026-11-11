# Evidence pipeline audit — 2026-11-04

Scope: three SMD entities, reusing SMD_link_v1 scores. No new DINO computation,
no VLM API invocation, no threshold selection using test labels. Current score
generator still lacks the previously discussed forward-calibration/relational
implementation. This audit does not restore or validate those experiments.

## Changes checked

- Preserve review evidence surrounding high events instead of dropping the
  entire merged review interval. High/review parent ranges do not overlap;
  their padded display tiles may overlap intentionally.
- Cover every parent tick with exact 224-tick tiles. No peak-only budget cap.
- Route top eight finite-score channels. This is a heuristic, not FDR control.
  Maximum score ties use mean tile evidence, then channel ID; cutoff ties are
  recorded. This tie-break is an explicit experimental choice, not validated
  as optimal.
- Check train period agreement against both train halves. Retrieval uses only
  pre-focus data; modifying the candidate focus does not change references.
- Store reference starts, context length, period strength and context RMSE in
  image-adjacent .reference.json files. These are diagnostics, not calibrated
  reliability measures.
- Hash renderer source, data, item and prompt in VLM cache keys. Old responses
  cannot be silently reused after a change to those inputs.
- Reject malformed/contradictory JSON and duplicate/unshown channel IDs.
- Full traces and zooms now share asinh display coordinates. Full traces no
  longer silently clip negative departures below -0.35.
- Load DINO lazily once rather than evaluating load_dino() on every setdefault.

## Results on identical stored scores

| Entity | Tiles | Display overlaps GT events | Parent tick coverage | Mean candidates | Event-union channel recall | Cutoff-tie tiles |
|---|---:|---:|---:|---:|---:|---:|
| machine-1-1 | 216 | 8/8 | 100% | 8 | 0.9092 | 42 |
| machine-1-2 | 220 | 10/10 | 100% | 8 | 0.8065 | 18 |
| machine-1-6 | 122 | 30/30 | 100% | 8 | 0.2101 | 55 |

Display overlap requires only one GT tick; it is not event detection recall or
full-GT coverage. Event-union channel recall unions candidate channels from
all displayed tiles overlapping each interpretation event, then averages GT
channel recall across those events (interpretation channel IDs are 1-based).
It is an optimistic channel-routing diagnostic, not final VLM accuracy.

Against maximum-only tie ordering, mean-evidence tie ordering changed this
channel recall from .8966/.7880/.2386 to .9092/.8065/.2101 respectively. Mixed
results: the tie-break does not solve score saturation and must be ablated.
At 38.4% of machine-1-6 scored ticks, eight or more channels share the maximum
channel score (absolute tolerance 1e-6). Candidate ranking remains a blocker.

Five regression checks passed: full tile/review coverage, periodic/noise/flat
controls, focus-value independence, context boundaries, malformed VLM JSON.
Six post-hoc GT-overlap/non-overlap QA images were regenerated successfully.
Test labels were used only for reporting and QA image selection.

## Open research/functional issues

1. Existing empirical calibration reuses training distributions and can
   saturate. No deployment false-alarm guarantee follows from alpha.
2. Large merged false events remain. Full coverage increases call count:
   current 558 tiles vs the earlier 230. Event quality has not improved.
3. Top-eight routing still misses many GT channels in machine-1-6.
4. Reference matching has no held-out train reliability calibration. Median
   may erase spikes, unstable/multiple periods remain possible, and anomalous
   pre-focus context may contaminate retrieval. LIMITED is only a sufficiency
   flag, not a reliability guarantee.
5. Heatmap candidate labels remain crowded in eight-channel pages; retrieval
   metadata needs a readable quality display before publication use.
6. Final event aggregation and uncertain-output policy are not implemented;
   VLM confidence is self-report, not calibrated probability.
7. No new end-to-end SMD/ASD performance claim is supported by this audit.

Outputs: stage1_final/output_statistical/SMD_link_v3_audit and
stage2/results/SMD_link_v3_checked. Keep these separate from earlier runs.
