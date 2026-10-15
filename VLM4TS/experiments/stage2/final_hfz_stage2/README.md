# Final Stage-2 HFZ pipeline

This folder is the clean entry point and specification for the final Stage-2
channel-diagnosis system.  The tested implementation remains in `../active/`
so the experiment history is preserved; `run_stage2_hfz.py` delegates to that
canonical implementation.

## Final design

```text
Provided anomaly interval
        │  make a common 224-tick context window
        ▼
Stage 1: adaptive DINO patch-kNN candidate screening
        │  all variable-count candidates passing train-only hysteresis
        ▼
Stage 2: HFZ evidence rendering
        │  all-channel heatmap + full trace + DINO-column 60-tick zoom
        ▼
VLM: validate candidates and output the anomalous/related channel subset
```

Stage 1 uses a train-only empirical-tail DINO score.  Channels with `p < 0.01`
are strong seeds; channels with `p < 0.1` are retained only when their absolute
training correlation with a seed exceeds `0.5`.  There is no fixed top-k:
every selected candidate is sent to Stage 2, in pages of at most eight channels.

The DINO column score is a localizer, not the final anomaly classifier.  It
centers a 60-tick zoom on the most novel image-time column, while the VLM sees
the full 224-tick trace to judge persistence and system context.

## Reproduce

Run all 48 SMD diagnosis segments:

```bash
python VLM4TS/experiments/stage2/final_hfz_stage2/run_stage2_hfz.py --run
```

Render one Stage-2 input image without a VLM call:

```bash
python VLM4TS/experiments/stage2/final_hfz_stage2/run_stage2_hfz.py \
  --render-example --example-index 3 --render-layout HFZ
```

## Final evaluation (48 segments)

| Metric | Value |
|---|---:|
| Stage-1 candidate precision / recall / F1 | 0.3949 / 0.7169 / 0.4460 |
| Final Stage-2 macro precision / recall / F1 | 0.6074 / 0.5818 / **0.5486** |
| HFZ improvement over HZ | +0.0805, paired Wilcoxon p=0.0082 |

The K40 normal-reference-band overlay was tested as a controlled ablation and
was not retained: HFZR=0.5472 versus HFZ=0.5486 (paired p=1.000).

## Canonical implementation files

- `../active/experiment_stage2_k35_evidence_contract.py`: HFZ renderer,
  VLM interface, paging, and evaluation.
- `../active/experiment_stage2_k4_adaptive.py`: train-only candidate screening
  and DINO calibration.
- `../../stage1/active/colab_multivariate_v2.py`: DINO image/embedding helpers.
