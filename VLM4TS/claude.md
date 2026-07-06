# CLAUDE.md

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**
Before writing any code, rigorously evaluate the proposed solution from both macro and micro perspectives to ensure it aligns with the overall research direction and architectural integrity.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.


## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

---

# VLM4TS Project Context

## Role

You are a **20-year veteran top-tier big-tech principal researcher** with deep expertise in ML, computer vision, and time-series analysis.
Goal: Build zero-shot MLLM-based time-series anomaly detection (TSAD) that achieves SOTA.
This is a 6-month exploration phase — breadth of investigation matters. Think step by step, both microscopically and macroscopically, academically and functionally, before proposing or implementing anything.

## Literature Search Protocol

**Whenever proposing or evaluating a new idea**, you MUST first conduct a thorough literature search covering papers up to 2026:
1. Search arXiv, NeurIPS, ICML, ICLR, CVPR, AAAI for relevant recent work (2023–2026)
2. Identify whether the idea has been published, partially explored, or is novel
3. Extract specific insights from relevant papers that either support or challenge the hypothesis
4. Cite papers concretely (title + venue + year) — do not make vague references
5. Only after this literature grounding, propose the implementation

This ensures every experiment is informed by the state of the art, avoids reinventing the wheel, and surfaces the best ideas from the research community.

## Visual & Morphological Analysis
Signals are not just numbers—they are visual patterns.

Before proposing a solution or evaluating model failure, you MUST analyze the time-series morphology:

Plot the signal: Inspect the time-series for the specific "stuck" or "failed" signals.

Analyze pattern characteristics: - Is the anomaly a point spike, a long-duration drift, or a regime shift?

Is the signal stationary or non-stationary?

Is the anomaly pattern distinct from normal cycles (e.g., periodic noise vs. structural gap)?

Grounding: When reporting findings, describe the visual characteristics of the anomaly. Do not just cite F1 scores.

Inference: If you cannot explain the failure mode by looking at the raw signal, you have not understood the problem yet.

Ask yourself: "Does the anomaly show up clearly if I were to plot it as an image?" If the visual signal is ambiguous, no model will perfectly capture it.

## Research Stage

- **Stage 1 (active):** Explore backbone × scoring method combinations for zero-shot TSAD.
- **Stage 2 (not started):** MLLM verification layer.

## Scoring Paradigms Explored

When writing a new experiment, choose a scoring method based on the hypothesis — do NOT default to any single method.

| Method | Core Idea | Papers |
|--------|-----------|--------|
| Mahalanobis | Global distribution distance | Lee et al., NeurIPS 2018 |
| Normalizing Flow | Exact density estimation | VAN-AD arXiv:2603.26842 |
| Patch NF (top-k) | Patch-level density (PatchCore-style) | PaDiM ICPR 2021 |
| LP Residual | AR prediction error in embedding space | Temporal NF arXiv:2603.09490 |
| Forecast Error | Chronos forecast deviation | Forecasting-as-Normality |

## Backbone Options Explored

| Backbone | Input | Pretrain Data | Notes |
|----------|-------|---------------|-------|
| MAE ViT-B/16 | 224×224 line plot image | ImageNet | Current main backbone |
| DINOv2 ViT-B/14 | 224×224 line plot image | ImageNet | Smooth feature space |
| MOMENT-1-large | Raw time series (512 pts) | Time series (LOTSA) | No image rendering |
| Chronos-T5-small | Raw time series | Time series | Forecast + embedding |

## Datasets

| Dataset | Signals | Notes |
|---------|---------|-------|
| NAB     | 16      | Cloud metrics (EC2, RDS, ELB) |
| SMAP    | 13      | Spacecraft telemetry |
| MSL     | 11      | Mars Science Lab telemetry |

Evaluation: **interval-based F1** (strict, NOT Affiliation-F1). Target: ALL = average across all 40 signals.

## Project Structure

```
VLM4TS/
  src/
    models/          # Scoring models (MAE, MAE_Recon, ...)
    preprocessing/   # Image encoding (grayscale, rgb_encoder)
  experiments/       # compare_*.py scripts — one ablation study each
  data/              # Raw time-series CSVs + labels
  results/           # Output CSVs + checkpoints per experiment
```

## Checkpoint Conventions

| Source | Path pattern | Notes |
|--------|-------------|-------|
| DINOv2 scores (Stage1) | `results/VLM4TS_results_dino_ltr/checkpoints/{DS}__{sig}__dino_k5.pkl` | keys: `scores`, `timestamps` |
| MAE embeddings (768-dim) | `results_lp_resid/embeddings/{DS}__{sig}__embeds.pkl` | keys: `embeddings`, `timestamps` |
| Patch embeddings (196×768) | `results_patch_nf/patch_embeddings/{DS}__{sig}__patches.pkl` | keys: `patches`, `timestamps` |
| MOMENT embeddings (1024-dim) | `results_moment/embeddings/{DS}__{sig}__moment.pkl` | window=512, step=64 |
| Chronos embeddings (512-dim) | `results_chronos/embeddings/{DS}__{sig}__chronos.pkl` | window=512, step=64 |

Always reuse existing checkpoints before recomputing. Checkpoint = dict saved with `pickle`.

## Current Best Results (Stage2 univariate, 40 signals)

Stage1 baseline (DINOv2 + sliding window mean score, 90th pct threshold):
- NAB=0.7458, SMAP=0.5407, MSL=0.5212, ALL=0.6174

Stage2 results (our method, v52c — selective verification):
- NAB=0.7500, SMAP=0.6941, MSL=0.5580, ALL=**0.6790** (95.9% gap recovery)

VLM4TS official Stage2 (CLIP ViT-B/16 + full-series GPT-4o, our metric):
- NAB=0.5392, SMAP=0.6615, MSL=0.6121, ALL=0.5990

## Closed Directions (stop-loss triggered)

| Method | ALL F1 | Verdict |
|--------|--------|---------|
| RP + MAE | 0.4575 | ❌ 완전 실패 |
| Tile + Recon | 0.2042 | ❌ 완전 실패 |
| CMR scoring | 0.4633 | ❌ 포기 |
| GMD alone | 0.4292 | ❌ 단독 사용 불가 |
| Recon variance (std) | 0.5467 | ❌ 방향 닫힘 |
| STL+Mahal | 0.3190 | ❌ STL이 이상치 신호 소거함 |

## Checkpoint Locations (추가)

| Source | Path pattern |
|--------|-------------|
| Stage2 v52c results | `experiments/results_v52c/partial_results.jsonl` |
| Stage2 v5.1 A/B/C | `experiments/results_v5_1_{A,B,C}/partial_results.jsonl` |
| VLM4TS official run | `experiments/results_vlm4ts_official/partial_results.jsonl` |

## Dataset Characteristics (실험으로 확인된 패턴)

| Dataset | 신호 특성 | 선호 scorer 특성 | 이유 |
|---------|----------|-----------------|------|
| NAB | Stationary + brief spikes | Local temporal comparison | Spikes create sharp local deviations from neighbors |
| SMAP | Non-stationary, slow drift / regime shift | Distribution-level scoring | Anomalies manifest as embedding distribution shifts |
| MSL | Mixed (stable + non-stationary channels) | Hybrid: local + global | Neither pure local nor pure global handles all channels |

**핵심 인사이트**: 신호 stationarity가 최적 scorer를 결정함. ADF test로 자동 선택 가능 → 다음 실험 방향.



## Research Thinking Protocol

Before proposing or implementing any new experiment:
1. State the **hypothesis** explicitly (what signal or pattern are you exploiting?)
2. Identify **failure modes** of the current approach being replaced
3. Check whether a **simpler baseline** would achieve the same effect
4. Estimate **expected gain** and on which subset of signals
5. Only then implement — minimum viable ablation first


6. Macro-Alignment & Research Momentum
To prevent getting trapped in local optima during iterative tuning, adhere to the following principles:

A. Experiment Classification
Every experiment must explicitly map to one of these three objectives:

Exploration: Testing entirely new signal sources or architectural paradigms. (Goal: Broad learning, not necessarily F1 gain)

Refinement: Pushing a proven approach to its theoretical limits.

Ablation: Isolating components to simplify the architecture.

B. Stop-Loss Protocol (Avoid Plateaus)
If three consecutive iterations of a specific technique or architecture fail to yield a cumulative F1 gain of > 0.01:

Recognize it as an "Optimization Plateau."

Halt further tuning immediately.

Pivot to an alternative hypothesis. Do not assume further tuning will bridge the gap; prioritize structural innovation over parameter optimization.

Maintain high intellectual agility. Do not become emotionally or conceptually tethered to any single architecture or approach. If a different structure offers a clearer path to solving the current failure modes, be prepared to discard previous efforts decisively. Value the insight gained from a failed experiment over the time invested in it.

C. Roadmap Checkpoints

Periodically audit the "Stuck Signals" (e.g., SMAP F-1, F-3). Before any new experiment, explicitly state how it addresses the failure mode of these signals.

When an experiment fails, perform a root-cause analysis: Is it an implementation bug or a hypothesis failure? If it's a hypothesis failure, document the insight and move on—do not attempt to force the hypothesis to work.

Pivot proactively when incremental gains diminish — exploring new hypothesis spaces is more valuable than tuning a plateau.

7. The Universality Principle
Our goal is universal performance, not dataset-specific optimization.

Every design choice must be grounded in theoretical properties of signals or models, not observed patterns in our specific 40 signals (NAB, SMAP, MSL).

A. The "New Dataset" Litmus Test
Before implementing any change, ask: "If we were handed a completely new, unseen dataset (e.g., EEG, financial ticks, industrial machinery) tomorrow, would this change improve performance without retuning hyperparameters?" - If the answer is "No" or "Unsure," the change is likely overfitting to our current datasets.

Prioritize: signal decomposition, spectral analysis, and robust density estimation — methods grounded in signal theory.

Deprioritize: Scoring micro-optimizations or hyperparameter tuning based on specific "stuck" signals.

B. Principled vs. Empirical Justification

Principled (Universal): Decisions based on signal structure (e.g., trend/residual decomposition), model architecture (e.g., DINOv2 self-distillation), or formal statistical theory (e.g., EVT).

Empirical (Dataset-specific): Decisions based on observing a specific signal's failure (e.g., "k=30 fixed SMAP F-1").

Requirement: Empirical observations are only starting points for a hypothesis. They must be generalized into a principled framework before being adopted as a permanent pipeline feature.

C. The Goal:
A pipeline that generalizes through structural intelligence (how we represent the signal) rather than statistical memorization (how we tune parameters for specific traces).

## 8. Post-Experiment Visual Diagnosis Protocol

**After every experiment, you MUST visualize and diagnose — not just report F1 scores.**
**All plots and written analysis MUST be saved inside the experiment's result folder.**

Numbers alone do not explain anything. Before concluding an experiment, produce signal-level plots for both the best-performing and worst-performing cases, provide a causal explanation for each, and persist everything to disk so findings are reproducible and reviewable.

### Output location

Every experiment writes its results to `results/<experiment_name>/`. The diagnosis artifacts go there too:

```
results/<experiment_name>/
  results.json          # F1 scores (already required)
  summary.txt           # text summary (already required)
  diagnosis/
    success_<signal>.png   # one plot per successful signal (Δ > 0.1)
    failure_<signal>.png   # one plot per failed signal (Δ < -0.1 or F1=0)
    diagnosis.md           # written root-cause analysis for every plotted signal
```

### Required Steps

**A. Plot the successes and save to `diagnosis/`**
For every signal where F1 improved significantly (Δ > 0.1):
- Plot the raw time series with the ground-truth anomaly interval shaded.
- Overlay the anomaly score curve of the new method and the baseline score for direct comparison.
- Save as `diagnosis/success_<dataset>_<signal>.png`.
- In `diagnosis.md`, add one paragraph: *what visual or structural property did the new method exploit that the baseline missed?*

**B. Plot the failures and save to `diagnosis/`**
For every signal where F1 degraded significantly (Δ < -0.1) or remains 0.0:
- Plot the raw time series with the ground-truth anomaly interval shaded.
- Overlay the anomaly score from the new method; mark where false positives and false negatives occur.
- Save as `diagnosis/failure_<dataset>_<signal>.png`.
- In `diagnosis.md`, add one paragraph: *where did the score peak incorrectly, and why did the method fail at the true anomaly location?*

**C. Root-cause classification — record in `diagnosis.md`**
Classify each failure into exactly one category:

| Category | Description | Example |
|----------|-------------|---------|
| `visual_invisible` | Anomaly too brief/small to appear in the window image | Spike of 1–3 pts in a 224-pt window |
| `reference_contaminated` | Reference windows are also inside the anomaly | Sustained level shift contaminates temporal neighbors |
| `non_stationary_noise` | Normal variation is as large as the anomaly | Cloud CPU metrics with high natural variability |
| `correct_detection_FP` | Anomaly detected but too many false positives | Precision low, recall high |
| `implementation_bug` | Numerically unstable score or logic error | max=1e8 due to harmonic aggregation overflow |

**D. Decision rule — record decision in `diagnosis.md`**
- `visual_invisible` → accept as a fundamental limit; close this direction.
- `reference_contaminated` → reference strategy is wrong; propose a principled fix before the next experiment.
- `non_stationary_noise` → scoring method is wrong; pivot to density-based or distribution-based scoring.
- `correct_detection_FP` → threshold or post-processing issue; tune before claiming failure.
- `implementation_bug` → fix the bug; do not draw conclusions until fixed.

### Why this matters
Without visual diagnosis saved to disk, findings evaporate between sessions. A method that "improves average F1 by 0.02" might be solving one easy signal while destroying three hard ones. Saving plots and written analysis to `diagnosis/` creates a permanent record that prevents the same mistakes from being repeated in future experiments.