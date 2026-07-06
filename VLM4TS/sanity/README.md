# Sanity Experiments

Sanity-1 tests whether GPT-4o can visually classify two overlaid synthetic time-series channels as `maintained` or `broken`.
Sanity-3 tests whether GPT-4o can also localize the time-step interval where the relationship breaks.
Sanity-4 tests whether GPT-4o can verify a highlighted candidate interval as `valid` or `invalid`.

Run from this directory:

```bash
python run_sanity1.py --dry-run
python run_sanity1.py --yes
python run_sanity1.py --resume <run_id> --yes
python run_sanity3.py --dry-run
python run_sanity3.py --yes
python run_sanity4.py --dry-run
python run_sanity4.py --yes
```

Outputs are written under `results/sanity1/runs/<run_id>/`. Each run keeps its own images, inference logs, raw CSVs, checkpoint, summary JSON, and diagnosis artifacts.

New sanity experiments reuse `data_gen.py`, `visualize.py`, `prompts.py`, `vlm_client.py`, `parser.py`, `metrics.py`, and `checkpoint.py`; add a new runner rather than modifying the existing experiment contracts.
