# Sanity Experiments

Sanity-1 tests whether GPT-4o can visually classify two overlaid synthetic time-series channels as `maintained` or `broken`.

Run from this directory:

```bash
python run_sanity1.py --dry-run
python run_sanity1.py --yes
python run_sanity1.py --resume <run_id> --yes
```

Outputs are written under `results/sanity1/runs/<run_id>/`. Each run keeps its own images, inference logs, raw CSVs, checkpoint, summary JSON, and diagnosis artifacts.

To add Sanity-2, reuse `data_gen.py`, `visualize.py`, `prompts.py`, `vlm_client.py`, `parser.py`, `metrics.py`, and `checkpoint.py`; add a new runner rather than modifying the Sanity-1 contract.
