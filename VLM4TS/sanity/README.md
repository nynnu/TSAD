# Sanity Experiments

Sanity-1 tests whether GPT-4o can visually classify two overlaid synthetic time-series channels as `maintained` or `broken`.
Sanity-3 tests whether GPT-4o can also localize the time-step interval where the relationship breaks (free-form coordinates).
Sanity-4 tests whether GPT-4o can verify a highlighted candidate interval as `valid` or `invalid`.
Sanity-5 tests constrained boundary *selection*: instead of free-form coordinates, GPT-4o picks the break start/end
from 4+4 pre-computed candidate time steps (`L0-L3`, `R0-R3`) derived from a rolling cross-correlation curve.
Primary evaluation set is C1/C2/C3 (cases Sanity-1 showed are reliably detected as broken); C4/C5 are boundary
cases run for reference only and excluded from pass/fail, per the Sanity-1 verdicts.
Sanity-6 tests multi-channel scaling: N channels (2/4/8), grouped into N/2 synchronized pairs, are overlaid in
one image and GPT-4o judges every pair's maintained/broken status in a single call. Scenarios vary both channel
count and how many pairs are broken at once (0 / 1 / multi), to find where per-image accuracy starts degrading
and whether simultaneous breaks get missed due to attention narrowing onto the most salient pair.

Run from this directory:

```bash
python run_sanity1.py --dry-run
python run_sanity1.py --yes
python run_sanity1.py --resume <run_id> --yes
python run_sanity3.py --dry-run
python run_sanity3.py --yes
python run_sanity4.py --dry-run
python run_sanity4.py --yes
python run_sanity5.py --dry-run
python run_sanity5.py --yes
python run_sanity6.py --dry-run
python run_sanity6.py --yes
```

Outputs are written under `results/sanity1/runs/<run_id>/`. Each run keeps its own images, inference logs, raw CSVs, checkpoint, summary JSON, and diagnosis artifacts.

New sanity experiments reuse `data_gen.py`, `visualize.py`, `prompts.py`, `vlm_client.py`, `parser.py`, `metrics.py`, and `checkpoint.py`; add a new runner rather than modifying the existing experiment contracts. Sanity-5 additionally introduces `boundary_candidates.py` (rolling-correlation candidate generation) and `sanity5_parser.py`, following the same pattern as the `sanity3_parser.py` / `sanity4_parser.py` per-experiment parsers. Sanity-6 introduces `multichannel_data_gen.py` (composes N/2 independent Sanity-1-style pairs into one scene) and `sanity6_parser.py` (parses one maintained/broken verdict per pair from a single JSON response).

Both Sanity-5 and Sanity-6 pick up `OPENAI_API_KEY` from a `.env` file in this directory (via `python-dotenv`, loaded in `vlm_client.py`) if the environment variable isn't already set.
