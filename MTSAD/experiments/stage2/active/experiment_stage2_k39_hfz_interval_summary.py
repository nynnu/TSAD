"""K39: controlled K35 HFZ ablation with one neutral, DINO-aligned interval summary per channel."""
from pathlib import Path

import numpy as np

import experiment_stage2_k38_hfz_index_aware_text as k38

k38.OUT_DIR = Path(__file__).resolve().parents[2] / "results_stage2_k39_hfz_interval_summary"
k38.OUT_DIR.mkdir(parents=True, exist_ok=True)
k38.LAYOUT_NAME = "HFZ-S"
k38.TEXT_INTRO = ("Neutral local summary for the same DINO-selected zoom interval. Values use the "
                  "same train-normalized scale as the plot; this is supporting context, not a verdict.")


def _longest_run(mask):
    best = run = 0
    for hit in mask:
        run = run + 1 if hit else 0
        best = max(best, run)
    return best


def interval_summary(page, window, train, ranges, intervals):
    lines = []
    for c in page:
        start, end = intervals[c]
        lo, hi = ranges[c]
        values = (window[start:end, c] - lo) / (hi - lo + 1e-8)
        above, below = values > 1.0, values < 0.0
        upper, lower = max(float(values.max() - 1.0), 0.0), max(float(-values.min()), 0.0)
        signed_peak = upper if upper >= lower else -lower
        outside = above | below
        lines.append(
            f"ch {c} | DINO focus [{start}, {end}) | outside training display range: "
            f"{int(outside.sum())}/{len(values)} ticks (longest run { _longest_run(outside) }); "
            f"signed range exceedance {signed_peak:+.3f}"
        )
    return "\n".join(lines)


k38.index_text = interval_summary


if __name__ == "__main__":
    k38.parser = None
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true", help="inspect fixed K4 candidates; no VLM calls")
    parser.add_argument("--run", action="store_true", help="render HFZ with interval summaries and call the VLM")
    parser.add_argument("--smoke", action="store_true", help="use first 12 paired segments")
    parser.add_argument("--confirm", action="store_true", help="use held-out 36 segments")
    args = parser.parse_args()
    if args.smoke and args.confirm:
        parser.error("choose only one of --smoke or --confirm")
    k38.run(execute=args.run, limit=12 if args.smoke else None,
            start_index=12 if args.confirm else 0,
            summary_name="summary_holdout36.json" if args.confirm else None)
